#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynthRetro-P 预训练模型实现 - 基于多任务预训练的半模板逆合成模型

这是一个新增的文件，应该放在 model/ 目录下，命名为 synthretro_pretrain.py
实现设计方案中的多任务预训练架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class GMPN(nn.Module):
    """
    图消息传递网络 - 共享编码器
    完全沿用G2Retro的核心组件
    """
    
    def __init__(self, hidden_size=256, embed_size=32, depth=10):
        super(GMPN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.depth = depth
        
        # 原子嵌入层
        self.atom_embedding = nn.Embedding(200, embed_size)  # 支持多种原子类型
        
        # 键嵌入层
        self.bond_embedding = nn.Embedding(10, embed_size)   # 支持多种键类型
        
        # 消息传递层
        self.message_layers = nn.ModuleList([
            nn.Linear(embed_size * 2, hidden_size) for _ in range(depth)
        ])
        
        # 更新层
        self.update_layers = nn.ModuleList([
            nn.GRU(hidden_size, embed_size, batch_first=True) for _ in range(depth)
        ])
        
        # 全局池化层
        self.global_pool = nn.Linear(embed_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, atom_features, bond_features, adjacency_matrix, batch_indices=None):
        """
        前向传播
        
        Args:
            atom_features: [num_atoms, atom_feature_dim] 原子特征
            bond_features: [num_bonds, bond_feature_dim] 键特征  
            adjacency_matrix: [num_atoms, num_atoms] 邻接矩阵
            batch_indices: [num_atoms] 原子属于哪个分子的标识
            
        Returns:
            atom_embeddings: [num_atoms, embed_size] 原子嵌入
            bond_embeddings: [num_bonds, embed_size] 键嵌入
            graph_embeddings: [batch_size, hidden_size] 图嵌入
        """
        # 原子特征嵌入
        atom_embeddings = self.atom_embedding(atom_features)  # [num_atoms, embed_size]
        
        # 消息传递
        for layer_idx in range(self.depth):
            # 收集邻居消息
            messages = []
            for atom_idx in range(atom_embeddings.size(0)):
                neighbors = torch.nonzero(adjacency_matrix[atom_idx]).squeeze(-1)
                if len(neighbors) > 0:
                    neighbor_embeddings = atom_embeddings[neighbors]
                    atom_neighbor_concat = torch.cat([
                        atom_embeddings[atom_idx].unsqueeze(0).repeat(len(neighbors), 1),
                        neighbor_embeddings
                    ], dim=-1)
                    message = self.message_layers[layer_idx](atom_neighbor_concat)
                    message = torch.mean(message, dim=0)  # 聚合邻居消息
                else:
                    message = torch.zeros(self.hidden_size, device=atom_embeddings.device)
                messages.append(message)
            
            messages = torch.stack(messages)  # [num_atoms, hidden_size]
            
            # 更新原子嵌入
            messages = messages.unsqueeze(1)  # [num_atoms, 1, hidden_size]
            atom_embeddings = atom_embeddings.unsqueeze(1)  # [num_atoms, 1, embed_size]
            
            updated_embeddings, _ = self.update_layers[layer_idx](messages, atom_embeddings.transpose(0, 1))
            atom_embeddings = updated_embeddings.squeeze(1)  # [num_atoms, embed_size]
            
            atom_embeddings = self.dropout(atom_embeddings)
        
        # 生成图级嵌入
        if batch_indices is not None:
            # 按批次聚合
            batch_size = batch_indices.max().item() + 1
            graph_embeddings = []
            
            for batch_idx in range(batch_size):
                mask = (batch_indices == batch_idx)
                if mask.any():
                    batch_atom_embeddings = atom_embeddings[mask]
                    graph_embedding = torch.mean(batch_atom_embeddings, dim=0)
                    graph_embeddings.append(graph_embedding)
                else:
                    graph_embeddings.append(torch.zeros(self.embed_size, device=atom_embeddings.device))
            
            graph_embeddings = torch.stack(graph_embeddings)  # [batch_size, embed_size]
        else:
            # 全局平均池化
            graph_embeddings = torch.mean(atom_embeddings, dim=0, keepdim=True)  # [1, embed_size]
        
        # 投影到更高维度
        graph_embeddings = self.global_pool(graph_embeddings)  # [batch_size, hidden_size]
        
        # 键嵌入（简化版本）
        bond_embeddings = self.bond_embedding(bond_features) if bond_features is not None else None
        
        return atom_embeddings, bond_embeddings, graph_embeddings


class ReactionCenterHead(nn.Module):
    """
    基础任务头 - 反应中心识别
    完全保留G2Retro的反应中心识别机制
    """
    
    def __init__(self, hidden_size=256, num_center_types=3):
        super(ReactionCenterHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_center_types = num_center_types
        
        # BF-center (新形成键) 预测头
        self.bf_center_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # BC-center (键类型变化) 预测头
        self.bc_center_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # A-center (原子失去片段) 预测头
        self.a_center_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, atom_embeddings, bond_embeddings, edge_indices=None):
        """
        预测反应中心
        
        Args:
            atom_embeddings: [num_atoms, hidden_size] 原子嵌入
            bond_embeddings: [num_bonds, hidden_size] 键嵌入
            edge_indices: [num_bonds, 2] 边的原子索引
            
        Returns:
            bf_scores: [num_bonds] BF-center分数
            bc_scores: [num_bonds] BC-center分数  
            a_scores: [num_atoms] A-center分数
        """
        # A-center预测（原子级别）
        a_scores = self.a_center_head(atom_embeddings).squeeze(-1)  # [num_atoms]
        
        if bond_embeddings is not None and edge_indices is not None:
            # BF-center和BC-center预测（键级别）
            atom1_embeddings = atom_embeddings[edge_indices[:, 0]]  # [num_bonds, hidden_size]
            atom2_embeddings = atom_embeddings[edge_indices[:, 1]]  # [num_bonds, hidden_size]
            
            # 拼接键两端的原子嵌入
            bond_pair_embeddings = torch.cat([atom1_embeddings, atom2_embeddings], dim=-1)  # [num_bonds, hidden_size*2]
            
            bf_scores = self.bf_center_head(bond_pair_embeddings).squeeze(-1)  # [num_bonds]
            bc_scores = self.bc_center_head(bond_pair_embeddings).squeeze(-1)  # [num_bonds]
        else:
            bf_scores = torch.zeros(0, device=atom_embeddings.device)
            bc_scores = torch.zeros(0, device=atom_embeddings.device)
        
        return {
            'bf_scores': bf_scores,
            'bc_scores': bc_scores,
            'a_scores': a_scores
        }


class MolecularRecoveryHead(nn.Module):
    """
    分子恢复任务头
    基于MolCLR的增强策略，预测被破坏的分子信息
    """
    
    def __init__(self, hidden_size=256, num_atom_types=12, num_bond_types=4):
        super(MolecularRecoveryHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        
        # 原子类型恢复头（用于原子掩码）
        self.atom_recovery_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_atom_types)
        )
        
        # 键类型恢复头（用于键删除）
        self.bond_recovery_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_bond_types)
        )
        
        # 子图恢复头（用于子图移除）
        self.subgraph_recovery_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_atom_types)
        )
        
    def forward(self, atom_embeddings, masked_indices, augment_type):
        """
        恢复被破坏的分子信息
        
        Args:
            atom_embeddings: [num_atoms, hidden_size] 原子嵌入
            masked_indices: [num_masked] 被掩码的原子/键索引
            augment_type: str 增强类型 ('atom_mask', 'bond_deletion', 'subgraph_removal')
            
        Returns:
            recovery_logits: [num_masked, num_classes] 恢复预测
        """
        if len(masked_indices) == 0:
            return torch.zeros(0, self.num_atom_types, device=atom_embeddings.device)
        
        if augment_type == 'atom_mask':
            # 原子掩码恢复
            masked_embeddings = atom_embeddings[masked_indices]  # [num_masked, hidden_size]
            recovery_logits = self.atom_recovery_head(masked_embeddings)  # [num_masked, num_atom_types]
            
        elif augment_type == 'bond_deletion':
            # 键删除恢复 - 简化版本，使用相邻原子的嵌入
            if len(masked_indices) > 0:
                # 假设masked_indices是边的索引，这里简化为使用原子嵌入
                masked_embeddings = atom_embeddings[masked_indices % len(atom_embeddings)]
                recovery_logits = self.atom_recovery_head(masked_embeddings)
            else:
                recovery_logits = torch.zeros(0, self.num_atom_types, device=atom_embeddings.device)
                
        elif augment_type == 'subgraph_removal':
            # 子图移除恢复
            masked_embeddings = atom_embeddings[masked_indices]  # [num_masked, hidden_size]
            recovery_logits = self.subgraph_recovery_head(masked_embeddings)  # [num_masked, num_atom_types]
            
        else:
            raise ValueError(f"未知的增强类型: {augment_type}")
        
        return recovery_logits


class ProductSynthonContrastiveHead(nn.Module):
    """
    产物-合成子对比学习任务头
    核心创新：利用产物与合成子的自然差异进行对比学习
    """
    
    def __init__(self, hidden_size=256, projection_dim=128, temperature=0.1):
        super(ProductSynthonContrastiveHead, self).__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 投影网络（遵循MolCLR和PMSR的标准做法）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
    def forward(self, product_features, synthon_features):
        """
        计算产物-合成子对比学习损失
        
        Args:
            product_features: [batch_size, hidden_size] 产物分子图特征
            synthon_features: [batch_size, hidden_size] 合成子组合特征
            
        Returns:
            contrastive_loss: 对比学习损失
            similarity_matrix: [batch_size, batch_size] 相似度矩阵
        """
        batch_size = product_features.size(0)
        
        # 投影到对比学习空间
        product_proj = self.projection_head(product_features)  # [batch_size, projection_dim]
        synthon_proj = self.projection_head(synthon_features)   # [batch_size, projection_dim]
        
        # L2归一化
        product_proj = F.normalize(product_proj, p=2, dim=1)
        synthon_proj = F.normalize(synthon_proj, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(product_proj, synthon_proj.t()) / self.temperature  # [batch_size, batch_size]
        
        # 构建正样本标签（对角线为正样本对）
        labels = torch.arange(batch_size, device=product_features.device)
        
        # 计算对比学习损失（NT-Xent损失）
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        return contrastive_loss, similarity_matrix


class SynthRetroPretrainModel(nn.Module):
    """
    SynthRetro-P 完整预训练模型
    包含共享编码器和三个任务头
    """
    
    def __init__(self, 
                 hidden_size=256, 
                 embed_size=32, 
                 depth=10,
                 num_atom_types=12,
                 num_bond_types=4,
                 projection_dim=128,
                 temperature=0.1):
        super(SynthRetroPretrainModel, self).__init__()
        
        # 共享编码器
        self.shared_encoder = GMPN(hidden_size, embed_size, depth)
        
        # 三个任务头
        self.reaction_center_head = ReactionCenterHead(hidden_size)
        self.molecular_recovery_head = MolecularRecoveryHead(hidden_size, num_atom_types, num_bond_types)
        self.contrastive_head = ProductSynthonContrastiveHead(hidden_size, projection_dim, temperature)
        
        # 损失权重
        self.base_weight = 1.0
        self.recovery_weight = 1.0  
        self.contrastive_weight = 0.1  # 设为0.1避免主导训练过程
        
    def forward(self, batch_data):
        """
        前向传播 - 多任务训练
        
        Args:
            batch_data: 包含三个任务数据的批次
            
        Returns:
            losses: 各任务损失
            predictions: 各任务预测结果
        """
        losses = {}
        predictions = {}
        
        # 1. 基础任务（反应中心识别）
        if batch_data.get('base_task') is not None:
            base_task_data = batch_data['base_task']
            
            # 编码产物分子
            atom_emb, bond_emb, graph_emb = self.shared_encoder(
                base_task_data['atom_features'],
                base_task_data.get('bond_features'),
                base_task_data.get('adjacency_matrix'),
                base_task_data.get('batch_indices')
            )
            
            # 反应中心预测
            center_preds = self.reaction_center_head(atom_emb, bond_emb, base_task_data.get('edge_indices'))
            predictions['base_task'] = center_preds
            
            # 计算基础任务损失
            if 'reaction_center_labels' in base_task_data:
                base_loss = self.compute_reaction_center_loss(center_preds, base_task_data['reaction_center_labels'])
                losses['base_loss'] = base_loss * self.base_weight
        
        # 2. 分子恢复任务
        if batch_data.get('recovery_task') is not None:
            recovery_losses = []
            
            for aug_type, aug_data in batch_data['recovery_task'].items():
                if len(aug_data.get('masked_indices', [])) > 0:
                    # 编码增强后的分子
                    atom_emb, _, _ = self.shared_encoder(
                        aug_data['augmented_atom_features'],
                        aug_data.get('augmented_bond_features'),
                        aug_data.get('augmented_adjacency_matrix'),
                        aug_data.get('batch_indices')
                    )
                    
                    # 恢复预测
                    recovery_logits = self.molecular_recovery_head(
                        atom_emb, 
                        aug_data['masked_indices'], 
                        aug_type
                    )
                    
                    # 计算恢复损失
                    recovery_loss = F.cross_entropy(recovery_logits, aug_data['target_labels'])
                    recovery_losses.append(recovery_loss)
            
            if recovery_losses:
                losses['recovery_loss'] = torch.mean(torch.stack(recovery_losses)) * self.recovery_weight
        
        # 3. 产物-合成子对比学习
        if batch_data.get('contrastive_task') is not None:
            contrastive_data = batch_data['contrastive_task']
            
            # 编码产物和合成子
            product_emb = self.encode_molecules(contrastive_data['product_features'])
            synthon_emb = self.encode_molecules(contrastive_data['synthon_features'])
            
            # 对比学习
            contrastive_loss, similarity_matrix = self.contrastive_head(product_emb, synthon_emb)
            losses['contrastive_loss'] = contrastive_loss * self.contrastive_weight
            predictions['contrastive_task'] = similarity_matrix
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses, predictions
    
    def encode_molecules(self, molecule_features):
        """简化的分子编码（占位符）"""
        # 这里应该根据实际的分子特征格式进行编码
        # 暂时返回特征本身
        return molecule_features
    
    def compute_reaction_center_loss(self, predictions, labels):
        """计算反应中心识别损失"""
        losses = []
        
        # BF-center损失
        if len(predictions['bf_scores']) > 0 and len(labels.get('bf_centers', [])) > 0:
            bf_loss = F.binary_cross_entropy(predictions['bf_scores'], labels['bf_centers'].float())
            losses.append(bf_loss)
        
        # BC-center损失
        if len(predictions['bc_scores']) > 0 and len(labels.get('bc_centers', [])) > 0:
            bc_loss = F.binary_cross_entropy(predictions['bc_scores'], labels['bc_centers'].float())
            losses.append(bc_loss)
        
        # A-center损失
        if len(predictions['a_scores']) > 0 and len(labels.get('a_centers', [])) > 0:
            a_loss = F.binary_cross_entropy(predictions['a_scores'], labels['a_centers'].float())
            losses.append(a_loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=predictions['a_scores'].device)


def create_synthretro_pretrain_model(config):
    """
    创建SynthRetro-P预训练模型的工厂函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: SynthRetro-P模型实例
    """
    model = SynthRetroPretrainModel(
        hidden_size=config.get('hidden_size', 256),
        embed_size=config.get('embed_size', 32),
        depth=config.get('depth', 10),
        num_atom_types=config.get('num_atom_types', 12),
        num_bond_types=config.get('num_bond_types', 4),
        projection_dim=config.get('projection_dim', 128),
        temperature=config.get('temperature', 0.1)
    )
    
    return model


# 使用示例
if __name__ == "__main__":
    # 模型配置
    config = {
        'hidden_size': 256,
        'embed_size': 32,
        'depth': 10,
        'num_atom_types': 12,
        'num_bond_types': 4,
        'projection_dim': 128,
        'temperature': 0.1
    }
    
    # 创建模型
    model = create_synthretro_pretrain_model(config)
    
    print("🎉 SynthRetro-P预训练模型创建成功！")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 显示模型架构
    print("\n🏗️ 模型架构:")
    print("  🧠 共享编码器 (GMPN)")
    print("  📍 基础任务头 (反应中心识别)")  
    print("  🔄 分子恢复头 (MolCLR增强)")
    print("  🤝 对比学习头 (产物-合成子对比)")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class GMPN(nn.Module):
    """
    图消息传递网络 - 共享编码器
    完全沿用G2Retro的核心组件
    """
    
    def __init__(self, hidden_size=256, embed_size=32, depth=10):
        super(GMPN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.depth = depth
        
        # 原子嵌入层
        self.atom_embedding = nn.Embedding(200, embed_size)  # 支持多种原子类型
        
        # 键嵌入层
        self.bond_embedding = nn.Embedding(10, embed_size)   # 支持多种键类型
        
        # 消息传递层
        self.message_layers = nn.ModuleList([
            nn.Linear(embed_size * 2, hidden_size) for _ in range(depth)
        ])
        
        # 更新层
        self.update_layers = nn.ModuleList([
            nn.GRU(hidden_size, embed_size, batch_first=True) for _ in range(depth)
        ])
        
        # 全局池化层
        self.global_pool = nn.Linear(embed_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, atom_features, bond_features, adjacency_matrix, batch_indices=None):
        """
        前向传播
        
        Args:
            atom_features: [num_atoms, atom_feature_dim] 原子特征
            bond_features: [num_bonds, bond_feature_dim] 键特征  
            adjacency_matrix: [num_atoms, num_atoms] 邻接矩阵
            batch_indices: [num_atoms] 原子属于哪个分子的标识
            
        Returns:
            atom_embeddings: [num_atoms, embed_size] 原子嵌入
            bond_embeddings: [num_bonds, embed_size] 键嵌入
            graph_embeddings: [batch_size, hidden_size] 图嵌入
        """
        # 原子特征嵌入
        atom_embeddings = self.atom_embedding(atom_features)  # [num_atoms, embed_size]
        
        # 消息传递
        for layer_idx in range(self.depth):
            # 收集邻居消息
            messages = []
            for atom_idx in range(atom_embeddings.size(0)):
                neighbors = torch.nonzero(adjacency_matrix[atom_idx]).squeeze(-1)
                if len(neighbors) > 0:
                    neighbor_embeddings = atom_embeddings[neighbors]
                    atom_neighbor_concat = torch.cat([
                        atom_embeddings[atom_idx].unsqueeze(0).repeat(len(neighbors), 1),
                        neighbor_embeddings
                    ], dim=-1)
                    message = self.message_layers[layer_idx](atom_neighbor_concat)
                    message = torch.mean(message, dim=0)  # 聚合邻居消息
                else:
                    message = torch.zeros(self.hidden_size, device=atom_embeddings.device)
                messages.append(message)
            
            messages = torch.stack(messages)  # [num_atoms, hidden_size]
            
            # 更新原子嵌入
            messages = messages.unsqueeze(1)  # [num_atoms, 1, hidden_size]
            atom_embeddings = atom_embeddings.unsqueeze(1)  # [num_atoms, 1, embed_size]
            
            updated_embeddings, _ = self.update_layers[layer_idx](messages, atom_embeddings.transpose(0, 1))
            atom_embeddings = updated_embeddings.squeeze(1)  # [num_atoms, embed_size]
            
            atom_embeddings = self.dropout(atom_embeddings)
        
        # 生成图级嵌入
        if batch_indices is not None:
            # 按批次聚合
            batch_size = batch_indices.max().item() + 1
            graph_embeddings = []
            
            for batch_idx in range(batch_size):
                mask = (batch_indices == batch_idx)
                if mask.any():
                    batch_atom_embeddings = atom_embeddings[mask]
                    graph_embedding = torch.mean(batch_atom_embeddings, dim=0)
                    graph_embeddings.append(graph_embedding)
                else:
                    graph_embeddings.append(torch.zeros(self.embed_size, device=atom_embeddings.device))
            
            graph_embeddings = torch.stack(graph_embeddings)  # [batch_size, embed_size]
        else:
            # 全局平均池化
            graph_embeddings = torch.mean(atom_embeddings, dim=0, keepdim=True)  # [1, embed_size]
        
        # 投影到更高维度
        graph_embeddings = self.global_pool(graph_embeddings)  # [batch_size, hidden_size]
        
        # 键嵌入（简化版本）
        bond_embeddings = self.bond_embedding(bond_features) if bond_features is not None else None
        
        return atom_embeddings, bond_embeddings, graph_embeddings


class ReactionCenterHead(nn.Module):
    """
    基础任务头 - 反应中心识别
    完全保留G2Retro的反应中心识别机制
    """
    
    def __init__(self, hidden_size=256, num_center_types=3):
        super(ReactionCenterHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_center_types = num_center_types
        
        # BF-center (新形成键) 预测头
        self.bf_center_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # BC-center (键类型变化) 预测头
        self.bc_center_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # A-center (原子失去片段) 预测头
        self.a_center_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, atom_embeddings, bond_embeddings, edge_indices=None):
        """
        预测反应中心
        
        Args:
            atom_embeddings: [num_atoms, hidden_size] 原子嵌入
            bond_embeddings: [num_bonds, hidden_size] 键嵌入
            edge_indices: [num_bonds, 2] 边的原子索引
            
        Returns:
            bf_scores: [num_bonds] BF-center分数
            bc_scores: [num_bonds] BC-center分数  
            a_scores: [num_atoms] A-center分数
        """
        # A-center预测（原子级别）
        a_scores = self.a_center_head(atom_embeddings).squeeze(-1)  # [num_atoms]
        
        if bond_embeddings is not None and edge_indices is not None:
            # BF-center和BC-center预测（键级别）
            atom1_embeddings = atom_embeddings[edge_indices[:, 0]]  # [num_bonds, hidden_size]
            atom2_embeddings = atom_embeddings[edge_indices[:, 1]]  # [num_bonds, hidden_size]
            
            # 拼接键两端的原子嵌入
            bond_pair_embeddings = torch.cat([atom1_embeddings, atom2_embeddings], dim=-1)  # [num_bonds, hidden_size*2]
            
            bf_scores = self.bf_center_head(bond_pair_embeddings).squeeze(-1)  # [num_bonds]
            bc_scores = self.bc_center_head(bond_pair_embeddings).squeeze(-1)  # [num_bonds]
        else:
            bf_scores = torch.zeros(0, device=atom_embeddings.device)
            bc_scores = torch.zeros(0, device=atom_embeddings.device)
        
        return {
            'bf_scores': bf_scores,
            'bc_scores': bc_scores,
            'a_scores': a_scores
        }


class MolecularRecoveryHead(nn.Module):
    """
    分子恢复任务头
    基于MolCLR的增强策略，预测被破坏的分子信息
    """
    
    def __init__(self, hidden_size=256, num_atom_types=12, num_bond_types=4):
        super(MolecularRecoveryHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        
        # 原子类型恢复头（用于原子掩码）
        self.atom_recovery_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_atom_types)
        )
        
        # 键类型恢复头（用于键删除）
        self.bond_recovery_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_bond_types)
        )
        
        # 子图恢复头（用于子图移除）
        self.subgraph_recovery_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_atom_types)
        )
        
    def forward(self, atom_embeddings, masked_indices, augment_type):
        """
        恢复被破坏的分子信息
        
        Args:
            atom_embeddings: [num_atoms, hidden_size] 原子嵌入
            masked_indices: [num_masked] 被掩码的原子/键索引
            augment_type: str 增强类型 ('atom_mask', 'bond_deletion', 'subgraph_removal')
            
        Returns:
            recovery_logits: [num_masked, num_classes] 恢复预测
        """
        if len(masked_indices) == 0:
            return torch.zeros(0, self.num_atom_types, device=atom_embeddings.device)
        
        if augment_type == 'atom_mask':
            # 原子掩码恢复
            masked_embeddings = atom_embeddings[masked_indices]  # [num_masked, hidden_size]
            recovery_logits = self.atom_recovery_head(masked_embeddings)  # [num_masked, num_atom_types]
            
        elif augment_type == 'bond_deletion':
            # 键删除恢复 - 简化版本，使用相邻原子的嵌入
            if len(masked_indices) > 0:
                # 假设masked_indices是边的索引，这里简化为使用原子嵌入
                masked_embeddings = atom_embeddings[masked_indices % len(atom_embeddings)]
                recovery_logits = self.atom_recovery_head(masked_embeddings)
            else:
                recovery_logits = torch.zeros(0, self.num_atom_types, device=atom_embeddings.device)
                
        elif augment_type == 'subgraph_removal':
            # 子图移除恢复
            masked_embeddings = atom_embeddings[masked_indices]  # [num_masked, hidden_size]
            recovery_logits = self.subgraph_recovery_head(masked_embeddings)  # [num_masked, num_atom_types]
            
        else:
            raise ValueError(f"未知的增强类型: {augment_type}")
        
        return recovery_logits


class ProductSynthonContrastiveHead(nn.Module):
    """
    产物-合成子对比学习任务头
    核心创新：利用产物与合成子的自然差异进行对比学习
    """
    
    def __init__(self, hidden_size=256, projection_dim=128, temperature=0.1):
        super(ProductSynthonContrastiveHead, self).__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 投影网络（遵循MolCLR和PMSR的标准做法）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
    def forward(self, product_features, synthon_features):
        """
        计算产物-合成子对比学习损失
        
        Args:
            product_features: [batch_size, hidden_size] 产物分子图特征
            synthon_features: [batch_size, hidden_size] 合成子组合特征
            
        Returns:
            contrastive_loss: 对比学习损失
            similarity_matrix: [batch_size, batch_size] 相似度矩阵
        """
        batch_size = product_features.size(0)
        
        # 投影到对比学习空间
        product_proj = self.projection_head(product_features)  # [batch_size, projection_dim]
        synthon_proj = self.projection_head(synthon_features)   # [batch_size, projection_dim]
        
        # L2归一化
        product_proj = F.normalize(product_proj, p=2, dim=1)
        synthon_proj = F.normalize(synthon_proj, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(product_proj, synthon_proj.t()) / self.temperature  # [batch_size, batch_size]
        
        # 构建正样本标签（对角线为正样本对）
        labels = torch.arange(batch_size, device=product_features.device)
        
        # 计算对比学习损失（NT-Xent损失）
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        return contrastive_loss, similarity_matrix


class G2RetroPretrainModel(nn.Module):
    """
    G2Retro-P 完整预训练模型
    包含共享编码器和三个任务头
    """
    
    def __init__(self, 
                 hidden_size=256, 
                 embed_size=32, 
                 depth=10,
                 num_atom_types=12,
                 num_bond_types=4,
                 projection_dim=128,
                 temperature=0.1):
        super(G2RetroPretrainModel, self).__init__()
        
        # 共享编码器
        self.shared_encoder = GMPN(hidden_size, embed_size, depth)
        
        # 三个任务头
        self.reaction_center_head = ReactionCenterHead(hidden_size)
        self.molecular_recovery_head = MolecularRecoveryHead(hidden_size, num_atom_types, num_bond_types)
        self.contrastive_head = ProductSynthonContrastiveHead(hidden_size, projection_dim, temperature)
        
        # 损失权重
        self.base_weight = 1.0
        self.recovery_weight = 1.0  
        self.contrastive_weight = 0.1  # 设为0.1避免主导训练过程
        
    def forward(self, batch_data):
        """
        前向传播 - 多任务训练
        
        Args:
            batch_data: 包含三个任务数据的批次
            
        Returns:
            losses: 各任务损失
            predictions: 各任务预测结果
        """
        losses = {}
        predictions = {}
        
        # 1. 基础任务（反应中心识别）
        if batch_data.get('base_task') is not None:
            base_task_data = batch_data['base_task']
            
            # 编码产物分子
            atom_emb, bond_emb, graph_emb = self.shared_encoder(
                base_task_data['atom_features'],
                base_task_data.get('bond_features'),
                base_task_data.get('adjacency_matrix'),
                base_task_data.get('batch_indices')
            )
            
            # 反应中心预测
            center_preds = self.reaction_center_head(atom_emb, bond_emb, base_task_data.get('edge_indices'))
            predictions['base_task'] = center_preds
            
            # 计算基础任务损失
            if 'reaction_center_labels' in base_task_data:
                base_loss = self.compute_reaction_center_loss(center_preds, base_task_data['reaction_center_labels'])
                losses['base_loss'] = base_loss * self.base_weight
        
        # 2. 分子恢复任务
        if batch_data.get('recovery_task') is not None:
            recovery_losses = []
            
            for aug_type, aug_data in batch_data['recovery_task'].items():
                if len(aug_data.get('masked_indices', [])) > 0:
                    # 编码增强后的分子
                    atom_emb, _, _ = self.shared_encoder(
                        aug_data['augmented_atom_features'],
                        aug_data.get('augmented_bond_features'),
                        aug_data.get('augmented_adjacency_matrix'),
                        aug_data.get('batch_indices')
                    )
                    
                    # 恢复预测
                    recovery_logits = self.molecular_recovery_head(
                        atom_emb, 
                        aug_data['masked_indices'], 
                        aug_type
                    )
                    
                    # 计算恢复损失
                    recovery_loss = F.cross_entropy(recovery_logits, aug_data['target_labels'])
                    recovery_losses.append(recovery_loss)
            
            if recovery_losses:
                losses['recovery_loss'] = torch.mean(torch.stack(recovery_losses)) * self.recovery_weight
        
        # 3. 产物-合成子对比学习
        if batch_data.get('contrastive_task') is not None:
            contrastive_data = batch_data['contrastive_task']
            
            # 编码产物和合成子
            product_emb = self.encode_molecules(contrastive_data['product_features'])
            synthon_emb = self.encode_molecules(contrastive_data['synthon_features'])
            
            # 对比学习
            contrastive_loss, similarity_matrix = self.contrastive_head(product_emb, synthon_emb)
            losses['contrastive_loss'] = contrastive_loss * self.contrastive_weight
            predictions['contrastive_task'] = similarity_matrix
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses, predictions
    
    def encode_molecules(self, molecule_features):
        """简化的分子编码（占位符）"""
        # 这里应该根据实际的分子特征格式进行编码
        # 暂时返回特征本身
        return molecule_features
    
    def compute_reaction_center_loss(self, predictions, labels):
        """计算反应中心识别损失"""
        losses = []
        
        # BF-center损失
        if len(predictions['bf_scores']) > 0 and len(labels.get('bf_centers', [])) > 0:
            bf_loss = F.binary_cross_entropy(predictions['bf_scores'], labels['bf_centers'].float())
            losses.append(bf_loss)
        
        # BC-center损失
        if len(predictions['bc_scores']) > 0 and len(labels.get('bc_centers', [])) > 0:
            bc_loss = F.binary_cross_entropy(predictions['bc_scores'], labels['bc_centers'].float())
            losses.append(bc_loss)
        
        # A-center损失
        if len(predictions['a_scores']) > 0 and len(labels.get('a_centers', [])) > 0:
            a_loss = F.binary_cross_entropy(predictions['a_scores'], labels['a_centers'].float())
            losses.append(a_loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=predictions['a_scores'].device)


def create_g2retro_pretrain_model(config):
    """
    创建G2Retro-P预训练模型的工厂函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: G2Retro-P模型实例
    """
    model = G2RetroPretrainModel(
        hidden_size=config.get('hidden_size', 256),
        embed_size=config.get('embed_size', 32),
        depth=config.get('depth', 10),
        num_atom_types=config.get('num_atom_types', 12),
        num_bond_types=config.get('num_bond_types', 4),
        projection_dim=config.get('projection_dim', 128),
        temperature=config.get('temperature', 0.1)
    )
    
    return model


# 使用示例
if __name__ == "__main__":
    # 模型配置
    config = {
        'hidden_size': 256,
        'embed_size': 32,
        'depth': 10,
        'num_atom_types': 12,
        'num_bond_types': 4,
        'projection_dim': 128,
        'temperature': 0.1
    }
    
    # 创建模型
    model = create_g2retro_pretrain_model(config)
    
    print("🎉 G2Retro-P预训练模型创建成功！")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 显示模型架构
    print("\n🏗️ 模型架构:")
    print("  🧠 共享编码器 (GMPN)")
    print("  📍 基础任务头 (反应中心识别)")  
    print("  🔄 分子恢复头 (MolCLR增强)")
    print("  🤝 对比学习头 (产物-合成子对比)")