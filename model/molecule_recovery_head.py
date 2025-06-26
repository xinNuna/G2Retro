#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子恢复任务头 - 基于MolCLR的对比学习框架（图结构级别掩码版本）

这个模块实现了G2Retro-P的分子恢复任务，通过对比学习从增强后的分子中恢复原始分子表示。
完全参考MolCLR的实现，使用InfoNCE损失进行对比学习。

新增功能：
- 支持在MolTree图结构级别进行掩码操作
- 掩码感知编码机制
- 增强的对比学习框架

参考文献：
- MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks
- PMSR: Learning Chemical Rules of Retrosynthesis with Pre-training

核心思想：
1. 原始分子和增强分子应该有相似的表示（正样本对）
2. 不同分子的表示应该不同（负样本对）
3. 通过InfoNCE损失最大化正样本相似度，最小化负样本相似度
4. 掩码感知：理解哪些部分被掩码，学习从部分信息恢复完整表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import math
import copy

# 导入图结构级别掩码增强函数
def apply_molclr_graph_augmentation(mol_tree, masked_indices, augment_type):
    """
    按照MolCLR思想在MolTree的图结构上进行掩码操作
    不修改底层化学结构，只在图表示级别进行临时增强
    """
    import copy
    
    try:
        # 深拷贝MolTree以避免修改原始对象
        augmented_tree = copy.deepcopy(mol_tree)
        
        # 获取分子图 (NetworkX DiGraph)
        mol_graph = augmented_tree.mol_graph
        
        if augment_type == 'atom_mask':
            # 原子掩码：在图节点级别进行掩码，不改变化学结构
            for atom_idx in masked_indices:
                if atom_idx in mol_graph.nodes:
                    node_data = mol_graph.nodes[atom_idx]
                    # 保存原始特征
                    node_data['original_label'] = node_data.get('label', '')
                    node_data['original_aroma'] = node_data.get('aroma', False)
                    # 设置掩码标记
                    node_data['masked'] = True
                    node_data['label'] = '[MASK]'  # 掩码标记
                    node_data['aroma'] = False  # 重置芳香性
                    
        elif augment_type == 'bond_deletion':
            # 键删除：在图边级别进行掩码，不删除实际化学键
            edges_list = list(mol_graph.edges())
            for bond_idx in masked_indices:
                if bond_idx < len(edges_list):
                    edge = edges_list[bond_idx]
                    if mol_graph.has_edge(edge[0], edge[1]):
                        edge_data = mol_graph.edges[edge]
                        # 保存原始边特征
                        edge_data['original_bond_type'] = edge_data.get('bond_type', 1)
                        edge_data['original_is_conju'] = edge_data.get('is_conju', False)
                        # 设置掩码标记
                        edge_data['masked'] = True
                        edge_data['bond_type'] = 0  # 设为无键类型
                        edge_data['is_conju'] = False
                        
        elif augment_type == 'subgraph_removal':
            # 子图移除：在图节点级别进行掩码
            for atom_idx in masked_indices:
                if atom_idx in mol_graph.nodes:
                    node_data = mol_graph.nodes[atom_idx]
                    # 保存原始特征
                    node_data['original_label'] = node_data.get('label', '')
                    node_data['original_aroma'] = node_data.get('aroma', False)
                    # 设置掩码标记
                    node_data['masked'] = True
                    node_data['label'] = '[REMOVED]'  # 移除标记
                    node_data['aroma'] = False
                    
                    # 同时掩码相关的边
                    for neighbor in mol_graph.neighbors(atom_idx):
                        if mol_graph.has_edge(atom_idx, neighbor):
                            edge_data = mol_graph.edges[atom_idx, neighbor]
                            edge_data['masked'] = True
                            edge_data['original_bond_type'] = edge_data.get('bond_type', 1)
                            edge_data['bond_type'] = 0
        
        # 更新增强信息
        augmented_tree.augmented = True
        augmented_tree.augment_type = augment_type
        augmented_tree.masked_indices = masked_indices
        
        return augmented_tree
        
    except Exception as e:
        print(f"MolCLR图增强错误: {e}")
        return mol_tree  # 返回原始树作为备选

class MoleculeRecoveryHead(nn.Module):
    """
    分子恢复任务头 - 完全参考MolCLR实现，支持图结构级别掩码
    
    通过对比学习学习分子表示，使增强后的分子能够恢复到原始分子的表示空间
    新增：掩码感知机制，能够处理在图结构级别进行掩码的MolTree
    """
    
    def __init__(self, 
                 input_dim: int = 300,           # G2Retro分子图嵌入维度
                 projection_dim: int = 128,       # MolCLR投影维度
                 hidden_dim: int = 512,          # 隐藏层维度
                 temperature: float = 0.1,       # InfoNCE温度参数
                 dropout: float = 0.1):
        """
        初始化分子恢复任务头
        
        Args:
            input_dim: 输入分子图嵌入的维度
            projection_dim: 投影后的表示维度
            hidden_dim: 隐藏层维度
            temperature: InfoNCE损失的温度参数
            dropout: Dropout比例
        """
        super(MoleculeRecoveryHead, self).__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # MolCLR风格的投影头 - 两层MLP
        # 完全参考MolCLR的架构设计
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # 新增：掩码感知注意力机制
        self.mask_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 新增：掩码类型嵌入（区分不同的增强类型）
        self.mask_type_embedding = nn.Embedding(4, input_dim)  # none, atom_mask, bond_deletion, subgraph_removal
        self.mask_type_map = {
            'none': 0,
            'atom_mask': 1, 
            'bond_deletion': 2,
            'subgraph_removal': 3
        }
        
        # 新增：掩码信息融合层
        self.mask_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
        
        # 初始化权重（使用He初始化）
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重 - 参考MolCLR的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def apply_mask_aware_encoding(self, 
                                 mol_embeddings: torch.Tensor,
                                 mask_info: List[Dict]) -> torch.Tensor:
        """
        应用掩码感知编码
        
        Args:
            mol_embeddings: 分子嵌入 [batch_size, input_dim]
            mask_info: 掩码信息列表，每个元素包含type和indices
            
        Returns:
            掩码感知的分子嵌入
        """
        batch_size = mol_embeddings.size(0)
        device = mol_embeddings.device
        
        # 获取掩码类型嵌入
        mask_types = []
        for info in mask_info:
            mask_type = info.get('type', 'none')
            mask_type_id = self.mask_type_map.get(mask_type, 0)
            mask_types.append(mask_type_id)
        
        mask_type_tensor = torch.tensor(mask_types, device=device)
        mask_type_embeds = self.mask_type_embedding(mask_type_tensor)  # [batch_size, input_dim]
        
        # 融合原始嵌入和掩码类型信息
        combined_embeds = torch.cat([mol_embeddings, mask_type_embeds], dim=1)  # [batch_size, input_dim*2]
        mask_aware_embeds = self.mask_fusion(combined_embeds)  # [batch_size, input_dim]
        
        # 应用掩码感知注意力
        if len(mol_embeddings.shape) == 2:
            mol_embeddings_expanded = mol_embeddings.unsqueeze(1)  # [batch_size, 1, input_dim]
            mask_aware_expanded = mask_aware_embeds.unsqueeze(1)   # [batch_size, 1, input_dim]
        else:
            mol_embeddings_expanded = mol_embeddings
            mask_aware_expanded = mask_aware_embeds.unsqueeze(1).expand(-1, mol_embeddings.size(1), -1)
        
        # 掩码感知注意力：查询使用掩码感知嵌入，键值使用原始嵌入
        attended_embeds, _ = self.mask_attention(
            mask_aware_expanded,      # query
            mol_embeddings_expanded,  # key
            mol_embeddings_expanded   # value
        )
        
        # 如果输入是2D，压缩回2D
        if len(mol_embeddings.shape) == 2:
            attended_embeds = attended_embeds.squeeze(1)
        
        return attended_embeds
    
    def forward(self, 
                original_embeddings: torch.Tensor,
                augmented_data: List[Dict],
                pretrain_infos: List[Dict],
                mol_encoder=None,
                vocab=None,
                avocab=None) -> Tuple[torch.Tensor, float]:
        """
        前向传播 - 计算分子恢复的对比学习损失
        
        Args:
            original_embeddings: 原始分子的图嵌入 [batch_size, input_dim]
            augmented_data: 增强数据信息
            pretrain_infos: 预训练信息
            mol_encoder: 分子编码器（用于编码增强的MolTree）
            vocab: 词汇表
            avocab: 原子词汇表
            
        Returns:
            (loss, accuracy): 损失值和准确率
        """
        batch_size = original_embeddings.size(0)
        device = original_embeddings.device
        
        # 为每个样本创建掩码信息和真正的增强嵌入
        mask_info = []
        augmented_embeddings_list = []
        
        for i in range(batch_size):
            # 获取当前样本的增强信息
            sample_aug_data = []
            if i < len(augmented_data):
                sample_aug_data = augmented_data[i] if isinstance(augmented_data[i], list) else [augmented_data[i]]
            
            if sample_aug_data and mol_encoder is not None and vocab is not None and avocab is not None:
                # 使用第一个增强版本的信息
                aug_info = sample_aug_data[0]
                mask_info.append({
                    'type': aug_info.get('augment_type', 'none'),
                    'indices': aug_info.get('masked_indices', [])
                })
                
                # 创建真正的增强MolTree并获取其嵌入
                try:
                    product_smiles = pretrain_infos[i]['product_smiles']
                    
                    # 创建原始MolTree
                    from moltree import MolTree
                    original_tree = MolTree(product_smiles)
                    
                    # 应用图结构级别掩码增强
                    augmented_tree = apply_molclr_graph_augmentation(
                        original_tree,
                        aug_info.get('masked_indices', []),
                        aug_info.get('augment_type', 'none')
                    )
                    
                    # 使用编码器获取增强MolTree的嵌入
                    aug_batch, aug_tensors = MolTree.tensorize(
                        [augmented_tree], vocab, avocab, 
                        use_feature=True, product=True
                    )
                    
                    # 编码增强的分子树
                    with torch.no_grad():
                        aug_embed, _, _ = mol_encoder.encode_with_gmpn([aug_tensors])
                    
                    augmented_embeddings_list.append(aug_embed)
                    print(f"样本 {i}: 成功获取图结构级别增强嵌入")
                    
                except Exception as e:
                    print(f"样本 {i} 增强嵌入获取错误: {e}")
                    # 如果增强失败，使用原始嵌入
                    augmented_embeddings_list.append(original_embeddings[i:i+1])
                    mask_info.append({'type': 'none', 'indices': []})
            else:
                # 没有增强数据或缺少编码器，使用原始嵌入
                mask_info.append({'type': 'none', 'indices': []})
                augmented_embeddings_list.append(original_embeddings[i:i+1])
        
        # 拼接所有增强嵌入
        augmented_embeddings = torch.cat(augmented_embeddings_list, dim=0)
        
        # 应用掩码感知编码
        mask_aware_augmented = self.apply_mask_aware_encoding(augmented_embeddings, mask_info)
        
        # 通过投影头获得对比学习表示
        original_proj = self.projection_head(original_embeddings)     # [batch_size, projection_dim]
        augmented_proj = self.projection_head(mask_aware_augmented)   # [batch_size, projection_dim]
        
        # L2归一化（MolCLR的关键步骤）
        original_proj = F.normalize(original_proj, p=2, dim=1)
        augmented_proj = F.normalize(augmented_proj, p=2, dim=1)
        
        # 计算InfoNCE损失
        contrastive_loss = self._compute_infonce_loss(original_proj, augmented_proj)
        
        # 计算准确率
        similarity_matrix = torch.mm(original_proj, augmented_proj.t()) / self.temperature
        accuracy = self._compute_accuracy(similarity_matrix)
        
        return contrastive_loss, accuracy.item()
    
    def _compute_infonce_loss(self, 
                             z_i: torch.Tensor, 
                             z_j: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE对比学习损失 - 完全参考MolCLR实现
        
        InfoNCE公式：
        L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
        
        其中z_i是原始分子表示，z_j是对应的增强分子表示，z_k是批次中的所有负样本
        
        Args:
            z_i: 原始分子的归一化投影表示 [batch_size, projection_dim]
            z_j: 增强分子的归一化投影表示 [batch_size, projection_dim]
            
        Returns:
            InfoNCE损失值
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # 构建正样本和负样本的表示矩阵
        # 将z_i和z_j拼接，形成2*batch_size的表示矩阵
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        
        # 计算所有表示之间的相似度矩阵
        similarity_matrix = torch.mm(representations, representations.t()) / self.temperature
        # 形状: [2*batch_size, 2*batch_size]
        
        # 创建正样本mask
        # 对于索引i，其正样本是索引(i + batch_size) % (2 * batch_size)
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True          # z_i的正样本是z_j[i]
            positive_mask[i + batch_size, i] = True          # z_j[i]的正样本是z_i
        
        # 创建对角线mask（排除自身）
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        
        # 计算InfoNCE损失
        # 对于每个样本，计算其与正样本的相似度和与所有负样本的相似度
        losses = []
        for i in range(2 * batch_size):
            # 当前样本与所有样本的相似度
            current_similarities = similarity_matrix[i]
            
            # 排除自身
            current_similarities = current_similarities[~diagonal_mask[i]]
            positive_mask_current = positive_mask[i][~diagonal_mask[i]]
            
            # 正样本相似度
            positive_similarity = current_similarities[positive_mask_current]
            
            # 所有样本（包括正样本）的相似度用于分母
            denominator = torch.logsumexp(current_similarities, dim=0)
            
            # InfoNCE损失: -log(exp(pos_sim) / sum(exp(all_sim)))
            loss = -positive_similarity + denominator
            losses.append(loss)
        
        # 返回平均损失
        return torch.stack(losses).mean()
    
    def _compute_accuracy(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习的准确率 - 用于监控训练过程
        
        准确率定义：在一个批次中，有多少比例的样本的最相似样本确实是其正样本
        
        Args:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
            
        Returns:
            准确率值
        """
        batch_size = similarity_matrix.size(0)
        
        # 找到每个原始分子最相似的增强分子
        _, top_indices = torch.topk(similarity_matrix, k=1, dim=1)
        
        # 正确的匹配应该是对角线（每个分子最相似的应该是自己的增强版本）
        correct_matches = (top_indices.squeeze() == torch.arange(batch_size, device=similarity_matrix.device))
        
        return correct_matches.float().mean()

    def compute_molecular_similarity(self, 
                                   mol_embeddings_1: torch.Tensor,
                                   mol_embeddings_2: torch.Tensor) -> torch.Tensor:
        """
        计算两组分子之间的相似度 - 用于下游任务评估
        
        Args:
            mol_embeddings_1: 第一组分子嵌入 [N, input_dim]
            mol_embeddings_2: 第二组分子嵌入 [M, input_dim]
            
        Returns:
            相似度矩阵 [N, M]
        """
        # 投影到对比学习空间
        proj_1 = F.normalize(self.projection_head(mol_embeddings_1), p=2, dim=1)
        proj_2 = F.normalize(self.projection_head(mol_embeddings_2), p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.mm(proj_1, proj_2.t())
        
        return similarity


class InfoNCELoss(nn.Module):
    """
    独立的InfoNCE损失模块 - 可复用于其他对比学习任务
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        初始化InfoNCE损失
        
        Args:
            temperature: 温度参数，控制分布的尖锐程度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor,
                negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            anchor: 锚点表示 [batch_size, dim]
            positive: 正样本表示 [batch_size, dim] 
            negatives: 负样本表示 [batch_size, num_negatives, dim]，如果为None则使用批次内负采样
            
        Returns:
            InfoNCE损失
        """
        batch_size = anchor.size(0)
        device = anchor.device
        
        # 归一化
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        if negatives is None:
            # 批次内负采样：使用批次中其他样本作为负样本
            representations = torch.cat([anchor, positive], dim=0)
            similarity_matrix = torch.mm(representations, representations.t()) / self.temperature
            
            # 创建标签：对于anchor[i]，positive[i]是正样本
            labels = torch.arange(batch_size, device=device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
            
            # 排除自身
            mask = torch.eye(2 * batch_size, device=device).bool()
            similarity_matrix.masked_fill_(mask, float('-inf'))
            
            # 计算交叉熵损失
            loss = F.cross_entropy(similarity_matrix, labels)
            
        else:
            # 显式负样本
            negatives = F.normalize(negatives, p=2, dim=-1)
            
            # 计算正样本相似度
            pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [batch_size]
            
            # 计算负样本相似度
            neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature
            # [batch_size, num_negatives]
            
            # InfoNCE损失
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 正样本在第0位
            
            loss = F.cross_entropy(logits, labels)
        
        return loss


def create_molecular_recovery_head(config: Dict) -> MoleculeRecoveryHead:
    """
    工厂函数：根据配置创建分子恢复任务头
    
    Args:
        config: 配置字典，包含模型参数
        
    Returns:
        配置好的分子恢复任务头
    """
    return MoleculeRecoveryHead(
        input_dim=config.get('input_dim', 300),
        projection_dim=config.get('projection_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        temperature=config.get('temperature', 0.1),
        dropout=config.get('dropout', 0.1)
    )


# 测试代码
if __name__ == "__main__":
    # 测试分子恢复任务头
    print("🧪 测试分子恢复任务头...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    recovery_head = MoleculeRecoveryHead(
        input_dim=300,
        projection_dim=128,
        hidden_dim=512,
        temperature=0.1
    ).to(device)
    
    # 模拟数据
    batch_size = 32
    original_embeddings = torch.randn(batch_size, 300).to(device)
    augmented_data = [
        {'augment_type': 'none', 'masked_indices': []},
        {'augment_type': 'atom_mask', 'masked_indices': [0, 1, 2]},
        {'augment_type': 'bond_deletion', 'masked_indices': [0, 1]},
        {'augment_type': 'subgraph_removal', 'masked_indices': [0, 1, 2, 3]}
    ]
    pretrain_infos = [{'type': 'none', 'indices': []}] * batch_size
    
    # 前向传播
    results = recovery_head(original_embeddings, augmented_data, pretrain_infos)
    
    print(f"✅ 分子恢复损失: {results[0].item():.4f}")
    print(f"✅ 对比学习准确率: {results[1]:.4f}")
    print(f"✅ 原始分子投影形状: {original_embeddings.shape}")
    print(f"✅ 增强分子投影形状: {augmented_data[0]['augmented_embeddings'].shape}")
    
    # 测试相似度计算
    similarity = recovery_head.compute_molecular_similarity(
        original_embeddings[:10], 
        augmented_data[0]['augmented_embeddings'][:10]
    )
    print(f"✅ 分子相似度矩阵形状: {similarity.shape}")
    
    print("🎉 分子恢复任务头测试完成！") 