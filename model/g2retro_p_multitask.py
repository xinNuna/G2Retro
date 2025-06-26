#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2Retro-P多任务预训练模型 - 完整的三任务并行架构

这个模块实现了G2Retro-P的完整预训练架构，包含三个并行任务头：
1. 基础任务头：反应中心识别（复用G2Retro的MolCenter）
2. 分子恢复头：MolCLR风格的分子对比学习
3. 产物-合成子对比头：学习逆合成映射关系

参考文献：
- G2Retro: Graph-Guided Molecule Generation for Retrosynthesis Prediction
- PMSR: Learning Chemical Rules of Retrosynthesis with Pre-training
- MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks

设计理念：
通过多任务预训练让模型同时学习：
- 化学反应的基本规律（反应中心识别）
- 分子表示的鲁棒性（分子恢复）
- 逆合成的映射关系（产物-合成子对比）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings

# 导入G2Retro现有组件
try:
    from mol_tree import MolTree
    from vocab import Vocab
    from config import device
except ImportError:
    warnings.warn("无法导入G2Retro组件，某些功能可能受限")

# 导入我们实现的任务头
from molecule_recovery_head import MoleculeRecoveryHead
from product_synthon_contrastive_head import ProductSynthonContrastiveHead

class G2RetroPMultiTaskModel(nn.Module):
    """
    G2Retro-P多任务预训练模型
    
    集成三个并行任务头的完整预训练架构：
    1. 基础任务：反应中心识别（复用G2Retro MolCenter）
    2. 分子恢复：MolCLR对比学习
    3. 产物-合成子对比：逆合成映射学习
    """
    
    def __init__(self,
                 # 分子图编码器配置
                 vocab: Vocab,
                 atom_vocab: List[str],
                 encoder_hidden_dim: int = 300,
                 encoder_depth: int = 3,
                 
                 # 任务头配置
                 recovery_config: Optional[Dict] = None,
                 contrastive_config: Optional[Dict] = None,
                 
                 # 多任务学习配置
                 task_weights: Optional[Dict[str, float]] = None,
                 temperature_schedule: str = "static",  # static, cosine, linear
                 
                 # 训练配置
                 dropout: float = 0.1):
        """
        初始化G2Retro-P多任务预训练模型
        
        Args:
            vocab: 分子树词汇表
            atom_vocab: 原子词汇表
            encoder_hidden_dim: 编码器隐藏维度
            encoder_depth: 编码器深度
            recovery_config: 分子恢复任务头配置
            contrastive_config: 产物-合成子对比任务头配置
            task_weights: 任务权重字典
            temperature_schedule: 温度调度策略
            dropout: Dropout比例
        """
        super(G2RetroPMultiTaskModel, self).__init__()
        
        self.vocab = vocab
        self.atom_vocab = atom_vocab
        self.encoder_hidden_dim = encoder_hidden_dim
        self.vocab_size = len(vocab)
        self.atom_vocab_size = len(atom_vocab)
        
        # 默认任务权重
        self.task_weights = task_weights or {
            'base_task': 1.0,           # 基础任务（反应中心识别）
            'molecular_recovery': 1.0,   # 分子恢复任务
            'product_synthon': 1.0       # 产物-合成子对比任务
        }
        
        self.temperature_schedule = temperature_schedule
        self.current_epoch = 0
        
        # === 1. 分子图编码器 ===
        # 这里使用简化的图编码器，实际应该复用G2Retro的图编码器
        self.molecule_encoder = self._build_molecule_encoder()
        
        # === 2. 基础任务头（反应中心识别）===
        # 复用G2Retro的MolCenter组件
        self.base_task_head = self._build_base_task_head()
        
        # === 3. 分子恢复任务头 ===
        recovery_config = recovery_config or {}
        self.molecular_recovery_head = MoleculeRecoveryHead(
            input_dim=encoder_hidden_dim,
            projection_dim=recovery_config.get('projection_dim', 128),
            hidden_dim=recovery_config.get('hidden_dim', 512),
            temperature=recovery_config.get('temperature', 0.1),
            dropout=dropout
        )
        
        # === 4. 产物-合成子对比任务头 ===
        contrastive_config = contrastive_config or {}
        self.product_synthon_head = ProductSynthonContrastiveHead(
            product_input_dim=encoder_hidden_dim,
            synthon_input_dim=encoder_hidden_dim,
            projection_dim=contrastive_config.get('projection_dim', 256),
            hidden_dim=contrastive_config.get('hidden_dim', 512),
            temperature=contrastive_config.get('temperature', 0.07),
            dropout=dropout,
            fusion_method=contrastive_config.get('fusion_method', 'attention')
        )
        
        # 权重初始化
        self._init_weights()
    
    def _build_molecule_encoder(self) -> nn.Module:
        """
        构建分子图编码器
        
        在实际实现中，这里应该复用G2Retro的图神经网络编码器。
        目前使用简化的MLP作为占位符。
        """
        # 占位符编码器 - 实际应该使用G2Retro的GNN编码器
        return nn.Sequential(
            nn.Linear(self.vocab_size + self.atom_vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.encoder_hidden_dim),
            nn.ReLU()
        )
    
    def _build_base_task_head(self) -> nn.Module:
        """
        构建基础任务头（反应中心识别）
        
        在实际实现中，这里应该复用G2Retro的MolCenter组件。
        """
        # 占位符 - 实际应该使用G2Retro的MolCenter
        return nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.atom_vocab_size)  # 预测原子级别的反应中心
        )
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode_molecule(self, mol_tree: Union[MolTree, torch.Tensor]) -> torch.Tensor:
        """
        编码分子为向量表示
        
        Args:
            mol_tree: 分子树对象或特征张量
            
        Returns:
            分子嵌入向量 [hidden_dim]
        """
        if isinstance(mol_tree, torch.Tensor):
            # 如果输入已经是张量，直接使用
            mol_features = mol_tree
        else:
            # 如果是MolTree对象，需要提取特征
            # 这里使用占位符，实际应该调用G2Retro的特征提取方法
            mol_features = torch.randn(self.vocab_size + self.atom_vocab_size)
        
        # 通过编码器获得分子嵌入
        mol_embedding = self.molecule_encoder(mol_features)
        
        return mol_embedding
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 执行三个并行任务
        
        Args:
            batch_data: 批次数据字典，包含：
                - 'product_trees': 产物分子树列表
                - 'synthon_trees': 合成子树列表
                - 'augmented_data': 增强数据字典
                - 'reaction_centers': 反应中心标签（可选）
                
        Returns:
            包含所有任务结果的字典
        """
        batch_size = len(batch_data['product_trees'])
        device = next(self.parameters()).device
        
        # === 1. 分子编码阶段 ===
        
        # 编码产物分子
        product_embeddings = []
        for mol_tree in batch_data['product_trees']:
            # 这里使用随机嵌入作为占位符，实际应该调用真实的编码方法
            mol_emb = torch.randn(self.encoder_hidden_dim).to(device)
            product_embeddings.append(mol_emb)
        product_embeddings = torch.stack(product_embeddings)  # [batch_size, hidden_dim]
        
        # 编码合成子
        synthon_embeddings = []
        max_synthons = 0
        for synthon_list in batch_data['synthon_trees']:
            if synthon_list:
                batch_synthons = []
                for synthon in synthon_list:
                    # 同样使用占位符
                    synthon_emb = torch.randn(self.encoder_hidden_dim).to(device)
                    batch_synthons.append(synthon_emb)
                synthon_embeddings.append(torch.stack(batch_synthons))
                max_synthons = max(max_synthons, len(batch_synthons))
            else:
                # 空合成子列表的情况
                synthon_embeddings.append(torch.zeros(1, self.encoder_hidden_dim).to(device))
                max_synthons = max(max_synthons, 1)
        
        # 填充为相同长度
        padded_synthon_embeddings = []
        synthon_masks = []
        for synthon_batch in synthon_embeddings:
            current_len = synthon_batch.shape[0]
            if current_len < max_synthons:
                # 填充
                padding = torch.zeros(max_synthons - current_len, self.encoder_hidden_dim).to(device)
                padded_synthon = torch.cat([synthon_batch, padding], dim=0)
                mask = torch.cat([torch.ones(current_len), torch.zeros(max_synthons - current_len)])
            else:
                padded_synthon = synthon_batch
                mask = torch.ones(current_len)
            
            padded_synthon_embeddings.append(padded_synthon)
            synthon_masks.append(mask.bool())
        
        synthon_embeddings_tensor = torch.stack(padded_synthon_embeddings)  # [batch_size, max_synthons, hidden_dim]
        synthon_masks_tensor = torch.stack(synthon_masks).to(device)  # [batch_size, max_synthons]
        
        # === 2. 任务执行阶段 ===
        
        results = {}
        
        # 任务1：基础任务（反应中心识别）
        if 'base_task' in self.task_weights and self.task_weights['base_task'] > 0:
            base_logits = self.base_task_head(product_embeddings)  # [batch_size, atom_vocab_size]
            
            # 计算损失（如果有标签）
            if 'reaction_centers' in batch_data:
                base_loss = F.cross_entropy(base_logits, batch_data['reaction_centers'])
            else:
                base_loss = torch.tensor(0.0, device=device)
            
            results['base_task'] = {
                'logits': base_logits,
                'loss': base_loss
            }
        
        # 任务2：分子恢复（MolCLR对比学习）
        if 'molecular_recovery' in self.task_weights and self.task_weights['molecular_recovery'] > 0:
            # 生成增强的分子嵌入（这里使用随机噪声模拟）
            augmented_embeddings = product_embeddings + torch.randn_like(product_embeddings) * 0.1
            
            recovery_results = self.molecular_recovery_head(product_embeddings, augmented_embeddings)
            results['molecular_recovery'] = recovery_results
        
        # 任务3：产物-合成子对比学习
        if 'product_synthon' in self.task_weights and self.task_weights['product_synthon'] > 0:
            contrastive_results = self.product_synthon_head(
                product_embeddings, 
                synthon_embeddings_tensor, 
                synthon_masks_tensor
            )
            results['product_synthon'] = contrastive_results
        
        # === 3. 多任务损失整合 ===
        
        total_loss = torch.tensor(0.0, device=device)
        loss_details = {}
        
        for task_name, task_weight in self.task_weights.items():
            if task_name in results and task_weight > 0:
                task_loss = results[task_name]['loss']
                weighted_loss = task_weight * task_loss
                total_loss += weighted_loss
                
                loss_details[f'{task_name}_loss'] = task_loss.item()
                loss_details[f'{task_name}_weighted_loss'] = weighted_loss.item()
        
        results['total_loss'] = total_loss
        results['loss_details'] = loss_details
        
        return results
    
    def update_temperature_schedule(self, epoch: int, max_epochs: int):
        """
        更新温度调度
        
        Args:
            epoch: 当前轮数
            max_epochs: 最大轮数
        """
        self.current_epoch = epoch
        
        if self.temperature_schedule == "cosine":
            # 余弦退火温度调度
            progress = epoch / max_epochs
            for head in [self.molecular_recovery_head, self.product_synthon_head]:
                original_temp = 0.1 if hasattr(head, 'temperature') else 0.07
                head.temperature = original_temp * (0.5 * (1 + np.cos(np.pi * progress)))
        
        elif self.temperature_schedule == "linear":
            # 线性温度调度
            progress = epoch / max_epochs
            for head in [self.molecular_recovery_head, self.product_synthon_head]:
                original_temp = 0.1 if hasattr(head, 'temperature') else 0.07
                head.temperature = original_temp * (1 - 0.5 * progress)
    
    def set_task_weights(self, new_weights: Dict[str, float]):
        """
        动态调整任务权重
        
        Args:
            new_weights: 新的任务权重字典
        """
        self.task_weights.update(new_weights)
    
    def get_molecular_embeddings(self, mol_trees: List[MolTree]) -> torch.Tensor:
        """
        获取分子嵌入（用于下游任务）
        
        Args:
            mol_trees: 分子树列表
            
        Returns:
            分子嵌入矩阵 [num_molecules, hidden_dim]
        """
        self.eval()
        embeddings = []
        
        with torch.no_grad():
            for mol_tree in mol_trees:
                mol_emb = self.encode_molecule(mol_tree)
                embeddings.append(mol_emb)
        
        return torch.stack(embeddings)
    
    def save_pretrained_weights(self, save_path: str):
        """
        保存预训练权重
        
        Args:
            save_path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab': self.vocab,
            'atom_vocab': self.atom_vocab,
            'task_weights': self.task_weights,
            'encoder_hidden_dim': self.encoder_hidden_dim
        }, save_path)
        
        print(f"✅ 预训练权重已保存到: {save_path}")
    
    @classmethod
    def load_pretrained_weights(cls, load_path: str, **kwargs):
        """
        加载预训练权重
        
        Args:
            load_path: 加载路径
            **kwargs: 额外的模型参数
            
        Returns:
            加载权重后的模型
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # 从checkpoint恢复配置
        vocab = checkpoint['vocab']
        atom_vocab = checkpoint['atom_vocab']
        task_weights = checkpoint.get('task_weights', None)
        encoder_hidden_dim = checkpoint.get('encoder_hidden_dim', 300)
        
        # 创建模型
        model = cls(
            vocab=vocab,
            atom_vocab=atom_vocab,
            encoder_hidden_dim=encoder_hidden_dim,
            task_weights=task_weights,
            **kwargs
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ 预训练权重已从 {load_path} 加载")
        
        return model


def create_g2retro_p_model(config: Dict) -> G2RetroPMultiTaskModel:
    """
    工厂函数：根据配置创建G2Retro-P模型
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的G2Retro-P模型
    """
    # 这里需要实际的词汇表，暂时使用占位符
    vocab = config.get('vocab', None)
    atom_vocab = config.get('atom_vocab', ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    
    if vocab is None:
        # 创建占位符词汇表
        class PlaceholderVocab:
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
        vocab = PlaceholderVocab(100)
    
    return G2RetroPMultiTaskModel(
        vocab=vocab,
        atom_vocab=atom_vocab,
        encoder_hidden_dim=config.get('encoder_hidden_dim', 300),
        encoder_depth=config.get('encoder_depth', 3),
        recovery_config=config.get('recovery_config', {}),
        contrastive_config=config.get('contrastive_config', {}),
        task_weights=config.get('task_weights', {}),
        temperature_schedule=config.get('temperature_schedule', 'static'),
        dropout=config.get('dropout', 0.1)
    )


# 测试代码
if __name__ == "__main__":
    print("🚀 测试G2Retro-P多任务预训练模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建配置
    config = {
        'encoder_hidden_dim': 300,
        'recovery_config': {
            'projection_dim': 128,
            'temperature': 0.1
        },
        'contrastive_config': {
            'projection_dim': 256,
            'temperature': 0.07,
            'fusion_method': 'attention'
        },
        'task_weights': {
            'base_task': 1.0,
            'molecular_recovery': 1.0,
            'product_synthon': 1.0
        }
    }
    
    # 创建模型
    model = create_g2retro_p_model(config).to(device)
    
    print(f"✅ G2Retro-P模型创建成功")
    print(f"   总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟批次数据
    batch_size = 8
    batch_data = {
        'product_trees': [None] * batch_size,  # 占位符
        'synthon_trees': [[None, None] for _ in range(batch_size)],  # 每个产物对应2个合成子
        'augmented_data': {},
        'reaction_centers': torch.randint(0, 9, (batch_size,)).to(device)
    }
    
    print(f"\n🧪 测试多任务前向传播...")
    
    # 前向传播
    results = model(batch_data)
    
    print(f"✅ 总损失: {results['total_loss'].item():.4f}")
    print(f"✅ 损失详情: {results['loss_details']}")
    
    # 测试各个任务头
    for task_name in ['base_task', 'molecular_recovery', 'product_synthon']:
        if task_name in results:
            task_result = results[task_name]
            print(f"   {task_name}: Loss={task_result['loss'].item():.4f}")
            if 'accuracy' in task_result:
                print(f"                     Acc={task_result['accuracy'].item():.4f}")
    
    # 测试温度调度
    print(f"\n🌡️  测试温度调度...")
    original_temp = model.molecular_recovery_head.temperature
    model.update_temperature_schedule(epoch=50, max_epochs=100)
    new_temp = model.molecular_recovery_head.temperature
    print(f"   分子恢复温度: {original_temp:.4f} -> {new_temp:.4f}")
    
    # 测试权重调整
    print(f"\n⚖️  测试动态权重调整...")
    model.set_task_weights({'molecular_recovery': 2.0, 'product_synthon': 0.5})
    print(f"   更新后的任务权重: {model.task_weights}")
    
    print("🎉 G2Retro-P多任务模型测试完成！") 