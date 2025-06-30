#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
产物-合成子对比学习任务头 - G2Retro-P预训练架构核心组件

这个模块实现了产物分子与合成子组合之间的对比学习，学习反应的逆向映射关系。
通过对比学习使得真实的产物-合成子配对在表示空间中更相似，而错误配对更不相似。

参考文献：
- PMSR: Learning Chemical Rules of Retrosynthesis with Pre-training  
- MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks
- RetroExplainer: Retrosynthesis prediction with an interpretable deep-learning framework

核心思想：
1. 真实的产物-合成子配对应该在表示空间中相似（正样本）
2. 错误的产物-合成子配对应该不相似（负样本）
3. 通过InfoNCE损失学习反应的逆向化学规律
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List, Union
import math

class ProductSynthonContrastiveHead(nn.Module):
    """
    产物-合成子对比学习任务头
    
    学习产物分子与合成子组合之间的对应关系，这是逆合成预测的核心任务。
    通过对比学习使模型理解"这个产物应该由哪些合成子组成"的化学规律。
    """
    
    def __init__(self, 
                 product_input_dim: int = 300,      # 产物分子图嵌入维度
                 synthon_input_dim: int = 300,      # 合成子图嵌入维度
                 projection_dim: int = 256,         # 投影空间维度
                 hidden_dim: int = 512,            # 隐藏层维度
                 temperature: float = 0.07,         # InfoNCE温度参数
                 dropout: float = 0.1,
                 fusion_method: str = "attention"):  # 合成子融合方法
        """
        初始化产物-合成子对比学习任务头
        
        Args:
            product_input_dim: 产物分子图嵌入维度
            synthon_input_dim: 合成子图嵌入维度  
            projection_dim: 投影后的表示维度
            hidden_dim: 隐藏层维度
            temperature: InfoNCE损失的温度参数
            dropout: Dropout比例
            fusion_method: 合成子融合方法 ('attention', 'mean', 'max')
        """
        super(ProductSynthonContrastiveHead, self).__init__()
        
        self.product_input_dim = product_input_dim
        self.synthon_input_dim = synthon_input_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.fusion_method = fusion_method
        
        # 产物分子投影头 - 参考MolCLR架构
        self.product_projector = nn.Sequential(
            nn.Linear(product_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # 合成子投影头 - 对称设计
        self.synthon_projector = nn.Sequential(
            nn.Linear(synthon_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # G2Retro-P设计：不需要合成子融合模块
        # 合成子组合已经在数据预处理阶段完成，这里只处理单一的合成子组合分子图
        # fusion_method参数保留用于兼容性，但实际不使用
        if fusion_method not in ["attention", "mean", "max"]:
            raise ValueError(f"未支持的融合方法: {fusion_method}")
        self.fusion_method = fusion_method  # 保存用于日志记录
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重 - 参考MolCLR初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def fuse_synthon_embeddings(self, 
                               synthon_embeddings: torch.Tensor,
                               synthon_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合多个合成子的表示为单一表示
        
        Args:
            synthon_embeddings: 合成子嵌入 [batch_size, max_synthons, projection_dim]
            synthon_masks: 合成子掩码 [batch_size, max_synthons] (可选)
            
        Returns:
            融合后的合成子表示 [batch_size, projection_dim]
        """
        batch_size, max_synthons, embedding_dim = synthon_embeddings.shape
        
        if self.fusion_method == "attention":
            # 添加位置编码
            pos_encoding = self.position_encoding[:max_synthons].unsqueeze(0).expand(batch_size, -1, -1)
            synthon_embeddings = synthon_embeddings + pos_encoding
            
            # 自注意力融合
            if synthon_masks is not None:
                # 转换mask格式：True表示有效位置，False表示填充位置
                attention_mask = ~synthon_masks  # MultiheadAttention期望True表示需要mask的位置
            else:
                attention_mask = None
            
            # 转换为序列优先格式 (seq_len, batch_size, embed_dim)
            synthon_embeddings_t = synthon_embeddings.transpose(0, 1)  # [max_synthons, batch_size, projection_dim]
            
            # 使用第一个合成子作为query，所有合成子作为key和value
            query = synthon_embeddings_t[0:1, :, :]  # [1, batch_size, projection_dim]
            
            attended_output, attention_weights = self.synthon_attention(
                query=query,
                key=synthon_embeddings_t,
                value=synthon_embeddings_t,
                key_padding_mask=attention_mask
            )
            
            # 返回注意力加权的表示，转换回batch优先格式
            fused_embedding = attended_output.squeeze(0)  # [batch_size, projection_dim]
            
        elif self.fusion_method == "mean":
            # 均值聚合
            if synthon_masks is not None:
                # masked均值
                masked_embeddings = synthon_embeddings * synthon_masks.unsqueeze(-1)
                valid_counts = synthon_masks.sum(dim=1, keepdim=True).float()
                valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
                fused_embedding = masked_embeddings.sum(dim=1) / valid_counts
            else:
                fused_embedding = synthon_embeddings.mean(dim=1)
                
        elif self.fusion_method == "max":
            # 最大值聚合
            if synthon_masks is not None:
                # 对无效位置设置很小的值
                masked_embeddings = synthon_embeddings.clone()
                masked_embeddings[~synthon_masks] = float('-inf')
                fused_embedding = masked_embeddings.max(dim=1)[0]
            else:
                fused_embedding = synthon_embeddings.max(dim=1)[0]
        
        return fused_embedding
    
    def forward(self, 
                product_embeddings: torch.Tensor,
                synthon_embeddings: torch.Tensor,
                pretrain_infos: List[Dict],
                synthon_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """
        前向传播 - 计算产物-合成子对比学习损失
        
        Args:
            product_embeddings: 产物分子图嵌入 [batch_size, product_input_dim]
            synthon_embeddings: 合成子图嵌入 
                              - 2D: [batch_size, synthon_input_dim] (单合成子或已融合)
                              - 3D: [batch_size, num_synthons, synthon_input_dim] (多合成子)
            pretrain_infos: 预训练信息列表
            synthon_masks: 合成子有效性掩码 [batch_size, max_synthons] (可选)
            
        Returns:
            (loss, accuracy): 损失值和准确率
        """
        batch_size = product_embeddings.size(0)
        device = product_embeddings.device
        
        # 投影产物嵌入
        product_proj = self.product_projector(product_embeddings)  # [batch_size, projection_dim]
        
        # 处理合成子嵌入
        # G2Retro-P设计：合成子组合已经在预处理阶段组合为单一分子图
        # GMPN编码器总是输出2D张量 [batch_size, hidden_size]
        if synthon_embeddings.dim() != 2:
            raise ValueError(f"期望合成子嵌入为2D张量 [batch_size, hidden_size]，但得到 {synthon_embeddings.shape}")
        
        synthon_proj = self.synthon_projector(synthon_embeddings)  # [batch_size, projection_dim]
        
        # L2归一化（对比学习的标准做法）
        product_proj = F.normalize(product_proj, p=2, dim=1)
        synthon_proj = F.normalize(synthon_proj, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(product_proj, synthon_proj.t()) / self.temperature
        
        # 计算InfoNCE损失
        contrastive_loss = self._compute_infonce_loss(product_proj, synthon_proj)
        
        # 计算准确率
        accuracy = self._compute_accuracy(similarity_matrix)
        
        return contrastive_loss, accuracy.item()
    
    def _compute_infonce_loss(self, 
                             product_proj: torch.Tensor, 
                             synthon_proj: torch.Tensor) -> torch.Tensor:
        """
        计算产物-合成子InfoNCE对比学习损失
        
        损失函数旨在使真实的产物-合成子配对相似度更高，
        而随机配对的相似度更低。
        
        Args:
            product_proj: 归一化的产物投影表示 [batch_size, projection_dim]
            synthon_proj: 归一化的合成子投影表示 [batch_size, projection_dim]
            
        Returns:
            InfoNCE损失值
        """
        batch_size = product_proj.size(0)
        device = product_proj.device
        
        # 计算相似度矩阵
        # 行：产物，列：合成子
        similarity_matrix = torch.mm(product_proj, synthon_proj.t()) / self.temperature
        
        # 创建标签：对角线元素是正样本配对
        labels = torch.arange(batch_size, device=device)
        
        # 计算两个方向的交叉熵损失
        # 1. 产物到合成子：给定产物，找到对应的合成子
        loss_product_to_synthon = F.cross_entropy(similarity_matrix, labels)
        
        # 2. 合成子到产物：给定合成子，找到对应的产物
        loss_synthon_to_product = F.cross_entropy(similarity_matrix.t(), labels)
        
        # 对称InfoNCE损失
        total_loss = (loss_product_to_synthon + loss_synthon_to_product) / 2
        
        return total_loss
    
    def _compute_accuracy(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算top-1准确率 - 产物-合成子匹配准确率
        
        Args:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
            
        Returns:
            平均准确率
        """
        batch_size = similarity_matrix.size(0)
        
        # 产物到合成子的top-1准确率
        _, top_indices_p2s = torch.topk(similarity_matrix, k=1, dim=1)
        correct_p2s = (top_indices_p2s.squeeze() == torch.arange(batch_size, device=similarity_matrix.device)).float()
        
        # 合成子到产物的top-1准确率  
        _, top_indices_s2p = torch.topk(similarity_matrix.t(), k=1, dim=1)
        correct_s2p = (top_indices_s2p.squeeze() == torch.arange(batch_size, device=similarity_matrix.device)).float()
        
        # 平均准确率
        accuracy = (correct_p2s.mean() + correct_s2p.mean()) / 2
        
        return accuracy
    
    def predict_synthons_for_product(self, 
                                   product_embedding: torch.Tensor,
                                   candidate_synthon_embeddings: torch.Tensor,
                                   top_k: int = 5) -> Dict[str, torch.Tensor]:
        """
        给定产物分子，预测最可能的合成子组合
        
        Args:
            product_embedding: 单个产物嵌入 [1, product_input_dim]
            candidate_synthon_embeddings: 候选合成子嵌入 [num_candidates, synthon_input_dim]
            top_k: 返回top-k结果
            
        Returns:
            包含预测结果的字典
        """
        self.eval()
        with torch.no_grad():
            # 投影并归一化
            product_proj = F.normalize(self.product_projector(product_embedding), p=2, dim=1)
            
            if candidate_synthon_embeddings.dim() == 2:
                synthon_proj = F.normalize(self.synthon_projector(candidate_synthon_embeddings), p=2, dim=1)
            else:
                # 处理多合成子候选
                batch_size, max_synthons, synthon_dim = candidate_synthon_embeddings.shape
                synthon_flat = candidate_synthon_embeddings.view(-1, synthon_dim)
                synthon_proj_flat = self.synthon_projector(synthon_flat)
                synthon_proj_3d = synthon_proj_flat.view(batch_size, max_synthons, self.projection_dim)
                synthon_proj = F.normalize(self.fuse_synthon_embeddings(synthon_proj_3d), p=2, dim=1)
            
            # 计算相似度
            similarities = torch.mm(product_proj, synthon_proj.t()).squeeze(0)
            
            # 获取top-k结果
            top_similarities, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
            
            return {
                'top_indices': top_indices,
                'top_similarities': top_similarities,
                'all_similarities': similarities
            }


def create_product_synthon_contrastive_head(config: Dict) -> ProductSynthonContrastiveHead:
    """
    工厂函数：根据配置创建产物-合成子对比学习任务头
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的对比学习任务头
    """
    return ProductSynthonContrastiveHead(
        product_input_dim=config.get('product_input_dim', 300),
        synthon_input_dim=config.get('synthon_input_dim', 300),
        projection_dim=config.get('projection_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        temperature=config.get('temperature', 0.07),
        dropout=config.get('dropout', 0.1),
        fusion_method=config.get('fusion_method', 'attention')
    )


# 测试代码
if __name__ == "__main__":
    print("🧪 测试产物-合成子对比学习任务头...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建模型
    contrastive_head = ProductSynthonContrastiveHead(
        product_input_dim=300,
        synthon_input_dim=300,
        projection_dim=256,
        hidden_dim=512,
        temperature=0.07,
        fusion_method='attention'
    ).to(device)
    
    print(f"✅ 模型创建成功")
    print(f"   参数总数: {sum(p.numel() for p in contrastive_head.parameters()):,}")
    
    # 模拟数据
    batch_size = 16
    max_synthons = 4
    
    # 产物嵌入
    product_embeddings = torch.randn(batch_size, 300).to(device)
    
    # 合成子嵌入（多合成子情况）
    synthon_embeddings = torch.randn(batch_size, max_synthons, 300).to(device)
    
    # 合成子掩码（模拟不同反应有不同数量的合成子）
    synthon_masks = torch.ones(batch_size, max_synthons, dtype=torch.bool).to(device)
    for i in range(batch_size):
        num_valid = torch.randint(1, max_synthons + 1, (1,)).item()
        synthon_masks[i, num_valid:] = False
    
    print(f"\n🧪 测试前向传播...")
    
    # 前向传播
    results = contrastive_head(product_embeddings, synthon_embeddings, synthon_masks)
    
    print(f"✅ 对比学习损失: {results['loss'].item():.4f}")
    print(f"✅ Top-1准确率: {results['accuracy'].item():.4f}")
    print(f"✅ 产物投影形状: {results['product_proj'].shape}")
    print(f"✅ 合成子投影形状: {results['synthon_proj'].shape}")
    print(f"✅ 相似度矩阵形状: {results['similarity_matrix'].shape}")
    
    # 测试预测功能
    print(f"\n🔍 测试合成子预测...")
    
    # 模拟候选合成子
    num_candidates = 50
    candidate_synthons = torch.randn(num_candidates, 300).to(device)
    
    # 预测
    prediction_results = contrastive_head.predict_synthons_for_product(
        product_embeddings[:1], 
        candidate_synthons,
        top_k=5
    )
    
    print(f"✅ Top-5相似度: {prediction_results['top_similarities']}")
    print(f"✅ Top-5索引: {prediction_results['top_indices']}")
    
    # 测试不同融合方法
    print(f"\n🔄 测试不同合成子融合方法...")
    
    for fusion_method in ['attention', 'mean', 'max']:
        test_head = ProductSynthonContrastiveHead(
            fusion_method=fusion_method
        ).to(device)
        
        with torch.no_grad():
            test_results = test_head(product_embeddings, synthon_embeddings, synthon_masks)
        
        print(f"  {fusion_method:10s}: Loss={test_results['loss'].item():.4f}, "
              f"Acc={test_results['accuracy'].item():.4f}")
    
    print("🎉 产物-合成子对比学习任务头测试完成！") 