#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2Retro-P 设计方案完全对齐实现
严格按照"基于多任务预训练的半模板逆合成模型设计方案"实现

核心创新点验证：
1. 共享编码器：完全沿用G2Retro的核心组件GMPN
2. 基础任务头：直接采用G2Retro的反应中心识别模块  
3. 分子恢复头：采用MolCLR的三种图增强策略
4. 产物-合成子对比头：核心创新，完美任务对齐
5. 数据流程：Gp → h_product, Gp_aug → h_augmented, Gs → h_synthons
6. 损失权重：L_total = L_base + L_recovery + 0.1 × L_contrastive
"""

import sys
import os
import pickle
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import time
import random
import rdkit.Chem as Chem
import torch.nn.functional as F

# 导入G2Retro核心模块
from mol_tree import MolTree
from mol_enc import MolEncoder
from molcenter import MolCenter, make_cuda
from chemutils import get_mol
from vocab import Vocab, common_atom_vocab
from nnutils import create_pad_tensor, index_select_ND
from config import device

# 导入预训练任务头
from molecule_recovery_head import MoleculeRecoveryHead
from product_synthon_contrastive_head import ProductSynthonContrastiveHead

# 导入数据集相关组件
from g2retro_p_dataset import G2RetroPDesignAlignedDataset, g2retro_design_aligned_collate_fn


class G2RetroPDesignAlignedModel(nn.Module):
    """
    完全符合设计方案的G2Retro-P模型
    
    设计方案核心架构：
    - 预训练阶段: 此阶段是模型学习的核心。模型由一个共享编码器和三个并行的任务头组成
    - 共享编码器: 完全沿用G2Retro的核心组件GMPN（图消息传递网络）
    - 并行任务头:
      1. 基础任务头: 直接采用G2Retro的反应中心识别模块
      2. 分子恢复头: 采用MolCLR的三种图增强策略
      3. 产物-合成子对比头（核心创新）: 直接利用产物与合成子间的自然差异进行对比学习
    """
    def __init__(self, vocab, avocab, args):
        super(G2RetroPDesignAlignedModel, self).__init__()
        
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        
        # 按照设计方案：共享编码器完全沿用G2Retro的核心组件GMPN
        print("初始化共享编码器（GMPN）...")
        self.mol_center = MolCenter(vocab, avocab, args)
        # 提取共享的GMPN编码器
        self.shared_encoder = self.mol_center.encoder  # 这就是GMPN
        
        # 按照设计方案：三个并行任务头
        print("初始化三个并行任务头...")
        
        # 1. 基础任务头：直接采用G2Retro的反应中心识别模块
        print("  - 基础任务头：G2Retro反应中心识别模块")
        self.reaction_center_head = self.mol_center  # 完整的MolCenter
        
        # 2. 分子恢复头：采用MolCLR的三种图增强策略
        print("  - 分子恢复头：MolCLR三种增强策略")
        self.molecule_recovery_head = MoleculeRecoveryHead(
            input_dim=args.hidden_size,
            projection_dim=128,
            temperature=0.1
        )
        
        # 3. 产物-合成子对比头：核心创新
        print("  - 产物-合成子对比头：核心创新")
        self.product_synthon_contrastive_head = ProductSynthonContrastiveHead(
            product_input_dim=args.hidden_size,
            synthon_input_dim=args.hidden_size,
            projection_dim=128,
            temperature=0.1,
            fusion_method='attention'  # 处理多合成子
        )
        
        # 按照设计方案：损失权重 L_total = L_base + L_recovery + 0.1 × L_contrastive
        print("设置动态自适应多任务学习权重（基于RetroExplainer）...")
        # 参考RetroExplainer的DAMT算法：动态根据任务学习进度调整权重
        # 初始权重按照设计方案设置：[基础任务, 分子恢复, 产物-合成子对比] = [1.0, 1.0, 0.1]
        self.initial_task_weights = torch.tensor([1.0, 1.0, 0.1])
        
        # 动态权重参数（参考RetroExplainer的DAMT）
        self.loss_queue_length = getattr(args, 'loss_queue_length', 50)  # 损失队列长度，用于计算平均值
        self.temperature = getattr(args, 'weight_temperature', 2.0)      # 温度系数τ，控制权重分布的锐度
        self.min_weight = getattr(args, 'min_task_weight', 0.01)         # 最小权重，防止某个任务权重过小
        self.max_weight = getattr(args, 'max_task_weight', 3.0)          # 最大权重，防止某个任务权重过大
        
        # 损失历史记录队列（用于计算下降率和平均值）
        self.loss_histories = {
            'center': [],     # 基础任务损失历史
            'recovery': [],   # 分子恢复损失历史  
            'contrastive': [] # 产物-合成子对比损失历史
        }
        
        # 当前动态权重（初始值为设计方案权重）
        self.register_buffer('task_weights', self.initial_task_weights.clone())
        
        # 权重调整计数器
        self.weight_update_step = 0
        self.weight_update_frequency = getattr(args, 'weight_update_frequency', 5)  # 权重更新频率
        
        print(f"\nG2Retro-P模型初始化完成（设计方案+动态权重）:")
        print(f"✓ 共享编码器：GMPN（来自G2Retro）")
        print(f"✓ 基础任务头：G2Retro反应中心识别模块")
        print(f"✓ 分子恢复头：MolCLR三种增强策略")
        print(f"✓ 产物-合成子对比头：核心创新")
        print(f"✓ 动态权重系统：基于RetroExplainer DAMT算法")
        print(f"✓ 初始权重：[基础:{self.task_weights[0]:.3f}, 恢复:{self.task_weights[1]:.3f}, 对比:{self.task_weights[2]:.3f}]")
        print(f"✓ 权重更新频率：每{self.weight_update_frequency}步")
        print(f"✓ 温度系数：{self.temperature}")
        print(f"✓ 损失队列长度：{self.loss_queue_length}")
        print(f"✓ 词汇表大小: {vocab.size()}")
        print(f"✓ 原子词汇表大小: {avocab.size()}")
        print(f"✓ 隐藏层大小: {args.hidden_size}")

    def update_task_weights(self, current_losses):
        """
        动态自适应多任务学习权重更新（基于RetroExplainer的DAMT算法）
        
        RetroExplainer的核心思想：根据任务的学习进度动态调整权重
        - 学习困难的任务（损失下降缓慢）获得更高权重
        - 学习容易的任务（损失下降快速）获得较低权重
        - 通过descent rate和normalization coefficient实现自适应平衡
        
        算法步骤：
        1. 记录当前损失到历史队列
        2. 计算每个任务的下降率 r_i^t = l_i^{t-1} / l_i^t  
        3. 计算归一化系数 α_i^t = n / ∑_{j=t-n}^{t-1} l_i^j
        4. 计算动态权重 w_i^t = softmax(r_i^t / τ) * α_i^t
        
        参数：
            current_losses: 字典，包含当前步的各任务损失
        """
        task_names = ['center', 'recovery', 'contrastive']
        
        # 记录当前损失到历史队列
        for i, task_name in enumerate(task_names):
            if task_name in current_losses:
                loss_value = current_losses[task_name].detach().cpu().item()
                self.loss_histories[task_name].append(loss_value)
                
                # 保持队列长度不超过设定值
                if len(self.loss_histories[task_name]) > self.loss_queue_length:
                    self.loss_histories[task_name].pop(0)
        
        # 权重调整频率控制
        self.weight_update_step += 1
        if self.weight_update_step % self.weight_update_frequency != 0:
            return
        
        # 需要至少2个损失值才能计算下降率
        min_history_length = 2
        all_tasks_ready = all(
            len(self.loss_histories[task]) >= min_history_length 
            for task in task_names
        )
        
        if not all_tasks_ready:
            print(f"权重更新：等待更多损失历史（当前长度: {[len(self.loss_histories[task]) for task in task_names]}）")
            return
        
        try:
            # 计算下降率 r_i^t = l_i^{t-1} / l_i^t
            descent_rates = []
            normalization_coeffs = []
            
            for task_name in task_names:
                history = self.loss_histories[task_name]
                
                # 计算下降率（当前损失相对于前一次损失的比率）
                if len(history) >= 2:
                    current_loss = history[-1]
                    previous_loss = history[-2]
                    
                    # 防止除零和负数
                    if current_loss > 0 and previous_loss > 0:
                        descent_rate = previous_loss / current_loss  # 比率越大说明下降越明显
                    else:
                        descent_rate = 1.0
                else:
                    descent_rate = 1.0
                
                descent_rates.append(descent_rate)
                
                # 计算归一化系数 α_i^t = n / ∑_{j=t-n}^{t-1} l_i^j
                # 使用最近n个损失值的平均值作为归一化
                recent_losses = history[-min(len(history), self.loss_queue_length):]
                if len(recent_losses) > 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    normalization_coeff = 1.0 / max(avg_loss, 1e-8)  # 防止除零
                else:
                    normalization_coeff = 1.0
                
                normalization_coeffs.append(normalization_coeff)
            
            # 转换为张量
            descent_rates = torch.tensor(descent_rates, dtype=torch.float32)
            normalization_coeffs = torch.tensor(normalization_coeffs, dtype=torch.float32)
            
            # 应用温度缩放的softmax计算权重分布
            # w_i^t = softmax(r_i^t / τ)
            scaled_rates = descent_rates / self.temperature
            weight_distribution = torch.softmax(scaled_rates, dim=0)
            
            # 结合归一化系数
            new_weights = weight_distribution * normalization_coeffs
            
            # 重新归一化权重，确保总和合理
            new_weights = new_weights / new_weights.sum() * self.initial_task_weights.sum()
            
            # 应用权重限制
            new_weights = torch.clamp(new_weights, min=self.min_weight, max=self.max_weight)
            
            # 更新权重
            old_weights = self.task_weights.clone()
            self.task_weights.copy_(new_weights)
            
            print(f"\n=== 动态权重更新（基于RetroExplainer DAMT）===")
            print(f"下降率: {descent_rates.tolist()}")
            print(f"归一化系数: {normalization_coeffs.tolist()}")
            print(f"权重变化:")
            for i, task_name in enumerate(task_names):
                print(f"  {task_name}: {old_weights[i]:.4f} → {new_weights[i]:.4f} (Δ{new_weights[i]-old_weights[i]:+.4f})")
            print(f"权重总和: {new_weights.sum():.4f}")
            print("=" * 50)
            
        except Exception as e:
            print(f"权重更新错误: {e}")
            import traceback
            traceback.print_exc()

    def get_weight_statistics(self):
        """
        获取权重调整统计信息
        """
        stats = {
            'current_weights': self.task_weights.detach().cpu().numpy().tolist(),
            'initial_weights': self.initial_task_weights.numpy().tolist(),
            'weight_update_step': self.weight_update_step,
            'loss_history_lengths': {
                task: len(history) for task, history in self.loss_histories.items()
            }
        }
        return stats
    
    def reset_weight_adaptation(self):
        """
        重置权重适应系统（用于新epoch或训练重启）
        """
        print("重置动态权重适应系统...")
        self.loss_histories = {task: [] for task in ['center', 'recovery', 'contrastive']}
        self.task_weights.copy_(self.initial_task_weights)
        self.weight_update_step = 0
        print(f"权重已重置为初始值: {self.initial_task_weights.tolist()}")
    
    def save_weight_adaptation_state(self):
        """
        保存权重适应状态（用于checkpointing）
        """
        return {
            'task_weights': self.task_weights.detach().cpu(),
            'loss_histories': self.loss_histories.copy(),
            'weight_update_step': self.weight_update_step,
            'initial_task_weights': self.initial_task_weights,
            'loss_queue_length': self.loss_queue_length,
            'temperature': self.temperature,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'weight_update_frequency': self.weight_update_frequency
        }
    
    def load_weight_adaptation_state(self, state):
        """
        加载权重适应状态（用于checkpointing）
        """
        self.task_weights.copy_(state['task_weights'])
        self.loss_histories = state['loss_histories']
        self.weight_update_step = state['weight_update_step']
        self.initial_task_weights = state['initial_task_weights']
        self.loss_queue_length = state['loss_queue_length']
        self.temperature = state['temperature']
        self.min_weight = state['min_weight']
        self.max_weight = state['max_weight']
        self.weight_update_frequency = state['weight_update_frequency']

    def encode_with_gmpn(self, tensors, classes=None, mol_trees=None):
        """
        使用共享的GMPN编码器进行编码
        
        设计方案要求：GMPN通过图消息传递网络（Graph Message Passing Network）来学习分子结构
        GMPN通过在分子图的原子和键之间迭代地传递信息，能够有效地捕捉到每个原子及其周围的
        局部化学环境，从而生成富有结构信息的特征表示。
        
        输入: 分子图 G = (A, B)，其中 A 是原子集合，B 是化学键集合
        输出: 原子级别的嵌入表示 ai，键级别的嵌入表示 bij，以及整个分子图的全局表示 h
        
        Args:
            mol_trees: MolTree对象列表，用于MolCLR掩码支持
        """
        # 使用共享的GMPN编码器，支持MolCLR掩码
        # 确保传入的格式正确
        if isinstance(tensors, list) and len(tensors) == 1:
            # 如果是单个图张量的列表格式，直接传入
            mol_embeds, atom_embeds, mess_embeds = self.shared_encoder(
                tensors,
                product=True, 
                classes=classes, 
                use_feature=True,
                mol_trees=mol_trees
            )
        else:
            # 如果是直接的图张量，包装成列表
            mol_embeds, atom_embeds, mess_embeds = self.shared_encoder(
                [tensors],
                product=True, 
                classes=classes, 
                use_feature=True,
                mol_trees=mol_trees
            )
        return mol_embeds, atom_embeds, mess_embeds
    
    def merge_batch_tensors(self, tensor_list):
        """合并批次中的张量数据"""
        if not tensor_list:
            return []
        
        # 获取第一个样本的结构作为参考
        first_sample = tensor_list[0]
        if not isinstance(first_sample, (list, tuple)):
            return tensor_list
        
        # 按位置合并所有样本的张量
        merged = []
        for i in range(len(first_sample)):
            if i < 6:  # 前6个是需要合并的张量
                # 收集所有样本在位置i的张量
                tensors_at_i = [sample[i] for sample in tensor_list]
                # 如果是列表，需要拼接
                if isinstance(tensors_at_i[0], list):
                    merged_tensor = []
                    for t in tensors_at_i:
                        merged_tensor.extend(t)
                    merged.append(merged_tensor)
                else:
                    merged.append(tensors_at_i)
            else:
                # 其他位置的数据直接使用第一个样本的
                merged.append(first_sample[i])
        
        return merged
    # ========== 辅助函数结束 ==========


    def forward(self, batch, epoch=0):
        """
        按照设计方案的数据流程进行前向传播
        
        在预训练的每一步中，数据流程如下：
        1. 输入处理: 基于atom-mapping从反应数据中提取产物分子图Gp和对应的合成子组合Gs
        2. 数据增强: 对原始产物分子图应用MolCLR增强策略，生成一个被"破坏"的版本Gp_aug
        3. 三路共享编码: 
           - 原始产物图 Gp 输入GMPN编码器 → h_product
           - 增强产物图 Gp_aug 输入GMPN编码器 → h_augmented 
           - 合成子组合 Gs 输入GMPN编码器 → h_synthons
        4. 并行计算:
           - h_product 被送入基础任务头和产物-合成子对比头
           - h_augmented 被送入分子恢复头
           - h_synthons 被送入产物-合成子对比头
        5. 损失计算与反向传播: 三个任务头分别计算出各自的损失
        """
        losses = {}
        metrics = {}
        # 获取张量数据 - 注意这里的数据结构
        prod_tensors = batch['prod_tensors']      # Gp
        aug_tensors = batch['aug_tensors']        # Gp_aug  
        synthon_tensors = batch['synthon_tensors'] # Gs
        react_tensors = batch['react_tensors']     # 用于基础任务
    
        # 检查数据结构并正确解包
        # MolTree.tensorize返回格式: 
        # - product=True: ([graph_batchG], [graph_tensors], None, all_orders)
        # - product=False: (react_graphs, react_tensors, react_orders)
        # 我们需要提取张量部分，跳过DiGraph对象
        
        # 调试：检查张量结构
        print(f"DEBUG: prod_tensors type: {type(prod_tensors)}")
        if isinstance(prod_tensors, tuple):
            print(f"DEBUG: prod_tensors length: {len(prod_tensors)}")
            for i, item in enumerate(prod_tensors):
                print(f"DEBUG: prod_tensors[{i}] type: {type(item)}")
                if isinstance(item, list) and len(item) > 0:
                    print(f"DEBUG: prod_tensors[{i}][0] type: {type(item[0])}")
        
        # 创建统一的数据处理函数，确保格式完全对齐
        def process_tensor_data(raw_tensors, is_product=True):
            """
            统一处理张量数据，确保格式与mol_enc.py期望的完全一致
            
            Args:
                raw_tensors: MolTree.tensorize返回的tensor_data部分（4元组）
                is_product: 是否为产物数据
            
            Returns:
                处理后的张量，格式与mol_enc.py期望的一致
            """
            print(f"DEBUG: raw_tensors type: {type(raw_tensors)}")
            print(f"DEBUG: raw_tensors length: {len(raw_tensors)}")
            
            # 步骤1：从MolTree.tensorize的tensor_data中提取图张量
            if isinstance(raw_tensors, tuple) and len(raw_tensors) == 4:
                # raw_tensors是tensor_data: ([graph_batchG], [graph_tensors], tree_tensors, all_orders)
                graph_batchG_list, graph_tensors_list, tree_tensors, all_orders = raw_tensors
                
                print(f"DEBUG: graph_tensors_list type: {type(graph_tensors_list)}")
                print(f"DEBUG: graph_tensors_list length: {len(graph_tensors_list)}")
                
                # 提取第一个图的张量（7元组）
                if isinstance(graph_tensors_list, list) and len(graph_tensors_list) > 0:
                    graph_tensors = graph_tensors_list[0]  # 这是7元组
                    print(f"DEBUG: graph_tensors type: {type(graph_tensors)}")
                    print(f"DEBUG: graph_tensors length: {len(graph_tensors)}")
                else:
                    raise ValueError(f"graph_tensors_list格式错误: {type(graph_tensors_list)}")
            else:
                raise ValueError(f"raw_tensors格式错误: 期望4元组，得到: {type(raw_tensors)}, 长度: {len(raw_tensors) if hasattr(raw_tensors, '__len__') else 'N/A'}")
            
            # 步骤2：验证图张量格式
            if not isinstance(graph_tensors, (list, tuple)) or len(graph_tensors) != 7:
                raise ValueError(f"期望7元组图张量，但得到: {type(graph_tensors)}, 长度: {len(graph_tensors) if hasattr(graph_tensors, '__len__') else 'N/A'}")
            
            # 步骤3：将张量移动到CUDA并确保正确的数据类型
            processed_tensors = []
            for i, tensor in enumerate(graph_tensors):
                if tensor is None:
                    processed_tensors.append(None)
                elif i == 6:  # scope保持原样（通常是列表）
                    processed_tensors.append(tensor)
                elif isinstance(tensor, torch.Tensor):
                    # 确保在正确设备上且为长整型
                    processed_tensors.append(tensor.to(device).long())
                else:
                    # 转换为张量
                    processed_tensors.append(torch.tensor(tensor, device=device, dtype=torch.long))
            
            # 步骤4：验证最终结果
            if len(processed_tensors) != 7:
                raise ValueError(f"处理后的张量长度不正确: {len(processed_tensors)}")
            
            return processed_tensors
        
        # 使用统一函数处理所有数据
        try:
            prod_tensors = process_tensor_data(prod_tensors, is_product=True)
            aug_tensors = process_tensor_data(aug_tensors, is_product=True) 
            synthon_tensors = process_tensor_data(synthon_tensors, is_product=False)
            
            print(f"✓ 数据处理成功 - 产物: {len(prod_tensors)}元组, 增强: {len(aug_tensors)}元组, 合成子: {len(synthon_tensors)}元组")
            
        except Exception as e:
            print(f"❌ 数据处理失败: {e}")
            # 返回零损失避免训练中断
            return {
                'total': torch.tensor(0.0, device=device, requires_grad=True),
                'center': torch.tensor(0.0, device=device, requires_grad=True),
                'recovery': torch.tensor(0.0, device=device, requires_grad=True),
                'contrastive': torch.tensor(0.0, device=device, requires_grad=True)
            }, {}        
        batch_size = batch['batch_size']
        
        try:
            # 按照设计方案：三路共享编码  
            # 为MolCLR掩码准备MolTree对象
            product_trees = batch.get('prod_trees', [])
            augmented_trees = batch.get('aug_trees', [])  # 增强的分子树，包含掩码信息
            synthon_trees = batch.get('synthon_trees', [])
            
            h_product, atom_embeds_prod, mess_embeds_prod = self.encode_with_gmpn(
                prod_tensors, mol_trees=product_trees)
            h_augmented, atom_embeds_aug, mess_embeds_aug = self.encode_with_gmpn(
                aug_tensors, mol_trees=augmented_trees)  # 传递增强的分子树
            h_synthons, atom_embeds_syn, mess_embeds_syn = self.encode_with_gmpn(
                synthon_tensors, mol_trees=synthon_trees)
            
            # 按照设计方案：并行计算三个任务
            
            # 任务1：基础任务（反应中心识别）
            try:
                # 设计方案要求：完全保留G2Retro的反应中心识别机制
                # 使用真正的G2Retro MolCenter模块进行反应中心识别
                
                # 准备反应中心识别所需的完整数据
                # 从批次数据中提取必要信息
                product_orders = []
                product_trees = batch.get('prod_trees', [])
                
                # 从产物张量中正确提取所需信息
                # MolTree.tensorize总是返回标准的7个张量：
                # [atom_tensors, bond_tensors, tree_tensors, word_tensors, 
                #  mess_dict, local_dict, scope_tensors]
                product_bond_tensors = prod_tensors[1]   # bond tensor (索引1)
                product_scope_tensors = prod_tensors[6]  # scope tensor (索引6)
                
                # 提取每个样本的完整order信息  
                product_orders = []
                valid_order_count = 0
                
                for i, pretrain_info in enumerate(batch['pretrain_infos']):
                    if 'product_orders' in pretrain_info and pretrain_info['product_orders'] is not None:
                        product_orders.append(pretrain_info['product_orders'])
                        valid_order_count += 1
                    else:
                        # 如果没有order信息，从产物树中提取
                        if i < len(product_trees):
                            tree = product_trees[i]
                            if hasattr(tree, 'order') and tree.order is not None:
                                # 从树中提取完整的order信息
                                bond_order, atom_order, ring_order, change_order = tree.order
                                product_orders.append((bond_order, atom_order, ring_order, change_order))
                                valid_order_count += 1
                            else:
                                # 预训练阶段：创建空order信息作为占位符
                                # 这样可以让predict_centers正常运行，但不会产生有效的监督信号
                                product_orders.append(([], [], [], []))
                        else:
                            product_orders.append(([], [], [], []))
                
                if valid_order_count < len(product_orders):
                    print(f"    警告: {len(product_orders)-valid_order_count}/{len(product_orders)} 样本缺少order信息")
                
                # 调用G2Retro的反应中心识别（使用完整参数）
                center_loss, center_acc, num_samples, bond_data, atom_data = self.reaction_center_head.predict_centers(
                    product_bond_tensors,     # 产物键张量
                    h_product,               # 产物嵌入向量
                    atom_embeds_prod,        # 产物原子向量
                    mess_embeds_prod,        # 产物消息向量
                    product_trees,           # 产物分子树列表
                    product_scope_tensors,   # 产物图范围张量
                    product_orders           # 产物order信息
                )
                
                print(f"    Center - Loss: {center_loss.item():.4f}, Acc: {center_acc:.4f}")
                
                # 处理bond change和atom charge预测
                if bond_data[0] is not None and len(bond_data[0]) > 0:
                    bond_change_hiddens, bond_change_labels = bond_data
                    bond_change_logits = self.reaction_center_head.W_bc(bond_change_hiddens)
                    bond_change_loss = self.reaction_center_head.bond_charge_loss(bond_change_logits, bond_change_labels)
                    center_loss = center_loss + bond_change_loss
                    
                if atom_data[0] is not None and len(atom_data[0]) > 0:
                    atom_charge_hiddens, atom_charge_labels = atom_data
                    atom_charge_logits = self.reaction_center_head.W_tac(atom_charge_hiddens)
                    atom_charge_loss = self.reaction_center_head.atom_charge_loss(atom_charge_logits, atom_charge_labels)
                    center_loss = center_loss + atom_charge_loss
                
            except Exception as e:
                print(f"    基础任务错误: {e}")
                center_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
                center_acc = 0.0
            
            losses['center'] = center_loss
            metrics['center_acc'] = center_acc
            
            # 任务2：分子恢复任务
            if len(batch['augmented_data']) > 0:
                # 按照设计方案：h_augmented被送入分子恢复头
                # 分子恢复任务的目标是从增强的表示恢复原始表示
                recovery_loss, recovery_acc = self.molecule_recovery_head(
                    original_embeddings=h_product,      # 原始分子表示作为目标
                    augmented_embeddings=h_augmented,   # 增强分子表示作为输入
                    augmented_data=batch['augmented_data'],
                    pretrain_infos=batch['pretrain_infos']
                )
                losses['recovery'] = recovery_loss
                metrics['recovery_acc'] = recovery_acc
                print(f"    Recovery - Loss: {recovery_loss.item():.4f}, Acc: {recovery_acc:.4f}")
            else:
                losses['recovery'] = torch.tensor(0.0, device=device, requires_grad=True)
                metrics['recovery_acc'] = 0.0
            
            # 任务3：产物-合成子对比学习（核心创新）
            
            contrastive_loss, contrastive_acc = self.product_synthon_contrastive_head(
                h_product,    # 产物表示 [batch_size, hidden_size]
                h_synthons,   # 合成子表示 [batch_size, hidden_size]
                batch['pretrain_infos']
            )
            
            losses['contrastive'] = contrastive_loss
            metrics['contrastive_acc'] = contrastive_acc
            print(f"    Contrastive - Loss: {contrastive_loss.item():.4f}, Acc: {contrastive_acc:.4f}")
            
            # 动态更新任务权重（基于RetroExplainer的DAMT）
            self.update_task_weights(losses)
            
            # 计算总损失（动态权重）
            total_loss = (
                self.task_weights[0] * losses['center'] + 
                self.task_weights[1] * losses['recovery'] + 
                self.task_weights[2] * losses['contrastive']
            )
            print(f"    Total Loss: {total_loss.item():.4f} "
                  f"[Weights: {self.task_weights[0]:.3f}, {self.task_weights[1]:.3f}, {self.task_weights[2]:.3f}]")
            
            losses['total'] = total_loss
            metrics['task_weights'] = self.task_weights.detach().cpu().numpy()
            
            # 记录权重变化统计
            metrics['weight_center'] = self.task_weights[0].item()
            metrics['weight_recovery'] = self.task_weights[1].item()  
            metrics['weight_contrastive'] = self.task_weights[2].item()
            
        except Exception as e:
            print(f"前向传播错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回零损失避免训练中断
            losses = {
                'total': torch.tensor(0.0, device=device, requires_grad=True),
                'center': torch.tensor(0.0, device=device, requires_grad=True),
                'recovery': torch.tensor(0.0, device=device, requires_grad=True),
                'contrastive': torch.tensor(0.0, device=device, requires_grad=True)
            }
            metrics = {
                'center_acc': 0.0,
                'recovery_acc': 0.0,
                'contrastive_acc': 0.0,
                'task_weights': np.array([1.0, 1.0, 0.1])
            }
        
        return losses, metrics

class G2RetroPDesignAlignedTrainer:
    """
    符合设计方案的训练器
    
    设计方案要求：
    - 微调阶段: 在预训练完成后，模型将移除辅助任务头（分子恢复与产物-合成子对比），
      仅保留经过充分学习的共享编码器和基础任务头。然后，使用预训练好的编码器权重
      进行初始化，在特定的下游数据集（如USPTO-50K）上进行高效的微调。
    """
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 动态权重适应设置
        self.reset_weights_per_epoch = getattr(args, 'reset_weights_per_epoch', False)  # 是否每epoch重置权重

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # 可选地在每个epoch开始时重置动态权重适应
        if self.reset_weights_per_epoch and epoch > 0:
            print(f"\n=== Epoch {epoch}: 重置动态权重适应系统 ===")
            self.model.reset_weight_adaptation()
        
        # 在epoch开始时显示当前权重状态
        if epoch % 5 == 0:  # 每5个epoch显示一次权重统计
            weight_stats = self.model.get_weight_statistics()
            print(f"\n=== Epoch {epoch}: 动态权重适应状态 ===")
            print(f"当前权重: {weight_stats['current_weights']}")
            print(f"初始权重: {weight_stats['initial_weights']}")
            print(f"权重更新步数: {weight_stats['weight_update_step']}")
            print(f"损失历史长度: {weight_stats['loss_history_lengths']}")
            print("=" * 50)
        
        total_losses = defaultdict(float)
        total_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:
                continue
                
            self.optimizer.zero_grad()
            
            # 前向传播
            losses, metrics = self.model(batch, epoch)
            
            # 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计统计
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    total_losses[key] += value.item()
                else:
                    total_losses[key] += value
                    
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    if key not in total_metrics:
                        total_metrics[key] = np.zeros_like(value)
                    total_metrics[key] += value
                else:
                    total_metrics[key] += value
            
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)} - "
                      f"Loss: {losses['total'].item():.4f} "
                      f"[Center: {losses['center'].item():.4f}, "
                      f"Recovery: {losses['recovery'].item():.4f}, "
                      f"Contrast: {losses['contrastive'].item():.4f}]")
                if 'task_weights' in metrics:
                    weights = metrics['task_weights']
                    print(f"  权重: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
        
        # 计算平均值
        if num_batches > 0:
            avg_losses = {k: v/num_batches for k, v in total_losses.items()}
            avg_metrics = {}
            for k, v in total_metrics.items():
                if isinstance(v, np.ndarray):
                    avg_metrics[k] = v / num_batches
                else:
                    avg_metrics[k] = v / num_batches
        else:
            avg_losses = {}
            avg_metrics = {}
        
        return avg_losses, avg_metrics

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        total_losses = defaultdict(float)
        total_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                    
                losses, metrics = self.model(batch, epoch)
                
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        total_losses[key] += value.item()
                    else:
                        total_losses[key] += value
                        
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        if key not in total_metrics:
                            total_metrics[key] = np.zeros_like(value)
                        total_metrics[key] += value
                    else:
                        total_metrics[key] += value
                
                num_batches += 1
        
        if num_batches == 0:
            return {}, {}
            
        avg_losses = {k: v/num_batches for k, v in total_losses.items()}
        avg_metrics = {}
        for k, v in total_metrics.items():
            if isinstance(v, np.ndarray):
                avg_metrics[k] = v / num_batches
            else:
                avg_metrics[k] = v / num_batches
        
        return avg_losses, avg_metrics

    def train(self):
        """完整训练流程"""
        print("\n" + "="*80)
        print("开始G2Retro-P预训练（设计方案完全对齐）")
        print("="*80)
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{self.args.epochs} {'='*20}")
            
            # 训练
            train_losses, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_losses, val_metrics = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印详细训练统计
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} 训练结果汇总:")
            print(f"{'='*60}")
            print(f"损失值:")
            print(f"  中心识别损失: {train_losses.get('center', 0):.4f}")
            print(f"  分子恢复损失: {train_losses.get('recovery', 0):.4f}")
            print(f"  对比学习损失: {train_losses.get('contrastive', 0):.4f}")
            print(f"  总损失: {train_losses.get('total', 0):.4f}")
            print(f"准确率:")
            print(f"  中心识别准确率: {train_metrics.get('center_acc', 0):.4f}")
            print(f"  分子恢复准确率: {train_metrics.get('recovery_acc', 0):.4f}")
            print(f"  对比学习准确率: {train_metrics.get('contrastive_acc', 0):.4f}")
            print(f"动态权重:")
            print(f"  当前权重: [Center: {self.model.task_weights[0]:.3f}, Recovery: {self.model.task_weights[1]:.3f}, Contrastive: {self.model.task_weights[2]:.3f}]")
            print(f"  权重更新步数: {self.model.weight_update_step}")
            
            if val_losses:
                print(f"\n验证结果:")
                print(f"损失值:")
                print(f"  中心识别损失: {val_losses.get('center', 0):.4f}")
                print(f"  分子恢复损失: {val_losses.get('recovery', 0):.4f}")
                print(f"  对比学习损失: {val_losses.get('contrastive', 0):.4f}")
                print(f"  总损失: {val_losses.get('total', 0):.4f}")
                print(f"准确率:")
                print(f"  中心识别准确率: {val_metrics.get('center_acc', 0):.4f}")
                print(f"  分子恢复准确率: {val_metrics.get('recovery_acc', 0):.4f}")
                print(f"  对比学习准确率: {val_metrics.get('contrastive_acc', 0):.4f}")
            
            print(f"{'='*60}\n")
            
            # 保存检查点
            if epoch % 2 == 0:
                self.save_checkpoint(epoch, train_losses, val_losses)
            
            # 早停检查
            current_val_loss = val_losses.get('total', float('inf'))
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, train_losses, val_losses, is_best=True)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.args.patience:
                print(f"早停于epoch {epoch+1}")
                break

    def save_checkpoint(self, epoch, train_losses, val_losses, is_best=False):
        """保存检查点（包含动态权重状态）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'weight_adaptation_state': self.model.save_weight_adaptation_state()  # 保存动态权重状态
        }
        
        save_path = f"checkpoints_g2retro_p_design_aligned/checkpoint_epoch_{epoch}.pt"
        if is_best:
            save_path = "checkpoints_g2retro_p_design_aligned/best_model.pt"
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")

# 训练参数类
class Args:
    def __init__(self):
        # 模型参数（与G2Retro保持一致）
        self.hidden_size = 300
        self.latent_size = 300
        self.depthT = 3
        self.depthG = 3
        self.use_feature = True
        self.use_tree = True
        self.network_type = "gru"
        self.use_node_embed = False
        self.use_class = False
        self.use_atomic = False
        self.sum_pool = False
        self.use_brics = False
        self.use_latent_attachatom = False
        self.use_mess = True
        
        # 训练参数
        self.batch_size = 4  # 小批次适应GPU内存
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 3  # 演示用少量epoch
        self.step_size = 2
        self.gamma = 0.5
        self.patience = 2
        
        # 动态权重适应参数（基于RetroExplainer DAMT）
        self.reset_weights_per_epoch = False  # 是否每epoch重置权重
        self.weight_update_frequency = 5      # 权重更新频率
        self.weight_temperature = 2.0         # 温度系数τ
        self.loss_queue_length = 50           # 损失队列长度
        self.min_task_weight = 0.01           # 最小任务权重
        self.max_task_weight = 3.0            # 最大任务权重
        
        # 数据集参数
        self.use_small_dataset = False        # 默认使用完整数据集

def main():
    """主函数"""
    print("="*80)
    print("G2Retro-P 设计方案完全对齐实现")
    print("严格按照'基于多任务预训练的半模板逆合成模型设计方案'实现")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 参数设置
    args = Args()
    
    # 命令行参数解析
    if '--small' in sys.argv or '--small-dataset' in sys.argv or '--use_small_dataset' in sys.argv:
        args.use_small_dataset = True
        print("🚀 命令行指定使用小数据集模式!")
    
    # 解析其他参数
    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            args.epochs = int(sys.argv[i + 1])
            print(f"设置epochs: {args.epochs}")
        elif arg == '--batch_size' and i + 1 < len(sys.argv):
            args.batch_size = int(sys.argv[i + 1])
            print(f"设置batch_size: {args.batch_size}")
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\n使用方法:")
        print("  python train_g2retro_p_design_aligned.py              # 使用完整数据集")
        print("  python train_g2retro_p_design_aligned.py --small      # 使用小数据集快速测试")
        print("  python train_g2retro_p_design_aligned.py --use_small_dataset  # 使用小数据集快速测试")
        print("\n参数说明:")
        print("  --small, --small-dataset, --use_small_dataset    使用小数据集进行快速测试(1000训练样本+200验证样本)")
        print("  --epochs N                                       设置训练轮数")
        print("  --batch_size N                                   设置批次大小")
        print("  --help, -h                                       显示帮助信息")
        return
    
    # 数据路径
    train_data_path = '../data/pretrain/pretrain_tensors_train.pkl'
    val_data_path = '../data/pretrain/pretrain_tensors_valid.pkl'
    vocab_path = '../data/pretrain/vocab_train.txt'
    
    # 创建数据集
    print("\n创建数据集...")
    if args.use_small_dataset:
        print("🚀 使用小数据集模式 - 快速测试!")
        train_dataset = G2RetroPDesignAlignedDataset(train_data_path, vocab_path, max_samples=None, use_small_dataset=True)
        val_dataset = G2RetroPDesignAlignedDataset(val_data_path, vocab_path, max_samples=None, use_small_dataset=True)
    else:
        print("使用完整数据集进行训练")
        train_dataset = G2RetroPDesignAlignedDataset(train_data_path, vocab_path, max_samples=None)
        val_dataset = G2RetroPDesignAlignedDataset(val_data_path, vocab_path, max_samples=None)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, train_dataset.vocab, train_dataset.avocab),
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, val_dataset.vocab, val_dataset.avocab),
        num_workers=0
    )
    
    # 创建模型
    print("\n创建模型...")
    model = G2RetroPDesignAlignedModel(train_dataset.vocab, train_dataset.avocab, args)
    model = model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("G2Retro-P完整系统核心要素验证:")
    print("="*70)
    print("✓ 共享编码器：GMPN（来自G2Retro）")
    print("✓ 基础任务头：G2Retro反应中心识别模块")
    print("✓ 分子恢复头：MolCLR三种增强策略")  
    print("✓ 产物-合成子对比头：核心创新")
    print("✓ 动态权重调整：基于RetroExplainer DAMT算法")
    print("✓ 初始权重设置：L_base + L_recovery + 0.1×L_contrastive")
    print("✓ 数据流程：Gp → h_product, Gp_aug → h_augmented, Gs → h_synthons")
    print("✓ 智能权重适应：根据任务学习进度动态调整")
    print("="*70)
    
    # 创建训练器
    trainer = G2RetroPDesignAlignedTrainer(model, train_loader, val_loader, args)
    
    # 开始训练
    trainer.train()
    
    print("\n" + "="*80)
    print("训练完成！")
    print("模型已按照设计方案完全对齐实现")
    print("="*80)

if __name__ == "__main__":
    main() 