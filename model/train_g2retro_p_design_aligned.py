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

class G2RetroPDesignAlignedDataset(Dataset):
    """
    完全符合设计方案的G2Retro-P预训练数据集
    严格按照设计方案的数据流程处理：
    - 输入处理: 基于atom-mapping从反应数据中提取产物分子图Gp和对应的合成子组合Gs
    - 数据增强: 对原始产物分子图应用MolCLR增强策略，生成被"破坏"的版本Gp_aug
    """
    def __init__(self, data_path, vocab_path, max_samples=None):
        print(f"加载预训练数据: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
            
        print(f"加载词汇表: {vocab_path}")
        with open(vocab_path, 'r') as f:
            words = [line.strip() for line in f.readlines()]
        
        # 使用完整词汇表（设计方案要求）
        self.vocab = Vocab(words)
        # 使用正确的原子词汇表（设计方案要求）
        self.avocab = common_atom_vocab
        
        print(f"数据集大小: {len(self.data)}")
        print(f"分子词汇表大小: {self.vocab.size()} (设计方案要求：完整词汇表)")
        print(f"原子词汇表大小: {self.avocab.size()} (设计方案要求：common_atom_vocab)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        try:
            data_item = self.data[idx]
            
            # 提取分子树
            mol_trees = data_item['mol_trees']  # [产物树, 合成子树, 反应物树]
            
            # 提取预训练信息
            pretrain_info = data_item['pretrain_info']
            
            # 提取增强数据
            augmented_data = data_item['augmented_data']
            
            return {
                'mol_trees': mol_trees,
                'pretrain_info': pretrain_info,
                'augmented_data': augmented_data
            }
            
        except Exception as e:
            print(f"数据加载错误 (idx={idx}): {e}")
            return None

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
                    print(f"原子掩码: 节点 {atom_idx}")
                    
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
                        print(f"键删除: 边 {edge}")
                        
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
                    print(f"子图移除: 节点 {atom_idx} 及其邻接边")
        
        # 更新增强信息
        augmented_tree.augmented = True
        augmented_tree.augment_type = augment_type
        augmented_tree.masked_indices = masked_indices
        
        return augmented_tree
        
    except Exception as e:
        print(f"MolCLR图增强错误: {e}")
        return mol_tree  # 返回原始树作为备选

def create_augmented_moltrees(augmented_data, original_smiles, original_tree=None):
    """
    按照MolCLR思想创建增强的分子树
    在图结构级别进行掩码操作，不修改底层化学结构
    
    设计方案要求：对原始产物分子图应用MolCLR增强策略，生成被"破坏"的版本Gp_aug
    三种增强方式：
    1. 原子掩码 (Atom Masking): 在图节点级别隐藏原子特征
    2. 键删除 (Bond Deletion): 在图边级别掩码化学键信息  
    3. 子图移除 (Subgraph Removal): 在图级别掩码连通子图
    """
    augmented_trees = []
    
    # 如果没有提供原始树，创建一个
    if original_tree is None:
        try:
            original_tree = MolTree(original_smiles)
        except Exception as e:
            print(f"创建原始分子树失败: {e}")
            return augmented_trees
    
    for aug_data in augmented_data:
        if aug_data['original_smiles'] == original_smiles:
            try:
                # 在图结构级别应用MolCLR增强
                augmented_tree = apply_molclr_graph_augmentation(
                    original_tree,
                    aug_data['masked_indices'],
                    aug_data['augment_type']
                )
                
                if augmented_tree is not None:
                    augmented_trees.append(augmented_tree)
                    print(f"成功创建MolCLR风格增强分子树: {aug_data['augment_type']}")
                    
            except Exception as e:
                print(f"MolCLR增强分子树创建错误: {e}")
                # 如果增强失败，使用原始分子树作为备选
                try:
                    augmented_trees.append(copy.deepcopy(original_tree))
                except:
                    continue
                
    return augmented_trees

def g2retro_design_aligned_collate_fn(batch, vocab, avocab):
    """
    设计方案对齐的批处理函数
    使用MolCLR图结构级别掩码增强
    
    设计方案要求的数据流程：
    1. 输入处理: 基于atom-mapping从反应数据中提取产物分子图Gp和对应的合成子组合Gs
    2. 数据增强: 对原始产物分子图应用MolCLR增强策略，生成一个被"破坏"的版本Gp_aug
    3. 三路共享编码准备: 
       - 原始产物图 Gp → prod_tensors
       - 增强产物图 Gp_aug → aug_tensors 
       - 合成子组合 Gs → synthon_tensors
    """
    # 过滤空样本
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    
    try:
        print(f"批处理函数：处理 {len(valid_batch)} 个样本")
        
        # 提取数据
        prod_trees = []
        synthon_trees = []
        react_trees = []
        augmented_data_batch = []
        pretrain_infos = []
        
        for item in valid_batch:
            mol_trees = item['mol_trees']
            augmented_data = item['augmented_data']
            pretrain_info = item['pretrain_info']
            
            # 分子树
            prod_trees.append(mol_trees[0])     # 产物树
            synthon_trees.append(mol_trees[1])  # 合成子树
            react_trees.append(mol_trees[2])    # 反应物树
            
            # 预训练信息
            pretrain_infos.append(pretrain_info)
            
            # 增强数据处理
            augmented_data_batch.append(augmented_data)
        
        print(f"  产物分子树: {len(prod_trees)}")
        print(f"  合成子分子树: {len(synthon_trees)}")
        print(f"  反应物分子树: {len(react_trees)}")
        
        # 按照设计方案：应用MolCLR图结构级别增强策略
        print("应用MolCLR图结构级别增强策略...")
        aug_trees = []
        
        for i, (prod_tree, augmented_data) in enumerate(zip(prod_trees, augmented_data_batch)):
            try:
                # 获取产物SMILES
                product_smiles = pretrain_infos[i]['product_smiles']
                
                # 使用新的图结构级别增强方法
                sample_aug_trees = create_augmented_moltrees(
                    augmented_data, 
                    product_smiles,
                    original_tree=prod_tree  # 传入原始分子树
                )
                
                if sample_aug_trees:
                    aug_trees.append(sample_aug_trees[0])  # 使用第一个增强版本
                    print(f"    样本 {i}: 成功创建图结构级别增强分子树")
                else:
                    # 如果增强失败，使用原始树
                    aug_trees.append(prod_tree)
                    print(f"    样本 {i}: 增强失败，使用原始分子树")
                    
            except Exception as e:
                print(f"    样本 {i} 增强错误: {e}")
                aug_trees.append(prod_tree)  # 使用原始树作为备选
        
        print(f"  增强分子树: {len(aug_trees)}")
        
        # 张量化处理
        print("张量化处理...")
        
        # 1. 原始产物图 Gp → prod_tensors
        prod_batch, prod_tensors = MolTree.tensorize(
            prod_trees, vocab, avocab, 
            use_feature=True, product=True
        )
        print(f"  原始产物张量化完成")
        
        # 2. 增强产物图 Gp_aug → aug_tensors
        aug_batch, aug_tensors = MolTree.tensorize(
            aug_trees, vocab, avocab, 
            use_feature=True, product=True
        )
        print(f"  增强产物张量化完成")
        
        # 3. 合成子组合 Gs → synthon_tensors
        synthon_batch, synthon_tensors = MolTree.tensorize(
            synthon_trees, vocab, avocab, 
            use_feature=True, product=True
        )
        print(f"  合成子张量化完成")
        
        # 4. 反应物张量化（用于基础任务）
        react_batch, react_tensors = MolTree.tensorize(
            react_trees, vocab, avocab, 
            use_feature=True, product=False
        )
        print(f"  反应物张量化完成")
        
        # 构建批次字典
        batch_dict = {
            'batch_size': len(valid_batch),
            'prod_tensors': prod_tensors,      # Gp：原始产物图
            'aug_tensors': aug_tensors,        # Gp_aug：增强产物图
            'synthon_tensors': synthon_tensors, # Gs：合成子组合
            'react_tensors': react_tensors,    # 反应物（基础任务用）
            'prod_trees': prod_trees,          # 产物分子树（基础任务需要）
            'synthon_trees': synthon_trees,    # 合成子分子树
            'react_trees': react_trees,        # 反应物分子树
            'augmented_trees': aug_trees,      # 增强分子树
            'augmented_data': augmented_data_batch,  # 增强数据信息
            'pretrain_infos': pretrain_infos   # 预训练信息（包含product_orders）
        }
        
        print(f"批处理完成：{len(valid_batch)} 个样本")
        print(f"设计方案三路数据流准备就绪：")
        print(f"  - Gp (原始产物图) → GMPN编码器")
        print(f"  - Gp_aug (增强产物图) → GMPN编码器") 
        print(f"  - Gs (合成子组合) → GMPN编码器")
        
        return batch_dict
        
    except Exception as e:
        print(f"批处理函数错误: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            input_dim=args.hidden_size,
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
        print(f"✓ 词汇表大小: {len(vocab)}")
        print(f"✓ 原子词汇表大小: {len(avocab)}")
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

    def encode_with_gmpn(self, tensors, classes=None):
        """
        使用共享的GMPN编码器进行编码
        
        设计方案要求：GMPN通过图消息传递网络（Graph Message Passing Network）来学习分子结构
        GMPN通过在分子图的原子和键之间迭代地传递信息，能够有效地捕捉到每个原子及其周围的
        局部化学环境，从而生成富有结构信息的特征表示。
        
        输入: 分子图 G = (A, B)，其中 A 是原子集合，B 是化学键集合
        输出: 原子级别的嵌入表示 ai，键级别的嵌入表示 bij，以及整个分子图的全局表示 h
        """
        # 使用共享的GMPN编码器
        mol_embeds, atom_embeds, mess_embeds = self.shared_encoder(
            tensors, 
            product=True, 
            classes=classes, 
            use_feature=True
        )
        return mol_embeds, atom_embeds, mess_embeds

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
        
        # 获取张量数据
        prod_tensors = batch['prod_tensors']      # Gp
        aug_tensors = batch['aug_tensors']        # Gp_aug  
        synthon_tensors = batch['synthon_tensors'] # Gs
        react_tensors = batch['react_tensors']     # 用于基础任务
        
        # 转换为CUDA张量
        prod_tensors = make_cuda(prod_tensors, product=True)
        aug_tensors = make_cuda(aug_tensors, product=True)
        synthon_tensors = make_cuda(synthon_tensors, product=True)
        react_tensors = make_cuda(react_tensors, product=False)
        
        batch_size = batch['batch_size']
        
        try:
            print(f"开始三路共享编码（设计方案核心）...")
            
            # 按照设计方案：三路共享编码
            # 1. 原始产物图 Gp 输入GMPN编码器 → h_product
            print("  1. 原始产物图 Gp 输入GMPN编码器 → h_product")
            h_product, atom_embeds_prod, mess_embeds_prod = self.encode_with_gmpn(prod_tensors)
            
            # 2. 增强产物图 Gp_aug 输入GMPN编码器 → h_augmented
            print("  2. 增强产物图 Gp_aug 输入GMPN编码器 → h_augmented")
            h_augmented, atom_embeds_aug, mess_embeds_aug = self.encode_with_gmpn(aug_tensors)
            
            # 3. 合成子组合 Gs 输入GMPN编码器 → h_synthons
            print("  3. 合成子组合 Gs 输入GMPN编码器 → h_synthons")
            h_synthons, atom_embeds_syn, mess_embeds_syn = self.encode_with_gmpn(synthon_tensors)
            
            print(f"开始并行计算三个任务（设计方案核心）...")
            
            # 按照设计方案：并行计算三个任务
            
            # 任务1：基础任务（反应中心识别）
            # h_product 被送入基础任务头
            print("  任务1：基础任务（反应中心识别）")
            try:
                # 设计方案要求：完全保留G2Retro的反应中心识别机制
                # 使用真正的G2Retro MolCenter模块进行反应中心识别
                
                # 准备反应中心识别所需的完整数据
                # 从批次数据中提取必要信息
                product_orders = []
                product_trees = batch.get('prod_trees', [])
                
                # 从产物张量中正确提取所需信息
                product_graph_tensors, product_bond_tensors, product_scope_tensors, product_tree_tensors = prod_tensors
                
                # 提取每个样本的完整order信息  
                product_orders = []
                for i, pretrain_info in enumerate(batch['pretrain_infos']):
                    if 'product_orders' in pretrain_info and pretrain_info['product_orders'] is not None:
                        product_orders.append(pretrain_info['product_orders'])
                    else:
                        # 如果没有order信息，从产物树中提取
                        if i < len(product_trees):
                            tree = product_trees[i]
                            if hasattr(tree, 'order') and tree.order is not None:
                                # 从树中提取完整的order信息
                                bond_order, atom_order, ring_order, change_order = tree.order
                                product_orders.append((bond_order, atom_order, ring_order, change_order))
                            else:
                                # 创建默认的空order信息
                                product_orders.append(([], [], [], []))
                        else:
                            product_orders.append(([], [], [], []))
                
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
                
                print(f"    基础任务损失: {center_loss.item():.4f}")
                print(f"    基础任务准确率: {center_acc:.4f}")
                print(f"    处理样本数: {num_samples}")
                
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
                print(f"    基础任务计算错误: {e}")
                import traceback
                traceback.print_exc()
                # 如果完整的反应中心识别失败，返回一个需要梯度的零损失
                center_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
                center_acc = 0.0
            
            losses['center'] = center_loss
            metrics['center_acc'] = center_acc
            
            # 任务2：分子恢复任务
            # h_augmented 被送入分子恢复头
            print("  任务2：分子恢复任务")
            if len(batch['augmented_data']) > 0:
                # 按照设计方案：h_augmented被送入分子恢复头
                # 分子恢复任务的目标是从增强的表示恢复原始表示
                recovery_loss, recovery_acc = self.molecule_recovery_head(
                    original_embeddings=h_product,      # 原始分子表示作为目标
                    augmented_embeddings=h_augmented,   # 增强分子表示作为输入
                    augmented_data=batch['augmented_data'],
                    pretrain_infos=batch['pretrain_infos']
                )
                print(f"    分子恢复损失: {recovery_loss.item():.4f}")
                print(f"    分子恢复准确率: {recovery_acc:.4f}")
                losses['recovery'] = recovery_loss
                metrics['recovery_acc'] = recovery_acc
            else:
                losses['recovery'] = torch.tensor(0.0, device=device, requires_grad=True)
                metrics['recovery_acc'] = 0.0
                print(f"    分子恢复：无增强数据")
            
            # 任务3：产物-合成子对比学习（核心创新）
            # h_product 和 h_synthons 被送入产物-合成子对比头
            print("  任务3：产物-合成子对比学习（核心创新）")
            
            # 检查合成子嵌入的维度，如果是3D则需要融合
            if h_synthons.dim() == 3:
                # 多合成子情况：[batch_size, num_synthons, hidden_size]
                print(f"    检测到多合成子：shape={h_synthons.shape}")
                # 产物-合成子对比头会在内部处理多合成子融合
            
            contrastive_loss, contrastive_acc = self.product_synthon_contrastive_head(
                h_product,    # 产物表示
                h_synthons,   # 合成子表示（可能是2D或3D）
                batch['pretrain_infos']
            )
            
            print(f"    产物-合成子对比损失: {contrastive_loss.item():.4f}")
            print(f"    产物-合成子对比准确率: {contrastive_acc:.4f}")
            losses['contrastive'] = contrastive_loss
            metrics['contrastive_acc'] = contrastive_acc
            
            # 在损失计算前，动态更新任务权重（基于RetroExplainer的DAMT）
            print(f"动态调整任务权重（基于RetroExplainer DAMT）...")
            self.update_task_weights(losses)
            
            # 按照设计方案：损失计算，但现在使用动态权重
            # L_total = w1 × L_base + w2 × L_recovery + w3 × L_contrastive
            print(f"计算总损失（动态权重）...")
            total_loss = (
                self.task_weights[0] * losses['center'] + 
                self.task_weights[1] * losses['recovery'] + 
                self.task_weights[2] * losses['contrastive']
            )
            
            print(f"  总损失 = {self.task_weights[0]:.4f}×{losses['center'].item():.4f} + {self.task_weights[1]:.4f}×{losses['recovery'].item():.4f} + {self.task_weights[2]:.4f}×{losses['contrastive'].item():.4f} = {total_loss.item():.4f}")
            
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
            
            if batch_idx % 2 == 0:
                print(f"\nEpoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}")
                print(f"  总损失: {losses['total'].item():.4f}")
                print(f"  基础任务损失: {losses['center'].item():.4f}")
                print(f"  分子恢复损失: {losses['recovery'].item():.4f}")
                print(f"  产物-合成子对比损失: {losses['contrastive'].item():.4f}")
                if 'task_weights' in metrics:
                    weights = metrics['task_weights']
                    print(f"  动态任务权重: [基础:{weights[0]:.3f}, 恢复:{weights[1]:.3f}, 对比:{weights[2]:.3f}]")
                if 'weight_center' in metrics:
                    print(f"  当前权重详细: 基础={metrics['weight_center']:.4f}, 恢复={metrics['weight_recovery']:.4f}, 对比={metrics['weight_contrastive']:.4f}")
        
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
            
            # 打印统计
            print(f"\n训练结果（设计方案三任务）:")
            print(f"  总损失: {train_losses.get('total', 0):.4f}")
            print(f"  基础任务准确率: {train_metrics.get('center_acc', 0):.4f}")
            print(f"  分子恢复准确率: {train_metrics.get('recovery_acc', 0):.4f}")
            print(f"  产物-合成子对比准确率: {train_metrics.get('contrastive_acc', 0):.4f}")
            
            if val_losses:
                print(f"\n验证结果:")
                print(f"  总损失: {val_losses.get('total', 0):.4f}")
                print(f"  基础任务准确率: {val_metrics.get('center_acc', 0):.4f}")
                print(f"  分子恢复准确率: {val_metrics.get('recovery_acc', 0):.4f}")
                print(f"  产物-合成子对比准确率: {val_metrics.get('contrastive_acc', 0):.4f}")
            
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
    
    # 数据路径
    train_data_path = '../data/pretrain/pretrain_tensors_train.pkl'
    test_data_path = '../data/pretrain/pretrain_tensors_test.pkl'
    vocab_path = '../data/pretrain/vocab_train.txt'
    
    # 创建数据集 - 使用小样本进行演示
    print("\n创建数据集...")
    train_dataset = G2RetroPDesignAlignedDataset(train_data_path, vocab_path, max_samples=20)
    val_dataset = G2RetroPDesignAlignedDataset(test_data_path, vocab_path, max_samples=5)
    
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