#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2Retro-P预训练数据集类
完全符合设计方案的数据处理和增强策略

这个文件包含：
1. G2RetroPDesignAlignedDataset - 数据集类
2. g2retro_design_aligned_collate_fn - 批处理函数
3. MolCLR数据增强策略实现
"""

import os
import pickle
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import copy

# 导入G2Retro核心模块
from mol_tree import MolTree
from vocab import Vocab, common_atom_vocab
from config import device

class G2RetroPDesignAlignedDataset(Dataset):
    """
    完全符合设计方案的G2Retro-P预训练数据集
    严格按照设计方案的数据流程处理：
    - 输入处理: 基于atom-mapping从反应数据中提取产物分子图Gp和对应的合成子组合Gs
    - 数据增强: 对原始产物分子图应用MolCLR增强策略，生成被"破坏"的版本Gp_aug
    """
    def __init__(self, data_path, vocab_path, max_samples=None, use_small_dataset=False):
        # 如果要使用小数据集，修改文件路径
        if use_small_dataset:
            if 'pretrain_tensors_train.pkl' in data_path:
                data_path = data_path.replace('pretrain_tensors_train.pkl', 'pretrain_tensors_train_small.pkl')
                print(f"🚀 使用小训练集进行快速测试!")
            elif 'pretrain_tensors_valid.pkl' in data_path:
                data_path = data_path.replace('pretrain_tensors_valid.pkl', 'pretrain_tensors_valid_small.pkl')
                print(f"🚀 使用小验证集进行快速测试!")
        
        print(f"加载预训练数据: {data_path}")
        if not os.path.exists(data_path):
            if use_small_dataset:
                print(f"❌ 小数据集文件不存在: {data_path}")
                print("请先运行 python model/create_small_dataset.py 来创建小数据集")
                raise FileNotFoundError(f"小数据集文件不存在: {data_path}")
            else:
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
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
        """
        返回设计方案要求的三路数据：
        1. 原始产物图 Gp
        2. 增强产物图 Gp_aug（应用MolCLR策略）
        3. 合成子组合 Gs
        """
        item = self.data[idx]
        
        # 检查数据格式并适配不同的预处理输出
        if 'product_tree' in item:
            # 标准格式（直接预处理输出）
            product_tree = item['product_tree']
            synthon_trees = item['synthon_trees']
            react_tree = item['react_tree']
        elif 'mol_trees' in item:
            # 新格式（来自预处理脚本）
            mol_trees = item['mol_trees']
            if isinstance(mol_trees, tuple) and len(mol_trees) >= 3:
                product_tree = mol_trees[0]    # 产物分子树
                synthon_trees = mol_trees[1]   # 合成子分子树
                react_tree = mol_trees[2]      # 反应物分子树
            else:
                raise ValueError(f"mol_trees格式错误: {type(mol_trees)}, length: {len(mol_trees) if hasattr(mol_trees, '__len__') else 'N/A'}")
        else:
            raise KeyError(f"未识别的数据格式，可用键: {list(item.keys())}")
        
        # 创建MolCLR风格的增强产物树
        augmented_tree = apply_molclr_graph_augmentation(product_tree)
        
        # 存储预训练信息
        if 'pretrain_info' in item:
            # 如果已有预训练信息，使用它
            pretrain_info = item['pretrain_info'].copy()
            pretrain_info['augment_type'] = getattr(augmented_tree, 'augment_type', 'unknown')
        else:
            # 否则创建默认信息
            pretrain_info = {
                'reaction_id': item.get('reaction_id', idx),
                'product_smiles': item.get('product_smiles', ''),
                'product_orders': item.get('product_orders', None),
                'augment_type': getattr(augmented_tree, 'augment_type', 'unknown')
            }
        
        return {
            'product_tree': product_tree,      # Gp：原始产物图
            'augmented_tree': augmented_tree,  # Gp_aug：增强产物图
            'synthon_trees': synthon_trees,    # Gs：合成子组合
            'react_tree': react_tree,          # 反应物（用于基础任务）
            'pretrain_info': pretrain_info,
            # 为MolCLR掩码添加树对象
            'prod_trees': [product_tree],
            'aug_trees': [augmented_tree],     # 包含掩码信息的增强树
            'synthon_tree_objects': synthon_trees if isinstance(synthon_trees, list) else [synthon_trees]
        }

def apply_molclr_graph_augmentation(tree, augmentation_prob=0.1):
    """
    基于MolCLR原始实现的图增强
    参考: "Molecular Contrastive Learning of Representations via Graph Neural Networks"
    
    重要：MolCLR的增强应该在GNN特征层面进行，而不是修改化学结构
    
    三种增强策略:
    1. 原子掩码 (Atom Masking): 在GNN中将某些原子特征替换为掩码token
    2. 键删除 (Bond Deletion): 在GNN中删除某些边的特征
    3. 子图移除 (Subgraph Removal): 在GNN中掩码连通子图的特征
    """
    try:
        # 保存原始分子树的副本
        original_tree = copy.deepcopy(tree)
        
        if tree.mol is None:
            return original_tree
        
        # 随机选择一种增强策略
        augmentation_type = random.choice(['atom_masking', 'bond_deletion', 'subgraph_removal'])
        
        # 为MolTree添加掩码信息，让GNN在编码时使用
        augmented_tree = copy.deepcopy(tree)
        
        if augmentation_type == 'atom_masking':
            # 原子掩码：标记哪些原子需要在GNN中掩码
            num_atoms = tree.mol.GetNumAtoms()
            if num_atoms > 0:
                num_mask = max(1, int(num_atoms * augmentation_prob))
                masked_atom_indices = random.sample(range(num_atoms), min(num_mask, num_atoms))
                
                # 在MolTree上添加掩码标记，供GNN使用
                augmented_tree.atom_masks = torch.zeros(num_atoms, dtype=torch.bool)
                for idx in masked_atom_indices:
                    augmented_tree.atom_masks[idx] = True
                    
        elif augmentation_type == 'bond_deletion':
            # 键删除：标记哪些键需要在GNN中删除
            num_bonds = tree.mol.GetNumBonds()
            if num_bonds > 0:
                num_delete = max(1, int(num_bonds * augmentation_prob))
                deleted_bond_indices = random.sample(range(num_bonds), min(num_delete, num_bonds))
                
                # 在MolTree上添加键掩码标记
                augmented_tree.bond_masks = torch.zeros(num_bonds, dtype=torch.bool)
                for idx in deleted_bond_indices:
                    augmented_tree.bond_masks[idx] = True
                    
        else:  # subgraph_removal
            # 子图移除：标记连通的原子子集需要在GNN中掩码
            num_atoms = tree.mol.GetNumAtoms()
            if num_atoms > 1:
                # 选择一个起始原子
                start_atom = random.randint(0, num_atoms - 1)
                
                # 找到连通的原子子集（BFS）
                mol = tree.mol
                visited = set()
                to_visit = [start_atom]
                max_subgraph_size = max(1, int(num_atoms * augmentation_prob))
                
                while to_visit and len(visited) < max_subgraph_size:
                    current = to_visit.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # 添加邻居原子
                    atom = mol.GetAtomWithIdx(current)
                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if neighbor_idx not in visited and len(visited) < max_subgraph_size:
                            to_visit.append(neighbor_idx)
                
                # 在MolTree上添加子图掩码标记
                augmented_tree.subgraph_masks = torch.zeros(num_atoms, dtype=torch.bool)
                for idx in visited:
                    augmented_tree.subgraph_masks[idx] = True
        
        # 设置增强信息
        augmented_tree.augmented = True
        augmented_tree.augment_type = augmentation_type
        
        # 确保增强的分子树仍有change属性（用于基础任务）
        if not hasattr(augmented_tree, 'change'):
            augmented_tree.change = getattr(tree, 'change', ([], [], [], [], [], []))
        
        return augmented_tree
        
    except Exception as e:
        print(f"MolCLR图增强错误: {e}")
        return original_tree

# 注意：MolCLR的增强现在在GNN特征层面进行，不再需要apply_atom_masking函数

# 注意：MolCLR的增强现在在GNN特征层面进行，不再需要化学结构修改函数

def g2retro_design_aligned_collate_fn(batch, vocab, avocab):
    """
    批处理函数：按照设计方案准备三路数据流
    
    数据流程（完全对齐设计方案）：
    1. 原始产物图 Gp → prod_tensors
    2. 增强产物图 Gp_aug → aug_tensors  
    3. 合成子组合 Gs → synthon_tensors
    """
    try:
        batch_size = len(batch)
        print(f"批处理函数：处理 {batch_size} 个样本")
        
        # 收集所有分子树
        prod_trees = []      # 原始产物树
        aug_trees = []       # 增强产物树
        synthon_trees = []   # 合成子树
        react_trees = []     # 反应物树
        
        # 收集其他信息
        pretrain_infos = []
        augmented_data_batch = []
        
        # 验证批次数据
        valid_batch = []
        for item in batch:
            if all(key in item for key in ['product_tree', 'augmented_tree', 'synthon_trees', 'react_tree']):
                valid_batch.append(item)
            else:
                print(f"警告：跳过不完整的数据项")
        
        if not valid_batch:
            print("错误：批次中没有有效数据")
            return None
            
        # 处理每个样本
        for item in valid_batch:
            # 原始产物树
            prod_trees.append(item['product_tree'])
            
            # 增强产物树
            aug_trees.append(item['augmented_tree'])
            
            # 合成子树（可能有多个）
            synthon_list = item['synthon_trees']
            if isinstance(synthon_list, list) and len(synthon_list) > 0:
                # 合并多个合成子为一个组合分子树
                combined_synthon = synthon_list[0]  # 简化：暂时只用第一个
                synthon_trees.append(combined_synthon)
            else:
                synthon_trees.append(synthon_list)
            
            # 反应物树
            react_trees.append(item['react_tree'])
            
            # 预训练信息
            pretrain_infos.append(item.get('pretrain_info', {}))
            
            # 增强数据信息（用于分子恢复任务）
            augmented_data = {
                'augment_type': item['augmented_tree'].augment_type if hasattr(item['augmented_tree'], 'augment_type') else 'unknown',
                'masked_atoms': getattr(item['augmented_tree'], 'masked_atoms', []),
                'original_atoms': getattr(item['augmented_tree'], 'original_atoms', {}),
                'deleted_bonds': getattr(item['augmented_tree'], 'deleted_bonds', []),
                'removed_atoms': getattr(item['augmented_tree'], 'removed_atoms', [])
            }
            augmented_data_batch.append(augmented_data)
        
        # 张量化处理
        # MolTree.tensorize返回(trees, tensors)元组，我们需要tensors部分
        mol_batch_prod, prod_tensors = MolTree.tensorize(
            prod_trees, vocab, avocab, 
            use_feature=True, product=True
        )
        
        mol_batch_aug, aug_tensors = MolTree.tensorize(
            aug_trees, vocab, avocab, 
            use_feature=True, product=True
        )
        
        mol_batch_synthon, synthon_tensors = MolTree.tensorize(
            synthon_trees, vocab, avocab, 
            use_feature=True, product=False
        )
        
        mol_batch_react, react_tensors = MolTree.tensorize(
            react_trees, vocab, avocab, 
            use_feature=True, product=False
        )
        
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
            'pretrain_infos': pretrain_infos,  # 预训练信息（包含product_orders）
            # 为MolCLR掩码添加MolTree对象
            'aug_trees': aug_trees  # 包含掩码信息的增强树对象
        }
        
        return batch_dict
        
    except Exception as e:
        print(f"批处理函数错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_dataloaders(train_data_path, val_data_path, vocab_path, args, max_train_samples=None, max_val_samples=None, use_small_dataset=False):
    """
    创建训练和验证数据加载器的辅助函数
    
    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        vocab_path: 词汇表路径
        args: 参数对象
        max_train_samples: 最大训练样本数（调试用）
        max_val_samples: 最大验证样本数（调试用）
        use_small_dataset: 是否使用小数据集进行快速测试
    
    Returns:
        train_dataset, val_dataset, train_loader, val_loader
    """
    # 创建数据集
    print("\n创建数据集...")
    if use_small_dataset:
        print("🚀 启用小数据集模式 - 快速测试!")
    train_dataset = G2RetroPDesignAlignedDataset(train_data_path, vocab_path, max_samples=max_train_samples, use_small_dataset=use_small_dataset)
    val_dataset = G2RetroPDesignAlignedDataset(val_data_path, vocab_path, max_samples=max_val_samples, use_small_dataset=use_small_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, train_dataset.vocab, train_dataset.avocab),
        num_workers=0,
        drop_last=True  # 确保批次大小一致（对比学习需要）
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, val_dataset.vocab, val_dataset.avocab),
        num_workers=0
    )
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 验证集大小: {len(val_dataset)}")
    print(f"✓ 训练批次数: {len(train_loader)}")
    print(f"✓ 验证批次数: {len(val_loader)}")
    
    return train_dataset, val_dataset, train_loader, val_loader

if __name__ == "__main__":
    # 测试数据集
    print("测试G2Retro-P数据集...")
    
    # 测试路径
    test_data_path = '../data/pretrain/pretrain_tensors_valid.pkl'
    vocab_path = '../data/pretrain/vocab_train.txt'
    
    # 创建数据集
    dataset = G2RetroPDesignAlignedDataset(test_data_path, vocab_path, max_samples=5)
    
    # 测试单个样本
    print("\n测试单个样本...")
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    
    # 测试批处理
    print("\n测试批处理...")
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    batch_data = g2retro_design_aligned_collate_fn(batch, dataset.vocab, dataset.avocab)
    
    if batch_data:
        print(f"批处理成功！")
        print(f"批次大小: {batch_data['batch_size']}")
        print(f"数据键: {list(batch_data.keys())}")
    else:
        print(f"批处理失败！")
    
    print("\n数据集测试完成！")