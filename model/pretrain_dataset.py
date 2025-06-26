#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2Retro-P预训练数据集类 - 基于多任务预训练的半模板逆合成模型

这个文件实现了预训练阶段的数据加载，支持：
1. 三路数据加载（产物、增强产物、合成子）
2. 批次内负样本采样（用于产物-合成子对比学习）
3. MolCLR增强策略的应用
4. 与G2Retro现有组件的完全兼容

参考文献：
- G2Retro: https://www.nature.com/articles/s42004-023-00897-3
- PMSR: Learning Chemical Rules of Retrosynthesis with Pre-training
- MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks
"""

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from typing import List, Dict, Tuple, Optional, Union

# 导入G2Retro现有组件
from mol_tree import MolTree
from vocab import Vocab
from config import device

# 定义简单的make_cuda函数（复用G2Retro组件）
def make_cuda(tensors):
    """将张量转移到CUDA设备（复用G2Retro逻辑）"""
    if torch.cuda.is_available():
        if isinstance(tensors, (list, tuple)):
            return [t.to(device) if torch.is_tensor(t) else t for t in tensors]
        else:
            return tensors.to(device)
    return tensors

class PretrainDataset(Dataset):
    """
    G2Retro-P预训练数据集类
    
    按照设计方案实现：
    1. 基础任务数据：产物分子图 + 反应中心标注（复用G2Retro）
    2. 分子恢复任务数据：MolCLR增强的产物分子图 + 恢复目标（完全参考MolCLR）
    3. 产物-合成子对比数据：产物vs合成子的自然差异（核心创新）
    """
    
    def __init__(self, 
                 pkl_path: str,
                 vocab: Vocab, 
                 avocab: Vocab,
                 contrastive_sampling: bool = True,
                 augment_ratio: float = 0.15,
                 max_samples: Optional[int] = None,
                 use_atomic: bool = False,
                 use_feature: bool = True):
        """
        初始化预训练数据集
        
        Args:
            pkl_path: 预训练数据pkl文件路径（用户生成的pretrain_tensors_train.pkl）
            vocab: 子结构词汇表（G2Retro组件）
            avocab: 原子词汇表（G2Retro组件）
            contrastive_sampling: 是否启用对比学习负样本采样
            augment_ratio: MolCLR增强比例
            max_samples: 最大样本数（调试用）
            use_atomic: 是否使用原子特征（G2Retro兼容）
            use_feature: 是否使用分子特征（G2Retro兼容）
        """
        print(f"🔄 加载G2Retro-P预训练数据: {pkl_path}")
        
        # 检查文件存在性
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"预训练数据文件不存在: {pkl_path}")
        
        # 加载预处理好的数据
        with open(pkl_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        
        # 限制样本数量（调试用）
        if max_samples and len(self.raw_data) > max_samples:
            print(f"🔬 使用前{max_samples}个样本进行调试")
            self.raw_data = self.raw_data[:max_samples]
        
        print(f"✅ 数据加载成功: {len(self.raw_data):,} 条记录")
        
        # 保存参数
        self.vocab = vocab
        self.avocab = avocab
        self.contrastive_sampling = contrastive_sampling
        self.augment_ratio = augment_ratio
        self.use_atomic = use_atomic
        self.use_feature = use_feature
        
        # 验证数据质量
        self._validate_data_quality()
        
        # 为对比学习准备负样本采样
        if self.contrastive_sampling:
            self._prepare_contrastive_sampling()
    
    def _validate_data_quality(self):
        """验证加载数据的质量和格式"""
        print("🔍 验证数据质量...")
        
        required_keys = ['mol_trees', 'pretrain_info', 'augmented_data', 'vocab']
        valid_count = 0
        
        for i, entry in enumerate(self.raw_data[:100]):  # 检查前100个样本
            if not all(key in entry for key in required_keys):
                continue
                
            mol_trees = entry['mol_trees']
            if len(mol_trees) != 3:  # 应该有产物、合成子、反应物三个分子树
                continue
                
            prod_tree, synthon_tree, react_tree = mol_trees
            if prod_tree is None or synthon_tree is None or react_tree is None:
                continue
                
            # 检查MolCLR增强数据
            aug_data = entry['augmented_data']
            if not isinstance(aug_data, list) or len(aug_data) == 0:
                continue
                
            valid_count += 1
        
        valid_rate = valid_count / min(100, len(self.raw_data))
        print(f"📊 数据质量检查: {valid_rate:.1%} 有效率")
        
        if valid_rate < 0.8:
            print("⚠️  警告: 数据质量较低，请检查预处理脚本")
    
    def _prepare_contrastive_sampling(self):
        """
        为产物-合成子对比学习准备负样本采样策略
        
        按照设计方案：利用产物分子与其合成子组合之间的自然差异进行对比学习
        """
        print("🎯 准备产物-合成子对比学习负样本采样...")
        
        # 构建有效数据索引
        self.valid_indices = []
        for i, entry in enumerate(self.raw_data):
            if self._is_valid_entry(entry):
                self.valid_indices.append(i)
        
        print(f"✅ 有效对比学习样本: {len(self.valid_indices):,} 条")
        
        # 预计算产物SMILES的哈希，用于快速负样本采样
        self.product_smiles_hash = {}
        for idx in self.valid_indices:
            entry = self.raw_data[idx]
            product_smiles = entry['pretrain_info']['product_smiles']
            if product_smiles not in self.product_smiles_hash:
                self.product_smiles_hash[product_smiles] = []
            self.product_smiles_hash[product_smiles].append(idx)
    
    def _is_valid_entry(self, entry: Dict) -> bool:
        """检查数据条目是否有效"""
        try:
            mol_trees = entry.get('mol_trees', ())
            if len(mol_trees) != 3:
                return False
            
            prod_tree, synthon_tree, react_tree = mol_trees
            if prod_tree is None or synthon_tree is None:
                return False
            
            pretrain_info = entry.get('pretrain_info', {})
            if not pretrain_info.get('has_reaction_center', False):
                return False
            
            return True
        except:
            return False
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个训练样本
        
        按照设计方案返回三路数据：
        1. 原始产物分子图（用于基础任务和对比学习）
        2. 增强产物分子图（用于分子恢复任务）
        3. 合成子组合（用于对比学习）
        
        Returns:
            包含所有预训练任务所需数据的字典
        """
        entry = self.raw_data[idx]
        
        # 提取核心分子树（完全复用G2Retro组件）
        prod_tree, synthon_tree, react_tree = entry['mol_trees']
        pretrain_info = entry['pretrain_info']
        augmented_data = entry['augmented_data']
        
        # 1. 基础任务数据：产物分子图 + 反应中心标注（直接使用G2Retro）
        base_task_data = {
            'product_tree': prod_tree,
            'react_tree': react_tree,  # 包含反应中心标注
            'pretrain_info': pretrain_info
        }
        
        # 2. 分子恢复任务数据：应用MolCLR增强策略
        recovery_task_data = self._prepare_recovery_task(prod_tree, augmented_data)
        
        # 3. 产物-合成子对比数据：核心创新
        contrastive_task_data = self._prepare_contrastive_task(
            prod_tree, synthon_tree, idx
        )
        
        return {
            'idx': idx,
            'base_task': base_task_data,
            'recovery_task': recovery_task_data,
            'contrastive_task': contrastive_task_data,
            'vocab_set': entry.get('vocab', set())
        }
    
    def _prepare_recovery_task(self, prod_tree: MolTree, augmented_data: List[Dict]) -> Dict:
        """
        准备分子恢复任务数据
        
        完全参考MolCLR的三种增强策略：
        1. 原子掩码 (Atom Masking)
        2. 键删除 (Bond Deletion) 
        3. 子图移除 (Subgraph Removal)
        
        按照PMSR的span-mask策略进行mask生成
        """
        recovery_data = {
            'original_tree': prod_tree,
            'augmented_versions': [],
            'recovery_targets': []
        }
        
        # 遍历所有增强版本（MolCLR三种策略）
        for aug_item in augmented_data:
            aug_type = aug_item.get('augment_type', 'unknown')
            masked_indices = aug_item.get('masked_indices', [])
            original_values = aug_item.get('original_values', [])
            
            # 构造恢复目标（参考PMSR的span-mask）
            recovery_target = {
                'augment_type': aug_type,
                'masked_positions': masked_indices,
                'target_values': original_values,
                'mask_ratio': len(masked_indices) / max(prod_tree.mol.GetNumAtoms(), 1)
            }
            
            recovery_data['augmented_versions'].append(aug_item)
            recovery_data['recovery_targets'].append(recovery_target)
        
        return recovery_data
    
    def _prepare_contrastive_task(self, 
                                prod_tree: MolTree, 
                                synthon_tree: MolTree, 
                                current_idx: int) -> Dict:
        """
        准备产物-合成子对比学习任务数据
        
        核心创新：直接利用产物分子与其对应合成子组合之间的自然差异
        这种差异本身就包含了反应中心信息，与下游任务实现完美对齐
        """
        contrastive_data = {
            'anchor': prod_tree,      # 锚点：产物分子
            'positive': synthon_tree, # 正样本：对应的合成子组合
            'negatives': []           # 负样本：其他反应的合成子
        }
        
        # 批次内负样本采样（在collate_fn中处理，这里先准备索引）
        if self.contrastive_sampling and hasattr(self, 'valid_indices'):
            # 随机选择一些负样本候选
            negative_candidates = [
                idx for idx in self.valid_indices 
                if idx != current_idx
            ]
            
            # 限制负样本数量以避免内存问题
            max_negatives = min(10, len(negative_candidates))
            if max_negatives > 0:
                negative_indices = random.sample(negative_candidates, max_negatives)
                contrastive_data['negative_indices'] = negative_indices
        
        return contrastive_data
    
    def get_contrastive_batch(self, indices: List[int]) -> Dict:
        """
        为对比学习构造批次数据
        
        实现批次内负样本采样：
        - 正样本对：(产物分子图, 对应的合成子组合图)
        - 负样本对：批次内不同反应的产物-合成子对
        """
        batch_data = {
            'products': [],           # 产物分子图
            'synthons_positive': [],  # 对应的合成子
            'synthons_negative': [],  # 负样本合成子
            'labels': []              # 对比学习标签
        }
        
        # 收集所有产物和合成子
        all_products = []
        all_synthons = []
        
        for idx in indices:
            if idx < len(self.raw_data) and self._is_valid_entry(self.raw_data[idx]):
                entry = self.raw_data[idx]
                prod_tree, synthon_tree, _ = entry['mol_trees']
                
                all_products.append(prod_tree)
                all_synthons.append(synthon_tree)
        
        # 构造正负样本对
        for i, (prod, synthon_pos) in enumerate(zip(all_products, all_synthons)):
            batch_data['products'].append(prod)
            batch_data['synthons_positive'].append(synthon_pos)
            
            # 批次内负样本：其他反应的合成子
            negative_synthons = [
                all_synthons[j] for j in range(len(all_synthons)) if j != i
            ]
            batch_data['synthons_negative'].extend(negative_synthons)
            
            # 标签：0表示正样本，1表示负样本
            batch_data['labels'].extend([0] + [1] * len(negative_synthons))
        
        return batch_data

def collate_pretrain_batch(batch: List[Dict]) -> Dict:
    """
    预训练批次数据整理函数
    
    将多个样本组织成适合G2Retro-P三个并行任务头的批次格式：
    1. 基础任务批次（复用G2Retro的MolCenter）
    2. 分子恢复任务批次（参考MolCLR）
    3. 产物-合成子对比批次（核心创新）
    """
    
    batch_size = len(batch)
    collated = {
        'batch_size': batch_size,
        'indices': [item['idx'] for item in batch],
        'base_tasks': [],
        'recovery_tasks': [],
        'contrastive_tasks': {
            'products': [],
            'synthons_positive': [],
            'synthons_negative': [],
            'batch_labels': []
        }
    }
    
    # 1. 收集基础任务数据（直接传给G2Retro的MolCenter）
    for item in batch:
        base_task = item['base_task']
        collated['base_tasks'].append({
            'product_tree': base_task['product_tree'],
            'react_tree': base_task['react_tree'],
            'pretrain_info': base_task['pretrain_info']
        })
    
    # 2. 收集分子恢复任务数据（送给分子恢复头）
    for item in batch:
        recovery_task = item['recovery_task']
        collated['recovery_tasks'].append(recovery_task)
    
    # 3. 构造对比学习批次（批次内负样本采样）
    for i, item in enumerate(batch):
        contrastive_task = item['contrastive_task']
        anchor = contrastive_task['anchor']      # 产物
        positive = contrastive_task['positive']  # 对应合成子
        
        collated['contrastive_tasks']['products'].append(anchor)
        collated['contrastive_tasks']['synthons_positive'].append(positive)
        
        # 批次内负样本：其他样本的合成子
        negatives = [
            batch[j]['contrastive_task']['positive'] 
            for j in range(batch_size) if j != i
        ]
        collated['contrastive_tasks']['synthons_negative'].extend(negatives)
        
        # 对比学习标签
        collated['contrastive_tasks']['batch_labels'].extend(
            [i] + [-1] * len(negatives)  # i表示正样本，-1表示负样本
        )
    
    return collated

def create_pretrain_dataloader(pkl_path: str,
                              vocab: Vocab,
                              avocab: Vocab, 
                              batch_size: int = 32,
                              num_workers: int = 4,
                              shuffle: bool = True,
                              **kwargs) -> DataLoader:
    """
    创建G2Retro-P预训练数据加载器
    
    这是用户调用的主要接口，集成了所有预训练功能
    """
    
    # 创建数据集
    dataset = PretrainDataset(
        pkl_path=pkl_path,
        vocab=vocab, 
        avocab=avocab,
        **kwargs
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pretrain_batch,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致（对比学习需要）
    )
    
    print(f"✅ G2Retro-P预训练数据加载器创建成功")
    print(f"   - 数据集大小: {len(dataset):,} 条")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 总批次数: {len(dataloader):,}")
    
    return dataloader

# 测试和验证函数
def test_pretrain_dataset(pkl_path: str, vocab: Vocab, avocab: Vocab):
    """测试预训练数据集的功能"""
    print("🧪 测试G2Retro-P预训练数据集...")
    
    # 创建小规模测试数据集
    dataset = PretrainDataset(
        pkl_path=pkl_path,
        vocab=vocab,
        avocab=avocab,
        max_samples=100  # 仅测试100个样本
    )
    
    # 测试单个样本
    sample = dataset[0]
    print(f"✅ 样本结构: {list(sample.keys())}")
    
    # 测试数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        collate_fn=collate_pretrain_batch
    )
    
    batch = next(iter(dataloader))
    print(f"✅ 批次结构: {list(batch.keys())}")
    print(f"✅ 批次大小: {batch['batch_size']}")
    
    print("🎉 预训练数据集测试通过！")

if __name__ == "__main__":
    # 示例用法
    from vocab import common_atom_vocab
    
    # 这里需要根据实际情况设置路径和词汇表
    pkl_path = "../data/pretrain/pretrain_tensors_train.pkl"
    vocab_path = "../data/pretrain/vocab_train.txt"
    
    # 加载词汇表（复用G2Retro）
    vocab_list = [x.strip() for x in open(vocab_path)]
    vocab = Vocab(vocab_list)
    avocab = common_atom_vocab
    
    # 测试数据集
    test_pretrain_dataset(pkl_path, vocab, avocab) 