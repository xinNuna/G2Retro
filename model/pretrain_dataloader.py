#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据加载器 - 基于多任务预训练的半模板逆合成模型 (G2Retro-P)

这是一个新增的文件，应该放在 model/ 目录下，命名为 pretrain_dataloader.py
用于加载和处理预训练阶段的多任务数据
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import pickle
import random
import numpy as np
from rdkit import Chem
from mol_tree import MolTree
from nnutils import create_pad_tensor
import networkx as nx

class PretrainDataset(data.Dataset):
    """
    预训练数据集类
    支持三个任务：
    1. 基础任务（反应中心识别）
    2. 分子恢复任务（MolCLR增强）
    3. 产物-合成子对比学习
    """
    
    def __init__(self, data_path, vocab_path=None):
        """
        初始化预训练数据集
        
        Args:
            data_path: 预训练数据pkl文件路径
            vocab_path: 词汇表文件路径
        """
        print(f"加载预训练数据: {data_path}")
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"加载了 {len(self.data)} 条预训练数据")
        
        # 加载词汇表
        if vocab_path:
            self.vocab = self.load_vocab(vocab_path)
        else:
            self.vocab = set()
            
        # 增强类型映射
        self.augment_type_map = {
            'atom_mask': 0,
            'bond_deletion': 1, 
            'subgraph_removal': 2
        }
        
    def load_vocab(self, vocab_path):
        """加载词汇表"""
        vocab = set()
        with open(vocab_path, 'r') as f:
            for line in f:
                vocab.add(line.strip())
        return vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取一个预训练样本
        
        Returns:
            sample: 包含三个任务所需数据的字典
        """
        entry = self.data[idx]
        
        prod_moltree, synthon_tree, react_moltree = entry['mol_trees']
        pretrain_info = entry['pretrain_info']
        augmented_data = entry['augmented_data']
        
        # 1. 基础任务数据（反应中心识别）
        base_task_data = self.prepare_base_task_data(prod_moltree, react_moltree)
        
        # 2. 分子恢复任务数据
        recovery_task_data = self.prepare_recovery_task_data(prod_moltree, augmented_data)
        
        # 3. 产物-合成子对比学习数据
        contrastive_task_data = self.prepare_contrastive_task_data(
            prod_moltree, synthon_tree, pretrain_info
        )
        
        sample = {
            'base_task': base_task_data,
            'recovery_task': recovery_task_data,
            'contrastive_task': contrastive_task_data,
            'meta_info': {
                'idx': idx,
                'product_smiles': pretrain_info['product_smiles'],
                'synthon_smiles': pretrain_info['synthon_smiles']
            }
        }
        
        return sample
    
    def prepare_base_task_data(self, prod_moltree, react_moltree):
        """
        准备基础任务（反应中心识别）数据
        这部分完全沿用G2Retro的原始实现
        """
        try:
            # 获取产物分子图的原子和键特征
            prod_atom_features = []
            prod_bond_features = []
            
            for node in prod_moltree.mol_graph.nodes():
                atom_feat = self.get_atom_features(prod_moltree.mol_graph.nodes[node])
                prod_atom_features.append(atom_feat)
            
            for edge in prod_moltree.mol_graph.edges():
                bond_feat = self.get_bond_features(prod_moltree.mol_graph[edge[0]][edge[1]])
                prod_bond_features.append(bond_feat)
            
            # 获取反应中心标签
            reaction_center_labels = self.get_reaction_center_labels(prod_moltree, react_moltree)
            
            return {
                'atom_features': torch.FloatTensor(prod_atom_features),
                'bond_features': torch.FloatTensor(prod_bond_features), 
                'reaction_center_labels': reaction_center_labels,
                'mol_graph': prod_moltree.mol_graph
            }
            
        except Exception as e:
            print(f"准备基础任务数据时出错: {e}")
            return None
    
    def prepare_recovery_task_data(self, prod_moltree, augmented_data):
        """
        准备分子恢复任务数据
        基于MolCLR的增强策略
        """
        recovery_samples = []
        
        for aug_data in augmented_data:
            if aug_data is None:
                continue
                
            augment_type = aug_data['augment_type']
            masked_indices = aug_data['masked_indices']
            original_values = aug_data['original_values']
            
            # 创建增强后的分子图
            augmented_graph = self.create_augmented_graph(
                prod_moltree.mol_graph, augment_type, masked_indices
            )
            
            # 准备恢复目标
            recovery_targets = self.prepare_recovery_targets(
                augment_type, masked_indices, original_values
            )
            
            recovery_sample = {
                'augment_type': self.augment_type_map[augment_type],
                'augmented_graph': augmented_graph,
                'masked_indices': torch.LongTensor(masked_indices),
                'recovery_targets': recovery_targets,
                'original_graph': prod_moltree.mol_graph
            }
            
            recovery_samples.append(recovery_sample)
        
        return recovery_samples
    
    def prepare_contrastive_task_data(self, prod_moltree, synthon_tree, pretrain_info):
        """
        准备产物-合成子对比学习数据
        核心创新：利用产物与合成子的自然差异进行对比学习
        """
        try:
            # 产物分子的全局表示（正样本的锚点）
            product_features = self.get_global_mol_features(prod_moltree)
            
            # 合成子组合的全局表示（正样本）
            synthon_features = self.get_global_mol_features(synthon_tree)
            
            # 为对比学习准备标识符（在batch中用于构建负样本对）
            contrastive_data = {
                'product_features': product_features,
                'synthon_features': synthon_features,
                'product_smiles': pretrain_info['product_smiles'],
                'synthon_smiles': pretrain_info['synthon_smiles'],
                'pair_id': hash(pretrain_info['product_smiles'])  # 用于识别正样本对
            }
            
            return contrastive_data
            
        except Exception as e:
            print(f"准备对比学习数据时出错: {e}")
            return None
    
    def get_atom_features(self, atom_data):
        """获取原子特征向量"""
        # 基于G2Retro的原子特征提取
        features = [
            atom_data.get('atomic_num', 6),  # 原子序数
            atom_data.get('charge', 0),      # 电荷
            atom_data.get('valence', 4),     # 价数
            atom_data.get('num_h', 0),       # 氢原子数
            int(atom_data.get('aroma', False)),      # 芳香性
            int(atom_data.get('in_ring', False)),    # 是否在环中
        ]
        return features
    
    def get_bond_features(self, bond_data):
        """获取键特征向量"""
        features = [
            bond_data.get('label', 1),       # 键类型
            int(bond_data.get('is_conju', False)),   # 共轭性
            int(bond_data.get('is_aroma', False)),   # 芳香性
            int(bond_data.get('in_ring', False)),    # 是否在环中
        ]
        return features
    
    def get_reaction_center_labels(self, prod_moltree, react_moltree):
        """
        获取反应中心标签
        包括BF-center, BC-center, A-center
        """
        labels = {
            'bf_centers': [],  # 新形成的键
            'bc_centers': [],  # 键类型改变的键  
            'a_centers': []    # 失去片段的原子
        }
        
        # 基于G2Retro的反应中心识别逻辑
        # 这里需要根据prod_moltree和react_moltree的差异来识别反应中心
        
        try:
            # BF-centers: 产物中存在但反应物中不存在的键
            for edge in prod_moltree.mol_graph.edges():
                if 'attach' in prod_moltree.mol_graph[edge[0]][edge[1]]:
                    if prod_moltree.mol_graph[edge[0]][edge[1]]['attach'] == 1:
                        labels['bf_centers'].append(edge)
            
            # BC-centers: 键类型发生改变的键
            for edge in prod_moltree.mol_graph.edges():
                if 'change' in prod_moltree.mol_graph[edge[0]][edge[1]]:
                    if prod_moltree.mol_graph[edge[0]][edge[1]]['change'] >= 0:
                        labels['bc_centers'].append(edge)
            
            # A-centers: 失去片段的原子
            for node in prod_moltree.mol_graph.nodes():
                if 'delete' in prod_moltree.mol_graph.nodes[node]:
                    if prod_moltree.mol_graph.nodes[node]['delete'] == 1:
                        labels['a_centers'].append(node)
                        
        except Exception as e:
            print(f"获取反应中心标签时出错: {e}")
        
        # 转换为tensor格式
        for key in labels:
            if len(labels[key]) > 0:
                labels[key] = torch.LongTensor(labels[key])
            else:
                labels[key] = torch.LongTensor([])
        
        return labels
    
    def create_augmented_graph(self, original_graph, augment_type, masked_indices):
        """
        创建增强后的分子图
        实现MolCLR的三种增强策略
        """
        augmented_graph = original_graph.copy()
        
        if augment_type == 'atom_mask':
            # 原子掩码：将指定原子的特征设置为特殊值
            for idx in masked_indices:
                if idx in augmented_graph.nodes:
                    augmented_graph.nodes[idx]['masked'] = True
                    augmented_graph.nodes[idx]['original_label'] = augmented_graph.nodes[idx]['label']
                    augmented_graph.nodes[idx]['label'] = 'MASK'  # 掩码标记
                    
        elif augment_type == 'bond_deletion':
            # 键删除：移除指定的键
            edges_to_remove = []
            edge_list = list(augmented_graph.edges())
            
            for bond_idx in masked_indices:
                if bond_idx < len(edge_list):
                    edge = edge_list[bond_idx]
                    edges_to_remove.append(edge)
            
            for edge in edges_to_remove:
                if augmented_graph.has_edge(edge[0], edge[1]):
                    augmented_graph.remove_edge(edge[0], edge[1])
                    
        elif augment_type == 'subgraph_removal':
            # 子图移除：移除指定的原子及其相关键
            for idx in masked_indices:
                if idx in augmented_graph.nodes:
                    augmented_graph.nodes[idx]['masked'] = True
                    augmented_graph.nodes[idx]['removed'] = True
        
        return augmented_graph
    
    def prepare_recovery_targets(self, augment_type, masked_indices, original_values):
        """
        准备分子恢复任务的目标标签
        """
        targets = {
            'augment_type': augment_type,
            'masked_indices': masked_indices,
            'original_values': original_values
        }
        
        if augment_type == 'atom_mask':
            # 原子掩码恢复：预测被掩码原子的类型
            atom_type_map = {
                'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7,
                'P': 8, 'B': 9, 'Si': 10, 'H': 11, 'MASK': 12
            }
            
            target_labels = []
            for val in original_values:
                target_labels.append(atom_type_map.get(val, 0))
            
            targets['target_labels'] = torch.LongTensor(target_labels)
            
        elif augment_type == 'bond_deletion':
            # 键删除恢复：预测被删除键的类型
            bond_type_map = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3
            }
            
            target_labels = []
            for bond_type in original_values:
                target_labels.append(bond_type_map.get(bond_type, 0))
                
            targets['target_labels'] = torch.LongTensor(target_labels)
            
        elif augment_type == 'subgraph_removal':
            # 子图移除恢复：预测被移除原子的类型
            atom_type_map = {
                'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7,
                'P': 8, 'B': 9, 'Si': 10, 'H': 11
            }
            
            target_labels = []
            for val in original_values:
                target_labels.append(atom_type_map.get(val, 0))
                
            targets['target_labels'] = torch.LongTensor(target_labels)
        
        return targets
    
    def get_global_mol_features(self, mol_tree):
        """
        获取分子的全局特征表示
        用于对比学习任务
        """
        try:
            # 简单的分子全局特征提取
            # 在实际实现中，这将通过GMPN编码器获得
            
            num_atoms = len(mol_tree.mol_graph.nodes())
            num_bonds = len(mol_tree.mol_graph.edges())
            
            # 原子类型统计
            atom_counts = {}
            for node in mol_tree.mol_graph.nodes():
                atom_type = mol_tree.mol_graph.nodes[node].get('label', 'C')
                atom_counts[atom_type] = atom_counts.get(atom_type, 0) + 1
            
            # 构建简单的全局特征向量
            # 注意：在实际训练中，这些特征将被GMPN编码器的输出替代
            global_features = [
                num_atoms,
                num_bonds,
                atom_counts.get('C', 0),
                atom_counts.get('N', 0),
                atom_counts.get('O', 0),
                atom_counts.get('S', 0),
            ]
            
            return torch.FloatTensor(global_features)
            
        except Exception as e:
            print(f"获取全局分子特征时出错: {e}")
            return torch.zeros(6)  # 返回零向量作为后备


class PretrainDataLoader:
    """
    预训练数据加载器管理类
    处理批量数据和对比学习的负样本构建
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
    def get_dataloader(self):
        """获取PyTorch数据加载器"""
        return data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_pretrain_batch
        )
    
    def collate_pretrain_batch(self, batch):
        """
        批量数据整理函数
        处理变长数据并构建对比学习的正负样本对
        """
        batch_size = len(batch)
        
        # 分离三个任务的数据
        base_task_batch = []
        recovery_task_batch = []
        contrastive_task_batch = []
        meta_info_batch = []
        
        for sample in batch:
            if sample['base_task'] is not None:
                base_task_batch.append(sample['base_task'])
            if sample['recovery_task']:
                recovery_task_batch.extend(sample['recovery_task'])
            if sample['contrastive_task'] is not None:
                contrastive_task_batch.append(sample['contrastive_task'])
            meta_info_batch.append(sample['meta_info'])
        
        # 整理基础任务数据
        collated_base_task = self.collate_base_task(base_task_batch)
        
        # 整理分子恢复任务数据
        collated_recovery_task = self.collate_recovery_task(recovery_task_batch)
        
        # 整理对比学习任务数据（构建正负样本对）
        collated_contrastive_task = self.collate_contrastive_task(contrastive_task_batch)
        
        return {
            'base_task': collated_base_task,
            'recovery_task': collated_recovery_task,
            'contrastive_task': collated_contrastive_task,
            'meta_info': meta_info_batch,
            'batch_size': batch_size
        }
    
    def collate_base_task(self, base_task_batch):
        """整理基础任务批量数据"""
        if not base_task_batch:
            return None
            
        # 这里需要实现类似G2Retro中的图批量处理逻辑
        # 包括原子特征、键特征、图邻接信息等的批量打包
        
        collated = {
            'atom_features': [],
            'bond_features': [],
            'reaction_center_labels': {
                'bf_centers': [],
                'bc_centers': [],
                'a_centers': []
            },
            'graph_indices': [],  # 用于标识每个原子/键属于哪个分子
            'batch_size': len(base_task_batch)
        }
        
        atom_offset = 0
        for i, sample in enumerate(base_task_batch):
            collated['atom_features'].append(sample['atom_features'])
            collated['bond_features'].append(sample['bond_features'])
            
            # 调整反应中心标签的索引（加上偏移量）
            for center_type in ['bf_centers', 'bc_centers', 'a_centers']:
                centers = sample['reaction_center_labels'][center_type]
                if len(centers) > 0:
                    adjusted_centers = centers + atom_offset
                    collated['reaction_center_labels'][center_type].append(adjusted_centers)
            
            # 更新原子偏移量
            atom_offset += len(sample['atom_features'])
        
        # 拼接特征
        if collated['atom_features']:
            collated['atom_features'] = torch.cat(collated['atom_features'], dim=0)
        if collated['bond_features']:
            collated['bond_features'] = torch.cat(collated['bond_features'], dim=0)
        
        # 拼接反应中心标签
        for center_type in ['bf_centers', 'bc_centers', 'a_centers']:
            if collated['reaction_center_labels'][center_type]:
                collated['reaction_center_labels'][center_type] = torch.cat(
                    collated['reaction_center_labels'][center_type], dim=0
                )
            else:
                collated['reaction_center_labels'][center_type] = torch.LongTensor([])
        
        return collated
    
    def collate_recovery_task(self, recovery_task_batch):
        """整理分子恢复任务批量数据"""
        if not recovery_task_batch:
            return None
        
        # 按增强类型分组
        grouped_by_type = {}
        for sample in recovery_task_batch:
            aug_type = sample['augment_type']
            if aug_type not in grouped_by_type:
                grouped_by_type[aug_type] = []
            grouped_by_type[aug_type].append(sample)
        
        collated_by_type = {}
        for aug_type, samples in grouped_by_type.items():
            collated_samples = {
                'augment_type': aug_type,
                'masked_indices': [],
                'target_labels': [],
                'batch_size': len(samples)
            }
            
            for sample in samples:
                collated_samples['masked_indices'].append(sample['masked_indices'])
                collated_samples['target_labels'].append(sample['recovery_targets']['target_labels'])
            
            # 拼接数据
            if collated_samples['masked_indices']:
                collated_samples['masked_indices'] = torch.cat(collated_samples['masked_indices'], dim=0)
            if collated_samples['target_labels']:
                collated_samples['target_labels'] = torch.cat(collated_samples['target_labels'], dim=0)
            
            collated_by_type[aug_type] = collated_samples
        
        return collated_by_type
    
    def collate_contrastive_task(self, contrastive_task_batch):
        """
        整理对比学习任务批量数据
        构建产物-合成子正负样本对
        """
        if not contrastive_task_batch:
            return None
        
        batch_size = len(contrastive_task_batch)
        
        # 收集产物和合成子特征
        product_features = []
        synthon_features = []
        pair_ids = []
        
        for sample in contrastive_task_batch:
            product_features.append(sample['product_features'])
            synthon_features.append(sample['synthon_features'])
            pair_ids.append(sample['pair_id'])
        
        # 拼接特征
        product_features = torch.stack(product_features, dim=0)  # [batch_size, feature_dim]
        synthon_features = torch.stack(synthon_features, dim=0)  # [batch_size, feature_dim]
        
        # 构建正负样本标签矩阵
        # positive_mask[i][j] = 1 表示第i个产物和第j个合成子是正样本对
        positive_mask = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            positive_mask[i][i] = 1  # 对角线为正样本对
        
        collated_contrastive = {
            'product_features': product_features,
            'synthon_features': synthon_features,
            'positive_mask': positive_mask,
            'pair_ids': pair_ids,
            'batch_size': batch_size
        }
        
        return collated_contrastive


def create_pretrain_dataloaders(train_data_path, test_data_path, 
                              train_vocab_path=None, test_vocab_path=None,
                              batch_size=32, num_workers=4):
    """
    创建预训练数据加载器的便捷函数
    
    Args:
        train_data_path: 训练集数据路径
        test_data_path: 测试集数据路径  
        train_vocab_path: 训练集词汇表路径
        test_vocab_path: 测试集词汇表路径
        batch_size: 批量大小
        num_workers: 数据加载进程数
    
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    
    # 创建数据集
    train_dataset = PretrainDataset(train_data_path, train_vocab_path)
    test_dataset = PretrainDataset(test_data_path, test_vocab_path)
    
    # 创建数据加载器
    train_loader_manager = PretrainDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader_manager = PretrainDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    train_loader = train_loader_manager.get_dataloader()
    test_loader = test_loader_manager.get_dataloader()
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"批量大小: {batch_size}")
    
    return train_loader, test_loader


# 使用示例
if __name__ == "__main__":
    # 测试数据加载器
    train_data_path = "../data/pretrain/pretrain_tensors_train.pkl"
    test_data_path = "../data/pretrain/pretrain_tensors_test.pkl"
    train_vocab_path = "../data/pretrain/vocab_train.txt"
    test_vocab_path = "../data/pretrain/vocab_test.txt"
    
    try:
        train_loader, test_loader = create_pretrain_dataloaders(
            train_data_path, test_data_path,
            train_vocab_path, test_vocab_path,
            batch_size=4, num_workers=2
        )
        
        print("数据加载器创建成功！")
        
        # 测试加载一个批次
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n批次 {batch_idx}:")
            print(f"  基础任务数据: {batch['base_task'] is not None}")
            print(f"  恢复任务数据: {batch['recovery_task'] is not None}")
            print(f"  对比学习数据: {batch['contrastive_task'] is not None}")
            print(f"  批量大小: {batch['batch_size']}")
            
            if batch_idx >= 2:  # 只测试前几个批次
                break
                
    except Exception as e:
        print(f"测试数据加载器时出错: {e}")
        print("请确保数据文件存在且格式正确")