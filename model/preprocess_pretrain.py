#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据处理脚本 - 基于多任务预训练的半模板逆合成模型 (G2Retro-P)
处理USPTO_FULL数据集用于预训练阶段

这是一个新增的脚本文件，应该放在 model/ 目录下，命名为 preprocess_pretrain.py
"""

import sys, os
import pickle
import rdkit
import time
import pandas as pd
import pdb
import numpy as np
import multiprocessing as mp
from functools import partial
from mol_tree import MolTree, update_revise_atoms, get_synthon_trees
from chemutils import get_idx_from_mapnum, copy_edit_mol
from multiprocessing import Pool
from argparse import ArgumentParser
from rdkit import Chem
import random

def convert_uspto_full_format(row):
    """
    将USPTO_FULL格式转换为G2Retro预期的格式
    
    输入格式: id,reactants>reagents>production
    输出格式: id,rxn_smiles (reactants>>production)
    """
    try:
        id_val = row['id']
        reaction_parts = row['rxn_smiles'].split('>')
        
        if len(reaction_parts) != 3:
            return None
            
        reactants = reaction_parts[0]
        # 跳过试剂部分 reaction_parts[1] 
        products = reaction_parts[2]
        
        # 构造标准的反应SMILES格式: reactants>>products
        rxn_smiles = f"{reactants}>>{products}"
        
        return {
            'id': id_val,
            'rxn_smiles': rxn_smiles
        }
    except Exception as e:
        print(f"转换格式时出错: {e}")
        return None

def build_moltree_pretrain(data, use_dfs=True, shuffle=False):
    """
    为预训练构建分子树结构
    相比原版build_moltree，这个版本专门为预训练阶段设计
    """
    try:
        react_smiles = data['rxn_smiles'].split(">>")[0]
        prod_smiles = data['rxn_smiles'].split(">>")[1]
        
        # 构建产物分子树
        prod_moltree = MolTree(prod_smiles, use_brics=True, decompose_ring=True)
        react_moltree = MolTree(react_smiles)
        
        # 更新原子信息和反应中心识别
        update_revise_atoms(prod_moltree, react_moltree, use_dfs=use_dfs, shuffle=shuffle)
        
        vocab = set()
        
        # 获取合成子结构
        synthon_tree = get_synthon_trees(react_moltree)
        
        # 收集词汇表
        for node in react_moltree.mol_tree.nodes:
            if react_moltree.mol_tree.nodes[node]['revise'] == 1:
                vocab.add(react_moltree.mol_tree.nodes[node]['label'])
        
        # 验证合成子数量的一致性
        if len(synthon_tree.smiles.split(".")) != len(react_moltree.smiles.split(".")):
            return (None, None, None, set())
            
        # 为预训练添加额外信息
        # 1. 产物分子图用于基础任务和对比学习
        # 2. 合成子结构用于对比学习
        # 3. 反应中心信息用于基础任务
        pretrain_info = {
            'product_smiles': prod_smiles,
            'synthon_smiles': synthon_tree.smiles,
            'reactant_smiles': react_smiles,
            'has_reaction_center': True  # 标记是否包含有效的反应中心
        }
        
        return (prod_moltree, synthon_tree, react_moltree, vocab, pretrain_info)
    except Exception as e:
        print(f"构建分子树时出错: {e}")
        return (None, None, None, set(), None)

def augment_molecule_for_recovery(mol_smiles, augment_type='atom_mask', ratio=0.15):
    """
    为分子恢复任务生成增强数据
    实现MolCLR的三种增强策略：原子掩码、键删除、子图移除
    
    Args:
        mol_smiles: 分子SMILES字符串
        augment_type: 增强类型 ('atom_mask', 'bond_deletion', 'subgraph_removal')
        ratio: 增强比例
    
    Returns:
        augmented_data: 包含原始信息和掩码信息的字典
    """
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            return None
            
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        augmented_data = {
            'original_smiles': mol_smiles,
            'augment_type': augment_type,
            'masked_indices': [],
            'original_values': []
        }
        
        if augment_type == 'atom_mask':
            # 原子掩码：随机选择部分原子进行掩码
            num_mask = max(1, int(num_atoms * ratio))
            mask_indices = random.sample(range(num_atoms), min(num_mask, num_atoms))
            
            augmented_data['masked_indices'] = mask_indices
            augmented_data['original_values'] = [mol.GetAtomWithIdx(i).GetSymbol() for i in mask_indices]
            
        elif augment_type == 'bond_deletion':
            # 键删除：随机删除部分化学键
            if num_bonds > 0:
                num_delete = max(1, int(num_bonds * ratio))
                bond_indices = random.sample(range(num_bonds), min(num_delete, num_bonds))
                
                augmented_data['masked_indices'] = bond_indices
                augmented_data['original_values'] = [mol.GetBondWithIdx(i).GetBondType() for i in bond_indices]
                
        elif augment_type == 'subgraph_removal':
            # 子图移除：移除连通的子图
            if num_atoms > 2:
                # 选择起始原子
                start_atom = random.randint(0, num_atoms - 1)
                subgraph_size = max(1, int(num_atoms * ratio))
                
                # BFS获取连通子图
                visited = set()
                queue = [start_atom]
                subgraph_atoms = []
                
                while queue and len(subgraph_atoms) < subgraph_size:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        subgraph_atoms.append(current)
                        
                        # 添加邻接原子
                        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx not in visited:
                                queue.append(neighbor_idx)
                
                augmented_data['masked_indices'] = subgraph_atoms
                augmented_data['original_values'] = [mol.GetAtomWithIdx(i).GetSymbol() for i in subgraph_atoms]
        
        return augmented_data
        
    except Exception as e:
        print(f"分子增强时出错: {e}")
        return None

def process_pretrain_data_batch(data_batch, use_dfs=True, shuffle=False):
    """
    批量处理预训练数据，增加详细的错误统计
    """
    results = []
    error_stats = {
        "total_processed": 0,
        "successful": 0,
        "format_error": 0,
        "product_tree_error": 0,
        "reactant_tree_error": 0,
        "zero_connect_atoms": 0,
        "update_atoms_error": 0,
        "synthon_error": 0,
        "synthon_count_mismatch": 0,
        "unknown_error": 0,
        "conversion_error": 0
    }
    
    for data in data_batch:
        error_stats["total_processed"] += 1
        
        # 转换数据格式
        converted_data = convert_uspto_full_format(data)
        if converted_data is None:
            error_stats["conversion_error"] += 1
            continue
            
        # 构建分子树
        mol_tree_result = build_moltree_pretrain(converted_data, use_dfs, shuffle)
        
        if len(mol_tree_result) == 5:
            prod_moltree, synthon_tree, react_moltree, vocab, pretrain_info = mol_tree_result
            
            if mol_tree_result[0] is None:
                # 统计错误类型
                if len(mol_tree_result) > 4 and isinstance(mol_tree_result[4], dict):
                    error_info = mol_tree_result[4]
                    error_type = error_info.get("type", "unknown_error")
                    if error_type in error_stats:
                        error_stats[error_type] += 1
                    else:
                        error_stats["unknown_error"] += 1
                continue
            
            error_stats["successful"] += 1
            
            # 为每个产物分子生成三种增强数据（用于分子恢复任务）
            augment_types = ['atom_mask', 'bond_deletion', 'subgraph_removal']
            augmented_data = []
            
            for aug_type in augment_types:
                try:
                    aug_data = augment_molecule_for_recovery(
                        pretrain_info['product_smiles'], 
                        augment_type=aug_type, 
                        ratio=0.15
                    )
                    if aug_data is not None:
                        augmented_data.append(aug_data)
                except Exception as e:
                    # 增强失败不影响主要数据的使用
                    continue
            
            # 构建预训练数据条目
            pretrain_entry = {
                'mol_trees': (prod_moltree, synthon_tree, react_moltree),
                'vocab': vocab,
                'pretrain_info': pretrain_info,
                'augmented_data': augmented_data,
                'original_data': converted_data
            }
            
            results.append(pretrain_entry)
    
    return results, error_stats

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument('--uspto_full_train', type=str, required=True, 
                       help="USPTO_FULL训练集路径 (格式: id,reactants>reagents>production)")
    parser.add_argument('--uspto_full_test', type=str, required=True,
                       help="USPTO_FULL测试集路径")
    parser.add_argument('--output_dir', type=str, default="../data/pretrain/", 
                       help="预训练数据输出目录")
    parser.add_argument('--output_name', type=str, default="pretrain_tensors", 
                       help="输出文件名前缀")
    parser.add_argument('--use_bfs', action="store_true", help="使用BFS而不是DFS")
    parser.add_argument('--shuffle', action="store_true", help="是否打乱数据")
    parser.add_argument('--ncpu', type=int, default=10, help="CPU进程数")
    parser.add_argument('--batch_size', type=int, default=1000, help="批处理大小")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("开始处理USPTO_FULL数据用于预训练...")
    
    # 处理训练集和测试集
    datasets = {
        'train': args.uspto_full_train,
        'test': args.uspto_full_test
    }
    
    for split_name, data_path in datasets.items():
        print(f"\n处理{split_name}集: {data_path}")
        
        # 读取数据
        if data_path.endswith('.csv'):
            # USPTO_FULL格式的CSV文件
            all_data = pd.read_csv(data_path, sep=',')
            # 重命名列以匹配期望的格式
            if 'reactants>reagents>production' in all_data.columns:
                all_data = all_data.rename(columns={'reactants>reagents>production': 'rxn_smiles'})
        else:
            print(f"不支持的文件格式: {data_path}")
            continue
            
        print(f"读取到 {len(all_data)} 条反应数据")
        
        # 转换为列表格式
        all_data_list = [all_data.iloc[i,:].to_dict() for i in range(len(all_data))]
        
        # 批量处理数据
        processed_data = []
        batch_size = args.batch_size
        total_error_stats = {
            "total_processed": 0,
            "successful": 0,
            "format_error": 0,
            "product_tree_error": 0,
            "reactant_tree_error": 0,
            "zero_connect_atoms": 0,
            "update_atoms_error": 0,
            "synthon_error": 0,
            "synthon_count_mismatch": 0,
            "unknown_error": 0,
            "conversion_error": 0
        }
        
        for i in range(0, len(all_data_list), batch_size):
            batch = all_data_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(all_data_list)-1)//batch_size + 1
            
            print(f"处理批次 {batch_num}/{total_batches}")
            
            # 处理当前批次
            batch_results, batch_error_stats = process_pretrain_data_batch(
                batch, 
                use_dfs=not args.use_bfs, 
                shuffle=args.shuffle
            )
            processed_data.extend(batch_results)
            
            # 累计错误统计
            for key in total_error_stats:
                total_error_stats[key] += batch_error_stats[key]
            
            # 每10个批次报告一次进度
            if batch_num % 10 == 0:
                success_rate = (total_error_stats["successful"] / max(total_error_stats["total_processed"], 1)) * 100
                print(f"  - 已处理: {total_error_stats['total_processed']} 条")
                print(f"  - 成功: {total_error_stats['successful']} 条 ({success_rate:.1f}%)")
                print(f"  - 主要错误: 连接原子为零({total_error_stats['zero_connect_atoms']}), "
                      f"格式错误({total_error_stats['format_error']}), "
                      f"更新原子错误({total_error_stats['update_atoms_error']})")
        
        print(f"\n{split_name}集处理完成:")
        print(f"总计处理: {total_error_stats['total_processed']} 条反应")
        print(f"成功处理: {total_error_stats['successful']} 条反应")
        success_rate = (total_error_stats['successful'] / max(total_error_stats['total_processed'], 1)) * 100
        print(f"成功率: {success_rate:.1f}%")
        
        print("\n详细错误统计:")
        for error_type, count in total_error_stats.items():
            if error_type not in ['total_processed', 'successful'] and count > 0:
                error_rate = (count / max(total_error_stats['total_processed'], 1)) * 100
                print(f"  - {error_type}: {count} 条 ({error_rate:.1f}%)")
        
        if total_error_stats['successful'] == 0:
            print(f"\n警告: {split_name}集没有成功处理任何数据！请检查数据格式和依赖项。")
            continue
        
        # 收集全局词汇表
        global_vocab = set()
        for entry in processed_data:
            global_vocab.update(entry['vocab'])
        
        # 保存词汇表
        vocab_path = os.path.join(args.output_dir, f"vocab_{split_name}.txt")
        with open(vocab_path, 'w') as f:
            for word in sorted(global_vocab):
                f.write(f"{word}\n")
        print(f"词汇表已保存到: {vocab_path}")
        
        # 保存处理后的数据
        output_path = os.path.join(args.output_dir, f"{args.output_name}_{split_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f, pickle.HIGHEST_PROTOCOL)
        print(f"预训练数据已保存到: {output_path}")
        
        # 保存数据统计信息
        stats = {
            'total_reactions': len(processed_data),
            'vocab_size': len(global_vocab),
            'augmentation_types': ['atom_mask', 'bond_deletion', 'subgraph_removal'],
            'data_format': 'pretrain_multitask'
        }
        
        stats_path = os.path.join(args.output_dir, f"stats_{split_name}.txt")
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print(f"统计信息已保存到: {stats_path}")
    
    print("\n预训练数据处理完成！")
    print(f"输出目录: {args.output_dir}")
    print("文件结构:")
    print("  - pretrain_tensors_train.pkl  # 训练集预训练数据")
    print("  - pretrain_tensors_test.pkl   # 测试集预训练数据") 
    print("  - vocab_train.txt             # 训练集词汇表")
    print("  - vocab_test.txt              # 测试集词汇表")
    print("  - stats_train.txt             # 训练集统计信息")
    print("  - stats_test.txt              # 测试集统计信息")

if __name__ == "__main__":
    main()