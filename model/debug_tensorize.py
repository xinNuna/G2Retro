#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
from mol_tree import MolTree
from vocab import Vocab, common_atom_vocab

def debug_tensorize():
    """调试MolTree.tensorize的输出格式"""
    print("调试MolTree.tensorize输出格式...")
    
    # 加载小数据集
    with open('../data/pretrain/pretrain_tensors_train_small.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 加载词汇表
    with open('../data/pretrain/vocab_train.txt', 'r') as f:
        words = [line.strip() for line in f.readlines()]
    vocab = Vocab(words)
    avocab = common_atom_vocab
    
    print(f"数据集大小: {len(data)}")
    print(f"词汇表大小: {vocab.size()}")
    print(f"原子词汇表大小: {avocab.size()}")
    
    # 检查前几个样本
    for i in range(min(3, len(data))):
        print(f"\n=== 样本 {i} ===")
        sample = data[i]
        print(f"样本键: {list(sample.keys())}")
        
        if 'mol_trees' in sample:
            mol_trees = sample['mol_trees']
            print(f"mol_trees类型: {type(mol_trees)}")
            print(f"mol_trees长度: {len(mol_trees)}")
            
            if len(mol_trees) >= 1:
                product_tree = mol_trees[0]
                print(f"product_tree类型: {type(product_tree)}")
                
                # 测试tensorize
                try:
                    print("\n测试MolTree.tensorize (use_feature=True, product=True)...")
                    mol_batch, tensors = MolTree.tensorize(
                        [product_tree], vocab, avocab, 
                        use_feature=True, product=True
                    )
                    
                    print(f"返回值类型: mol_batch={type(mol_batch)}, tensors={type(tensors)}")
                    print(f"tensors长度: {len(tensors)}")
                    
                    if len(tensors) >= 2:
                        graph_tensors = tensors[1][0]  # 第一个图的张量
                        print(f"graph_tensors类型: {type(graph_tensors)}")
                        print(f"graph_tensors长度: {len(graph_tensors)}")
                        
                        if len(graph_tensors) > 0:
                            fnode = graph_tensors[0]
                            print(f"fnode类型: {type(fnode)}")
                            if hasattr(fnode, 'shape'):
                                print(f"fnode形状: {fnode.shape}")
                                print(f"fnode维度: {fnode.dim()}")
                                print(f"fnode前3个元素: {fnode[:3]}")
                            else:
                                print(f"fnode没有shape属性: {fnode}")
                                
                except Exception as e:
                    print(f"tensorize错误: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # 测试另一种参数组合
                try:
                    print("\n测试MolTree.tensorize (use_feature=False, product=True)...")
                    mol_batch2, tensors2 = MolTree.tensorize(
                        [product_tree], vocab, avocab, 
                        use_feature=False, product=True
                    )
                    
                    if len(tensors2) >= 2:
                        graph_tensors2 = tensors2[1][0]
                        if len(graph_tensors2) > 0:
                            fnode2 = graph_tensors2[0]
                            print(f"use_feature=False时，fnode形状: {fnode2.shape if hasattr(fnode2, 'shape') else 'N/A'}")
                            
                except Exception as e:
                    print(f"use_feature=False测试错误: {e}")

if __name__ == "__main__":
    debug_tensorize() 