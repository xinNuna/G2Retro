#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import torch
from mol_tree import MolTree
from vocab import Vocab, common_atom_vocab
from config import device
import numpy as np

def debug_cuda_error():
    """调试CUDA索引错误"""
    print(f"当前设备: {device}")
    print("调试CUDA索引错误...")
    
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
    
    # 测试前几个样本
    for i in range(min(3, len(data))):
        print(f"\n=== 样本 {i} ===")
        sample = data[i]
        product_tree = sample['mol_trees'][0]
        
        try:
            # 测试MolTree.tensorize
            mol_batch, tensors = MolTree.tensorize([product_tree], vocab, avocab, use_feature=True, product=True)
            graph_tensors = tensors[1][0]  # 提取图张量
            
            print(f"图张量长度: {len(graph_tensors)}")
            
            # 检查每个张量的属性
            tensor_names = ['fnode', 'fmess', 'agraph', 'bgraph', 'fgraph', 'egraph', 'scope']
            for j, (name, tensor) in enumerate(zip(tensor_names, graph_tensors)):
                print(f"  {name} ({j}): 类型={type(tensor)}", end="")
                if tensor is not None:
                    if hasattr(tensor, 'shape'):
                        print(f", 形状={tensor.shape}, 设备={tensor.device}, 数据类型={tensor.dtype}")
                        # 检查数值范围
                        if tensor.numel() > 0:
                            if tensor.dtype in [torch.long, torch.int32, torch.int64]:
                                print(f"    数值范围: [{tensor.min().item()}, {tensor.max().item()}]")
                            # 检查是否有负数或异常值
                            if torch.any(tensor < 0):
                                print(f"    警告: 包含负数!")
                            if j in [2, 3, 4]:  # agraph, bgraph, fgraph - 这些是索引张量
                                max_val = tensor.max().item() if tensor.numel() > 0 else -1
                                print(f"    最大索引值: {max_val}")
                                # 检查索引是否超出fnode的范围
                                fnode = graph_tensors[0]
                                if fnode is not None and hasattr(fnode, 'shape'):
                                    fnode_size = fnode.shape[0]
                                    if max_val >= fnode_size:
                                        print(f"    错误: 索引{max_val}超出fnode大小{fnode_size}!")
                    else:
                        print(f", 值={tensor}")
                else:
                    print(", None")
            
            # 创建我们的make_cuda函数并测试
            def make_cuda_graph_tensors(graph_tensors):
                """正确处理图张量的make_cuda函数"""
                make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x, requires_grad=False)
                new_tensors = []
                for idx, tensor in enumerate(graph_tensors):
                    if tensor is None:
                        new_tensors.append(None)
                    elif idx == 6:  # scope张量保持原样
                        new_tensors.append(tensor)
                    else:
                        # 确保张量在正确的设备上，并且是长整型
                        t = make_tensor(tensor)
                        if t.device != device:
                            t = t.to(device)
                        if t.dtype != torch.long:
                            t = t.long()
                        new_tensors.append(t)
                return new_tensors
            
            print(f"\n  测试make_cuda转换...")
            cuda_tensors = make_cuda_graph_tensors(graph_tensors)
            
            # 再次检查转换后的张量
            for j, (name, tensor) in enumerate(zip(tensor_names, cuda_tensors)):
                print(f"  CUDA {name} ({j}): 类型={type(tensor)}", end="")
                if tensor is not None:
                    if hasattr(tensor, 'shape'):
                        print(f", 形状={tensor.shape}, 设备={tensor.device}, 数据类型={tensor.dtype}")
                        if j in [2, 3, 4] and tensor.numel() > 0:  # 索引张量
                            max_val = tensor.max().item()
                            fnode = cuda_tensors[0]
                            if fnode is not None:
                                fnode_size = fnode.shape[0]
                                if max_val >= fnode_size:
                                    print(f"    错误: 转换后索引{max_val}仍超出fnode大小{fnode_size}!")
                                else:
                                    print(f"    OK: 最大索引{max_val} < fnode大小{fnode_size}")
                    else:
                        print(f", 值={tensor}")
                else:
                    print(", None")
                    
        except Exception as e:
            print(f"样本{i}处理失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_cuda_error() 