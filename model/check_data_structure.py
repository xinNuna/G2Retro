#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

def check_data_structure():
    """
    检查小数据集的实际数据结构
    """
    files_to_check = [
        "../data/pretrain/pretrain_tensors_train_small.pkl",
        "../data/pretrain/pretrain_tensors_valid_small.pkl"
    ]
    
    for file_path in files_to_check:
        print(f"\n{'='*60}")
        print(f"检查文件: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"数据类型: {type(data)}")
            print(f"数据长度: {len(data)}")
            
            if len(data) > 0:
                # 检查第一个样本
                sample = data[0]
                print(f"\n第一个样本:")
                print(f"  类型: {type(sample)}")
                
                if isinstance(sample, dict):
                    print(f"  键: {list(sample.keys())}")
                    for key, value in sample.items():
                        print(f"    {key}: {type(value)}")
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            try:
                                print(f"      长度: {len(value)}")
                            except:
                                pass
                elif isinstance(sample, (list, tuple)):
                    print(f"  长度: {len(sample)}")
                    for i, item in enumerate(sample[:3]):  # 只看前3个元素
                        print(f"    [{i}]: {type(item)}")
                else:
                    print(f"  值: {sample}")
                    
                # 检查更多样本的键是否一致
                if isinstance(sample, dict) and len(data) > 1:
                    print(f"\n检查键的一致性...")
                    first_keys = set(sample.keys())
                    inconsistent_samples = []
                    
                    for i in range(1, min(10, len(data))):  # 检查前10个样本
                        if isinstance(data[i], dict):
                            current_keys = set(data[i].keys())
                            if current_keys != first_keys:
                                inconsistent_samples.append(i)
                    
                    if inconsistent_samples:
                        print(f"  ⚠️  发现不一致的样本: {inconsistent_samples}")
                        for idx in inconsistent_samples[:3]:  # 只显示前3个
                            print(f"    样本{idx}的键: {list(data[idx].keys())}")
                    else:
                        print(f"  ✅ 前10个样本的键都一致")
                        
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("G2Retro-P 数据结构检查工具")
    check_data_structure() 