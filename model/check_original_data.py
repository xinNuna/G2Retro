#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

def check_original_data():
    """
    检查原始大数据集的格式
    """
    original_file = "../data/pretrain/pretrain_tensors_train.pkl"
    small_file = "../data/pretrain/pretrain_tensors_train_small.pkl"
    
    print("检查原始数据集 vs 小数据集格式...")
    
    for file_path, name in [(original_file, "原始大数据集"), (small_file, "小数据集")]:
        print(f"\n{'='*60}")
        print(f"检查: {name}")
        print(f"文件: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        try:
            print("正在加载数据...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"数据类型: {type(data)}")
            print(f"数据长度: {len(data)}")
            
            if len(data) > 0:
                # 检查前3个样本
                for i in range(min(3, len(data))):
                    sample = data[i]
                    print(f"\n样本 {i}:")
                    print(f"  类型: {type(sample)}")
                    
                    if isinstance(sample, dict):
                        print(f"  键: {list(sample.keys())}")
                    elif isinstance(sample, (list, tuple)):
                        print(f"  长度: {len(sample)}")
                        print(f"  元素类型: {[type(x) for x in sample[:3]]}")
                    else:
                        print(f"  值: {sample}")
                    
                    # 只检查第一个样本的详细信息
                    if i == 0:
                        break
                        
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_original_data() 