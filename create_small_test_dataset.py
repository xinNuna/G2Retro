#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
from tqdm import tqdm

def create_small_test_dataset():
    """
    从大的训练数据文件中提取小部分数据用于快速测试
    """
    
    # 文件路径
    large_train_file = "data/pretrain/pretrain_tensors_train.pkl"
    small_train_file = "data/pretrain/pretrain_tensors_train_small.pkl"
    
    # 检查原始文件是否存在
    if not os.path.exists(large_train_file):
        print(f"错误: 找不到文件 {large_train_file}")
        return
    
    print(f"正在从 {large_train_file} 提取小数据集...")
    print(f"原始文件大小: {os.path.getsize(large_train_file) / (1024**3):.2f} GB")
    
    # 设置提取数量
    extract_count = 1000  # 只取前1000个样本
    
    try:
        # 读取原始数据
        print("正在加载原始数据...")
        with open(large_train_file, 'rb') as f:
            full_data = pickle.load(f)
        
        print(f"原始数据总量: {len(full_data)} 个样本")
        
        # 提取前N个样本
        small_data = full_data[:extract_count]
        print(f"提取样本数量: {len(small_data)} 个")
        
        # 保存小数据集
        print(f"正在保存到 {small_train_file}...")
        with open(small_train_file, 'wb') as f:
            pickle.dump(small_data, f)
        
        print(f"小数据集已保存!")
        print(f"新文件大小: {os.path.getsize(small_train_file) / (1024**2):.2f} MB")
        print(f"压缩比例: {len(small_data) / len(full_data) * 100:.2f}%")
        
        # 验证数据格式
        print("\n验证数据格式:")
        sample = small_data[0]
        print(f"样本键: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
        
        if isinstance(sample, dict):
            for key, value in sample.items():
                if hasattr(value, '__len__'):
                    print(f"  {key}: 长度 {len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def create_small_datasets_all():
    """
    为所有数据文件创建小版本
    """
    files_to_process = [
        ("data/pretrain/pretrain_tensors_train.pkl", "data/pretrain/pretrain_tensors_train_small.pkl", 1000),
        ("data/pretrain/pretrain_tensors_valid.pkl", "data/pretrain/pretrain_tensors_valid_small.pkl", 200)
    ]
    
    for source_file, target_file, extract_count in files_to_process:
        if not os.path.exists(source_file):
            print(f"跳过 {source_file} (文件不存在)")
            continue
            
        print(f"\n{'='*60}")
        print(f"处理文件: {source_file}")
        print(f"目标文件: {target_file}")
        print(f"提取数量: {extract_count}")
        print(f"{'='*60}")
        
        try:
            # 检查文件大小
            file_size_gb = os.path.getsize(source_file) / (1024**3)
            print(f"原始文件大小: {file_size_gb:.2f} GB")
            
            # 读取数据
            print("正在加载数据...")
            with open(source_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"原始数据量: {len(data)} 个样本")
            
            # 提取小部分
            small_data = data[:extract_count]
            actual_count = len(small_data)
            
            # 保存
            print(f"保存 {actual_count} 个样本到 {target_file}...")
            with open(target_file, 'wb') as f:
                pickle.dump(small_data, f)
            
            new_size_mb = os.path.getsize(target_file) / (1024**2)
            print(f"✓ 完成! 新文件大小: {new_size_mb:.2f} MB")
            print(f"压缩比例: {actual_count / len(data) * 100:.2f}%")
            
        except Exception as e:
            print(f"✗ 处理 {source_file} 时出错: {e}")

if __name__ == "__main__":
    print("G2Retro-P 小数据集创建工具")
    print("="*50)
    
    # 创建所有小数据集
    create_small_datasets_all()
    
    print(f"\n{'='*50}")
    print("处理完成!")
    print("现在可以在训练脚本中使用这些小数据集进行快速测试:")
    print("- pretrain_tensors_train_small.pkl (1000 samples)")
    print("- pretrain_tensors_valid_small.pkl (200 samples)")
    print("="*50) 