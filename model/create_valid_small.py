#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

def create_valid_small():
    """
    专门创建验证集小文件
    """
    source_file = "../data/pretrain/pretrain_tensors_valid.pkl"
    target_file = "../data/pretrain/pretrain_tensors_valid_small.pkl"
    extract_count = 200
    
    print("创建验证集小数据文件...")
    print(f"源文件: {source_file}")
    print(f"目标文件: {target_file}")
    print(f"提取数量: {extract_count}")
    
    if not os.path.exists(source_file):
        print(f"❌ 源文件不存在: {source_file}")
        return
        
    try:
        # 显示文件大小
        file_size = os.path.getsize(source_file) / (1024**3)
        print(f"源文件大小: {file_size:.2f} GB")
        
        print("正在加载验证数据...")
        with open(source_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"原始验证数据量: {len(data):,} 个样本")
        
        # 提取数据
        small_data = data[:extract_count]
        actual_count = len(small_data)
        
        print(f"提取样本: {actual_count:,} 个")
        
        # 保存
        print("正在保存小验证集...")
        with open(target_file, 'wb') as f:
            pickle.dump(small_data, f)
        
        # 检查新文件
        new_size = os.path.getsize(target_file) / (1024**2)
        compression_ratio = actual_count / len(data) * 100
        
        print(f"✅ 验证集小文件创建完成!")
        print(f"   新文件大小: {new_size:.1f} MB")
        print(f"   压缩比例: {compression_ratio:.1f}%")
        print(f"   加载速度提升: ~{100/compression_ratio:.1f}x")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_valid_small() 