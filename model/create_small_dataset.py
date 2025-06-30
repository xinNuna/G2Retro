#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import sys

def create_small_datasets():
    """
    从大数据集中提取小部分用于快速测试
    """
    base_path = "../data/pretrain/"
    
    files_config = [
        {
            'source': os.path.join(base_path, "pretrain_tensors_train.pkl"),
            'target': os.path.join(base_path, "pretrain_tensors_train_small.pkl"),
            'count': 1000,
            'desc': "训练集"
        },
        {
            'source': os.path.join(base_path, "pretrain_tensors_valid.pkl"), 
            'target': os.path.join(base_path, "pretrain_tensors_valid_small.pkl"),
            'count': 200,
            'desc': "验证集"
        }
    ]
    
    for config in files_config:
        source_file = config['source']
        target_file = config['target']
        extract_count = config['count']
        desc = config['desc']
        
        print(f"\n{'='*50}")
        print(f"处理{desc}: {os.path.basename(source_file)}")
        print(f"目标: {os.path.basename(target_file)}")
        print(f"提取数量: {extract_count}")
        print(f"{'='*50}")
        
        if not os.path.exists(source_file):
            print(f"⚠️  源文件不存在: {source_file}")
            continue
            
        try:
            # 显示文件大小
            file_size = os.path.getsize(source_file) / (1024**3)
            print(f"源文件大小: {file_size:.2f} GB")
            
            print("正在加载数据...")
            with open(source_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"原始数据量: {len(data):,} 个样本")
            
            # 提取数据
            small_data = data[:extract_count]
            actual_count = len(small_data)
            
            print(f"提取样本: {actual_count:,} 个")
            
            # 保存
            print("正在保存小数据集...")
            with open(target_file, 'wb') as f:
                pickle.dump(small_data, f)
            
            # 检查新文件
            new_size = os.path.getsize(target_file) / (1024**2)
            compression_ratio = actual_count / len(data) * 100
            
            print(f"✅ 完成!")
            print(f"   新文件大小: {new_size:.1f} MB")
            print(f"   压缩比例: {compression_ratio:.1f}%")
            print(f"   加载速度提升: ~{100/compression_ratio:.1f}x")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("G2Retro-P 小数据集创建工具")
    print("用于从大训练集中提取小部分数据进行快速测试")
    
    create_small_datasets()
    
    print(f"\n{'='*50}")
    print("✅ 处理完成!")
    print("\n现在可以在数据集加载时使用小文件:")
    print("- pretrain_tensors_train_small.pkl (1000 samples)")  
    print("- pretrain_tensors_valid_small.pkl (200 samples)")
    print("\n这将大幅提升加载速度，适合调试和快速测试!")
    print(f"{'='*50}") 