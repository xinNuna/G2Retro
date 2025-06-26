#!/usr/bin/env python3
"""
G2Retro-P 多任务预训练启动脚本
基于设计方案 + RetroExplainer动态权重调整

使用方法:
1. 快速启动（演示模式）:
   cd model && python run_g2retro_p_pretrain.py --mode demo

2. 完整预训练:
   cd model && python run_g2retro_p_pretrain.py --mode full

3. 自定义配置:
   cd model && python run_g2retro_p_pretrain.py --epochs 100 --batch_size 8 --learning_rate 1e-4
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
from datetime import datetime

# 导入训练模块（在model目录中）
try:
    from train_g2retro_p_design_aligned import (
        G2RetroPDesignAlignedModel, 
        G2RetroPDesignAlignedTrainer,
        G2RetroPDesignAlignedDataset,
        g2retro_design_aligned_collate_fn
    )
    from torch.utils.data import DataLoader
    print("✓ 成功导入G2Retro-P训练模块")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在model目录下运行: cd model && python run_g2retro_p_pretrain.py")
    sys.exit(1)

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  使用CPU（建议使用GPU加速）")
    return device

def check_data_files():
    """检查数据文件"""
    files = {
        'train_data': '../data/pretrain/pretrain_tensors_train.pkl',
        'valid_data': '../data/pretrain/pretrain_tensors_valid.pkl', 
        'vocab': '../data/pretrain/vocab_train.txt'
    }
    
    print("\n检查数据文件:")
    all_exist = True
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**3)  # GB
            print(f"✓ {name}: {path} ({size:.1f} GB)")
        else:
            print(f"❌ {name}: {path} - 文件不存在")
            all_exist = False
    
    return all_exist, files

class PretrainConfig:
    """预训练配置类"""
    def __init__(self, mode='demo'):
        # 模型参数（与G2Retro保持一致）
        self.hidden_size = 300
        self.latent_size = 300
        self.depthT = 3
        self.depthG = 3
        self.use_feature = True
        self.use_tree = True
        self.network_type = "gru"
        self.use_node_embed = False
        self.use_class = False
        self.use_atomic = False
        self.sum_pool = False
        self.use_brics = False
        self.use_latent_attachatom = False
        self.use_mess = True
        
        # 根据模式设置训练参数
        if mode == 'demo':
            self._setup_demo_config()
        elif mode == 'full':
            self._setup_full_config()
        else:
            self._setup_custom_config()
            
        # 动态权重适应参数（基于RetroExplainer DAMT）
        self.reset_weights_per_epoch = False
        self.weight_update_frequency = 10
        self.weight_temperature = 2.0
        self.loss_queue_length = 50
        self.min_task_weight = 0.01
        self.max_task_weight = 3.0
        
        # 输出路径（相对于model目录）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"checkpoints_g2retro_p_{mode}_{timestamp}/"
        
    def _setup_demo_config(self):
        """演示模式配置（快速验证）"""
        print("📋 配置模式: 演示模式（快速验证）")
        self.batch_size = 2
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 3
        self.step_size = 2
        self.gamma = 0.7
        self.patience = 2
        self.max_train_samples = 50
        self.max_val_samples = 10
        
    def _setup_full_config(self):
        """完整预训练配置"""
        print("📋 配置模式: 完整预训练")
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 100
        self.step_size = 20
        self.gamma = 0.8
        self.patience = 10
        self.max_train_samples = None  # 使用全部数据
        self.max_val_samples = None
        
    def _setup_custom_config(self):
        """自定义配置（默认中等规模）"""
        print("📋 配置模式: 自定义配置")
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 50
        self.step_size = 10
        self.gamma = 0.9
        self.patience = 5
        self.max_train_samples = 10000
        self.max_val_samples = 1000

def create_datasets_and_loaders(config, data_files):
    """创建数据集和数据加载器"""
    print(f"\n创建数据集...")
    
    # 创建训练数据集
    train_dataset = G2RetroPDesignAlignedDataset(
        data_files['train_data'], 
        data_files['vocab'], 
        max_samples=config.max_train_samples
    )
    
    # 创建验证数据集  
    val_dataset = G2RetroPDesignAlignedDataset(
        data_files['valid_data'],
        data_files['vocab'],
        max_samples=config.max_val_samples
    )
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 验证集大小: {len(val_dataset)}")
    print(f"✓ 词汇表大小: {train_dataset.vocab.size()}")
    print(f"✓ 原子词汇表大小: {train_dataset.avocab.size()}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, train_dataset.vocab, train_dataset.avocab),
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: g2retro_design_aligned_collate_fn(batch, val_dataset.vocab, val_dataset.avocab),
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ 训练批次数: {len(train_loader)}")
    print(f"✓ 验证批次数: {len(val_loader)}")
    
    return train_dataset, val_dataset, train_loader, val_loader

def run_pretrain(config):
    """运行预训练"""
    print("\n" + "="*80)
    print("🚀 G2Retro-P 多任务预训练启动")
    print("📝 设计方案完全对齐 + RetroExplainer动态权重调整")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设置设备
    device = setup_device()
    
    # 检查数据文件
    data_available, data_files = check_data_files()
    if not data_available:
        print("❌ 数据文件缺失，无法启动训练")
        print("💡 提示：请确保已运行预处理脚本生成数据文件")
        return False
    
    # 创建数据集和加载器
    train_dataset, val_dataset, train_loader, val_loader = create_datasets_and_loaders(config, data_files)
    
    # 创建模型
    print(f"\n创建G2Retro-P模型...")
    model = G2RetroPDesignAlignedModel(train_dataset.vocab, train_dataset.avocab, config)
    model = model.to(device)
    
    # 打印模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型统计:")
    print(f"参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: ~{total_params * 4 / 1e6:.1f} MB")
    
    # 显示核心架构
    print(f"\n🏗️  核心架构验证:")
    print(f"✓ 共享编码器: GMPN（隐藏维度:{config.hidden_size}）")
    print(f"✓ 基础任务头: G2Retro反应中心识别")
    print(f"✓ 分子恢复头: MolCLR图结构增强")
    print(f"✓ 对比学习头: 产物-合成子对比")
    print(f"✓ 动态权重: RetroExplainer DAMT算法")
    print(f"✓ 权重更新频率: 每{config.weight_update_frequency}步")
    print(f"✓ 温度系数: {config.weight_temperature}")
    
    # 创建训练器
    trainer = G2RetroPDesignAlignedTrainer(model, train_loader, val_loader, config)
    
    # 开始训练
    print(f"\n🎯 开始预训练...")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"保存路径: {config.save_dir}")
    
    try:
        trainer.train()
        print(f"\n🎉 预训练完成！")
        print(f"最佳模型保存在: {config.save_dir}")
        return True
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='G2Retro-P多任务预训练')
    
    # 预设模式
    parser.add_argument('--mode', choices=['demo', 'full', 'custom'], default='demo',
                        help='预训练模式: demo(演示), full(完整), custom(自定义)')
    
    # 自定义参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--max_train_samples', type=int, help='最大训练样本数')
    parser.add_argument('--max_val_samples', type=int, help='最大验证样本数')
    
    # 动态权重参数
    parser.add_argument('--weight_temperature', type=float, default=2.0, help='权重温度系数')
    parser.add_argument('--weight_update_frequency', type=int, default=10, help='权重更新频率')
    parser.add_argument('--reset_weights_per_epoch', action='store_true', help='每epoch重置权重')
    
    args = parser.parse_args()
    
    # 创建配置
    config = PretrainConfig(mode=args.mode)
    
    # 应用自定义参数
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        config.max_val_samples = args.max_val_samples
    
    # 应用动态权重参数
    config.weight_temperature = args.weight_temperature
    config.weight_update_frequency = args.weight_update_frequency
    config.reset_weights_per_epoch = args.reset_weights_per_epoch
    
    # 显示配置
    print("=" * 50)
    print("🔧 预训练配置:")
    print("=" * 50)
    print(f"模式: {args.mode}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    if config.max_train_samples:
        print(f"最大训练样本: {config.max_train_samples:,}")
    if config.max_val_samples:
        print(f"最大验证样本: {config.max_val_samples:,}")
    print(f"权重温度: {config.weight_temperature}")
    print(f"权重更新频率: {config.weight_update_frequency}")
    print(f"每epoch重置权重: {config.reset_weights_per_epoch}")
    print("=" * 50)
    
    # 运行预训练
    success = run_pretrain(config)
    
    if success:
        print("\n🎯 预训练成功完成！")
        print("💡 下一步可以使用预训练模型进行fine-tuning")
    else:
        print("\n❌ 预训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 