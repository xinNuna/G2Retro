#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynthRetro-P预训练训练脚本

这是一个新增的文件，应该放在 model/ 目录下，命名为 train_pretrain.py
实现多任务预训练的完整训练流程，包含损失可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
import argparse
from datetime import datetime
import numpy as np
from collections import defaultdict
import gc
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns

# 导入我们的模块
from synthretro_pretrain import create_synthretro_pretrain_model
from pretrain_dataloader import create_pretrain_dataloaders

class PretrainTrainer:
    """SynthRetro-P预训练训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        # 创建模型
        self.model = create_synthretro_pretrain_model(config['model'])
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 损失历史记录（详细）
        self.loss_history = {
            'epoch': [],
            'train_total': [],
            'train_base': [],
            'train_recovery': [],
            'train_contrastive': [],
            'val_total': [],
            'val_base': [],
            'val_recovery': [],
            'val_contrastive': [],
            'learning_rate': []
        }
        
        # 创建输出目录
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建plots目录
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            try:
                # 数据移到GPU
                batch = self.move_batch_to_device(batch)
                
                # 前向传播
                self.optimizer.zero_grad()
                losses, predictions = self.model(batch)
                
                # 反向传播
                total_loss = losses['total_loss']
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        epoch_losses[loss_name].append(loss_value.item())
                
                # 打印进度
                if batch_idx % self.config['training']['log_interval'] == 0:
                    print(f"  批次 {batch_idx}/{len(train_loader)}: "
                          f"总损失={total_loss.item():.4f}")
                
                # 内存清理
                del losses, predictions, batch
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ 批次 {batch_idx} 训练失败: {e}")
                continue
        
        # 计算epoch平均损失
        avg_losses = {}
        for loss_name, loss_list in epoch_losses.items():
            if loss_list:
                avg_losses[loss_name] = np.mean(loss_list)
        
        return avg_losses
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                    
                try:
                    # 数据移到GPU
                    batch = self.move_batch_to_device(batch)
                    
                    # 前向传播
                    losses, predictions = self.model(batch)
                    
                    # 记录损失
                    for loss_name, loss_value in losses.items():
                        if isinstance(loss_value, torch.Tensor):
                            epoch_losses[loss_name].append(loss_value.item())
                    
                    
                except Exception as e:
                    print(f"⚠️ 验证批次 {batch_idx} 失败: {e}")
                    continue
        
        # 计算epoch平均损失
        avg_losses = {}
        for loss_name, loss_list in epoch_losses.items():
            if loss_list:
                avg_losses[loss_name] = np.mean(loss_list)
        
        return avg_losses
    
    def plot_loss_curves(self):
        """绘制并保存损失曲线图"""
        if len(self.loss_history['epoch']) < 2:
            return
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = self.loss_history['epoch']
        
        # 1. 总损失对比
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.loss_history['train_total'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.loss_history['val_total'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('总损失变化曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 各任务训练损失
        ax2 = axes[0, 1]
        if self.loss_history['train_base']:
            ax2.plot(epochs, self.loss_history['train_base'], 'g-', label='基础任务', linewidth=2)
        if self.loss_history['train_recovery']:
            ax2.plot(epochs, self.loss_history['train_recovery'], 'm-', label='分子恢复', linewidth=2)
        if self.loss_history['train_contrastive']:
            ax2.plot(epochs, self.loss_history['train_contrastive'], 'c-', label='对比学习', linewidth=2)
        ax2.set_title('训练损失分解', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 各任务验证损失
        ax3 = axes[1, 0]
        if self.loss_history['val_base']:
            ax3.plot(epochs, self.loss_history['val_base'], 'g--', label='基础任务', linewidth=2)
        if self.loss_history['val_recovery']:
            ax3.plot(epochs, self.loss_history['val_recovery'], 'm--', label='分子恢复', linewidth=2)
        if self.loss_history['val_contrastive']:
            ax3.plot(epochs, self.loss_history['val_contrastive'], 'c--', label='对比学习', linewidth=2)
        ax3.set_title('验证损失分解', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('损失值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率变化
        ax4 = axes[1, 1]
        ax4.plot(epochs, self.loss_history['learning_rate'], 'orange', linewidth=2)
        ax4.set_title('学习率变化', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('学习率')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.plots_dir, f'loss_curves_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存最新版本
        latest_path = os.path.join(self.plots_dir, 'loss_curves_latest.png')
        plt.figure(figsize=(15, 12))
        
        # 重新绘制并保存最新版本
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 复制上面的绘图逻辑
        epochs = self.loss_history['epoch']
        
        # 总损失
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.loss_history['train_total'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.loss_history['val_total'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('总损失变化曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练损失分解
        ax2 = axes[0, 1]
        if self.loss_history['train_base']:
            ax2.plot(epochs, self.loss_history['train_base'], 'g-', label='基础任务', linewidth=2)
        if self.loss_history['train_recovery']:
            ax2.plot(epochs, self.loss_history['train_recovery'], 'm-', label='分子恢复', linewidth=2)
        if self.loss_history['train_contrastive']:
            ax2.plot(epochs, self.loss_history['train_contrastive'], 'c-', label='对比学习', linewidth=2)
        ax2.set_title('训练损失分解', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 验证损失分解
        ax3 = axes[1, 0]
        if self.loss_history['val_base']:
            ax3.plot(epochs, self.loss_history['val_base'], 'g--', label='基础任务', linewidth=2)
        if self.loss_history['val_recovery']:
            ax3.plot(epochs, self.loss_history['val_recovery'], 'm--', label='分子恢复', linewidth=2)
        if self.loss_history['val_contrastive']:
            ax3.plot(epochs, self.loss_history['val_contrastive'], 'c--', label='对比学习', linewidth=2)
        ax3.set_title('验证损失分解', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('损失值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率
        ax4 = axes[1, 1]
        ax4.plot(epochs, self.loss_history['learning_rate'], 'orange', linewidth=2)
        ax4.set_title('学习率变化', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('学习率')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(latest_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 损失曲线已保存: {plot_path}")
        print(f"📊 最新损失曲线: {latest_path}")
    
    def save_loss_data(self):
        """保存损失数据为CSV格式"""
        import pandas as pd
        
        # 创建DataFrame
        loss_df = pd.DataFrame(self.loss_history)
        
        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'loss_history.csv')
        loss_df.to_csv(csv_path, index=False)
        
        # 保存为JSON（更详细的格式）
        json_path = os.path.join(self.output_dir, 'loss_history.json')
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        
        print(f"📈 损失数据已保存: {csv_path}")
    
    def update_loss_history(self, epoch, train_losses, val_losses, lr):
        """更新损失历史记录"""
        self.loss_history['epoch'].append(epoch)
        self.loss_history['learning_rate'].append(lr)
        
        # 训练损失
        self.loss_history['train_total'].append(train_losses.get('total_loss', 0))
        self.loss_history['train_base'].append(train_losses.get('base_loss', 0))
        self.loss_history['train_recovery'].append(train_losses.get('recovery_loss', 0))
        self.loss_history['train_contrastive'].append(train_losses.get('contrastive_loss', 0))
        
        # 验证损失
        self.loss_history['val_total'].append(val_losses.get('total_loss', 0))
        self.loss_history['val_base'].append(val_losses.get('base_loss', 0))
        self.loss_history['val_recovery'].append(val_losses.get('recovery_loss', 0))
        self.loss_history['val_contrastive'].append(val_losses.get('contrastive_loss', 0))
    
    def move_batch_to_device(self, batch):
        """将批次数据移动到指定设备"""
        if isinstance(batch, dict):
            return {k: self.move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self.move_batch_to_device(item) for item in batch]
        else:
            return batch
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_history': self.loss_history,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💎 保存最佳模型: {best_path}")
        
        # 定期保存
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            print(f"📂 加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.loss_history = checkpoint.get('loss_history', self.loss_history)
            
            print(f"✅ 从epoch {self.current_epoch}继续训练")
            return True
        return False
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("🚀 开始SynthRetro-P预训练")
        print(f"📊 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 训练设备: {self.device}")
        print(f"📈 训练轮数: {self.config['training']['epochs']}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 图片保存: {self.plots_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 60)
            
            # 训练
            print("🔄 训练阶段...")
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            print("🔍 验证阶段...")
            val_losses = self.validate_epoch(val_loader)
            
            # 更新学习率
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            # 记录损失
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            # 更新详细损失历史
            self.update_loss_history(epoch + 1, train_losses, val_losses, current_lr)
            
            # 打印结果
            epoch_time = time.time() - epoch_start_time
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"   训练损失: {train_losses.get('total_loss', 0):.4f}")
            if 'base_loss' in train_losses:
                print(f"     - 基础任务: {train_losses['base_loss']:.4f}")
            if 'recovery_loss' in train_losses:
                print(f"     - 分子恢复: {train_losses['recovery_loss']:.4f}")
            if 'contrastive_loss' in train_losses:
                print(f"     - 对比学习: {train_losses['contrastive_loss']:.4f}")
            print(f"   验证损失: {val_losses.get('total_loss', 0):.4f}")
            print(f"   学习率: {current_lr:.6f}")
            print(f"   耗时: {epoch_time:.2f}秒")
            
            # 保存检查点
            current_val_loss = val_losses.get('total_loss', float('inf'))
            is_best = current_val_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_val_loss
                print(f"🎉 新的最佳验证损失: {self.best_loss:.4f}")
            
            self.save_checkpoint(epoch + 1, is_best)
            
            # 绘制并保存损失曲线
            if (epoch + 1) % 5 == 0 or is_best:  # 每5个epoch或最佳模型时保存图片
                self.plot_loss_curves()
                self.save_loss_data()
            
            # 早停检查
            if self.should_early_stop():
                print("⏹️ 早停触发，停止训练")
                break
        
        # 训练完成后的最终保存
        self.plot_loss_curves()
        self.save_loss_data()
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！总耗时: {total_time:.2f}秒")
        print(f"💎 最佳验证损失: {self.best_loss:.4f}")
        print(f"📊 损失曲线图: {self.plots_dir}/loss_curves_latest.png")
    
    def should_early_stop(self):
        """检查是否应该早停"""
        if len(self.val_losses) < self.config['training']['patience']:
            return False
        
        recent_losses = [loss.get('total_loss', float('inf')) 
                        for loss in self.val_losses[-self.config['training']['patience']:]]
        
        # 如果最近几个epoch损失都没有改善，触发早停
        return all(loss >= self.best_loss for loss in recent_losses)


def create_default_config():
    """创建默认配置"""
    return {
        'model': {
            'hidden_size': 256,
            'embed_size': 32,
            'depth': 10,
            'num_atom_types': 12,
            'num_bond_types': 4,
            'projection_dim': 128,
            'temperature': 0.1
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'lr_step_size': 20,
            'lr_gamma': 0.8,
            'batch_size': 8,  # 较小的批次以适应内存限制
            'num_workers': 2,
            'log_interval': 100,
            'save_interval': 10,
            'patience': 10,
            'output_dir': '../results/synthretro_pretrain_models/'
        },
        'data': {
            'train_data_path': '../data/pretrain/pretrain_tensors_train.pkl',
            'test_data_path': '../data/pretrain/pretrain_tensors_test.pkl',
            'train_vocab_path': '../data/pretrain/vocab_train.txt',
            'test_vocab_path': '../data/pretrain/vocab_test.txt',
            'use_lazy_loading': True
        },
        'device': 'cuda'
    }


def main():
    parser = argparse.ArgumentParser(description='SynthRetro-P预训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    config['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print("🔧 训练配置:")
    print(json.dumps(config, indent=2))
    
    # 检查数据文件
    train_path = config['data']['train_data_path']
    test_path = config['data']['test_data_path']
    
    if not os.path.exists(train_path):
        print(f"❌ 训练数据不存在: {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"❌ 测试数据不存在: {test_path}")
        return
    
    # 创建数据加载器
    print("📂 创建数据加载器...")
    try:
        train_loader, val_loader = create_pretrain_dataloaders(
            train_data_path=config['data']['train_data_path'],
            test_data_path=config['data']['test_data_path'],
            train_vocab_path=config['data']['train_vocab_path'],
            test_vocab_path=config['data']['test_vocab_path'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_lazy_loading=config['data']['use_lazy_loading']
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        return
    
    # 创建训练器
    trainer = PretrainTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        trainer.save_checkpoint(trainer.current_epoch)
        trainer.plot_loss_curves()
        trainer.save_loss_data()
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎯 训练脚本执行完成")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2Retro-P预训练训练脚本

这是一个新增的文件，应该放在 model/ 目录下，命名为 train_pretrain.py
实现多任务预训练的完整训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
import argparse
from datetime import datetime
import numpy as np
from collections import defaultdict
import gc

# 导入我们的模块
from synthretro_pretrain import create_synthretro_pretrain_model
from pretrain_dataloader import create_pretrain_dataloaders

class PretrainTrainer:
    """G2Retro-P预训练训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        # 创建模型
        self.model = create_g2retro_pretrain_model(config['model'])
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 创建输出目录
        self.output_dir = config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            try:
                # 数据移到GPU
                batch = self.move_batch_to_device(batch)
                
                # 前向传播
                self.optimizer.zero_grad()
                losses, predictions = self.model(batch)
                
                # 反向传播
                total_loss = losses['total_loss']
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        epoch_losses[loss_name].append(loss_value.item())
                
                # 打印进度
                if batch_idx % self.config['training']['log_interval'] == 0:
                    print(f"  批次 {batch_idx}/{len(train_loader)}: "
                          f"总损失={total_loss.item():.4f}")
                
                # 内存清理
                del losses, predictions, batch
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ 批次 {batch_idx} 训练失败: {e}")
                continue
        
        # 计算epoch平均损失
        avg_losses = {}
        for loss_name, loss_list in epoch_losses.items():
            if loss_list:
                avg_losses[loss_name] = np.mean(loss_list)
        
        return avg_losses
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                    
                try:
                    # 数据移到GPU
                    batch = self.move_batch_to_device(batch)
                    
                    # 前向传播
                    losses, predictions = self.model(batch)
                    
                    # 记录损失
                    for loss_name, loss_value in losses.items():
                        if isinstance(loss_value, torch.Tensor):
                            epoch_losses[loss_name].append(loss_value.item())
                    
                    # 内存清理
                    del losses, predictions, batch
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"⚠️ 验证批次 {batch_idx} 失败: {e}")
                    continue
        
        # 计算epoch平均损失
        avg_losses = {}
        for loss_name, loss_list in epoch_losses.items():
            if loss_list:
                avg_losses[loss_name] = np.mean(loss_list)
        
        return avg_losses
    
    def move_batch_to_device(self, batch):
        """将批次数据移动到指定设备"""
        if isinstance(batch, dict):
            return {k: self.move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self.move_batch_to_device(item) for item in batch]
        else:
            return batch
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💎 保存最佳模型: {best_path}")
        
        # 定期保存
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            print(f"📂 加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            print(f"✅ 从epoch {self.current_epoch}继续训练")
            return True
        return False
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("🚀 开始G2Retro-P预训练")
        print(f"📊 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 训练设备: {self.device}")
        print(f"📈 训练轮数: {self.config['training']['epochs']}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 60)
            
            # 训练
            print("🔄 训练阶段...")
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            print("🔍 验证阶段...")
            val_losses = self.validate_epoch(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录损失
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            # 打印结果
            epoch_time = time.time() - epoch_start_time
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"   训练损失: {train_losses.get('total_loss', 0):.4f}")
            print(f"   验证损失: {val_losses.get('total_loss', 0):.4f}")
            print(f"   学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"   耗时: {epoch_time:.2f}秒")
            
            # 保存检查点
            current_val_loss = val_losses.get('total_loss', float('inf'))
            is_best = current_val_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_val_loss
                print(f"🎉 新的最佳验证损失: {self.best_loss:.4f}")
            
            self.save_checkpoint(epoch + 1, is_best)
            
            # 早停检查
            if self.should_early_stop():
                print("⏹️ 早停触发，停止训练")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！总耗时: {total_time:.2f}秒")
        print(f"💎 最佳验证损失: {self.best_loss:.4f}")
    
    def should_early_stop(self):
        """检查是否应该早停"""
        if len(self.val_losses) < self.config['training']['patience']:
            return False
        
        recent_losses = [loss.get('total_loss', float('inf')) 
                        for loss in self.val_losses[-self.config['training']['patience']:]]
        
        # 如果最近几个epoch损失都没有改善，触发早停
        return all(loss >= self.best_loss for loss in recent_losses)


def create_default_config():
    """创建默认配置"""
    return {
        'model': {
            'hidden_size': 256,
            'embed_size': 32,
            'depth': 10,
            'num_atom_types': 12,
            'num_bond_types': 4,
            'projection_dim': 128,
            'temperature': 0.1
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'lr_step_size': 20,
            'lr_gamma': 0.8,
            'batch_size': 8,  # 较小的批次以适应内存限制
            'num_workers': 2,
            'log_interval': 100,
            'save_interval': 10,
            'patience': 10,
            'output_dir': '../results/pretrain_models/'
        },
        'data': {
            'train_data_path': '../data/pretrain/pretrain_tensors_train.pkl',
            'test_data_path': '../data/pretrain/pretrain_tensors_test.pkl',
            'train_vocab_path': '../data/pretrain/vocab_train.txt',
            'test_vocab_path': '../data/pretrain/vocab_test.txt',
            'use_lazy_loading': True
        },
        'device': 'cuda'
    }


def main():
    parser = argparse.ArgumentParser(description='G2Retro-P预训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    config['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print("🔧 训练配置:")
    print(json.dumps(config, indent=2))
    
    # 检查数据文件
    train_path = config['data']['train_data_path']
    test_path = config['data']['test_data_path']
    
    if not os.path.exists(train_path):
        print(f"❌ 训练数据不存在: {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"❌ 测试数据不存在: {test_path}")
        return
    
    # 创建数据加载器
    print("📂 创建数据加载器...")
    try:
        train_loader, val_loader = create_pretrain_dataloaders(
            train_data_path=config['data']['train_data_path'],
            test_data_path=config['data']['test_data_path'],
            train_vocab_path=config['data']['train_vocab_path'],
            test_vocab_path=config['data']['test_vocab_path'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            use_lazy_loading=config['data']['use_lazy_loading']
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        return
    
    # 创建训练器
    trainer = PretrainTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        trainer.save_checkpoint(trainer.current_epoch)
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎯 训练脚本执行完成")


if __name__ == "__main__":
    main()