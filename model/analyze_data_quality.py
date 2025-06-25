#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据质量分析脚本 - 内存优化版

这个文件应该替换现有的 analyze_data_quality.py
专门针对大文件和磁盘空间不足的情况进行优化
"""

import pickle
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import os
import gc

def analyze_pretrain_data_efficient(data_path, max_samples=5000):
    """
    高效分析预训练数据的质量 - 使用采样避免内存问题
    """
    print(f"🔍 高效分析预训练数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return None
    
    file_size = os.path.getsize(data_path) / 1024**3
    print(f"📊 文件大小: {file_size:.1f} GB")
    
    try:
        print("⏳ 正在加载数据（可能需要几分钟）...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        total_samples = len(data)
        print(f"✅ 数据加载成功: {total_samples:,} 条")
        
        # 使用采样分析以节省内存和时间
        if total_samples > max_samples:
            print(f"🔬 使用采样分析: {max_samples}/{total_samples}")
            # 均匀采样
            step = total_samples // max_samples
            sample_indices = list(range(0, total_samples, step))[:max_samples]
            sample_data = [data[i] for i in sample_indices]
        else:
            sample_data = data
        
        print(f"📈 分析样本数: {len(sample_data)}")
        
        # 分析统计信息
        stats = {
            'total_samples': total_samples,
            'analyzed_samples': len(sample_data),
            'vocab_sizes': [],
            'augmentation_counts': [],
            'product_lengths': [],
            'synthon_lengths': [],
            'reactant_lengths': [],
            'non_empty_vocabs': 0,
            'all_vocabs': set()
        }
        
        print("🔄 正在分析数据特征...")
        
        for i, entry in enumerate(sample_data):
            if i % 1000 == 0 and i > 0:
                print(f"   进度: {i}/{len(sample_data)}")
            
            try:
                # 分析vocab
                vocab = entry.get('vocab', set())
                if not isinstance(vocab, set):
                    vocab = set(vocab) if vocab else set()
                
                vocab_size = len(vocab)
                stats['vocab_sizes'].append(vocab_size)
                
                if vocab_size > 0:
                    stats['non_empty_vocabs'] += 1
                    stats['all_vocabs'].update(vocab)
                
                # 分析增强数据
                aug_data = entry.get('augmented_data', [])
                stats['augmentation_counts'].append(len(aug_data))
                
                # 分析SMILES长度
                pretrain_info = entry.get('pretrain_info', {})
                if pretrain_info:
                    product_smiles = pretrain_info.get('product_smiles', '')
                    synthon_smiles = pretrain_info.get('synthon_smiles', '')
                    reactant_smiles = pretrain_info.get('reactant_smiles', '')
                    
                    stats['product_lengths'].append(len(product_smiles))
                    stats['synthon_lengths'].append(len(synthon_smiles))
                    stats['reactant_lengths'].append(len(reactant_smiles))
                
            except Exception as e:
                print(f"   ⚠️ 分析第{i}条数据时出错: {e}")
                continue
        
        # 立即释放大数据对象
        del data
        del sample_data
        gc.collect()
        
        print(f"✅ 分析完成")
        return stats
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def analyze_original_data_efficient(data_path, max_lines=20000):
    """
    高效分析原始数据的错误模式
    """
    print(f"\n🔍 高效分析原始数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return None
    
    file_size = os.path.getsize(data_path) / 1024**2
    print(f"📊 文件大小: {file_size:.1f} MB")
    
    try:
        # 先读取小样本检查格式
        print("🔍 检查数据格式...")
        sample_df = pd.read_csv(data_path, nrows=100)
        columns = sample_df.columns.tolist()
        print(f"📋 列名: {columns}")
        
        # 确定反应列名
        rxn_col = None
        if 'rxn_smiles' in columns:
            rxn_col = 'rxn_smiles'
        elif len(columns) >= 2:
            rxn_col = columns[1]  # 通常第二列是反应
        else:
            print("❌ 无法确定反应SMILES列")
            return None
        
        print(f"✅ 使用反应列: {rxn_col}")
        
        # 读取指定数量的数据进行分析
        print(f"📖 读取前 {max_lines} 行数据进行分析...")
        df = pd.read_csv(data_path, nrows=max_lines)
        
        # 获取总行数（不加载全部数据）
        total_lines = sum(1 for line in open(data_path, 'r')) - 1  # 减去header
        print(f"📊 总数据量: {total_lines:,} 行")
        print(f"🔬 分析样本: {len(df):,} 行 ({len(df)/total_lines*100:.1f}%)")
        
        error_analysis = {
            'total_reactions': total_lines,
            'analyzed_reactions': len(df),
            'empty_reactions': 0,
            'invalid_format': 0,
            'no_products': 0,
            'no_reactants': 0,
            'complex_reactions': 0,
            'simple_reactions': 0,
            'reaction_lengths': [],
            'reactant_counts': [],
            'product_counts': []
        }
        
        print("🔄 正在分析反应质量...")
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0 and idx > 0:
                print(f"   进度: {idx}/{len(df)}")
            
            rxn_smiles = str(row.get(rxn_col, ''))
            
            if pd.isna(rxn_smiles) or rxn_smiles.strip() == '' or rxn_smiles == 'nan':
                error_analysis['empty_reactions'] += 1
                continue
            
            error_analysis['reaction_lengths'].append(len(rxn_smiles))
            
            if '>>' not in rxn_smiles:
                error_analysis['invalid_format'] += 1
                continue
            
            try:
                parts = rxn_smiles.split('>')
                if len(parts) != 3:
                    error_analysis['invalid_format'] += 1
                    continue
                
                reactants = parts[0].strip()
                products = parts[2].strip()
                
                if not reactants:
                    error_analysis['no_reactants'] += 1
                    continue
                
                if not products:
                    error_analysis['no_products'] += 1
                    continue
                
                # 统计反应物和产物数量
                reactant_count = len(reactants.split('.')) if reactants else 0
                product_count = len(products.split('.')) if products else 0
                
                error_analysis['reactant_counts'].append(reactant_count)
                error_analysis['product_counts'].append(product_count)
                
                if reactant_count > 2 or product_count > 1:
                    error_analysis['complex_reactions'] += 1
                else:
                    error_analysis['simple_reactions'] += 1
                    
            except Exception:
                error_analysis['invalid_format'] += 1
        
        print(f"✅ 原始数据分析完成")
        return error_analysis
        
    except Exception as e:
        print(f"❌ 分析原始数据失败: {e}")
        return None

def calculate_statistics(data_list):
    """计算统计信息"""
    if not data_list:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    return {
        "mean": np.mean(data_list),
        "std": np.std(data_list),
        "min": np.min(data_list),
        "max": np.max(data_list),
        "median": np.median(data_list)
    }

def generate_optimized_report(stats, error_analysis, output_dir):
    """
    生成优化的数据质量报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "data_quality_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("           预训练数据质量分析报告 (优化版)\n")
        f.write("=" * 80 + "\n\n")
        
        # 处理后数据分析
        if stats:
            f.write("📊 处理后数据统计:\n")
            f.write("-" * 60 + "\n")
            f.write(f"总样本数: {stats['total_samples']:,}\n")
            f.write(f"分析样本数: {stats['analyzed_samples']:,}\n")
            
            # 词汇表分析
            if stats['vocab_sizes']:
                vocab_stats = calculate_statistics(stats['vocab_sizes'])
                f.write(f"\n📝 词汇表分析:\n")
                f.write(f"  平均大小: {vocab_stats['mean']:.1f} ± {vocab_stats['std']:.1f}\n")
                f.write(f"  大小范围: {vocab_stats['min']:.0f} - {vocab_stats['max']:.0f}\n")
                f.write(f"  中位数: {vocab_stats['median']:.1f}\n")
                f.write(f"  非空词汇表: {stats['non_empty_vocabs']}/{stats['analyzed_samples']} ")
                f.write(f"({stats['non_empty_vocabs']/stats['analyzed_samples']*100:.1f}%)\n")
                f.write(f"  累计不重复词汇: {len(stats['all_vocabs'])}个\n")
            
            # 增强数据分析
            if stats['augmentation_counts']:
                aug_stats = calculate_statistics(stats['augmentation_counts'])
                f.write(f"\n🎯 数据增强分析:\n")
                f.write(f"  平均增强数据量: {aug_stats['mean']:.1f}\n")
                f.write(f"  增强数据范围: {aug_stats['min']:.0f} - {aug_stats['max']:.0f}\n")
            
            # SMILES长度分析
            if stats['product_lengths']:
                prod_stats = calculate_statistics(stats['product_lengths'])
                f.write(f"\n🧬 分子SMILES长度分析:\n")
                f.write(f"  产物长度: {prod_stats['mean']:.1f} ± {prod_stats['std']:.1f}\n")
                
            if stats['synthon_lengths']:
                syn_stats = calculate_statistics(stats['synthon_lengths'])
                f.write(f"  合成子长度: {syn_stats['mean']:.1f} ± {syn_stats['std']:.1f}\n")
                
            if stats['reactant_lengths']:
                react_stats = calculate_statistics(stats['reactant_lengths'])
                f.write(f"  反应物长度: {react_stats['mean']:.1f} ± {react_stats['std']:.1f}\n")
            
            f.write("\n")
        
        # 原始数据分析
        if error_analysis:
            f.write("📋 原始数据质量分析:\n")
            f.write("-" * 60 + "\n")
            total = error_analysis['total_reactions']
            analyzed = error_analysis['analyzed_reactions']
            
            f.write(f"总反应数: {total:,}\n")
            f.write(f"分析样本: {analyzed:,} ({analyzed/total*100:.1f}%)\n\n")
            
            f.write(f"📊 错误统计 (基于分析样本):\n")
            f.write(f"  空反应: {error_analysis['empty_reactions']:,} ({error_analysis['empty_reactions']/analyzed*100:.1f}%)\n")
            f.write(f"  格式错误: {error_analysis['invalid_format']:,} ({error_analysis['invalid_format']/analyzed*100:.1f}%)\n")
            f.write(f"  无产物: {error_analysis['no_products']:,} ({error_analysis['no_products']/analyzed*100:.1f}%)\n")
            f.write(f"  无反应物: {error_analysis['no_reactants']:,} ({error_analysis['no_reactants']/analyzed*100:.1f}%)\n")
            f.write(f"  复杂反应: {error_analysis['complex_reactions']:,} ({error_analysis['complex_reactions']/analyzed*100:.1f}%)\n")
            f.write(f"  简单反应: {error_analysis['simple_reactions']:,} ({error_analysis['simple_reactions']/analyzed*100:.1f}%)\n\n")
            
            if error_analysis['reactant_counts']:
                react_stats = calculate_statistics(error_analysis['reactant_counts'])
                f.write(f"⚗️ 反应复杂度:\n")
                f.write(f"  平均反应物数量: {react_stats['mean']:.1f}\n")
                
            if error_analysis['product_counts']:
                prod_stats = calculate_statistics(error_analysis['product_counts'])
                f.write(f"  平均产物数量: {prod_stats['mean']:.1f}\n")
                
            if error_analysis['reaction_lengths']:
                len_stats = calculate_statistics(error_analysis['reaction_lengths'])
                f.write(f"  反应SMILES平均长度: {len_stats['mean']:.1f} ± {len_stats['std']:.1f}\n")
            
            f.write("\n")
        
        # 质量评估
        f.write("🎯 数据质量评估:\n")
        f.write("-" * 60 + "\n")
        
        if stats and error_analysis:
            success_rate = stats['total_samples'] / error_analysis['total_reactions']
            f.write(f"📈 数据处理成功率: {success_rate*100:.1f}%\n")
            
            if success_rate > 0.7:
                f.write("✅ 优秀: 处理成功率很高，数据质量优秀\n")
            elif success_rate > 0.5:
                f.write("✅ 良好: 处理成功率可接受，数据质量良好\n")
            elif success_rate > 0.3:
                f.write("⚠️ 一般: 处理成功率中等，可以使用但需关注\n")
            else:
                f.write("❌ 较低: 处理成功率偏低，建议优化数据处理流程\n")
        
        # 词汇表质量评估
        if stats and stats['all_vocabs']:
            vocab_quality = len(stats['all_vocabs'])
            if vocab_quality > 1000:
                f.write("✅ 词汇表丰富: 包含足够多样的化学基团\n")
            elif vocab_quality > 500:
                f.write("✅ 词汇表适中: 化学基团数量合理\n")
            elif vocab_quality > 100:
                f.write("⚠️ 词汇表较小: 化学基团多样性有限\n")
            else:
                f.write("❌ 词汇表过小: 可能影响预训练效果\n")
        
        # 非空词汇表比例评估
        if stats and stats['analyzed_samples'] > 0:
            non_empty_ratio = stats['non_empty_vocabs'] / stats['analyzed_samples']
            if non_empty_ratio > 0.8:
                f.write("✅ 反应中心识别率高: 大部分反应成功识别反应中心\n")
            elif non_empty_ratio > 0.5:
                f.write("✅ 反应中心识别率中等: 多数反应识别成功\n")
            elif non_empty_ratio > 0.2:
                f.write("⚠️ 反应中心识别率较低: 需要关注处理质量\n")
            else:
                f.write("❌ 反应中心识别率过低: 处理流程可能有问题\n")
        
        f.write(f"\n🚀 使用建议:\n")
        f.write(f"-" * 60 + "\n")
        
        if stats and stats['total_samples'] > 100000:
            f.write("✅ 数据规模充足，可以开始预训练\n")
        elif stats and stats['total_samples'] > 50000:
            f.write("✅ 数据规模适中，适合预训练\n")
        else:
            f.write("⚠️ 数据规模较小，建议增加数据量\n")
        
        f.write("📋 下一步操作:\n")
        f.write("  1. 使用预训练数据加载器验证数据完整性\n")
        f.write("  2. 设计并实现G2Retro-P预训练模型\n")
        f.write("  3. 开始多任务预训练实验\n")
        f.write("  4. 在USPTO-50K上进行微调验证\n")
    
    print(f"✅ 详细报告已保存到: {report_path}")
    
    # 生成简化摘要
    summary_path = os.path.join(output_dir, "quality_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("数据质量摘要\n")
        f.write("=" * 30 + "\n\n")
        
        if stats:
            f.write(f"处理后数据: {stats['total_samples']:,} 条\n")
            if stats['vocab_sizes']:
                avg_vocab = np.mean(stats['vocab_sizes'])
                f.write(f"平均词汇表大小: {avg_vocab:.1f}\n")
            f.write(f"累计词汇数: {len(stats['all_vocabs'])}\n")
            f.write(f"非空词汇表比例: {stats['non_empty_vocabs']/stats['analyzed_samples']*100:.1f}%\n")
        
        if error_analysis:
            f.write(f"原始数据: {error_analysis['total_reactions']:,} 条\n")
            if stats and error_analysis:
                success_rate = stats['total_samples'] / error_analysis['total_reactions'] * 100
                f.write(f"处理成功率: {success_rate:.1f}%\n")
        
        if stats and len(stats['all_vocabs']) > 0 and stats['non_empty_vocabs'] > 0:
            f.write("\n状态: ✅ 数据质量良好，可用于预训练\n")
        else:
            f.write("\n状态: ⚠️ 数据质量需要关注\n")
    
    print(f"✅ 摘要已保存到: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="优化版预训练数据质量分析")
    parser.add_argument('--processed_data', type=str, help="处理后的预训练数据路径(.pkl)")
    parser.add_argument('--original_data', type=str, help="原始数据路径(.csv)")
    parser.add_argument('--output_dir', type=str, default="./quality_analysis/", help="输出目录")
    parser.add_argument('--max_samples', type=int, default=5000, help="最大分析样本数")
    parser.add_argument('--max_lines', type=int, default=20000, help="原始数据最大分析行数")
    
    args = parser.parse_args()
    
    print("🔍 优化版预训练数据质量分析")
    print("=" * 80)
    print("⚡ 针对大文件和磁盘空间限制进行了优化")
    print("=" * 80)
    
    stats = None
    error_analysis = None
    
    # 分析处理后的数据
    if args.processed_data and os.path.exists(args.processed_data):
        try:
            stats = analyze_pretrain_data_efficient(args.processed_data, args.max_samples)
            if stats:
                print("✅ 处理后数据分析完成")
            else:
                print("❌ 处理后数据分析失败")
        except Exception as e:
            print(f"❌ 分析处理后数据时出错: {e}")
    elif args.processed_data:
        print(f"❌ 处理后数据文件不存在: {args.processed_data}")
    
    # 分析原始数据
    if args.original_data and os.path.exists(args.original_data):
        try:
            error_analysis = analyze_original_data_efficient(args.original_data, args.max_lines)
            if error_analysis:
                print("✅ 原始数据分析完成")
            else:
                print("❌ 原始数据分析失败")
        except Exception as e:
            print(f"❌ 分析原始数据时出错: {e}")
    elif args.original_data:
        print(f"❌ 原始数据文件不存在: {args.original_data}")
    
    # 生成报告
    if stats or error_analysis:
        try:
            generate_optimized_report(stats, error_analysis, args.output_dir)
            print(f"\n🎉 分析完成！")
            print(f"📁 结果保存在: {args.output_dir}")
            print(f"📄 查看详细报告: {os.path.join(args.output_dir, 'data_quality_report.txt')}")
            print(f"📋 查看摘要: {os.path.join(args.output_dir, 'quality_summary.txt')}")
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
    else:
        print("❌ 没有有效数据进行分析")
        print("💡 请检查数据文件路径是否正确")

if __name__ == "__main__":
    main()

def analyze_pretrain_data(data_path):
    """
    分析预训练数据的质量
    """
    print(f"分析预训练数据: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"总数据量: {len(data)} 条")
    
    # 基本统计信息
    stats = {
        'total_samples': len(data),
        'product_smiles': [],
        'synthon_smiles': [],
        'reactant_smiles': [],
        'vocab_sizes': [],
        'augmentation_counts': [],
        'molecular_weights': [],
        'num_atoms': [],
        'num_bonds': [],
        'reaction_center_types': {'bf': 0, 'bc': 0, 'a': 0}
    }
    
    # 分析每个样本
    for entry in data:
        pretrain_info = entry['pretrain_info']
        
        stats['product_smiles'].append(pretrain_info['product_smiles'])
        stats['synthon_smiles'].append(pretrain_info['synthon_smiles'])
        stats['reactant_smiles'].append(pretrain_info['reactant_smiles'])
        stats['vocab_sizes'].append(pretrain_info.get('vocab_size', 0))
        stats['augmentation_counts'].append(len(entry['augmented_data']))
        
        # 分析产物分子属性
        try:
            mol = Chem.MolFromSmiles(pretrain_info['product_smiles'])
            if mol:
                stats['molecular_weights'].append(Descriptors.MolWt(mol))
                stats['num_atoms'].append(mol.GetNumAtoms())
                stats['num_bonds'].append(mol.GetNumBonds())
        except:
            stats['molecular_weights'].append(0)
            stats['num_atoms'].append(0)
            stats['num_bonds'].append(0)
    
    return stats

def analyze_error_patterns(original_data_path, processed_data_path=None):
    """
    分析数据处理过程中的错误模式
    """
    print(f"分析原始数据: {original_data_path}")
    
    # 读取原始数据
    if original_data_path.endswith('.csv'):
        df = pd.read_csv(original_data_path)
        if 'reactants>reagents>production' in df.columns:
            df = df.rename(columns={'reactants>reagents>production': 'rxn_smiles'})
    else:
        print("不支持的原始数据格式")
        return None
    
    print(f"原始数据量: {len(df)} 条")
    
    # 分析反应SMILES的基本特征
    error_analysis = {
        'total_reactions': len(df),
        'empty_reactions': 0,
        'invalid_format': 0,
        'no_products': 0,
        'no_reactants': 0,
        'complex_reactions': 0,  # 多个产物或反应物
        'simple_reactions': 0,   # 单个产物和反应物
        'atom_counts': [],
        'reactant_counts': [],
        'product_counts': []
    }
    
    for idx, row in df.iterrows():
        rxn_smiles = str(row.get('rxn_smiles', ''))
        
        if pd.isna(rxn_smiles) or rxn_smiles.strip() == '':
            error_analysis['empty_reactions'] += 1
            continue
        
        if '>>' not in rxn_smiles:
            error_analysis['invalid_format'] += 1
            continue
        
        try:
            parts = rxn_smiles.split('>')
            if len(parts) != 3:
                error_analysis['invalid_format'] += 1
                continue
            
            reactants = parts[0].strip()
            products = parts[2].strip()
            
            if not reactants:
                error_analysis['no_reactants'] += 1
                continue
            
            if not products:
                error_analysis['no_products'] += 1
                continue
            
            # 统计反应物和产物数量
            reactant_count = len(reactants.split('.')) if reactants else 0
            product_count = len(products.split('.')) if products else 0
            
            error_analysis['reactant_counts'].append(reactant_count)
            error_analysis['product_counts'].append(product_count)
            
            if reactant_count > 2 or product_count > 1:
                error_analysis['complex_reactions'] += 1
            else:
                error_analysis['simple_reactions'] += 1
            
            # 估算原子数量
            try:
                total_atoms = 0
                for smi in reactants.split('.') + products.split('.'):
                    mol = Chem.MolFromSmiles(smi.strip())
                    if mol:
                        total_atoms += mol.GetNumAtoms()
                error_analysis['atom_counts'].append(total_atoms)
            except:
                error_analysis['atom_counts'].append(0)
                
        except Exception as e:
            error_analysis['invalid_format'] += 1
    
    return error_analysis

def generate_quality_report(stats, error_analysis, output_dir):
    """
    生成数据质量报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文字报告
    report_path = os.path.join(output_dir, "data_quality_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 预训练数据质量分析报告 ===\n\n")
        
        if stats:
            f.write("1. 成功处理的数据统计:\n")
            f.write(f"   总样本数: {stats['total_samples']}\n")
            f.write(f"   平均词汇表大小: {np.mean(stats['vocab_sizes']):.2f}\n")
            f.write(f"   平均增强数据量: {np.mean(stats['augmentation_counts']):.2f}\n")
            f.write(f"   平均分子量: {np.mean(stats['molecular_weights']):.2f}\n")
            f.write(f"   平均原子数: {np.mean(stats['num_atoms']):.2f}\n")
            f.write(f"   平均键数: {np.mean(stats['num_bonds']):.2f}\n\n")
        
        if error_analysis:
            f.write("2. 原始数据错误分析:\n")
            total = error_analysis['total_reactions']
            f.write(f"   总反应数: {total}\n")
            f.write(f"   空反应: {error_analysis['empty_reactions']} ({error_analysis['empty_reactions']/total*100:.1f}%)\n")
            f.write(f"   格式错误: {error_analysis['invalid_format']} ({error_analysis['invalid_format']/total*100:.1f}%)\n")
            f.write(f"   无产物: {error_analysis['no_products']} ({error_analysis['no_products']/total*100:.1f}%)\n")
            f.write(f"   无反应物: {error_analysis['no_reactants']} ({error_analysis['no_reactants']/total*100:.1f}%)\n")
            f.write(f"   复杂反应: {error_analysis['complex_reactions']} ({error_analysis['complex_reactions']/total*100:.1f}%)\n")
            f.write(f"   简单反应: {error_analysis['simple_reactions']} ({error_analysis['simple_reactions']/total*100:.1f}%)\n\n")
            
            if error_analysis['reactant_counts']:
                f.write(f"   平均反应物数量: {np.mean(error_analysis['reactant_counts']):.2f}\n")
                f.write(f"   平均产物数量: {np.mean(error_analysis['product_counts']):.2f}\n")
                f.write(f"   平均原子总数: {np.mean(error_analysis['atom_counts']):.2f}\n\n")
        
        f.write("3. 数据质量建议:\n")
        
        if stats and stats['total_samples'] > 0:
            success_indicators = []
            if np.mean(stats['vocab_sizes']) > 5:
                success_indicators.append("✓ 词汇表大小正常")
            if np.mean(stats['num_atoms']) > 10:
                success_indicators.append("✓ 分子复杂度适中")
            if np.mean(stats['augmentation_counts']) >= 2:
                success_indicators.append("✓ 数据增强正常")
            
            for indicator in success_indicators:
                f.write(f"   {indicator}\n")
        
        if error_analysis:
            total = error_analysis['total_reactions']
            if error_analysis['invalid_format'] / total > 0.1:
                f.write("   ⚠ 格式错误率较高，建议检查数据预处理\n")
            if error_analysis['complex_reactions'] / total > 0.5:
                f.write("   ⚠ 复杂反应较多，可能影响处理成功率\n")
            if len(error_analysis['atom_counts']) > 0 and np.mean(error_analysis['atom_counts']) > 100:
                f.write("   ⚠ 分子过于复杂，可能导致处理困难\n")
        
        f.write("\n4. 改进建议:\n")
        f.write("   - 过滤掉格式不正确的反应\n")
        f.write("   - 优先处理简单反应（单一产物）\n")
        f.write("   - 对复杂分子进行预筛选\n")
        f.write("   - 提高atom mapping质量\n")
    
    print(f"质量报告已保存到: {report_path}")
    
    # 生成可视化图表
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('数据质量分析图表', fontsize=16)
        
        if stats and len(stats['molecular_weights']) > 0:
            # 分子量分布
            axes[0, 0].hist(stats['molecular_weights'], bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('分子量分布')
            axes[0, 0].set_xlabel('分子量')
            axes[0, 0].set_ylabel('频数')
            
            # 原子数分布
            axes[0, 1].hist(stats['num_atoms'], bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('原子数分布')
            axes[0, 1].set_xlabel('原子数')
            axes[0, 1].set_ylabel('频数')
            
            # 词汇表大小分布
            axes[0, 2].hist(stats['vocab_sizes'], bins=20, alpha=0.7, color='orange')
            axes[0, 2].set_title('词汇表大小分布')
            axes[0, 2].set_xlabel('词汇表大小')
            axes[0, 2].set_ylabel('频数')
        
        if error_analysis and len(error_analysis['reactant_counts']) > 0:
            # 反应物数量分布
            reactant_counter = Counter(error_analysis['reactant_counts'])
            axes[1, 0].bar(reactant_counter.keys(), reactant_counter.values(), alpha=0.7, color='red')
            axes[1, 0].set_title('反应物数量分布')
            axes[1, 0].set_xlabel('反应物数量')
            axes[1, 0].set_ylabel('反应数')
            
            # 产物数量分布
            product_counter = Counter(error_analysis['product_counts'])
            axes[1, 1].bar(product_counter.keys(), product_counter.values(), alpha=0.7, color='purple')
            axes[1, 1].set_title('产物数量分布')
            axes[1, 1].set_xlabel('产物数量')
            axes[1, 1].set_ylabel('反应数')
            
            # 错误类型分布
            error_types = ['空反应', '格式错误', '无产物', '无反应物']
            error_counts = [
                error_analysis['empty_reactions'],
                error_analysis['invalid_format'],
                error_analysis['no_products'],
                error_analysis['no_reactants']
            ]
            axes[1, 2].bar(error_types, error_counts, alpha=0.7, color='gray')
            axes[1, 2].set_title('错误类型统计')
            axes[1, 2].set_ylabel('错误数量')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "data_quality_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"质量分析图表已保存到: {plot_path}")
        
    except ImportError:
        print("matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="分析预训练数据质量")
    parser.add_argument('--processed_data', type=str, help="处理后的预训练数据路径(.pkl)")
    parser.add_argument('--original_data', type=str, help="原始数据路径(.csv)")
    parser.add_argument('--output_dir', type=str, default="./quality_analysis/", help="输出目录")
    
    args = parser.parse_args()
    
    print("=== 预训练数据质量分析 ===\n")
    
    stats = None
    error_analysis = None
    
    # 分析处理后的数据
    if args.processed_data and os.path.exists(args.processed_data):
        try:
            stats = analyze_pretrain_data(args.processed_data)
            print("✓ 成功分析处理后的数据")
        except Exception as e:
            print(f"✗ 分析处理后数据失败: {e}")
    
    # 分析原始数据的错误模式
    if args.original_data and os.path.exists(args.original_data):
        try:
            error_analysis = analyze_error_patterns(args.original_data)
            print("✓ 成功分析原始数据错误模式")
        except Exception as e:
            print(f"✗ 分析原始数据失败: {e}")
    
    # 生成质量报告
    if stats or error_analysis:
        try:
            generate_quality_report(stats, error_analysis, args.output_dir)
            print("✓ 成功生成质量报告")
        except Exception as e:
            print(f"✗ 生成报告失败: {e}")
    else:
        print("✗ 没有有效数据进行分析")
    
    # 输出关键建议
    print("\n=== 关键建议 ===")
    
    if error_analysis:
        total = error_analysis['total_reactions']
        format_error_rate = error_analysis['invalid_format'] / total
        complex_rate = error_analysis['complex_reactions'] / total
        
        print(f"原始数据格式错误率: {format_error_rate*100:.1f}%")
        print(f"复杂反应比例: {complex_rate*100:.1f}%")
        
        if format_error_rate > 0.05:
            print("⚠ 建议: 格式错误率较高，需要改进数据预处理")
        
        if complex_rate > 0.3:
            print("⚠ 建议: 复杂反应较多，考虑分批处理或预筛选")
    
    if stats:
        success_rate = stats['total_samples']
        if args.original_data and error_analysis:
            success_rate = stats['total_samples'] / error_analysis['total_reactions']
            print(f"数据处理成功率: {success_rate*100:.1f}%")
            
            if success_rate < 0.5:
                print("⚠ 建议: 处理成功率较低，需要优化处理流程")
            elif success_rate < 0.8:
                print("⚠ 建议: 处理成功率中等，可进一步优化")
            else:
                print("✓ 数据处理成功率良好")
    
    print(f"\n详细分析结果请查看: {args.output_dir}")

if __name__ == "__main__":
    main()