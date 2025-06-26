#!/usr/bin/env python3
"""
G2Retro-P 数据预处理简单测试
用于快速验证预处理功能是否正常工作
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit import rdkit

# 屏蔽RDKit警告
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def test_single_reaction():
    """测试单个反应的预处理"""
    print("🧪 测试单个反应的预处理...")
    
    try:
        from preprocess_pretrain import build_moltree_pretrain, convert_uspto_full_format
        from mol_tree import MolTree
        
        # 简单的酯化反应
        test_reaction = {
            'id': 'test_1',
            'rxn_smiles': 'CCO.CC(=O)O>>CC(=O)OCC'
        }
        
        print(f"测试反应: {test_reaction['rxn_smiles']}")
        
        # 转换格式
        converted_data = convert_uspto_full_format(test_reaction)
        if converted_data is None:
            print("❌ 格式转换失败")
            return False
        
        print(f"✓ 格式转换成功")
        
        # 构建分子树
        result = build_moltree_pretrain(converted_data, use_dfs=True, shuffle=False)
        
        if len(result) != 5:
            print(f"❌ 分子树构建失败: 返回{len(result)}个元素，期望5个")
            return False
        
        prod_moltree, synthon_tree, react_moltree, vocab, pretrain_info = result
        
        if prod_moltree is None:
            print("❌ 产物分子树构建失败")
            return False
        
        print(f"✓ 产物分子树构建成功")
        print(f"✓ 合成子树构建成功: {synthon_tree.smiles}")
        print(f"✓ 反应物分子树构建成功")
        print(f"✓ 词汇表大小: {len(vocab)}")
        
        # 检查预训练信息
        if pretrain_info is None:
            print("❌ 预训练信息缺失")
            return False
        
        required_keys = ['product_smiles', 'synthon_smiles', 'reactant_smiles', 'product_orders']
        for key in required_keys:
            if key not in pretrain_info:
                print(f"❌ 预训练信息缺失关键字段: {key}")
                return False
        
        print(f"✓ 预训练信息完整")
        
        # 检查product_orders
        product_orders = pretrain_info['product_orders']
        if not isinstance(product_orders, tuple) or len(product_orders) != 6:
            print(f"❌ product_orders格式错误: {type(product_orders)}, 长度{len(product_orders) if isinstance(product_orders, (list, tuple)) else 'N/A'}")
            return False
        
        bond_order, change_order, ring_order, atom_order, attach_atoms, revise_bonds = product_orders
        print(f"✓ product_orders格式正确")
        print(f"  - bond_order: {len(bond_order) if isinstance(bond_order, list) else 'N/A'}")
        print(f"  - change_order: {change_order}")
        print(f"  - ring_order: {ring_order}")
        print(f"  - atom_order: {len(atom_order) if isinstance(atom_order, list) else 'N/A'}")
        print(f"  - attach_atoms: {len(attach_atoms) if isinstance(attach_atoms, list) else 'N/A'}")
        print(f"  - revise_bonds: {len(revise_bonds) if isinstance(revise_bonds, dict) else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_augmentation():
    """测试分子增强功能"""
    print("\n🧪 测试分子增强功能...")
    
    try:
        from preprocess_pretrain import augment_molecule_for_recovery
        
        test_smiles = "CC(=O)OCC"  # 乙酸乙酯
        print(f"测试分子: {test_smiles}")
        
        augment_types = ['atom_mask', 'bond_deletion', 'subgraph_removal']
        
        for aug_type in augment_types:
            aug_data = augment_molecule_for_recovery(test_smiles, augment_type=aug_type, ratio=0.15)
            
            if aug_data is None:
                print(f"❌ {aug_type} 增强失败")
                return False
            
            required_keys = ['original_smiles', 'augment_type', 'masked_indices', 'original_values']
            for key in required_keys:
                if key not in aug_data:
                    print(f"❌ {aug_type} 增强数据缺失字段: {key}")
                    return False
            
            print(f"✓ {aug_type} 增强成功")
            print(f"  - 掩码位置: {aug_data['masked_indices']}")
            print(f"  - 原始值: {aug_data['original_values']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """测试批量处理功能"""
    print("\n🧪 测试批量处理功能...")
    
    try:
        from preprocess_pretrain import process_pretrain_data_batch
        
        # 测试反应集
        test_reactions = [
            {'id': 'test_1', 'rxn_smiles': 'CCO.CC(=O)O>>CC(=O)OCC'},
            {'id': 'test_2', 'rxn_smiles': 'CC(=O)OCC>>CCO.CC(=O)O'},
            {'id': 'test_3', 'rxn_smiles': 'c1ccccc1Br.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1'},  # Suzuki偶联
        ]
        
        print(f"测试 {len(test_reactions)} 个反应的批量处理...")
        
        results, error_stats = process_pretrain_data_batch(test_reactions, use_dfs=True, shuffle=False)
        
        print(f"✓ 批量处理完成")
        print(f"  - 处理总数: {error_stats['total_processed']}")
        print(f"  - 成功数量: {error_stats['successful']}")
        print(f"  - 成功率: {error_stats['successful']/error_stats['total_processed']*100:.1f}%")
        
        if error_stats['successful'] == 0:
            print("❌ 批量处理没有成功处理任何反应")
            return False
        
        # 检查结果格式
        if len(results) == 0:
            print("❌ 批量处理结果为空")
            return False
        
        sample_result = results[0]
        required_keys = ['mol_trees', 'vocab', 'pretrain_info', 'augmented_data', 'original_data']
        for key in required_keys:
            if key not in sample_result:
                print(f"❌ 批量处理结果缺失字段: {key}")
                return False
        
        print(f"✓ 批量处理结果格式正确")
        print(f"  - 生成数据条目: {len(results)}")
        print(f"  - 增强数据数量: {len(sample_result['augmented_data'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*70)
    print("🧪 G2Retro-P 数据预处理功能测试")
    print("="*70)
    
    # 检查当前目录
    current_dir = os.getcwd()
    if not current_dir.endswith("model"):
        print("⚠️  请在model目录中运行此脚本")
        print(f"当前目录: {current_dir}")
        print("建议: cd model")
        return 1
    
    tests = [
        ("单个反应预处理", test_single_reaction),
        ("分子增强功能", test_augmentation),
        ("批量处理功能", test_batch_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*50}")
        print(f"📋 {test_name}")
        print(f"{'-'*50}")
        
        if test_func():
            print(f"✅ {test_name}: 通过")
            passed += 1
        else:
            print(f"❌ {test_name}: 失败")
    
    print(f"\n{'='*70}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有预处理功能测试通过!")
        print("✅ 数据预处理组件工作正常")
        
        print(f"\n💡 下一步:")
        print(f"   1. 运行完整预处理:")
        print(f"      python run_preprocess_pretrain.py --test_mode")
        print(f"   2. 开始预训练:")
        print(f"      python run_g2retro_p_pretrain.py --mode demo")
        return 0
    else:
        print("❌ 部分测试失败")
        print("💡 请检查相关依赖和代码实现")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 