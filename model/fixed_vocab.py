#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复的Vocab类包装器

解决原始Vocab类没有__len__方法的问题
"""

from vocab import Vocab as OriginalVocab

class FixedVocab:
    """修复的Vocab类，添加了__len__方法"""
    
    def __init__(self, smiles_list):
        """初始化修复的Vocab"""
        self.original_vocab = OriginalVocab(smiles_list)
        self.smiles_list = smiles_list
    
    def __len__(self):
        """返回词汇表大小"""
        # 尝试多种方式获取词汇表大小
        if hasattr(self.original_vocab, 'vocab') and hasattr(self.original_vocab.vocab, '__len__'):
            return len(self.original_vocab.vocab)
        elif hasattr(self.original_vocab, 'smiles_list'):
            return len(self.original_vocab.smiles_list)
        elif hasattr(self.original_vocab, 'size'):
            return self.original_vocab.size
        else:
            return len(self.smiles_list)
    
    def __getattr__(self, name):
        """代理所有其他属性到原始Vocab对象"""
        return getattr(self.original_vocab, name)

def create_vocab(smiles_list):
    """创建修复的Vocab对象"""
    return FixedVocab(smiles_list) 