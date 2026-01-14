#!/usr/bin/env python
# encoding: utf-8
"""
Extractor 基类
"""
from abc import abstractmethod


class Extractor:
    """Extractor 抽象基类"""
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def step(self, input_text: str) -> str:
        """
        从输入文本中提取信息
        
        Args:
            input_text: 输入文本
        
        Returns:
            提取的信息
        """
        pass

    @classmethod
    def init_instance(cls, **kwargs):
        """工厂方法，创建 Extractor 实例"""
        return cls(**kwargs)
