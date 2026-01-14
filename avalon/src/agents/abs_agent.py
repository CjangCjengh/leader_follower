#!/usr/bin/env python
# encoding: utf-8
"""
Agent 基类
"""
from abc import abstractmethod


class Agent:
    """Agent 抽象基类"""
    name = None
    role = None

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.role = kwargs.get('role')

    @abstractmethod
    def step(self, message: str) -> str:
        """
        与 agent 交互
        
        Args:
            message: 输入给 agent 的消息
        
        Returns:
            agent 的响应
        """
        pass

    @abstractmethod
    def receive(self, name: str, message: str) -> None:
        """
        接收来自其他 agent 的消息
        
        Args:
            name: 发送消息的 agent 名称
            message: 消息内容
        """
        pass

    def set_night_info(self, info: str) -> None:
        """
        设置夜晚阶段获得的信息，会被合并到系统提示中
        
        Args:
            info: 夜晚阶段获得的信息
        """
        self.night_info = info

    def identify_intent(self, next_player: str) -> dict:
        """
        意图识别：识别希望和不希望后置位玩家说的内容
        
        基于论文中的 Intent Identification 方法：
        - 识别 K 个期望的响应（对当前玩家有利）
        - 识别 K 个不期望的响应（对当前玩家不利）
        
        Args:
            next_player: 下一个发言的玩家名称
            
        Returns:
            dict: 包含 desired_responses 和 undesired_responses 的字典，
                  如果未实现则返回 None
        """
        return None

    @classmethod
    def init_instance(cls, **kwargs):
        """工厂方法，创建 agent 实例"""
        return cls(**kwargs)
