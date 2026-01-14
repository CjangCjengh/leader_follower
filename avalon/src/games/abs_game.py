#!/usr/bin/env python
# encoding: utf-8
"""
Game 基类
"""
from abc import abstractmethod


class Game:
    """Game 抽象基类"""
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def start(self):
        """启动游戏"""
        pass

    @abstractmethod
    def add_players(self, players: list):
        """添加玩家"""
        pass
