#!/usr/bin/env python
# encoding: utf-8
"""
工具函数
"""
import time
import os
import json
from colorama import Fore


def print_text_animated(text, delay=0.01):
    """动画打印文本
    
    Args:
        text: 要打印的文本
        delay: 每个字符之间的延迟时间（秒），默认 0.01
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)


# 玩家颜色映射
COLOR = {
    "player 1": Fore.BLUE,
    "player 2": Fore.GREEN,
    "player 3": Fore.YELLOW,
    "player 4": Fore.RED,
    "player 5": Fore.LIGHTGREEN_EX,
    "player 6": Fore.CYAN,
}


def create_dir(dir_path):
    """创建目录"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_data(data, path):
    """写入文本数据"""
    with open(path, mode='a+', encoding='utf-8') as f:
        f.write(data)
        f.write('\n')


def write_json(data, path):
    """写入 JSON 数据"""
    with open(path, mode='w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(path):
    """读取 JSON 数据"""
    with open(path, mode="r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data
