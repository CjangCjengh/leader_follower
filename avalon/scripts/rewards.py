#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Avalon 游戏 GRPO 训练的 Reward 函数

基于论文中的 Stackelberg 博弈方法：
- Impact Measurement: 计算 leader 发言对 follower 响应概率的影响
- Reward: R(u_t) = Σ log P_F(desired | context + u_t) - Σ log P_F(undesired | context + u_t)

调用本地 reward_server 来计算 reward
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from typing import List, Dict, Any, Optional

# 默认 reward 值（当 API 调用失败时使用）
DEFAULT_REWARD_VALUE = 0.0

# Reward Server 配置
REWARD_SERVER_HOST = os.environ.get('REWARD_SERVER_HOST', '127.0.0.1')
REWARD_SERVER_PORT = os.environ.get('REWARD_SERVER_PORT', '8000')
REWARD_SERVER_URL = f'http://{REWARD_SERVER_HOST}:{REWARD_SERVER_PORT}'

# 配置 requests session
session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)
session.mount('http://', adapter)


# leader 发言的占位符（与 convert_logs_to_grpo_data.py 中保持一致）
LEADER_RESPONSE_PLACEHOLDER = "{{LEADER_RESPONSE}}"


def my_reward_function(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """
    计算 Avalon 游戏中 leader 发言的 persuasive impact reward
    
    基于论文公式：
    R(u_t) = Σ log P_F(desired | context + u_t) - Σ log P_F(undesired | context + u_t)
    
    Args:
        data_sources: 数据来源列表
        solution_strs: 模型生成的 leader 发言列表
        ground_truths: follower 的实际响应（这里不使用，reward 基于概率计算）
        extra_infos: 额外信息列表，每个元素包含：
            - follower_prompt_template: follower 的 prompt 模板（messages 格式）
            - intent_identification: 包含 desired_responses 和 undesired_responses
        **kwargs: 其他参数
    
    Returns:
        List[float]: 每个 leader 发言的 reward 值
    """
    if extra_infos is None:
        extra_infos = [{} for _ in range(len(solution_strs))]
    
    # 构建请求 payload
    requests_payload = []
    for i in range(len(solution_strs)):
        extra_info = extra_infos[i] or {}
        
        # 获取 follower 的 prompt template
        follower_prompt_template = extra_info.get('follower_prompt_template', [])
        
        # 获取 intent identification（desired 和 undesired responses）
        intent_identification = extra_info.get('intent_identification', {})
        desired_responses = intent_identification.get('desired_responses', [])
        undesired_responses = intent_identification.get('undesired_responses', [])
        
        requests_payload.append({
            'leader_response': solution_strs[i],
            'follower_prompt_template': follower_prompt_template,
            'desired_responses': desired_responses,
            'undesired_responses': undesired_responses,
        })
    
    # 调用 reward server
    url = f'{REWARD_SERVER_URL}/compute_reward'
    
    try:
        resp = session.post(url, json={'requests': requests_payload}, timeout=6000)
        if resp.status_code == 200:
            data = resp.json() or {}
            rewards = data.get('rewards', None)
            if not isinstance(rewards, list) or len(rewards) != len(solution_strs):
                print(f'[Reward API Warning] Invalid response format, using default values')
                rewards = [DEFAULT_REWARD_VALUE for _ in range(len(solution_strs))]
        else:
            print(f'[Reward API Error] Status: {resp.status_code}, Response: {resp.text}')
            rewards = [DEFAULT_REWARD_VALUE for _ in range(len(solution_strs))]
    except Exception as e:
        print(f'[Reward API Exception] {e}')
        rewards = [DEFAULT_REWARD_VALUE for _ in range(len(solution_strs))]
    
    # 后处理：确保 reward 在合理范围内
    final_rewards = []
    for reward in rewards:
        try:
            reward = float(reward)
        except (ValueError, TypeError):
            reward = DEFAULT_REWARD_VALUE
        
        # 限制 reward 范围，避免数值不稳定
        reward = max(-50.0, min(50.0, reward))
        final_rewards.append(reward)
    
    return final_rewards


def compute_single_reward(
    leader_response: str,
    follower_prompt_template: List[Dict[str, str]],
    desired_responses: List[str],
    undesired_responses: List[str],
) -> float:
    """
    计算单个 leader 发言的 reward（便于调试和测试）
    
    Args:
        leader_response: leader 的发言
        follower_prompt_template: follower 的 prompt 模板
        desired_responses: 期望的 follower 响应列表
        undesired_responses: 不期望的 follower 响应列表
    
    Returns:
        float: reward 值
    """
    rewards = my_reward_function(
        data_sources=['custom_api'],
        solution_strs=[leader_response],
        ground_truths=[''],
        extra_infos=[{
            'follower_prompt_template': follower_prompt_template,
            'intent_identification': {
                'desired_responses': desired_responses,
                'undesired_responses': undesired_responses,
            }
        }]
    )
    return rewards[0]


if __name__ == '__main__':
    # 简单测试
    import json
    
    # 加载测试数据
    test_data_path = '/data/workspace/avalon-battle/test_output.jsonl'
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_sample = json.loads(f.readline().strip())
        
        extra_info = test_sample.get('extra_info', {})
        
        # 测试 leader response
        test_leader_response = "I think we should include Player 2 and myself for this quest."
        
        reward = compute_single_reward(
            leader_response=test_leader_response,
            follower_prompt_template=extra_info.get('follower_prompt_template', []),
            desired_responses=extra_info.get('intent_identification', {}).get('desired_responses', []),
            undesired_responses=extra_info.get('intent_identification', {}).get('undesired_responses', []),
        )
        
        print(f'Test Leader Response: {test_leader_response}')
        print(f'Reward: {reward}')
        
    except FileNotFoundError:
        print(f'Test data file not found: {test_data_path}')
    except Exception as e:
        print(f'Test failed: {e}')
