#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Avalon 游戏日志转换为 verl GRPO 训练数据格式

用法：
    python scripts/convert_logs_to_grpo_data.py --log_dir logs/avalon/battle --output output.jsonl

输出格式：
    每行为一个JSON对象，包含：
    - prompt: 对话历史（messages格式）
    - extra_info: 额外信息，包括system_prompt、完整messages、intent信息等
    - reward_model: 奖励模型配置
    - data_source: 数据来源
    - ability: 能力标签
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt.avalon_prompts import (
    system_prompt as avalon_system_prompt_template,
    role_introduction,
    role_target,
    init_strategies
)


def extract_player_info(player_key: str) -> Tuple[str, str]:
    """
    从日志key中提取玩家名称和角色
    
    Args:
        player_key: 如 "player 2(Morgana)" 格式的key
    
    Returns:
        Tuple[player_name, role]: 如 ("player 2", "Morgana")
    """
    match = re.match(r"(player \d+)\(([^)]+)\)", player_key)
    if match:
        return match.group(1), match.group(2)
    return None, None


def build_system_prompt(name: str, role: str, strategy: str = None, 
                        suggestion: str = "", other_strategy: str = "") -> str:
    """
    构建系统提示词
    
    Args:
        name: 玩家名称
        role: 玩家角色
        strategy: 玩家策略
        suggestion: 建议（来自之前游戏的经验）
        other_strategy: 其他角色的策略
    
    Returns:
        格式化后的系统提示词
    """
    if strategy is None:
        strategy = init_strategies.get(role, "Play strategically to help your side win.")
    
    return avalon_system_prompt_template.format(
        name=name,
        role=role,
        strategy=strategy,
        suggestion=suggestion,
        other_strategy=other_strategy
    )


def build_role_intro_prompt(role: str) -> str:
    """获取角色介绍"""
    return role_introduction.get(role.lower(), "")


def build_game_goal(role: str) -> str:
    """获取游戏目标"""
    return role_target.get(role, "Win the game for your side.")


def parse_process_json(process_file: str) -> Dict[str, List[Dict]]:
    """
    解析 process.json 文件
    
    Args:
        process_file: process.json文件路径
    
    Returns:
        解析后的游戏日志，按回合组织
    """
    with open(process_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_player_mapping(process_data: Dict) -> Dict[str, str]:
    """
    从日志中提取玩家名称到角色的映射
    
    Returns:
        Dict[player_name, role]: 如 {"player 1": "Loyal Servant", ...}
    """
    player_mapping = {}
    
    for round_key, events in process_data.items():
        for event in events:
            for key in event.keys():
                if key.startswith("player ") and "(" in key:
                    player_name, role = extract_player_info(key)
                    if player_name and role:
                        player_mapping[player_name] = role
    
    return player_mapping


def build_dialogue_summary(process_data: Dict, current_round: str, 
                           current_event_idx: int) -> str:
    """
    构建到当前事件为止的对话历史摘要（文本格式）
    
    按照 Avalon 的实际请求格式，对话历史是以文本摘要的形式嵌入的：
    格式为 "host: xxx" 和 "player X: xxx"
    
    Args:
        process_data: 完整的游戏日志
        current_round: 当前回合key
        current_event_idx: 当前事件在回合中的索引
    
    Returns:
        对话历史摘要文本
    """
    summary_lines = []
    rounds = list(process_data.keys())
    current_round_idx = rounds.index(current_round)
    
    # 遍历所有之前的回合
    for i in range(current_round_idx + 1):
        round_key = rounds[i]
        events = process_data[round_key]
        
        # 如果是当前回合，只处理到当前事件
        max_idx = current_event_idx if round_key == current_round else len(events)
        
        for j in range(max_idx):
            event = events[j]
            
            # 提取Host消息
            if "Host" in event:
                summary_lines.append(f"host: {event['Host']}")
            
            # 提取玩家消息
            for key, value in event.items():
                if key.startswith("player ") and "(" in key:
                    player_name, role = extract_player_info(key)
                    if player_name and value:  # 排除空响应
                        summary_lines.append(f"{player_name}: {value}")
    
    return "\n".join(summary_lines)


def build_dialogue_context(process_data: Dict, current_round: str, 
                          current_event_idx: int,
                          current_player_name: str = None) -> List[Dict[str, str]]:
    """
    构建到当前事件为止的对话历史（兼容旧接口，但实际不使用）
    
    注意：这个函数保留用于兼容，但实际的 Avalon 格式是将历史嵌入到 user 消息中
    
    Args:
        process_data: 完整的游戏日志
        current_round: 当前回合key
        current_event_idx: 当前事件在回合中的索引
        current_player_name: 当前玩家名称（不使用）
    
    Returns:
        空列表（因为历史会嵌入到 user 消息中）
    """
    # 返回空列表，因为历史会作为 summary 嵌入到 user 消息中
    return []


def build_avalon_user_message(
    host_question: str,
    summary: str = "",
    player_name: str = "",
    role: str = "",
    plan: str = "",
    actions: str = ""
) -> str:
    """
    构建 Avalon 格式的 user 消息
    
    按照实际的请求格式，包含：
    - information: 玩家信息
    - environment: 环境状态（包括 summary）
    - question: Host 的问题
    
    Args:
        host_question: Host 的问题/指令
        summary: 之前回合的摘要
        player_name: 玩家名称
        role: 玩家角色
        plan: 玩家的计划
        actions: 当前动作
    
    Returns:
        格式化的 user 消息
    """
    parts = []
    
    # 添加玩家信息
    if player_name and role:
        parts.append(f"the information of yourself is <information>")
        parts.append(f"your name is <name>{player_name}</name>")
        parts.append(f"your role is <role>{role}</role>")
        role_intro = build_role_intro_prompt(role)
        if role_intro:
            parts.append(f"the role introduction is <introduction>{role_intro}</introduction>")
        strategy = init_strategies.get(role, "")
        if strategy:
            parts.append(f"your playing strategy <strategy>{strategy}</strategy>")
        parts.append("</information>")
    
    # 添加环境状态
    parts.append("the environment state is <environment>")
    if summary:
        parts.append(f"the summary of previous turns <summary> {summary} </summary>")
    else:
        parts.append("the summary of previous turns <summary> None </summary>")
    
    if plan:
        parts.append(f"your current playing plan is <plan> {plan} </plan>")
    
    parts.append(f"the Host's question is <question> {host_question} </question>")
    
    if actions:
        parts.append(f"current actions <actions>{actions}</actions>")
    
    parts.append("</environment>")
    
    return "\n".join(parts)


# leader 发言的占位符
LEADER_RESPONSE_PLACEHOLDER = "{{LEADER_RESPONSE}}"


def build_follower_prompt_template(
    process_data: Dict,
    round_key: str,
    current_event_idx: int,
    leader_name: str,
    follower_name: str,
    follower_role: str,
    follower_system_prompt: str,
    current_host_instruction: str = "",
    next_host_instruction: str = ""
) -> List[Dict[str, str]]:
    """
    构建 follower 的 prompt template，leader 发言处使用占位符
    
    按照 Avalon 的实际请求格式：
    - system: 游戏规则和角色设定
    - user: 包含 summary 的环境信息和 Host 问题，leader 发言处为占位符
    
    Args:
        process_data: 完整的游戏日志
        round_key: 当前回合key
        current_event_idx: leader 事件的索引
        leader_name: leader 玩家名称
        follower_name: follower 玩家名称
        follower_role: follower 的角色
        follower_system_prompt: follower 的系统提示词
        current_host_instruction: 当前（leader）的 Host 指令
        next_host_instruction: follower 收到的 Host 指令
    
    Returns:
        follower 的 prompt template（messages 格式），leader 发言处为 {{LEADER_RESPONSE}} 占位符
    """
    # 1. 构建到 leader 发言之前的对话摘要
    summary = build_dialogue_summary(
        process_data, round_key, current_event_idx
    )
    
    # 2. 添加 leader 收到的 Host 指令和 leader 发言的占位符到 summary
    if current_host_instruction:
        summary += f"\nhost: {current_host_instruction}"
    # 使用占位符代替实际的 leader 发言
    summary += f"\n{leader_name}: {LEADER_RESPONSE_PLACEHOLDER}"
    
    # 3. 构建 user 消息
    user_message = build_avalon_user_message(
        host_question=next_host_instruction if next_host_instruction else "Please respond.",
        summary=summary,
        player_name=follower_name,
        role=follower_role
    )
    
    # 4. 构建完整的 messages
    messages = [
        {"role": "system", "content": follower_system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return messages


def convert_discuss_event_to_training_sample(
    event: Dict,
    process_data: Dict,
    round_key: str,
    event_idx: int,
    player_mapping: Dict[str, str],
    game_dir: str
) -> Optional[Dict[str, Any]]:
    """
    将讨论事件转换为训练样本
    
    Args:
        event: 当前事件
        process_data: 完整游戏日志
        round_key: 当前回合key
        event_idx: 事件索引
        player_mapping: 玩家名称到角色的映射
        game_dir: 游戏日志目录
    
    Returns:
        训练样本或None
    """
    # 找到当前说话的玩家
    speaker_key = None
    speaker_response = None
    
    for key, value in event.items():
        if key.startswith("player ") and "(" in key:
            speaker_key = key
            speaker_response = value
            break
    
    if not speaker_key or not speaker_response:
        return None
    
    # 跳过空响应
    if not speaker_response.strip():
        return None
    
    player_name, role = extract_player_info(speaker_key)
    if not player_name or not role:
        return None
    
    # 构建系统提示词
    system_prompt = build_system_prompt(
        name=player_name,
        role=role,
        strategy=init_strategies.get(role)
    )
    
    # 构建对话历史摘要
    summary = build_dialogue_summary(
        process_data, round_key, event_idx
    )
    
    # 添加当前Host指令
    host_instruction = event.get("Host", "")
    
    # 构建 user 消息（按照 Avalon 的实际格式）
    user_message = build_avalon_user_message(
        host_question=host_instruction,
        summary=summary,
        player_name=player_name,
        role=role
    )
    
    # 构建messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # 构建prompt（用于推理）
    prompt = messages.copy()
    
    # 查找下一个说话的玩家（作为follower）
    rounds = list(process_data.keys())
    current_round_idx = rounds.index(round_key)
    round_events = process_data[round_key]
    
    next_speaker_key = None
    next_speaker_name = None
    next_speaker_role = None
    next_response = None
    next_event_idx = None
    
    # 查找下一个说话的事件
    for i in range(event_idx + 1, len(round_events)):
        next_event = round_events[i]
        for key, value in next_event.items():
            if key.startswith("player ") and "(" in key:
                _next_speaker_name, _next_speaker_role = extract_player_info(key)
                if _next_speaker_name and value and value.strip():
                    next_speaker_key = key
                    next_speaker_name = _next_speaker_name
                    next_speaker_role = _next_speaker_role
                    next_response = value
                    next_event_idx = i
                    break
        if next_speaker_key:
            break
    
    # 如果没有找到下一个玩家，跳过这个样本（无法计算reward）
    if not next_speaker_key:
        return None
    
    # ===== 构建 follower 的 prompt template 用于计算 reward =====
    # 按照论文方法，需要用 follower 的视角计算 P_F(response | context)
    # leader 发言处使用占位符 {{LEADER_RESPONSE}}，方便计算 GRPO reward 时替换
    
    # 1. 构建 follower 的 system prompt
    follower_system_prompt = build_system_prompt(
        name=next_speaker_name,
        role=next_speaker_role,
        strategy=init_strategies.get(next_speaker_role)
    )
    
    # 2. 查找 next_event 中的 Host 指令
    next_event = round_events[next_event_idx]
    next_host_instruction = next_event.get("Host", "")
    
    # 3. 构建 follower 的 prompt template（带占位符）
    follower_prompt_template = build_follower_prompt_template(
        process_data=process_data,
        round_key=round_key,
        current_event_idx=event_idx,
        leader_name=player_name,
        follower_name=next_speaker_name,
        follower_role=next_speaker_role,
        follower_system_prompt=follower_system_prompt,
        current_host_instruction=host_instruction,
        next_host_instruction=next_host_instruction
    )
    
    # extra_info 只包含 follower 的 prompt template
    # leader 发言处为 {{LEADER_RESPONSE}} 占位符，计算 reward 时替换成 leader sample 出的发言
    extra_info = {
        "follower_prompt_template": follower_prompt_template,
    }
    
    # 如果有intent信息，添加到extra_info中
    if "intent_identification" in event:
        extra_info["intent_identification"] = event["intent_identification"]
    
    # reward_model 只包含 style 和 ground_truth
    reward_model = {
        "style": "rule",
        "ground_truth": next_response,  # follower 的实际响应
    }
    
    return {
        "prompt": prompt,
        "extra_info": extra_info,
        "reward_model": reward_model,
        "data_source": "custom_api",
        "ability": "strategic_dialogue"
    }


def convert_game_logs_to_grpo_data(
    log_dir: str,
    output_file: str,
    only_discuss: bool = True,
    include_intent: bool = True
):
    """
    将游戏日志转换为GRPO训练数据
    
    Args:
        log_dir: 日志目录（包含多个游戏目录）
        output_file: 输出文件路径
        only_discuss: 是否只处理讨论阶段的数据
        include_intent: 是否包含intent_identification信息
    """
    samples = []
    
    # 查找所有游戏目录
    log_path = Path(log_dir)
    game_dirs = []
    
    # 支持两种目录结构：
    # 1. log_dir 直接包含 process.json
    # 2. log_dir 下有多个游戏子目录
    if (log_path / "process.json").exists():
        game_dirs = [log_path]
    else:
        for item in log_path.iterdir():
            if item.is_dir() and (item / "process.json").exists():
                game_dirs.append(item)
    
    print(f"找到 {len(game_dirs)} 个游戏目录")
    
    for game_dir in game_dirs:
        process_file = game_dir / "process.json"
        print(f"处理: {process_file}")
        
        try:
            process_data = parse_process_json(str(process_file))
        except Exception as e:
            print(f"  警告: 无法解析 {process_file}: {e}")
            continue
        
        # 提取玩家映射
        player_mapping = extract_player_mapping(process_data)
        print(f"  玩家映射: {player_mapping}")
        
        # 遍历所有回合和事件
        for round_key, events in process_data.items():
            for event_idx, event in enumerate(events):
                # 判断是否为讨论事件
                is_discuss_event = "discuss" in event.get("Host", "").lower() or \
                                   "speak" in event.get("Host", "").lower()
                
                if only_discuss and not is_discuss_event:
                    continue
                
                # 检查是否有intent_identification（如果需要）
                if include_intent and "intent_identification" not in event:
                    # 如果要求包含intent但事件中没有，跳过或继续处理
                    pass
                
                # 转换为训练样本
                sample = convert_discuss_event_to_training_sample(
                    event=event,
                    process_data=process_data,
                    round_key=round_key,
                    event_idx=event_idx,
                    player_mapping=player_mapping,
                    game_dir=str(game_dir)
                )
                
                if sample:
                    samples.append(sample)
    
    print(f"生成 {len(samples)} 个训练样本")
    
    # 写入输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"输出保存到: {output_file}")
    
    return samples


def convert_single_game(
    game_dir: str,
    output_file: str = None,
    only_discuss: bool = True
) -> List[Dict]:
    """
    转换单个游戏的日志
    
    Args:
        game_dir: 单个游戏的日志目录
        output_file: 输出文件路径（可选）
        only_discuss: 是否只处理讨论阶段
    
    Returns:
        训练样本列表
    """
    game_path = Path(game_dir)
    process_file = game_path / "process.json"
    
    if not process_file.exists():
        raise FileNotFoundError(f"找不到 process.json: {process_file}")
    
    process_data = parse_process_json(str(process_file))
    player_mapping = extract_player_mapping(process_data)
    
    samples = []
    
    for round_key, events in process_data.items():
        for event_idx, event in enumerate(events):
            is_discuss_event = "discuss" in event.get("Host", "").lower() or \
                               "speak" in event.get("Host", "").lower()
            
            if only_discuss and not is_discuss_event:
                continue
            
            sample = convert_discuss_event_to_training_sample(
                event=event,
                process_data=process_data,
                round_key=round_key,
                event_idx=event_idx,
                player_mapping=player_mapping,
                game_dir=str(game_path)
            )
            
            if sample:
                samples.append(sample)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="将 Avalon 游戏日志转换为 verl GRPO 训练数据格式"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="游戏日志目录，可以是单个游戏目录或包含多个游戏的父目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grpo_training_data.jsonl",
        help="输出文件路径（默认: grpo_training_data.jsonl）"
    )
    parser.add_argument(
        "--only_discuss",
        action="store_true",
        default=True,
        help="是否只处理讨论阶段的数据（默认: True）"
    )
    parser.add_argument(
        "--all_phases",
        action="store_true",
        help="处理所有阶段的数据（覆盖 --only_discuss）"
    )
    parser.add_argument(
        "--include_intent",
        action="store_true",
        default=True,
        help="是否包含intent_identification信息（默认: True）"
    )
    
    args = parser.parse_args()
    
    only_discuss = not args.all_phases
    
    convert_game_logs_to_grpo_data(
        log_dir=args.log_dir,
        output_file=args.output,
        only_discuss=only_discuss,
        include_intent=args.include_intent
    )


if __name__ == "__main__":
    main()
