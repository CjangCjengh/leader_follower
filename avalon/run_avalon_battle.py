#!/usr/bin/env python
# encoding: utf-8
"""
Avalon Battle Game Runner
支持通过 JSON 配置文件启动游戏

Agent 类型说明：
- direct: 直接生成回复（1次API调用，最快）
- react: ReAct框架（思考-行动循环，2次API调用）
- recon: 关系一致性框架（跨玩家关系分析，3次API调用）
- lasi: LASI框架（局势分析-计划-行动-回复，4次API调用）

使用方式：
    python run_avalon_battle.py -c config.json
"""
import argparse
import json
import random
import os
import sys

from colorama import Fore, Style

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extractor.llm_extractor.chatgpt_extractor import ChatGPTBasedExtractor
from src.games.avalon.avalon import Avalon
from src.agents import DirectAgent, ReActAgent, ReConAgent, LASIAgent
from prompt.avalon_prompts import (
    summary_prompt, plan_prompt, response_prompt, system_prompt,
    action_prompt, suggestion_prompt, update_prompt, analysis_prompt,
    strategy_prompt, candidate_actions, init_strategies, role_introduction, role_target
)
from src.games.avalon.extract_demos import (
    number_extract_prompt, player_extractor_demos, vote_extractor_demos,
    quest_extractor_demos, choose_identify_extractor_demos, select_merlin_extractor_demos,
    bool_extract_prompt, quest_extract_prompt
)
from src.utils import create_dir, read_json, write_json, print_text_animated


# 默认角色配置
DEFAULT_ROLES = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"]

# 思考过程颜色（用于 watch 模式）
THINKING_COLOR = Fore.CYAN + Style.DIM


def create_thinking_callback(player_name: str, mode: str):
    """创建思考过程回调函数"""
    if mode != 'watch':
        return None
    
    def callback(stage: str, content: str):
        """显示思考过程"""
        # 使用青色和暗淡样式显示思考过程
        print_text_animated(
            THINKING_COLOR + 
            f"    💭 [{player_name}] {stage}:\n" +
            f"    {content[:500]}{'...' if len(content) > 500 else ''}\n" +
            Style.RESET_ALL, 
            delay=0.001
        )
    
    return callback


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_model_config(player_model: dict, default_model: dict) -> dict:
    """获取模型配置，优先使用玩家配置，没有则使用默认配置"""
    if player_model is None:
        return default_model.copy()
    
    # 合并配置：玩家配置优先级更高
    merged = default_model.copy()
    for key, value in player_model.items():
        if value is not None:
            merged[key] = value
    return merged


def assign_roles(players: list, roles: list) -> list:
    """
    给玩家分配角色
    如果玩家配置中指定了角色，则使用指定的角色
    否则从可用角色中随机分配
    """
    available_roles = roles.copy()
    random.shuffle(available_roles)
    
    assigned_players = []
    used_indices = set()
    
    # 第一轮：处理已指定角色的玩家
    for player in players:
        if player.get('role') is not None:
            role = player['role']
            if role in available_roles:
                available_roles.remove(role)
                assigned_players.append({**player, 'role': role})
            else:
                raise ValueError(f"角色 {role} 不在可用角色列表中或已被分配")
        else:
            assigned_players.append(player.copy())
    
    # 第二轮：为未指定角色的玩家随机分配
    for i, player in enumerate(assigned_players):
        if player.get('role') is None:
            if not available_roles:
                raise ValueError("可用角色不足，无法为所有玩家分配角色")
            player['role'] = available_roles.pop(0)
    
    return assigned_players


def get_base_agent_args(player: dict, model_config: dict, log_dir: str, 
                        game_idx: int, output_dir: str, mode: str,
                        enable_intent_identification: bool = False) -> dict:
    """获取所有 Agent 类型共用的基础参数"""
    name = player['name']
    role = player['role']
    
    # 加载历史经验
    if game_idx == 0:
        role_strategy = init_strategies.get(role, "Play strategically.")
        other_strategy = "None"
        suggestion = "None"
    else:
        load_file = os.path.join(output_dir.format(game_idx - 1), f"{role}_reflection.json")
        if os.path.exists(load_file):
            experience = read_json(load_file)
            role_strategy = experience.get("strategy", init_strategies.get(role, "Play strategically."))
            other_strategy = experience.get("other_strategy", "None")
            suggestion = experience.get("suggestion", "None")
        else:
            role_strategy = init_strategies.get(role, "Play strategically.")
            other_strategy = "None"
            suggestion = "None"
    
    role_system_prompt = system_prompt.format(
        name=name, role=role, strategy=role_strategy,
        suggestion=suggestion, other_strategy=other_strategy
    )
    
    return {
        "name": name,
        "role": role,
        "role_intro": role_introduction.get(role.lower(), ""),
        "game_goal": role_target.get(role, "Win the game."),
        "strategy": role_strategy,
        "system_prompt": role_system_prompt,
        "model": model_config['model_name'],
        "temperature": model_config.get('temperature', 0.3),
        "api_key": model_config['api_key'],
        "api_base": model_config.get('api_base'),
        "output_dir": log_dir,
        "thinking_callback": create_thinking_callback(name, mode),
        "enable_intent_identification": enable_intent_identification,
        # 以下用于 LASI 的 reflection
        "suggestion": suggestion,
        "other_strategy": other_strategy,
    }


def create_direct_agent_args(player: dict, model_config: dict, log_dir: str,
                              game_idx: int, output_dir: str, mode: str,
                              enable_intent_identification: bool = False) -> tuple:
    """创建 Direct Agent 参数"""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        DirectAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_react_agent_args(player: dict, model_config: dict, log_dir: str,
                             game_idx: int, output_dir: str, mode: str,
                             enable_intent_identification: bool = False) -> tuple:
    """创建 ReAct Agent 参数"""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        ReActAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_recon_agent_args(player: dict, model_config: dict, log_dir: str,
                             game_idx: int, output_dir: str, mode: str,
                             enable_intent_identification: bool = False) -> tuple:
    """创建 ReCon Agent 参数"""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        ReConAgent,
        {
            **base_args,
            "response_prompt": response_prompt,
        }
    )


def create_lasi_agent_args(player: dict, model_config: dict, log_dir: str,
                            game_idx: int, output_dir: str, mode: str,
                            enable_intent_identification: bool = False) -> tuple:
    """创建 LASI Agent 参数"""
    base_args = get_base_agent_args(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
    
    return (
        LASIAgent,
        {
            **base_args,
            "analysis_prompt": analysis_prompt,
            "plan_prompt": plan_prompt,
            "action_prompt": action_prompt,
            "response_prompt": response_prompt,
            "suggestion_prompt": suggestion_prompt,
            "strategy_prompt": strategy_prompt,
            "update_prompt": update_prompt,
            "candidate_actions": candidate_actions,
        }
    )


# Agent 类型到创建函数的映射
AGENT_CREATORS = {
    'direct': create_direct_agent_args,
    'react': create_react_agent_args,
    'recon': create_recon_agent_args,
    'lasi': create_lasi_agent_args,
}


def run_game(config: dict, game_idx: int):
    """运行单局游戏"""
    game_config = config['game']
    default_model = config['default_model']
    players_config = config['players']
    roles = config.get('roles', DEFAULT_ROLES)
    extractor_config = config.get('extractors', {})
    
    # 构建输出目录
    output_dir = os.path.join(
        game_config.get('output_dir', 'logs/avalon/battle'),
        f"{game_config.get('exp_name', 'battle')}-{game_config.get('camp', 'good')}-game_{{}}"
    )
    game_output_dir = output_dir.format(game_idx)
    create_dir(game_output_dir)
    
    # 分配角色
    assigned_players = assign_roles(players_config, roles)
    
    # 根据阵营过滤（如果指定了阵营）
    camp = game_config.get('camp')
    if camp == "good":
        camp_roles = ["Merlin", "Percival", "Loyal Servant"]
    elif camp == "evil":
        camp_roles = ["Morgana", "Assassin"]
    else:
        camp_roles = None  # 不过滤
    
    # 创建游戏实例
    player_nums = game_config.get('player_nums', 6)
    language = game_config.get('language', 'english')
    mode = game_config.get('mode', 'watch')
    ai_model = default_model.get('model_name', 'gpt-3.5-turbo-16k')
    enable_intent_identification = game_config.get('enable_intent_identification', False)
    
    game = Avalon(player_nums, language, mode, ai_model, game_output_dir,
                  enable_intent_identification=enable_intent_identification)
    
    # 创建玩家参数
    player_args = []
    player_mapping = {}
    
    for i, player in enumerate(assigned_players):
        log_dir = os.path.join(game_output_dir, player['name'])
        create_dir(log_dir)
        
        model_config = get_model_config(player.get('model'), default_model)
        agent_type = player.get('agent_type', 'direct').lower()  # 默认使用 direct
        role = player['role']
        
        player_mapping[player['name']] = role
        
        # 根据阵营配置决定使用哪种 Agent 类型
        if camp_roles is not None:
            if role in camp_roles:
                use_type = agent_type  # 目标阵营使用配置的 agent_type
            else:
                use_type = 'direct'  # 非目标阵营使用最简单的 direct
        else:
            use_type = agent_type
        
        # 获取 agent 创建函数
        creator = AGENT_CREATORS.get(use_type)
        if creator is None:
            print(f"警告: 未知的 agent_type '{use_type}'，使用 'direct' 替代")
            creator = AGENT_CREATORS['direct']
        
        args = creator(player, model_config, log_dir, game_idx, output_dir, mode, enable_intent_identification)
        player_args.append(args)
    
    game.add_players(player_args)
    
    # 创建 Extractor 配置
    ext_model = extractor_config.get('model_name') or default_model.get('model_name', 'gpt-4o')
    ext_api_key = extractor_config.get('api_key') or default_model.get('api_key')
    ext_api_base = extractor_config.get('api_base') or default_model.get('api_base')
    ext_temp = extractor_config.get('temperature', 0)
    
    extractor_args = [
        (ChatGPTBasedExtractor, {
            "extractor_name": "player extractor",
            "model_name": ext_model,
            "extract_prompt": number_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": player_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "vote extractor",
            "model_name": ext_model,
            "extract_prompt": bool_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": vote_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "quest extractor",
            "model_name": ext_model,
            "extract_prompt": quest_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": quest_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "identify extractor",
            "model_name": ext_model,
            "extract_prompt": bool_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": choose_identify_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base
        }),
        (ChatGPTBasedExtractor, {
            "extractor_name": "merlin extractor",
            "model_name": ext_model,
            "extract_prompt": number_extract_prompt,
            "system_prompt": "You are a helpful assistant.",
            "temperature": ext_temp,
            "few_shot_demos": select_merlin_extractor_demos,
            "output_dir": game_output_dir,
            "api_key": ext_api_key,
            "api_base": ext_api_base
        })
    ]
    
    game.init_extractor(
        player_extractor=extractor_args[0],
        vote_extractor=extractor_args[1],
        quest_extractor=extractor_args[2],
        choose_identify_extractor=extractor_args[3],
        select_merlin_extractor=extractor_args[4]
    )
    
    # 开始游戏
    game.start()
    
    # 保存反思结果（仅 LASI Agent 会执行实际反思）
    for player_name, agent in game.players.items():
        agent.reflection(
            player_mapping,
            file_name=os.path.join(game_output_dir, f"{player_mapping.get(player_name)}_reflection.json"),
            winners=game.winners,
            duration=game.game_round
        )
    
    return game


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Avalon Battle Game Runner')
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='配置文件路径 (JSON格式)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅验证配置，不运行游戏'
    )
    return parser.parse_args()


def validate_config(config: dict) -> bool:
    """验证配置文件"""
    errors = []
    
    # 检查必要字段
    if 'game' not in config:
        errors.append("缺少 'game' 配置")
    
    if 'default_model' not in config:
        errors.append("缺少 'default_model' 配置")
    elif not config['default_model'].get('api_key'):
        errors.append("缺少 default_model.api_key")
    
    if 'players' not in config:
        errors.append("缺少 'players' 配置")
    else:
        player_nums = config.get('game', {}).get('player_nums', 6)
        if len(config['players']) != player_nums:
            errors.append(f"玩家数量不匹配：配置了 {len(config['players'])} 个玩家，但 game.player_nums 设置为 {player_nums}")
        
        # 检查 agent_type 是否有效
        valid_types = list(AGENT_CREATORS.keys())
        for i, player in enumerate(config['players']):
            agent_type = player.get('agent_type', 'direct').lower()
            if agent_type not in valid_types:
                errors.append(f"玩家 {i+1} 的 agent_type '{agent_type}' 无效，有效值: {valid_types}")
    
    if errors:
        print("配置验证失败：")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过！")
    print(f"支持的 Agent 类型: {list(AGENT_CREATORS.keys())}")
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 验证配置
    if not validate_config(config):
        sys.exit(1)
    
    if args.dry_run:
        print("Dry run 模式，配置验证通过，不运行游戏")
        sys.exit(0)
    
    # 运行游戏
    game_config = config['game']
    start_idx = game_config.get('start_game_idx', 0)
    game_count = game_config.get('game_count', 10)
    
    for game_round in range(start_idx, game_count):
        print(f"\n{'='*50}")
        print(f"开始游戏 {game_round + 1}/{game_count}")
        print(f"{'='*50}")
        
        try:
            game = run_game(config, game_round)
            print(f"\n游戏 {game_round} 完成！")
            print(f"获胜方: {game.winners}")
        except Exception as e:
            print(f"游戏 {game_round} 发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n所有游戏完成！")


if __name__ == '__main__':
    main()
