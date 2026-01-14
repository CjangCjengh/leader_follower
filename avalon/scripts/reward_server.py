#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Avalon 游戏 GRPO 训练的 Reward Server

基于论文中的 Impact Measurement 方法：
- 使用 Qwen2.5-72B-Instruct 作为 Measurer 模拟 follower 的响应模式
- 计算 P_F(response | context) 的 log probability
- Reward: R(u_t) = Σ log P_F(desired | context + u_t) - Σ log P_F(undesired | context + u_t)

启动方式：
    python scripts/reward_server.py --model_path /path/to/Qwen2.5-72B-Instruct --port 8000
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Body
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Leader 发言的占位符（与 convert_logs_to_grpo_data.py 和 rewards.py 保持一致）
LEADER_RESPONSE_PLACEHOLDER = "{{LEADER_RESPONSE}}"

# 默认 reward 值
DEFAULT_REWARD_VALUE = 0.0


@dataclass
class RewardServerConfig:
    """Reward Server 配置"""
    model_path: str = "/data/models/Qwen2.5-72B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_length: int = 8192
    port: int = 8000


class AvalonRewardMeasurer:
    """
    Avalon 游戏 Reward Measurer
    
    使用本地 LLM（如 Qwen2.5-72B-Instruct）计算 follower 响应的 log probability
    基于论文中的 Impact Measurement 方法
    """
    
    def __init__(self, config: RewardServerConfig):
        """
        初始化 Measurer
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.device
        
        logger.info(f"Loading model from {config.model_path}...")
        
        # 设置 torch dtype
        if config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 确保有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _build_follower_prompt(
        self,
        follower_prompt_template: List[Dict[str, str]],
        leader_response: str
    ) -> str:
        """
        构建 follower 的完整 prompt（替换占位符）
        
        Args:
            follower_prompt_template: follower 的 prompt 模板（messages 格式）
            leader_response: leader 的发言（用于替换占位符）
        
        Returns:
            str: 完整的 prompt 字符串
        """
        # 深拷贝模板，避免修改原始数据
        messages = []
        for msg in follower_prompt_template:
            new_msg = {
                'role': msg['role'],
                'content': msg['content'].replace(LEADER_RESPONSE_PLACEHOLDER, leader_response)
            }
            messages.append(new_msg)
        
        # 使用 tokenizer 的 apply_chat_template 方法
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 添加生成 prompt
        )
        
        return prompt
    
    def compute_log_probability(
        self,
        prompt: str,
        target_response: str
    ) -> float:
        """
        计算给定 prompt 下生成 target_response 的 log probability
        
        基于论文公式：
        log P_F(response | context) = Σ log p(w_i | w_{<i}, context)
        
        Args:
            prompt: 完整的输入 prompt
            target_response: 目标响应文本
        
        Returns:
            float: log probability（归一化到 per-token）
        """
        # 编码 prompt 和完整序列
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
        
        # 编码 target response
        # 注意：这里我们需要编码完整序列，然后计算 target 部分的概率
        full_text = prompt + target_response
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt', add_special_tokens=False)
        
        # 如果序列太长，截断
        if full_ids.shape[1] > self.config.max_length:
            logger.warning(f"Sequence too long ({full_ids.shape[1]}), truncating to {self.config.max_length}")
            full_ids = full_ids[:, :self.config.max_length]
        
        full_ids = full_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        # 如果 target response 为空或太短
        if full_ids.shape[1] <= prompt_len:
            return 0.0
        
        with torch.no_grad():
            # 前向传播获取 logits
            outputs = self.model(full_ids)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # 计算 log softmax
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 提取 target response 部分的 log probability
            # logits[i] 预测的是 token i+1
            # 所以我们需要 logits[prompt_len-1:] 来预测 token[prompt_len:]
            target_log_probs = []
            for i in range(prompt_len, full_ids.shape[1]):
                # logits[i-1] 预测 token[i]
                token_id = full_ids[0, i].item()
                token_log_prob = log_probs[0, i - 1, token_id].item()
                target_log_probs.append(token_log_prob)
            
            # 计算总的 log probability（归一化到 per-token 以便比较）
            if len(target_log_probs) > 0:
                total_log_prob = sum(target_log_probs)
                # 返回 per-token 平均 log probability，避免长度偏差
                avg_log_prob = total_log_prob / len(target_log_probs)
                return avg_log_prob
            else:
                return 0.0
    
    def compute_reward(
        self,
        leader_response: str,
        follower_prompt_template: List[Dict[str, str]],
        desired_responses: List[str],
        undesired_responses: List[str]
    ) -> float:
        """
        计算 leader 发言的 persuasive impact reward
        
        基于论文公式：
        R(u_t) = Σ log P_F(desired | context + u_t) - Σ log P_F(undesired | context + u_t)
        
        Args:
            leader_response: leader 的发言
            follower_prompt_template: follower 的 prompt 模板
            desired_responses: 期望的 follower 响应列表
            undesired_responses: 不期望的 follower 响应列表
        
        Returns:
            float: reward 值
        """
        # 如果没有 prompt template 或 intent，返回默认值
        if not follower_prompt_template:
            logger.warning("Empty follower_prompt_template, returning default reward")
            return DEFAULT_REWARD_VALUE
        
        if not desired_responses and not undesired_responses:
            logger.warning("No desired or undesired responses, returning default reward")
            return DEFAULT_REWARD_VALUE
        
        # 构建替换了 leader 发言的 follower prompt
        follower_prompt = self._build_follower_prompt(
            follower_prompt_template,
            leader_response
        )
        
        # 计算 desired responses 的 log probability 之和
        desired_log_prob_sum = 0.0
        for response in desired_responses:
            if response:  # 跳过空响应
                log_prob = self.compute_log_probability(follower_prompt, response)
                desired_log_prob_sum += log_prob
                logger.debug(f"Desired response log prob: {log_prob}")
        
        # 计算 undesired responses 的 log probability 之和
        undesired_log_prob_sum = 0.0
        for response in undesired_responses:
            if response:  # 跳过空响应
                log_prob = self.compute_log_probability(follower_prompt, response)
                undesired_log_prob_sum += log_prob
                logger.debug(f"Undesired response log prob: {log_prob}")
        
        # 计算 reward = desired - undesired
        reward = desired_log_prob_sum - undesired_log_prob_sum
        
        logger.info(f"Reward computed: desired={desired_log_prob_sum:.4f}, "
                   f"undesired={undesired_log_prob_sum:.4f}, reward={reward:.4f}")
        
        return reward
    
    def compute_rewards_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[float]:
        """
        批量计算 rewards
        
        Args:
            requests: 请求列表，每个请求包含：
                - leader_response: leader 的发言
                - follower_prompt_template: follower 的 prompt 模板
                - desired_responses: 期望的响应列表
                - undesired_responses: 不期望的响应列表
        
        Returns:
            List[float]: reward 列表
        """
        rewards = []
        
        for i, req in enumerate(requests):
            try:
                reward = self.compute_reward(
                    leader_response=req.get('leader_response', ''),
                    follower_prompt_template=req.get('follower_prompt_template', []),
                    desired_responses=req.get('desired_responses', []),
                    undesired_responses=req.get('undesired_responses', [])
                )
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Error computing reward for request {i}: {e}")
                rewards.append(DEFAULT_REWARD_VALUE)
        
        return rewards


# 全局 Measurer 实例（在启动时初始化）
measurer: Optional[AvalonRewardMeasurer] = None

# FastAPI 应用
app = FastAPI(title="Avalon Reward Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    logger.info("Reward Server is starting...")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    if measurer is None:
        return {"status": "error", "message": "Measurer not initialized"}
    return {"status": "ok", "model": measurer.config.model_path}


@app.post("/compute_reward")
async def compute_reward_endpoint(data: Dict[str, Any] = Body(...)):
    """
    计算 reward 的 API 接口
    
    请求格式：
    {
        "requests": [
            {
                "leader_response": "...",
                "follower_prompt_template": [...],
                "desired_responses": [...],
                "undesired_responses": [...]
            },
            ...
        ]
    }
    
    响应格式：
    {
        "rewards": [float, ...]
    }
    """
    if measurer is None:
        return {"error": "Measurer not initialized", "rewards": []}
    
    requests_list = data.get('requests', [])
    
    if not requests_list:
        # 单个请求的兼容模式
        requests_list = [data]
    
    try:
        rewards = measurer.compute_rewards_batch(requests_list)
        return {"rewards": rewards}
    except Exception as e:
        logger.error(f"Error in compute_reward: {e}")
        return {"error": str(e), "rewards": [DEFAULT_REWARD_VALUE] * len(requests_list)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Avalon 游戏 GRPO 训练的 Reward Server"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/Qwen2.5-72B-Instruct",
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda 或 cpu）"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="模型精度"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="最大序列长度"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务地址"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = RewardServerConfig(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        port=args.port
    )
    
    # 初始化 Measurer
    global measurer
    measurer = AvalonRewardMeasurer(config)
    
    # 启动服务
    logger.info(f"Starting Reward Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
