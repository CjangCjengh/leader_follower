#!/usr/bin/env python
# encoding: utf-8
"""
LLM Agent 实现
支持四种类型：
- DirectAgent: 直接生成回复（1次API调用）
- ReActAgent: ReAct 框架（思考-行动循环）
- ReConAgent: 关系一致性框架（跨模态/玩家关系分析）
- LASIAgent: LASI 框架（局势分析-计划-行动-回复，4次API调用）
"""
import json
import re
import time
from typing import List, Optional, Callable

import openai

from ..abs_agent import Agent
from ..utils import write_json
from ...apis.chatgpt_api import chatgpt

try:
    OPENAI_MAX_TOKENS_ERROR = openai.error.InvalidRequestError
except AttributeError:
    OPENAI_MAX_TOKENS_ERROR = openai.BadRequestError


def extract_response(output: str) -> str:
    """
    从 LLM 输出中提取响应内容
    支持多种格式：
    1. <response>...</response> (完整闭合标签)
    2. <response>... (未闭合标签，输出被截断的情况)
    3. my response is <...>
    4. 直接返回原文（如果没有匹配）
    """
    # 首先尝试完整的 <response>...</response> 标签
    match = re.search(r"(?<=<response>).*?(?=</response>)", output, re.S)
    if match:
        return match.group().strip()
    
    # 如果没有闭合标签，尝试提取 <response> 之后的所有内容（处理输出被截断的情况）
    match = re.search(r"<response>(.+)", output, re.S)
    if match:
        return match.group(1).strip()
    
    # 尝试 my response is <...> 格式
    match = re.search(r"my response is\s*<(.+?)>", output, re.S | re.I)
    if match:
        return match.group(1).strip()
    
    # 尝试 my response is ... 格式（不带尖括号）
    match = re.search(r"my response is\s*[:\-]?\s*(.+)", output, re.S | re.I)
    if match:
        return match.group(1).strip()
    
    # 如果都没匹配，返回原文
    return output.strip()


class BaseAvalonAgent(Agent):
    """
    Avalon 游戏 Agent 基类
    提供通用的夜晚信息管理、对话历史管理等功能
    """
    
    # 意图识别提示词模板
    INTENT_IDENTIFICATION_PROMPT = """You are {name}, playing as {role} in the game of Avalon.
Your goal: {goal}

Current game state and dialogue history:
{context}

The next player to speak after you is {next_player}.

Based on your role and the current situation, identify:
1. THREE responses you would DESIRE the next player ({next_player}) to say (responses that would benefit your goals)
2. THREE responses you would NOT DESIRE the next player ({next_player}) to say (responses that would harm your goals)

Think strategically about what the next player might say and how it could affect the game.

Output your response in the following format:
<desired_responses>
1. [First desired response from the next player]
2. [Second desired response from the next player]
3. [Third desired response from the next player]
</desired_responses>
<undesired_responses>
1. [First undesired response from the next player]
2. [Second undesired response from the next player]
3. [Third undesired response from the next player]
</undesired_responses>"""

    def __init__(self, name: str, role: str, role_intro: str, game_goal: str, 
                 strategy: str, system_prompt: str, model: str, temperature: float,
                 api_key: str, output_dir: str, api_base: Optional[str] = None,
                 thinking_callback: Optional[Callable[[str, str], None]] = None,
                 enable_intent_identification: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.role = role
        self.introduction = role_intro
        self.game_goal = game_goal
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.output_dir = output_dir
        
        # 夜晚阶段获得的信息，会合并到系统提示中
        self.night_info = ""
        
        # 简化的对话历史记录
        self.conversation_history = []
        
        # 当前游戏阶段
        self.phase = 0
        
        # 思考过程回调函数（用于 watch 模式显示中间过程）
        self.thinking_callback = thinking_callback
        
        # 是否启用意图识别（论文中的Intent Identification）
        self.enable_intent_identification = enable_intent_identification
        
        # 存储最近一次的意图识别结果
        self.last_intent = None

    def get_system_prompt_with_night_info(self) -> str:
        """获取包含夜晚信息的系统提示"""
        if self.night_info:
            return f"{self.system_prompt}\n\n{self.night_info}"
        return self.system_prompt

    def get_conversation_context(self) -> str:
        """获取当前对话历史的简要上下文"""
        if not self.conversation_history:
            return "None"
        recent = self.conversation_history[-20:]
        return "\n".join([f"{item['name']}: {item['message']}" for item in recent])

    def send_messages(self, messages: List[dict]) -> str:
        """发送消息到 LLM"""
        output = chatgpt(self.model, messages, self.temperature, 
                        api_key=self.api_key, api_base=self.api_base)
        return output

    def receive(self, name: str, message: str) -> None:
        """接收来自其他玩家的消息"""
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]
        self.conversation_history.append({"name": name, "message": message})

    def emit_thinking(self, stage: str, content: str):
        """发出思考过程（用于 watch 模式显示）"""
        if self.thinking_callback:
            self.thinking_callback(stage, content)

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        """游戏结束后的反思（默认不做任何操作）"""
        pass

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)

    def identify_intent(self, next_player: str) -> dict:
        """
        意图识别：识别希望和不希望后置位玩家说的内容
        
        基于论文中的 Intent Identification 方法：
        - 识别 K 个期望的响应（对当前玩家有利）
        - 识别 K 个不期望的响应（对当前玩家不利）
        
        Args:
            next_player: 下一个发言的玩家名称
            
        Returns:
            dict: 包含 desired_responses 和 undesired_responses 的字典
        """
        if not self.enable_intent_identification:
            return None
        
        context = self.get_conversation_context()
        
        prompt = self.INTENT_IDENTIFICATION_PROMPT.format(
            name=self.name,
            role=self.role,
            goal=self.game_goal,
            context=context,
            next_player=next_player
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        
        output = self.send_messages(messages)
        
        # 解析期望的响应
        desired_responses = []
        desired_match = re.search(r"<desired_responses>(.+?)</desired_responses>", output, re.S)
        if desired_match:
            desired_text = desired_match.group(1)
            # 提取每行的响应
            for line in desired_text.strip().split('\n'):
                line = line.strip()
                if line:
                    # 移除序号前缀（如 "1. ", "2. " 等）
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        desired_responses.append(cleaned)
        
        # 解析不期望的响应
        undesired_responses = []
        undesired_match = re.search(r"<undesired_responses>(.+?)</undesired_responses>", output, re.S)
        if undesired_match:
            undesired_text = undesired_match.group(1)
            for line in undesired_text.strip().split('\n'):
                line = line.strip()
                if line:
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        undesired_responses.append(cleaned)
        
        # 限制为前3个（K=3）
        desired_responses = desired_responses[:3]
        undesired_responses = undesired_responses[:3]
        
        intent_result = {
            "desired_responses": desired_responses,
            "undesired_responses": undesired_responses
        }
        
        # 保存到日志
        self.log(f"{self.output_dir}/intent_identification.txt",
                 f"phase:{self.phase}\nnext_player:{next_player}\ninput:{prompt}\noutput:\n{output}\n" +
                 f"parsed_intent:{json.dumps(intent_result, ensure_ascii=False)}\n--------------------\n")
        
        # 显示思考过程
        self.emit_thinking("Intent Identification", 
                          f"Desired: {desired_responses}\nUndesired: {undesired_responses}")
        
        self.last_intent = intent_result
        return intent_result
    
    def get_last_intent(self) -> dict:
        """获取最近一次的意图识别结果"""
        return self.last_intent


class DirectAgent(BaseAvalonAgent):
    """
    DirectAgent: 直接生成回复
    最简单的实现，仅 1 次 API 调用
    """

    def __init__(self, response_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # 直接生成回复
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role, 
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan="None", question=message, actions="None"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        
        # 记录对话历史
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class ReActAgent(BaseAvalonAgent):
    """
    ReActAgent: ReAct 框架（Reasoning + Acting）
    思考-行动循环，2 次 API 调用
    """

    def __init__(self, response_prompt: str, 
                 react_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt
        # ReAct 思考提示
        self.react_prompt = react_prompt or """You are {name}, playing as {role} in Avalon.
Current phase: {phase}
Your goal: {goal}
Your strategy: {strategy}

Conversation history:
{summary}

Current question/situation: {question}

Think step by step about:
1. Thought: What do I observe? What does this mean?
2. Action: What should I do? (options: speak supportively, speak suspiciously, deflect, accuse, defend, stay neutral)
3. Observation: What response might this action get?

Provide your reasoning in this format:
<thinking>
Thought: [your analysis]
Action: [your chosen action]
Observation: [expected outcome]
</thinking>"""

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # Step 1: ReAct 思考
        t_think_start = time.time()
        react_prompt = self.react_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            goal=self.game_goal, strategy=self.strategy,
            summary=context, question=message
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": react_prompt}
        ]
        
        thinking = self.send_messages(messages)
        t_think = time.time() - t_think_start
        
        # 提取思考内容并显示
        think_match = re.search("(?<=<thinking>).*?(?=</thinking>)", thinking, re.S)
        think_content = think_match.group().strip() if think_match else thinking
        self.emit_thinking("ReAct Thinking", think_content)
        
        self.log(f"{self.output_dir}/react_thinking.txt",
                 f"phase:{self.phase}\ninput:{react_prompt}\noutput:\n{thinking}\n--------------------")

        # Step 2: 生成回复
        t_response_start = time.time()
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan=think_content, question=message, actions="Based on ReAct thinking"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        t_response = time.time() - t_response_start
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Think: {t_think}\nResponse: {t_response}\n")
        
        # 记录对话历史
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class ReConAgent(BaseAvalonAgent):
    """
    ReConAgent: 关系一致性框架（Relation Consistency）
    基于跨玩家关系分析，3 次 API 调用
    
    核心思想：
    1. 分析玩家间的关系一致性（谁支持谁、谁怀疑谁）
    2. 通过关系网络识别可能的阵营
    3. 基于关系分析做出决策
    """

    def __init__(self, response_prompt: str,
                 relation_prompt: str = None,
                 consistency_prompt: str = None, **kwargs):
        super().__init__(**kwargs)
        self.response_prompt = response_prompt
        
        # 关系分析提示
        self.relation_prompt = relation_prompt or """You are {name}, playing as {role} in Avalon.
Current phase: {phase}

Analyze the relationships between players based on the conversation:
{summary}

For each player pair, identify:
1. Support relationships (who defended/agreed with whom)
2. Suspicion relationships (who accused/doubted whom)
3. Neutral relationships

Output in format:
<relations>
- [Player A] -> [Player B]: [support/suspicion/neutral] (reason: [brief reason])
...
</relations>"""

        # 一致性分析提示
        self.consistency_prompt = consistency_prompt or """Based on the relationship analysis:
{relations}

Now check for consistency:
1. Cross-modal consistency: Do the stated intentions match the actual behaviors?
2. Intra-modal consistency: Are there contradictions in a player's statements?
3. Camp prediction: Based on relationship patterns, which players likely belong to Evil?

Your role: {role}
Your goal: {goal}

<analysis>
Consistency check: [your analysis]
Suspected evil players: [list with confidence]
Recommended action: [what to do]
</analysis>"""

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        context = self.get_conversation_context()
        
        # Step 1: 关系分析
        t_relation_start = time.time()
        relation_prompt = self.relation_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            summary=context
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": relation_prompt}
        ]
        
        relations = self.send_messages(messages)
        t_relation = time.time() - t_relation_start
        
        # 提取关系并显示
        rel_match = re.search("(?<=<relations>).*?(?=</relations>)", relations, re.S)
        rel_content = rel_match.group().strip() if rel_match else relations
        self.emit_thinking("Relation Analysis", rel_content)
        
        self.log(f"{self.output_dir}/relation_analysis.txt",
                 f"phase:{self.phase}\noutput:\n{relations}\n--------------------")

        # Step 2: 一致性分析
        t_consist_start = time.time()
        consist_prompt = self.consistency_prompt.format(
            relations=rel_content, role=self.role, goal=self.game_goal
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": consist_prompt}
        ]
        
        analysis = self.send_messages(messages)
        t_consist = time.time() - t_consist_start
        
        # 提取分析并显示
        analysis_match = re.search("(?<=<analysis>).*?(?=</analysis>)", analysis, re.S)
        analysis_content = analysis_match.group().strip() if analysis_match else analysis
        self.emit_thinking("Consistency Analysis", analysis_content)
        
        self.log(f"{self.output_dir}/consistency_analysis.txt",
                 f"phase:{self.phase}\noutput:\n{analysis}\n--------------------")

        # Step 3: 生成回复
        t_response_start = time.time()
        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan=analysis_content, question=message, 
            actions=f"Based on relation analysis"
        )
        
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)
        t_response = time.time() - t_response_start
        
        response = extract_response(output)
        
        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\noutput:\n{output}\n--------------------")
        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Relation: {t_relation}\nConsistency: {t_consist}\nResponse: {t_response}\n")
        
        # 记录对话历史
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})
        
        return response


class LASIAgent(BaseAvalonAgent):
    """
    LASIAgent: LASI 框架（Landscape Analysis - Strategy - Implementation）
    原 SAPAR 的完整流程，4 次 API 调用
    
    流程：
    1. Analysis: 局势分析
    2. Plan: 制定计划
    3. Action: 选择行动
    4. Response: 生成回复
    """

    def __init__(self, analysis_prompt: str, plan_prompt: str, 
                 action_prompt: str, response_prompt: str,
                 suggestion_prompt: str, strategy_prompt: str, update_prompt: str,
                 suggestion: str, other_strategy: str, candidate_actions: list,
                 use_analysis: bool = True, use_plan: bool = True, use_action: bool = True,
                 reflection_other: bool = True, improve_strategy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.analysis_prompt = analysis_prompt
        self.plan_prompt = plan_prompt
        self.action_prompt = action_prompt
        self.response_prompt = response_prompt
        self.suggestion_prompt = suggestion_prompt
        self.strategy_prompt = strategy_prompt
        self.update_prompt = update_prompt
        self.previous_suggestion = suggestion
        self.previous_other_strategy = other_strategy
        self.candidate_actions = candidate_actions
        
        self.use_analysis = use_analysis
        self.use_plan = use_plan
        self.use_action = use_action
        self.reflection_other = reflection_other
        self.improve_strategy = improve_strategy
        
        self.plan = {}

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        pattern = r"\d+"
        matches = re.findall(pattern, temp_phase)
        phase = matches[-1] if matches else "0"
        
        context = self.get_conversation_context()
        
        # Step 1: Analysis
        t_analysis_start = time.time()
        if self.use_analysis and context != "None":
            analysis = self.make_analysis(phase, context)
            self.emit_thinking("Analysis", analysis)
        else:
            analysis = "None"
        t_analysis = time.time() - t_analysis_start
        
        # Step 2: Plan
        t_plan_start = time.time()
        if self.use_plan:
            format_plan = self.make_plan(phase, context, analysis)
            self.emit_thinking("Plan", format_plan)
        else:
            format_plan = "None"
        t_plan = time.time() - t_plan_start

        # Step 3: Action
        t_action_start = time.time()
        if self.use_action:
            action = self.make_action(phase, context, format_plan, analysis, message)
            action_str = str(action) if action else "None"
            self.emit_thinking("Action", action_str)
        else:
            action = None
        t_action = time.time() - t_action_start

        # Step 4: Response
        t_response_start = time.time()
        response = self.make_response(phase, context, format_plan, action, message)
        t_response = time.time() - t_response_start
        
        # 记录对话历史
        self.conversation_history.append({"name": "Host", "message": message})
        self.conversation_history.append({"name": self.name, "message": response})

        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Analysis: {t_analysis}\nPlan: {t_plan}\nAction: {t_action}\nResponse: {t_response}\n")
        return response

    def make_analysis(self, phase, context):
        prompt = self.analysis_prompt.format(
            name=self.name, phase=self.phase, role=self.role, summary=context
        )
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/step_reflection.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        return output

    def make_plan(self, phase, context, analysis):
        if self.plan:
            format_previous_plan = '\n'.join(
                [
                    f"Quest Phase Turn {i}: {self.plan.get(str(i), 'None')}" if i != 0 else f"Reveal Phase: {self.plan.get(str(i), 'None')}"
                    for i in range(int(phase) + 1)]
            )
        else:
            format_previous_plan = "None"

        following_format = '\n'.join(
            [f"Quest Phase Turn {i}: <your_plan_{i}>" if
             i != 0 else f"Reveal Phase: <your_plan_0>" for i in range(int(phase), 6)]
        )
        prompt = self.plan_prompt.format(
            name=self.name, phase=self.phase, role=self.role, introduction=self.introduction, goal=self.game_goal,
            strategy=self.strategy,
            previous_plan=format_previous_plan, summary=context, analysis=analysis, plan=following_format)
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        match = re.search('<plan>(.*?)</plan>', output, re.S)
        format_plans = match.group().split('\n') if match else output.split('\n')
        plans = [f.split(":", 1) for f in format_plans]
        dict_plans = {}
        for plan in plans:
            if len(plan) == 2:
                match = re.search(r'\d+', plan[0])
                c_phase = match.group() if match else None
                if c_phase is None and plan[0].lower().startswith("reveal phase"):
                    c_phase = "0"
                c_plan = plan[1].strip()
                if c_phase:
                    dict_plans[c_phase] = c_plan
        self.plan.update(dict_plans)
        self.log(f"{self.output_dir}/plan.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        format_plan = '\n'.join([
            f"Quest Phase Round {str(c_phase)}:{self.plan.get(str(c_phase))}" if str(
                c_phase) != "0" else f"Reveal Phase: {self.plan.get(str(c_phase))}"
            for c_phase in range(int(phase), 6)])
        return format_plan

    def make_action(self, phase, context, format_plan, analysis, message):
        prompt = self.action_prompt.format(name=self.name, phase=self.phase, role=self.role,
                                           introduction=self.introduction, goal=self.game_goal,
                                           strategy=self.strategy, candidate_actions=self.candidate_actions,
                                           summary=context, analysis=analysis, plan=format_plan,
                                           question=message)

        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/actions.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        actions = re.findall("(?<=<actions>).*?(?=</actions>)", output, re.S)
        if not actions:
            actions = re.findall("(?<=<output>).*?(?=</output>)", output, re.S)
            if not actions:
                return output
        return actions

    def make_response(self, phase, context, format_plan, actions, message):
        if self.use_action:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=context, plan=format_plan, question=message, actions=actions)
        else:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=context, plan=format_plan, question=message, actions="None")
        messages = [
            {"role": 'system', "content": self.get_system_prompt_with_night_info()},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_messages(messages)

        self.log(f"{self.output_dir}/response.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        response = extract_response(output)
        return response

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        """游戏结束后的反思"""
        p_r_mapping = '\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()])
        context = self.get_conversation_context()
        
        if self.reflection_other:
            prompt = self.strategy_prompt.format(
                name=self.name, roles=p_r_mapping, summaries=context, strategies=self.previous_other_strategy
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            role_strategy = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{role_strategy}\n--------------------")
        else:
            role_strategy = "None"

        if self.improve_strategy:
            prompt = self.suggestion_prompt.format(
                name=self.name, role=self.role, roles=p_r_mapping, summaries=context, goal=self.game_goal,
                strategy=self.strategy, previous_suggestions=self.previous_suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            suggestion = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{suggestion}\n--------------------")

            prompt = self.update_prompt.format(
                name=self.name, role=self.role, strategy=self.strategy, suggestions=suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_messages(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{output}\n--------------------")
            match = re.search("(?<=<strategy>).*?(?=</strategy>)", output)
            strategy = match.group() if match else output
        else:
            suggestion = "None"
            strategy = self.strategy

        write_json(
            data={"strategy": strategy, "suggestion": suggestion, "other_strategy": role_strategy},
            path=file_name
        )
