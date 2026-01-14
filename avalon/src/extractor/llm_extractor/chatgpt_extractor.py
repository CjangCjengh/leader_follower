#!/usr/bin/env python
# encoding: utf-8
"""
ChatGPT 基于的 Extractor 实现
"""
from typing import List, Tuple, Optional

from ..abs_extractor import Extractor
from ...apis.chatgpt_api import chatgpt


class ChatGPTBasedExtractor(Extractor):
    """基于 ChatGPT 的信息提取器"""
    
    def __init__(self, extractor_name: str, model_name: str, system_prompt: str, extract_prompt: str,
                 temperature: float, few_shot_demos: List[Tuple[str, str]] = None, 
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 output_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.extractor_name = extractor_name
        self.model = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.extract_prompt = extract_prompt
        self.few_shot_demos = few_shot_demos if few_shot_demos else []
        self.log_file = f"{output_dir}/extractor.txt" if output_dir else None
        self.api_key = api_key
        self.api_base = api_base

    def extract(self, message: str) -> str:
        """从消息中提取信息"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for demo in self.few_shot_demos:
            messages.append(demo)
        instruction = self.extract_prompt.format(message)
        messages.append({"role": 'user', "content": instruction})

        output = chatgpt(self.model, messages, self.temperature, 
                        api_key=self.api_key, api_base=self.api_base)
        self.log(instruction, output)
        return output

    def step(self, message: str) -> str:
        """提取步骤"""
        return self.extract(message)
    
    def log(self, input_text: str, output_text: str):
        """记录日志"""
        if self.log_file:
            with open(self.log_file, mode='a+', encoding='utf-8') as f:
                f.write(f"[{self.extractor_name}]\n")
                f.write(f"Input: {input_text}\n")
                f.write(f"Output: {output_text}\n")
                f.write("-" * 50 + "\n")
