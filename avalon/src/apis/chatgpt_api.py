#!/usr/bin/env python
# encoding: utf-8
"""
ChatGPT API 封装
支持自定义 api_key 和 api_base
"""
import time
import warnings
from typing import List, Optional

import openai
from openai import OpenAI


def chatgpt(model: str, messages: List[dict], temperature: float, 
            api_key: Optional[str] = None, api_base: Optional[str] = None) -> str:
    """
    调用 ChatGPT API
    
    Args:
        model: 模型名称
        messages: 消息列表
        temperature: 温度参数
        api_key: API Key，如果为 None 则使用全局配置
        api_base: API Base URL，如果为 None 则使用默认 OpenAI API
    
    Returns:
        模型输出的文本
    """
    # 优先使用传入的参数，否则使用全局配置
    _api_key = api_key or openai.api_key
    _api_base = api_base or getattr(openai, 'base_url', None)
    
    if _api_base:
        client = OpenAI(api_key=_api_key, base_url=_api_base)
    else:
        client = OpenAI(api_key=_api_key)
    
    retry = 0
    max_retry = 10
    flag = False
    out = ''
    
    while retry < max_retry and not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )
            out = response.choices[0].message.content
            flag = True
        except openai.APIStatusError as e:
            if e.message == "Error code: 307":
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                time.sleep(1)
                continue
            else:
                if retry < max_retry:
                    retry += 1
                    warnings.warn(f"{e} retry:{retry}")
                    time.sleep(2)
                    continue
                else:
                    raise e
        except openai.RateLimitError as e:
            retry += 1
            warnings.warn(f"Rate limit error: {e}, retry:{retry}")
            time.sleep(5)
            continue
        except Exception as e:
            if retry < max_retry:
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                time.sleep(2)
                continue
            else:
                raise e
    
    client.close()
    return out
