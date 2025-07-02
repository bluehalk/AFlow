#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Groq API支持的模型名称
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.async_llm import AsyncLLM, LLMConfig

# 常见的Groq Llama模型名称
GROQ_MODELS = [
    "llama3-70b-8192",
    # "llama-3.1-70b-versatile", 
    # "llama3-8b-8192",
    # "llama-3.1-8b-instant",
    # "llama-3.1-405b-reasoning"
]

async def test_model(model_name: str, api_key: str) -> bool:
    """测试指定模型是否可用"""
    try:
        config = LLMConfig({
            "model": model_name,
            "temperature": 0,
            "key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "top_p": 1
        })
        
        llm = AsyncLLM(config)
        
        # 发送简单测试请求
        response = await llm("计算 2+2 等于多少？请只回答数字。")
        
        print(f"✅ {model_name}: 可用 - 响应: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name}: 不可用 - 错误: {str(e)}")
        return False

async def find_working_model():
    """找到可用的模型"""
    # 从配置文件读取API key
    try:
        import yaml
        with open("config/config2.yaml", 'r') as f:
            config_data = yaml.safe_load(f)
            api_key = config_data['models']['llama3-70b-8192']['api_key']
    except Exception as e:
        print(f"❌ 无法读取API key: {e}")
        return
    
    print("🔍 测试Groq API支持的模型...")
    print("=" * 50)
    
    working_models = []
    
    for model in GROQ_MODELS:
        print(f"测试模型: {model}")
        is_working = await test_model(model, api_key)
        if is_working:
            working_models.append(model)
        print()
        
        # 添加延迟避免API限制
        await asyncio.sleep(1)
    
    print("=" * 50)
    if working_models:
        print("🎉 可用的模型:")
        for model in working_models:
            print(f"  ✅ {model}")
        
        print(f"\n💡 建议修改 config/config2.yaml 使用以下模型名称:")
        print(f"'{working_models[0]}'")
        
    else:
        print("❌ 没有找到可用的模型")
        print("请检查:")
        print("1. API key是否正确")
        print("2. 网络连接是否正常")
        print("3. Groq API服务是否可用")

if __name__ == "__main__":
    asyncio.run(find_working_model()) 