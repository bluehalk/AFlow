#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from scripts.async_llm import LLMsConfig, create_llm_instance

async def test_config():
    """测试配置是否正确加载extra_headers"""
    
    print("🔧 测试配置加载...")
    
    # 加载默认配置
    config = LLMsConfig.default()
    
    # 获取meta-llama/llama-3.3-70b-instruct:free配置
    llm_config = config.get('meta-llama/llama-3.3-70b-instruct:free')
    
    print(f"模型: {llm_config.model}")
    print(f"Base URL: {llm_config.base_url}")
    print(f"API Key: {llm_config.key[:20]}..." if llm_config.key else "No API Key")
    print(f"Extra Headers: {llm_config.extra_headers}")
    print(f"Temperature: {llm_config.temperature}")
    
    # 创建LLM实例
    print("\n🚀 创建LLM实例...")
    llm = create_llm_instance(llm_config)
    
    print(f"LLM配置的extra_headers: {llm.config.extra_headers}")
    
    # 测试一个简单的调用
    print("\n💬 测试简单调用...")
    try:
        result = await llm("What is 2+2? Answer only with the number.")
        print(f"✅ 测试成功!")
        print(f"响应: {result}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    return llm.get_usage_summary()

if __name__ == "__main__":
    summary = asyncio.run(test_config())
    print(f"\n📊 使用摘要: {summary}") 