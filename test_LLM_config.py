#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 config2.yaml 配置和API连接
"""

import asyncio
import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.async_llm import LLMsConfig, AsyncLLM
from scripts.logs import logger

async def test_api_connection():
    """测试API连接"""
    print("🔍 开始测试API配置...")
    
    try:
        # 1. 测试配置加载
        print("\n1️⃣ 测试配置加载...")
        config = LLMsConfig.default()
        print(f"✅ 配置加载成功")
        print(f"📋 可用模型: {config.get_all_names()}")
        
        # 2. 测试模型配置获取
        print("\n2️⃣ 测试模型配置获取...")
        opt_config = config.get('llama3-70b-8192')
        exec_config = config.get('llama3-70b-8192')
        
        print(f"🔧 优化模型配置:")
        print(f"   - base_url: {opt_config.base_url}")
        print(f"   - model: {opt_config.model}")
        print(f"   - temperature: {opt_config.temperature}")
        print(f"   - api_key: {opt_config.key[:10]}..." if opt_config.key else "   - api_key: None")
        
        print(f"⚡ 执行模型配置:")
        print(f"   - base_url: {exec_config.base_url}")
        print(f"   - model: {exec_config.model}")
        print(f"   - temperature: {exec_config.temperature}")
        print(f"   - api_key: {exec_config.key[:10]}..." if exec_config.key else "   - api_key: None")
        
        # 3. 测试API连接
        print("\n3️⃣ 测试API连接...")
        
        # 测试执行模型
        print("🔍 测试执行模型连接...")
        exec_llm = AsyncLLM('llama3-70b-8192')
        
        test_prompt = "请回答：1 + 1 = ?"
        print(f"📝 发送测试请求: {test_prompt}")
        
        try:
            response = await exec_llm(test_prompt)
            print(f"✅ 执行模型响应成功!")
            print(f"📄 响应内容: {response}")
            # 获取成本信息
            usage_summary = exec_llm.get_usage_summary()
            cost = usage_summary['total_cost']
            print(f"💰 成本: ${cost:.6f}")
        except Exception as e:
            print(f"❌ 执行模型连接失败: {e}")
            return False
        
        # 测试优化模型
        print("\n🔍 测试优化模型连接...")
        opt_llm = AsyncLLM('llama3-70b-8192')
        
        test_prompt = "请简单介绍一下你自己"
        print(f"📝 发送测试请求: {test_prompt}")
        
        try:
            response = await opt_llm(test_prompt)
            print(f"✅ 优化模型响应成功!")
            print(f"📄 响应内容: {response}")
            # 获取成本信息
            usage_summary = opt_llm.get_usage_summary()
            cost = usage_summary['total_cost']
            print(f"💰 成本: ${cost:.6f}")
        except Exception as e:
            print(f"❌ 优化模型连接失败: {e}")
            return False
        
        print("\n🎉 所有测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

async def test_groq_specific():
    """测试Groq API特定配置"""
    print("\n🔧 测试Groq API特定配置...")
    
    try:
        # 测试不同的模型名称
        test_models = [
            "llama3-70b-8192",
        ]
        
        config = LLMsConfig.default()
        
        for model_name in test_models:
            print(f"\n🔍 测试模型: {model_name}")
            try:
                llm_config = config.get(model_name)
                print(f"   ✅ 配置获取成功")
                print(f"   📍 base_url: {llm_config.base_url}")
                print(f"   🤖 model: {llm_config.model}")
                
                # 测试连接
                llm = AsyncLLM(model_name)
                response = await llm("Hello, test message")
                print(f"   ✅ API连接成功")
                # 获取成本信息
                usage_summary = llm.get_usage_summary()
                cost = usage_summary['total_cost']
                print(f"   💰 成本: ${cost:.6f}")
                
            except Exception as e:
                print(f"   ❌ 失败: {e}")
                
    except Exception as e:
        print(f"❌ Groq测试失败: {e}")

def check_config_file():
    """检查配置文件内容"""
    print("📁 检查配置文件...")
    
    config_path = "config/config2.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"✅ 配置文件存在: {config_path}")
        print(f"📄 文件大小: {len(content)} 字符")
        
        # 检查关键配置
        if "llama3-70b-8192" in content:
            print("✅ 找到执行模型配置")
        else:
            print("❌ 未找到执行模型配置")
            
        if "https://api.groq.com/openai/v1" in content:
            print("✅ 找到正确的Groq API URL")
        else:
            print("❌ 未找到正确的Groq API URL")
            
        if "gsk_" in content:
            print("✅ 找到API密钥配置")
        else:
            print("❌ 未找到API密钥配置")
    
    return True

async def main():
    """主测试函数"""
    print("🚀 开始AFlow配置测试")
    print("=" * 50)
    
    # 检查配置文件
    if not check_config_file():
        return
    
    # 测试API连接
    success = await test_api_connection()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 配置测试全部通过！可以运行AFlow实验了")
        print("\n💡 运行命令:")
        print("python run.py --dataset GSM8K \\")
        print("    --opt_model_name 'llama3-70b-8192' \\")
        print("    --exec_model_name 'llama3-70b-8192' \\")
        print("    --sample 2 --max_rounds 2 --validation_rounds 1")
    else:
        print("\n" + "=" * 50)
        print("❌ 配置测试失败，请检查以下问题:")
        print("1. API密钥是否正确")
        print("2. 网络连接是否正常")
        print("3. Groq API服务是否可用")
        print("4. 模型名称是否正确")
        
        # 额外测试
        await test_groq_specific()

if __name__ == "__main__":
    asyncio.run(main()) 