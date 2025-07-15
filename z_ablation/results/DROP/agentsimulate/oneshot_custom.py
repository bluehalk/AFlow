#!/usr/bin/env python3
"""
DROP数据集一次性激活自定义操作器
"""

import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.async_llm import AsyncLLM
from z_ablation.drop_one_shot_prompt import DROP_ONE_SHOT_PROMPT, DROP_SIMPLIFIED_ONE_SHOT_PROMPT


class OneshotCustom:
    """DROP数据集一次性激活自定义操作器"""
    
    def __init__(self, llm: AsyncLLM, use_simplified: bool = False):
        self.llm = llm
        self.prompt_template = DROP_SIMPLIFIED_ONE_SHOT_PROMPT if use_simplified else DROP_ONE_SHOT_PROMPT
    
    async def __call__(self, input: str, instruction: str = ""):
        """
        使用一次性激活prompt处理问题
        
        Args:
            input: 输入问题/上下文
            instruction: 指令（为了兼容接口，实际不使用）
        
        Returns:
            dict: {'response': '提取的答案'}
        """
        # 格式化prompt
        formatted_prompt = self.prompt_template.format(input=input)
        
        # 调用LLM
        response = await self.llm(formatted_prompt)
        
        # 提取最终答案
        final_answer = self._extract_final_answer(response)
        
        return {'response': final_answer}
    
    def _extract_final_answer(self, response: str) -> str:
        """从响应中提取<ans>标签包裹的最终答案"""
        try:
            # 首先尝试提取<ans>标签中的内容
            ans_pattern = r'<ans>(.*?)</ans>'
            ans_matches = re.findall(ans_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if ans_matches:
                # 取最后一个匹配的答案
                answer = ans_matches[-1].strip()
                # 清理答案格式
                answer = self._clean_answer(answer)
                return answer
            
            # 如果没有找到<ans>标签，尝试其他方法
            # 查找 "Final Answer" 部分
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if 'final answer' in line.lower():
                    # 查找答案内容
                    for j in range(i + 1, min(i + 5, len(lines))):
                        candidate = lines[j].strip()
                        if candidate and not candidate.startswith('#') and not candidate.startswith('*'):
                            # 尝试提取<ans>标签
                            ans_in_line = re.search(ans_pattern, candidate, re.IGNORECASE)
                            if ans_in_line:
                                return self._clean_answer(ans_in_line.group(1))
                            # 如果没有标签，直接返回清理后的内容
                            return self._clean_answer(candidate)
            
            # 最后尝试从结尾提取
            for line in reversed(lines[-10:]):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('*') and not line.startswith('```'):
                    # 检查是否包含<ans>标签
                    ans_in_line = re.search(ans_pattern, line, re.IGNORECASE)
                    if ans_in_line:
                        return self._clean_answer(ans_in_line.group(1))
                    # 如果看起来像一个简单的答案，返回它
                    if len(line) < 100 and not line.endswith(':'):
                        return self._clean_answer(line)
            
            return "Could not extract answer"
            
        except Exception as e:
            return f"Extraction error: {str(e)}"
    
    def _clean_answer(self, answer: str) -> str:
        """清理答案格式"""
        if not answer:
            return "Could not extract answer"
        
        # 移除markdown格式
        answer = answer.replace('```', '').replace('**', '').replace('*', '')
        
        # 移除多余的空格和换行
        answer = ' '.join(answer.split())
        
        # 移除常见的前缀
        prefixes_to_remove = [
            'the answer is',
            'answer:',
            'final answer:',
            'result:',
            'therefore,',
            'so,',
            'thus,',
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # 移除引号
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1]
        
        return answer.strip() 