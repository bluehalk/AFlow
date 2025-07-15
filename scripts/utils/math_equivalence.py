#!/usr/bin/env python3
"""
数学等价性检测模块
基于 MATH 数据集的等价性检测逻辑
"""

import re
import math
from math import isclose
from typing import Optional, Union


class MathEquivalenceChecker:
    """
    数学等价性检测器
    
    用于检测两个数学表达式是否等价，支持：
    - 分数与小数的转换
    - 百分比处理
    - 平方根计算
    - 度数符号处理
    - 等式简化
    - 格式化和空格处理
    - 数学常数计算（π、e等）
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        初始化数学等价性检测器
        
        Args:
            tolerance: 数值比较的容差，默认为1e-6
        """
        self.tolerance = tolerance
        
        # 支持的答案提取正则表达式模式
        self.answer_patterns = [
            r"Answer is \$\\boxed\{(.*?)\}\$",
            r"The answer is \$\\boxed\{(.*?)\}\$", 
            r"\\boxed\{(.*?)\}",
            r"Answer: \$\\boxed\{(.*?)\}\$",
            r"The answer is: \$\\boxed\{(.*?)\}\$"
        ]
        
        # 数学常数映射
        self.math_constants = {
            'pi': math.pi,
            '\\pi': math.pi,
            'e': math.e,
            '\\e': math.e,
        }
    
    def _fix_fracs(self, string: str) -> str:
        """修复分数格式，将 \frac12 转换为 \frac{1}{2}"""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str
    
    def _fix_a_slash_b(self, string: str) -> str:
        """将 a/b 格式转换为 \frac{a}{b}"""
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string
    
    def _remove_right_units(self, string: str) -> str:
        """移除右侧的单位"""
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string
    
    def _fix_sqrt(self, string: str) -> str:
        """修复平方根格式，将 \sqrt2 转换为 \sqrt{2}"""
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    
    def strip_string(self, string: str) -> str:
        """
        规范化数学表达式字符串
        
        Args:
            string: 原始字符串
            
        Returns:
            规范化后的字符串
        """
        # 移除换行符
        string = string.replace("\n", "")
        
        # 移除反向空格
        string = string.replace("\\!", "")
        
        # 替换双反斜杠
        string = string.replace("\\\\", "\\")
        
        # 替换分数格式
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        
        # 移除 \left 和 \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        
        # 移除度数符号
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        
        # 移除美元符号
        string = string.replace("\\$", "")
        
        # 移除右侧单位
        string = self._remove_right_units(string)
        
        # 处理百分比
        if string.endswith("\\%") or string.endswith("%"):
            if string.endswith("\\%"):
                num_str = string[:-2]
            else:
                num_str = string[:-1]
            
            try:
                num = float(num_str) / 100
                if num == int(num):
                    string = str(int(num))
                else:
                    string = str(num)
            except:
                string = string.replace("\\%", "").replace("%", "")
        else:
            string = string.replace("\\%", "")
            string = string.replace(r"\%", "")
        
        # 移除逗号
        string = string.replace(",", "")
        
        # 处理小数点
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        
        # 处理等式
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
        
        # 修复平方根
        string = self._fix_sqrt(string)
        
        # 移除空格
        string = string.replace(" ", "")
        
        # 修复分数
        string = self._fix_fracs(string)
        
        # 修复 a/b 格式
        string = self._fix_a_slash_b(string)
        
        return string
    
    def _evaluate_symbolic_expression(self, expr: str) -> Optional[float]:
        """
        计算包含数学常数的符号表达式
        
        Args:
            expr: 符号表达式
            
        Returns:
            计算结果，如果无法计算则返回None
        """
        try:
            # 创建一个安全的计算环境
            safe_dict = {
                'pi': math.pi,
                'e': math.e,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'ln': math.log,
                'exp': math.exp,
                'abs': abs,
                'pow': pow,
                '__builtins__': {},
            }
            
            # 替换常见的数学符号
            expr = expr.replace('\\pi', 'pi')
            expr = expr.replace('\\e', 'e')
            
            # 处理隐式乘法，如 3pi -> 3*pi
            expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
            
            # 处理分数
            if '\\frac{' in expr:
                # 简单的分数处理
                frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
                def replace_frac(match):
                    num = match.group(1)
                    den = match.group(2)
                    return f'({num})/({den})'
                expr = re.sub(frac_pattern, replace_frac, expr)
            
            # 处理平方根
            if '\\sqrt{' in expr:
                sqrt_pattern = r'\\sqrt\{([^}]+)\}'
                def replace_sqrt(match):
                    content = match.group(1)
                    return f'sqrt({content})'
                expr = re.sub(sqrt_pattern, replace_sqrt, expr)
            
            # 尝试计算表达式
            result = eval(expr, safe_dict)
            return float(result)
            
        except Exception:
            return None
    
    def _try_parse_number(self, s: str) -> Optional[float]:
        """
        尝试将字符串解析为数字
        
        Args:
            s: 待解析的字符串
            
        Returns:
            解析后的数字，如果无法解析则返回None
        """
        try:
            return float(s)
        except:
            pass
        
        # 尝试计算符号表达式
        symbolic_result = self._evaluate_symbolic_expression(s)
        if symbolic_result is not None:
            return symbolic_result
        
        try:
            # 处理负分数
            if s.startswith("-\\frac{") and s.endswith("}"):
                content = s[7:-1]
                parts = content.split("}{")
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return -numerator / denominator
            # 处理正分数
            elif s.startswith("\\frac{") and s.endswith("}"):
                content = s[6:-1]
                parts = content.split("}{")
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return numerator / denominator
        except:
            pass
        
        try:
            # 处理平方根
            if s.startswith("\\sqrt{") and s.endswith("}"):
                content = s[6:-1]
                num = float(content)
                return num ** 0.5
        except:
            pass
        
        return None
    
    def is_equiv(self, str1: Union[str, None], str2: Union[str, None], verbose: bool = False) -> bool:
        """
        检查两个数学表达式是否等价
        
        Args:
            str1: 第一个表达式
            str2: 第二个表达式
            verbose: 是否输出详细信息
            
        Returns:
            是否等价
        """
        if str1 is None and str2 is None:
            if verbose:
                print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False
        
        try:
            ss1 = self.strip_string(str1)
            ss2 = self.strip_string(str2)
            if verbose:
                print(f"规范化: '{str1}' -> '{ss1}', '{str2}' -> '{ss2}'")
            
            # 首先检查字符串是否相等
            if ss1 == ss2:
                return True
            
            # 尝试数值比较
            try:
                num1 = self._try_parse_number(ss1)
                num2 = self._try_parse_number(ss2)
                
                if verbose and (num1 is not None or num2 is not None):
                    print(f"数值解析: '{ss1}' -> {num1}, '{ss2}' -> {num2}")
                
                if num1 is not None and num2 is not None:
                    return isclose(num1, num2, abs_tol=self.tolerance)
            except:
                pass
            
            return False
        except:
            return str1 == str2
    
    def extract_answer(self, model_response: str) -> Optional[str]:
        """
        从模型输出中提取答案
        
        Args:
            model_response: 模型的完整输出
            
        Returns:
            提取并规范化后的答案，如果无法提取则返回None
        """
        for pattern in self.answer_patterns:
            match = re.search(pattern, model_response)
            if match:
                # 提取boxed中的内容
                answer = match.group(1).replace(",", "")
                # 进行规范化处理
                normalized_answer = self.strip_string(answer)
                return normalized_answer
        
        return None
    
    def is_correct(self, model_completion: str, gt_answer: str) -> bool:
        """
        检查模型输出是否正确
        
        Args:
            model_completion: 模型的完整输出
            gt_answer: 标准答案
            
        Returns:
            是否正确
        """
        try:
            extracted = self.extract_answer(model_completion)
            if extracted is None:
                return False
            return self.is_equiv(extracted, gt_answer)
        except:
            extracted = self.extract_answer(model_completion)
            return extracted == gt_answer if extracted is not None else False


# 为了兼容性，提供函数式接口
_default_checker = MathEquivalenceChecker()

def is_equiv(str1: Union[str, None], str2: Union[str, None], verbose: bool = False) -> bool:
    """检查两个数学表达式是否等价（函数式接口）"""
    return _default_checker.is_equiv(str1, str2, verbose)

def extract_answer(model_response: str) -> Optional[str]:
    """从模型输出中提取答案（函数式接口）"""
    return _default_checker.extract_answer(model_response)

def is_correct(model_completion: str, gt_answer: str) -> bool:
    """检查模型输出是否正确（函数式接口）"""
    return _default_checker.is_correct(model_completion, gt_answer)

def strip_string(string: str) -> str:
    """规范化数学表达式字符串（函数式接口）"""
    return _default_checker.strip_string(string)


# 为了与现有代码兼容，提供别名
judge_symbolic_equality = is_equiv 