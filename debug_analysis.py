#!/usr/bin/env python3
"""
调试分析数学关系
"""

def debug_math_relationship():
    print("=== 调试数学关系分析 ===\n")
    
    # 从分析结果中得到的数据
    total_questions = 486
    
    # 成功案例
    success_A_Mathemation = 263
    success_A_Programmer = 267
    success_intersection = 185
    success_union = 345
    
    # 失败案例
    fail_A_Mathemation = 223
    fail_A_Programmer = 219
    fail_intersection = 141
    fail_union = 301
    
    print("📊 成功案例分析:")
    print(f"A_Mathemation成功: {success_A_Mathemation}")
    print(f"A_Programmer成功: {success_A_Programmer}")
    print(f"成功交集: {success_intersection}")
    print(f"成功并集: {success_union}")
    
    print(f"\n📊 失败案例分析:")
    print(f"A_Mathemation失败: {fail_A_Mathemation}")
    print(f"A_Programmer失败: {fail_A_Programmer}")
    print(f"失败交集: {fail_intersection}")
    print(f"失败并集: {fail_union}")
    
    print(f"\n🔍 验证单个方法的成功+失败:")
    print(f"A_Mathemation: {success_A_Mathemation} + {fail_A_Mathemation} = {success_A_Mathemation + fail_A_Mathemation}")
    print(f"A_Programmer: {success_A_Programmer} + {fail_A_Programmer} = {success_A_Programmer + fail_A_Programmer}")
    
    print(f"\n🔍 验证集合运算关系:")
    
    # 验证并集公式: |A ∪ B| = |A| + |B| - |A ∩ B|
    expected_success_union = success_A_Mathemation + success_A_Programmer - success_intersection
    expected_fail_union = fail_A_Mathemation + fail_A_Programmer - fail_intersection
    
    print(f"成功并集验证: {success_A_Mathemation} + {success_A_Programmer} - {success_intersection} = {expected_success_union}")
    print(f"实际成功并集: {success_union}")
    print(f"成功并集验证: {'✅' if expected_success_union == success_union else '❌'}")
    
    print(f"\n失败并集验证: {fail_A_Mathemation} + {fail_A_Programmer} - {fail_intersection} = {expected_fail_union}")
    print(f"实际失败并集: {fail_union}")
    print(f"失败并集验证: {'✅' if expected_fail_union == fail_union else '❌'}")
    
    print(f"\n🔍 理论关系验证:")
    print(f"成功交集 + 失败交集应该等于总数？")
    print(f"{success_intersection} + {fail_intersection} = {success_intersection + fail_intersection}")
    print(f"总数: {total_questions}")
    print(f"{'✅ 正确' if success_intersection + fail_intersection == total_questions else '❌ 错误'}")
    
    print(f"\n🤔 发现问题：")
    print(f"成功交集 + 失败交集 = {success_intersection + fail_intersection} ≠ {total_questions}")
    print(f"差值: {total_questions - (success_intersection + fail_intersection)}")
    
    print(f"\n💡 正确的关系应该是:")
    print(f"每个问题要么两者都成功，要么至少一个失败")
    print(f"成功交集 + (总数 - 成功交集) = 总数")
    print(f"其中 (总数 - 成功交集) 就是至少一个失败的情况")
    
    at_least_one_fail = total_questions - success_intersection
    print(f"至少一个失败的情况数: {total_questions} - {success_intersection} = {at_least_one_fail}")
    
    print(f"\n🔢 重新分析失败情况:")
    print(f"两者都失败: {fail_intersection}")
    print(f"仅A_Mathemation失败: {fail_A_Mathemation - fail_intersection}")
    print(f"仅A_Programmer失败: {fail_A_Programmer - fail_intersection}")
    print(f"至少一个失败总数: {fail_intersection + (fail_A_Mathemation - fail_intersection) + (fail_A_Programmer - fail_intersection)}")
    
    calculated_at_least_one_fail = fail_intersection + (fail_A_Mathemation - fail_intersection) + (fail_A_Programmer - fail_intersection)
    print(f"计算的至少一个失败: {calculated_at_least_one_fail}")
    print(f"理论的至少一个失败: {at_least_one_fail}")
    print(f"验证: {'✅' if calculated_at_least_one_fail == at_least_one_fail else '❌'}")

if __name__ == "__main__":
    debug_math_relationship() 