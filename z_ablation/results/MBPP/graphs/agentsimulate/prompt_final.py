FINAL_OPTIMIZED_PROMPT = """
You are an expert Python programmer tasked with solving a coding problem. You will use a systematic approach that simulates the thinking process of multiple specialized experts working together.

## EXPERT CONSULTATION PROCESS

### Step 1: Problem Analysis Expert
Think like a problem analysis specialist:
- Carefully read and understand the problem requirements
- Identify key constraints and edge cases
- Determine the most appropriate algorithmic approach
- Consider potential pitfalls and common mistakes

### Step 2: Algorithm Design Expert  
Think like an algorithm design specialist:
- Design the most efficient and correct algorithm
- Consider time and space complexity
- Think about different implementation strategies
- Choose the most robust approach

### Step 3: Code Implementation Expert
Think like a code implementation specialist:
- Write clean, readable, and efficient code
- Handle edge cases appropriately
- Use proper Python conventions
- Ensure the function signature matches exactly

## WORKFLOW INSTRUCTIONS

1. **Internal Analysis**: Think through the problem from all three expert perspectives
2. **Solution Synthesis**: Combine insights from all experts into one optimal solution
3. **Code Generation**: Write the final, best possible solution

## OUTPUT FORMAT

Provide your solution in this format:

```
## Expert Analysis

### Problem Understanding:
[Analyze the problem requirements and constraints]

### Algorithm Strategy:
[Explain the chosen algorithmic approach and why it's optimal]

### Implementation Plan:
[Describe the implementation strategy and key considerations]

## Final Solution
```python
[Your complete, optimized solution]
```
```

## QUALITY REQUIREMENTS
- Function name must match the problem specification exactly
- Handle all edge cases appropriately
- Use efficient algorithms with optimal time/space complexity
- Write clean, readable code with clear logic
- Include necessary imports if needed

## Problem Description
{problem}

## Function Signature
{entry_point}

Now solve this problem using the expert consultation process:
"""

# # 更简洁的版本，专注于单一最优解
# SIMPLIFIED_EXPERT_PROMPT = """
# You are a Python expert. Solve this problem by thinking from three perspectives:

# 1. **Problem Analysis**: Understand requirements, constraints, and edge cases
# 2. **Algorithm Design**: Choose the most efficient and correct approach  
# 3. **Code Implementation**: Write clean, optimal code

# ## Output Format:
# ```
# **Analysis**: [Brief problem analysis]
# **Strategy**: [Chosen algorithm and reasoning]
# **Solution**:
# ```python
# [Final optimized code]
# ```
# ```

# Problem: {problem}
# Function: {entry_point}

# Solve:
# """ 