ONE_SHOT_PROMPT = """
You are a professional Python coding expert. You need to solve the given programming problem through a comprehensive thinking process that simulates multi-solution generation and consistency checking.

## Workflow Instructions
Follow these steps in your thinking and implementation:

### Step 1: Multi-Solution Generation
Generate 3 different solutions, each should:
- Carefully analyze the problem requirements
- Consider different implementation approaches
- Include necessary imports and clear variable names
- Add appropriate comments for clarity

### Step 2: Solution Consistency Check & Final Selection
Compare these 3 solutions to find the most reliable approach:
- Analyze which method appears most frequently or robust
- Evaluate correctness and efficiency of each approach
- Select the best solution and make final optimizations
- Ensure function signature matches requirements exactly

## Output Format Requirements
Please output your thinking process and final code in the following format:

```
## Solution Generation Phase

### Solution A:
[Describe the first approach]
```python
[Code for Solution A]
```

### Solution B:
[Describe the second approach]
```python
[Code for Solution B]
```

### Solution C:
[Describe the third approach]
```python
[Code for Solution C]
```

## Solution Evaluation & Selection Phase
[Analyze and compare the 3 solutions, explain selection reasoning and any final optimizations]

## Final Solution
```python
[Final optimized code]
```

## Code Quality Requirements
- Ensure function name matches exactly what's specified in the problem
- Include all necessary import statements
- Use clear variable names and appropriate comments
- Handle boundary conditions properly
- Clear code structure with correct logic

## Problem Description
{problem}

## Function Signature
{entry_point}

Please complete this programming problem following the workflow above.
"""

# 2025-07-06 15:30:00 - INFO - Average score on MBPP dataset: 0.65396
# 2025-07-06 15:30:00 - INFO - Tokens: 103145 + 130579 = 233724
# 2025-07-06 15:30:00 - INFO - Avg tokens:685.407624633431




# Tokens: 142019 + 199917 = 341936
# Total calls: 341
# Avg tokens: 1002.7448680351906
# Avg calls: 1.0
# Avg score: 0.6891495601173021
# Fail total: 106