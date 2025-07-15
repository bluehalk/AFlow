# z_ablation/results/MATH/oneshotprompt/prompt.py

REFINE_ANSWER_PROMPT = """
Given the mathematical problem and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your response should be comprehensive, mathematically rigorous, and easy to follow.
"""

GENERATE_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
"""

DETAILED_SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""

ONESHOT_PROMPT_1 = """
You are an expert mathematician. Solve the given mathematical problem using a comprehensive multi-perspective approach that combines rigorous analysis, systematic calculation, and educational insight.

## Solution Framework:

### 1. Problem Understanding & Conceptual Foundation
- Clearly restate the problem in your own words
- Identify the key mathematical concepts, theorems, and principles involved
- Explain why these concepts are relevant to solving this problem

### 2. Multi-Method Solution Approach
Solve the problem using multiple complementary methods to ensure accuracy and provide deep understanding:

**Primary Method**: Use the most direct and rigorous mathematical approach
- Provide detailed step-by-step calculations with clear reasoning
- Show all mathematical work using proper LaTeX notation
- Explain the logic behind each mathematical step

**Alternative Verification**: Approach the problem from a different angle
- Use an alternative method (e.g., if primary was algebraic, try geometric; if computational, try analytical)
- This serves as both verification and deeper insight
- Compare results to ensure consistency

**Educational Insight**: Provide broader mathematical context
- Explain the significance of the solution method
- Connect to related mathematical concepts or theorems
- Describe any special properties or patterns observed

### 3. Solution Integration & Validation
- Synthesize insights from different approaches
- Verify consistency across methods
- Check the reasonableness of the final answer
- Present the final answer clearly enclosed in \\boxed{} LaTeX notation

## Quality Requirements:
- All mathematical expressions must use proper LaTeX formatting
- Each step should include clear explanations and reasoning
- The solution should be both rigorous and accessible
- Demonstrate how different mathematical perspectives complement each other
- Ensure the final answer is mathematically sound and properly formatted

Your response should showcase the power of multi-perspective mathematical reasoning while maintaining clarity and educational value.
"""



ONESHOT_PROMPT_2 = """
You are an expert mathematician. Solve the given mathematical problem using a comprehensive multi-perspective approach. Your solution should demonstrate the robustness of mathematical reasoning by considering different viewpoints and methods.

## Instructions:

### Phase 1: Problem Analysis & Multiple Approaches
First, analyze the problem from multiple mathematical perspectives:

1. **Conceptual Understanding**: Clearly restate the problem and identify the key mathematical concepts, theorems, and principles involved.

2. **Strategic Planning**: Consider different solution approaches:
   - Algebraic/Analytical approach (equations, formulas, symbolic manipulation)
   - Geometric/Visual approach (areas, shapes, visual reasoning)
   - Computational/Systematic approach (step-by-step calculations, case analysis)
   - Probabilistic/Combinatorial approach (if applicable)

### Phase 2: Multi-Method Solution
Solve the problem using at least 2-3 different methods or perspectives:

**Method A (Primary Approach)**: Use the most direct mathematical method
- Provide detailed step-by-step calculations
- Show all mathematical work with LaTeX notation
- Explain the reasoning behind each step

**Method B (Alternative Approach)**: Use a different perspective or method
- This could be geometric if Method A was algebraic, or vice versa
- Or use a more systematic/computational approach
- Verify that this method leads to the same result

**Method C (Verification/Educational)**: If applicable, provide additional insight
- Use a third method or special case analysis
- Explain the broader mathematical significance
- Connect to related concepts or theorems

### Phase 3: Solution Integration & Final Answer
- Compare results from different methods to ensure consistency
- If methods disagree, identify and resolve the discrepancy
- Present the final answer clearly enclosed in \\boxed{} LaTeX notation
- Provide a brief explanation of why this answer makes sense

## Quality Standards:
- All mathematical expressions must use proper LaTeX notation
- Each step should be clearly explained with reasoning
- The solution should be educational and easy to follow
- Demonstrate mathematical rigor and logical consistency
- Show how different approaches complement each other

Your response should be comprehensive, mathematically sound, and demonstrate the power of multi-perspective problem solving.
"""


ONESHOT_PROMPT_3 = """
You are an expert mathematician. Solve the given mathematical problem using a comprehensive multi-perspective approach. 
Let's think from different perspectives.
For each perspective, let's think step by step.
Present the final answer clearly enclosed in \\boxed{} LaTeX notation
"""