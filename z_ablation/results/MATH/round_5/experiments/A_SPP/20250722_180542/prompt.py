REFINE_ANSWER_PROMPT = """
You are a professional mathematician. Given the mathematical problem and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \\boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your response should be comprehensive, mathematically rigorous, and easy to follow.
"""

# Ablation_Refine_Answer_PROMPT = """
# Given the mathematical problem, the code and code output from the programmer.
# Please translate the code output to the final answer, enclosed in \\boxed{} LaTeX notation.

# IMPORTANT: Do NOT re-analyze the problem or provide your own mathematical reasoning. 
# Simply take the numerical output from the code and format it as the final answer.

# Example:
# - If code output is "21.991148575128562" and the answer should be "7π", 
#   then output: \\boxed{7\\pi}

# - If code output is "10" and the answer should be "10", 
#   then output: \\boxed{10}

# Just format the code output as the final answer in \\boxed{} notation.
# """

GENERATE_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \\boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
"""

DETAILED_SOLUTION_PROMPT = """
You are a professional mathematician. Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \\boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""


PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer in the required format.

Problem description: {problem}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""
# 4. The return value should be the final answer, and must be formatted in \\boxed{{}} LaTeX notation.



MERGE_PROMPT = """
You are a professional mathematician and professional Python programmer. 
Your task is to solve the given mathematical problem with two perspectives: mathematical perspective and coding perspective.

## Perspective 1: Python Code Solution
Think as a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer in the required format.

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.
4. The return value should be the final answer, and must be formatted in \\boxed{{}} LaTeX notation.
Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.


## Perspective 2: Mathematical Solution
Think as a professional mathematician. Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \\boxed{{}} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.

Problem: {problem}

## Solution Format: 

<python_code>
[solve function]
</python_code>

<math_solution>
[math solution]
</math_solution>
"""


RECHECK_PROMPT = """
There is inconsistency between your answers:
- Mathematical answer: {math_answer}
- Code execution result: {code_answer}

Please critically analyze the root cause of the different answers, and provide the correct answer.
Neither method is inherently more trustworthy - both can contain errors and should be verified against each other.

Please respond in the standard format:
<analysis>
[analysis]
</analysis>

<python_code>
[python code]
</python_code>

<math_solution>
[math solution]
</math_solution>
"""



JUDGER_PROMPT = """
You are an expert in mathematics and programming, acting as an impartial judge.
Your task is to analyze two different solutions to the same problem—one mathematical and one computational—that have produced conflicting answers. Your goal is to identify the source of the error and provide clear guidance for correction.

**Problem Statement**: 
{problem}

**Solution 1: Mathematical Approach**
*Full Solution*:
{math_solution}
*Claimed Answer*: {math_answer}

**Solution 2: Computational Approach (Python)**
*Full Code*:
{code_solution}
*Execution Result*: {code_answer}

**Your Analysis Task**:

1.  **Acknowledge Solution Natures**: Recognize that the computational approach, if it executes successfully, yields a deterministic and highly reliable numerical result. The mathematical approach, while aiming for logical rigor, is susceptible to reasoning errors.
2.  **Prioritize Scrutiny**:
    *   For the **Mathematical Solution**, meticulously verify each logical step and calculation. The error is most likely to be found here.
    *   For the **Computational Solution**, if the code ran successfully, trust its output. Focus your analysis on whether the code's *logic* correctly models the problem, not on the result itself.
3.  **Identify the Root Cause**: Pinpoint the exact location and nature of the error. Is it a logical flaw in the math, a bug in the code's logic (less likely if answers are just different), or a misinterpretation of the problem in one of the approaches?
4.  **Provide Actionable Guidance**: Give a clear and concise explanation of what went wrong. Your guidance can be a textual explanation or a Python code snippet intended for debugging. If providing code, wrap it in ```python.

**Output Format (Strictly follow this format)**:
<analysis>
[Provide a detailed breakdown of your analysis, comparing the two approaches and identifying the error based on the principles above.]
</analysis>
<conclusion>
[State which solution's *approach* is correct and why.]
</conclusion>
<suggestion>
[Provide a clear and actionable suggestion. This can be a textual explanation of the error or a Python code snippet to help debug. If providing code, ensure it is executable and wrap it in ```python.]
</suggestion>
"""



RECHECK_PROMPT_WITH_JUDGEMENT = """
An expert judge has reviewed your previous attempt and found an error. Use the judge's feedback to correct your work.

**Previous Inconsistent Answers**:
- Mathematical Answer: {math_answer}
- Code Execution Result: {code_answer}

**Judge's Analysis and Guidance**:
{judgement}
{executable_feedback_result}
**Your Task**:
Based on the judge's feedback (and the result of their executable suggestion, if provided), please re-evaluate the problem and provide a single, consistent, and correct solution in the original dual-perspective format.
Additionally, ensure the final Python code is clean, free of debugging statements, and ready for production.

## Solution Format: (must follow this format)
<python_code>
[corrected solve function]
</python_code>

<math_solution>
[corrected math solution]
</math_solution>
"""


SPP_PROMPT_LESS = '''When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here is an example:
---
Example Task: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:", enclosed in \\boxed{} LaTeX notation.

Task: {input}
'''





SPP_PROMPT = '''When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:", enclosed in \\boxed{} LaTeX notation.
Task:
'''