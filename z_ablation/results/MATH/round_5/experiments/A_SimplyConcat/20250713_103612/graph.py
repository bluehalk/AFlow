from typing import Literal
from scripts.operators import Custom, Programmer, ScEnsemble
from .prompt import REFINE_ANSWER_PROMPT, DETAILED_SOLUTION_PROMPT, GENERATE_SOLUTION_PROMPT# , Ablation_Refine_Answer_PROMPT, PYTHON_CODE_VERIFIER_PROMPT
from scripts.async_llm import create_llm_instance
from scripts.logs import logger

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

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


MERGE_PROMPT = """
You are a professional mathematician and professional Python programmer. 
Your task is to solve the given mathematical problem with two perspectives: mathematical perspective and coding perspective.

## Perspective 1: Python Code Solution
Think as a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer in the required format.

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.


## Perspective 2: Mathematical Solution
Think as a professional mathematician. Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \\boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.

Problem: {problem}

"""

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm_config = llm_config
    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        llm = create_llm_instance(self.llm_config)
        custom = Custom(llm)
        programmer = Programmer(llm)
        sc_ensemble = ScEnsemble(llm)

        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        # Use Programmer to generate and execute Python code
        # code_solution = await programmer(problem=problem) # Programmer会自动生成代码并执行，循环三次。如果执行失败，会返回错误信息
        # logger.info(f"Code solution: {code_solution['output']}")
        
        # Use Custom to refine and format the answer from Programmer（执行结果）
        # refined_solution = await custom(input=problem + f"\nCode: {code_solution['code']}\n Code output: {code_solution['output']}", instruction=REFINE_ANSWER_PROMPT)
        
        # Generate a detailed step-by-step solution using Custom
        detailed_solution = await custom(input=problem, instruction=MERGE_PROMPT)
        
        # # Generate multiple solutions using Custom
        # solutions = [
        #     refined_solution['response'],
        #     detailed_solution['response']
        # ]
        # for _ in range(2):
        #     solution = await custom(input=problem, instruction=GENERATE_SOLUTION_PROMPT)
        #     solutions.append(solution['response'])
        
        # # Use ScEnsemble to select the best solution
        # final_solution = await sc_ensemble(solutions=solutions, problem=problem)
        
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']

        return detailed_solution['response'], input_tokens, output_tokens, call_count
                    