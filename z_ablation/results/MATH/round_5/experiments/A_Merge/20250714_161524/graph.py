from typing import Literal
from scripts.operators import Custom, Programmer, ScEnsemble
from .prompt import REFINE_ANSWER_PROMPT, DETAILED_SOLUTION_PROMPT, GENERATE_SOLUTION_PROMPT# , Ablation_Refine_Answer_PROMPT, PYTHON_CODE_VERIFIER_PROMPT
from scripts.async_llm import create_llm_instance
from scripts.logs import logger

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


MERGE_PROMPT = """
[System]
You are a world-class “Mathematician-Programmer” capable of both rigorous proofs and correct Python implementations. Solve the given mathematical problem through the structured pipeline below. Follow every instruction exactly.

[User]
★★ INPUT PROBLEM ★★
{problem}
★★ END OF PROBLEM ★★

===== STAGE 1 : ANALYSIS =====
1. Think step by step to understand the problem, cite relevant formulas or lemmas, and design an algorithm.  
2. Keep this section concise (≤ 120 words) yet capture the core idea.

===== STAGE 2 : PYTHON CODE =====
1. Write complete, self-contained Python 3 code inside **one** Markdown ```python``` block.  
   • Define a function `solve()` with **no parameters**; it must return (not print) the final answer.  
   • Use only the Python standard library.  
2. Do not place explanations inside the code block—only comments and code.

===== STAGE 3 : CODE OUTPUT (mental run) =====
1. Execute the code mentally or by formal reasoning.  
2. Output the return value of `solve()` in a Markdown ```output``` block, e.g.  
   ```output
   42
   ```

===== STAGE 4 : EXPLANATION & PROOF =====
1. Restate the problem in one sentence, then provide a step-by-step derivation; all formulas must use LaTeX.  
2. Cite key snippets from **Stage 2** and explain their mathematical meaning.  
3. Quote the result from **Stage 3** and explain how it confirms correctness.  
4. Keep the logic rigorous and the narrative clear.

===== STAGE 5 : FINAL ANSWER =====
Output only the final numerical or algebraic answer wrapped in LaTeX, e.g.  
\\boxed{42}

===== GLOBAL CONSTRAINTS =====
• Preserve the exact order and headings of all STAGES; do not add extra titles or text.  
• Ensure the value in **Stage 3** matches the \\boxed{} content in **Stage 5**.  
• The response must be text-only; do not output any files or non-text formats.
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
        
        # refined_solution = await custom(input=problem + f"\nCode: {code_solution['code']}\n Code output: {code_solution['output']}", instruction=REFINE_ANSWER_PROMPT)
        
        detailed_solution = await custom(input=problem, instruction=MERGE_PROMPT)
        
        # solutions = [
        #     refined_solution['response'],
        #     detailed_solution['response']
        # ]
        # for _ in range(2):
        #     solution = await custom(input=problem, instruction=GENERATE_SOLUTION_PROMPT)
        #     solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        # final_solution = await sc_ensemble(solutions=solutions, problem=problem)
        
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']

        return detailed_solution['response'], input_tokens, output_tokens, call_count
                    