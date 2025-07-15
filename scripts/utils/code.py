import re
import json
import asyncio
import sys
from enum import Enum
from typing import Any, List, Tuple, Optional


class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"


async def exec_code(code: str, timeout: int = 5, only_solve_function: bool = True) -> Tuple[str, str]:
    """
    Execute code in a subprocess with timeout.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        only_solve_function: If True, only returns the solve() function result (for problem solving).
                            If False, captures all output including prints (for debugging).
    
    Returns:
        Tuple[status, output], where status is "Success" or "Error".
    """
    try:
        # Add system import
        code = "\nimport sys\n" + code
        
        if only_solve_function:
            # For problem solving: only get solve() function result
            if "def solve():" in code:
                # Suppress all prints and only output solve() result
                code = code + """
import sys
from io import StringIO

# Redirect stdout to capture prints
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    # Call solve function
    result = solve()
    # Restore stdout and write only the result
    sys.stdout = old_stdout
    sys.stdout.write(str(result))
except Exception as e:
    # Restore stdout and write error
    sys.stdout = old_stdout
    sys.stdout.write(f"Error in solve(): {str(e)}")
"""
            else:
                # For snippets without solve function, just execute them
                pass
        # If not only_solve_function, execute code as-is and capture all output (including prints)
        
        code = code + "\nsys.stdout.flush()"
        
        # Handle LaTeX boxed expressions
        code = re.sub(r'(?<!\\)\\boxed\{', r'\\\\boxed{', code)
        
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", code,  # -u disables buffering
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Wait for process completion or timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="ignore").strip().replace('\x08', '\\b') if stdout else ""
            error = stderr.decode("utf-8", errors="ignore").strip().replace('\x08', '\\b') if stderr else ""

            if process.returncode == 0:
                return "Success", output
            else:
                return "Error", f"Process failed with return code {process.returncode}: {error}"
        
        except asyncio.TimeoutError:
            # Kill process on timeout
            process.kill()
            await process.wait()
            return "Error", "Code execution timed out (killed)"

    except Exception as e:
        return "Error", f"Subprocess error: {str(e)}"


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from markdown code blocks.
    
    Args:
        text: Text containing possible markdown code blocks
        
    Returns:
        Extracted Python code or None if no code blocks found
    """
    # Look for Python code blocks (```python ... ```)
    python_pattern = r"```python\n(.*?)```"
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Look for generic code blocks (``` ... ```)
    generic_pattern = r"```\n(.*?)```"
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def extract_test_cases_from_jsonl(entry_point: str, dataset: CodeDataset = CodeDataset.HUMAN_EVAL):


    # print(f"dataset: {dataset}"
    # )
    # if dataset == CodeDataset.HUMAN_EVAL.value:
    #     file_path = "data/datasets/humaneval_public_test.jsonl"
    #     # Retain the original hardcoded test cases
    #     hardcoded_cases = {
    #         "find_zero": "",
    #         "decode_cyclic": "",
    #         "decode_shift": "",
    #         "by_length": "",
    #         "add": "",
    #         "triangle_area": "",
    #         "correct_bracketing": "",
    #         "solve": "",
    #         "sum_squares": "",
    #         "starts_one_ends": "",
    #     }
    # elif dataset == CodeDataset.MBPP.value:
    file_path = "data/datasets/mbpp_public_test.jsonl"
    hardcoded_cases = {
        "remove_odd": "",
        "replace_spaces": "",
        "snake_to_camel": "",
        "Split": "",
        "swap_List": "",
        "square_Sum": "",
        "sort_sublists": "",
        "unique_sublists": "",
    }
    
    # print(f"file_path: {file_path}")
    # print(f"hardcoded_cases: {hardcoded_cases}")
    # Check if there are hardcoded test cases
    if entry_point in hardcoded_cases:
        return hardcoded_cases[entry_point]

    # If there are no hardcoded test cases, read from the file
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") == entry_point:
                return data.get("test")

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function
