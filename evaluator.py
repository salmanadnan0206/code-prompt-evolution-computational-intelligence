"""
evaluator.py — Code extraction and sandboxed test-case execution.

Execution model (Codewars APPS fn_name problems):
    inputs[i]  = Python list already parsed, e.g. [1] or ["hello", 2]
    outputs[i] = [expected_return_value], e.g. [5] or [True]

    result = fn_name(*inputs[i])
    passed = ([result] == outputs[i])

Fitness = (test cases passed) / (total test cases)   — partial credit.
"""

import re
import subprocess
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """
    Pull the Python code block out of an LLM response.
    Tries ```python … ``` first, then any ``` … ```, then the raw text.
    """
    # Markdown fenced code block with python tag
    m = re.search(r"```python\s*\n([\s\S]*?)```", response)
    if m:
        return m.group(1).strip()

    # Any fenced code block
    m = re.search(r"```\s*\n?([\s\S]*?)```", response)
    if m:
        return m.group(1).strip()

    # No fences — assume the whole response is code
    return response.strip()


def has_function(code: str, fn_name: str) -> bool:
    """Return True if `def fn_name(` appears in the code."""
    return bool(re.search(rf"\bdef\s+{re.escape(fn_name)}\s*\(", code))


# ---------------------------------------------------------------------------
# Subprocess test runner
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = """\
import sys
sys.set_int_max_str_digits(1_000_000)

{code}

def __run__(inp, expected):
    try:
        result = {fn_name}(*inp)
        sys.stdout.write("__R__:" + ("1" if [result] == expected else "0") + "\\n")
        sys.stdout.flush()
    except Exception:
        sys.stdout.write("__R__:0\\n")
        sys.stdout.flush()

{calls}
"""


def _build_script(code: str, fn_name: str, inputs: list, outputs: list) -> str:
    """Build the standalone test script that will be executed in a subprocess."""
    call_lines = "\n".join(
        f"__run__({repr(inp)}, {repr(out)})"
        for inp, out in zip(inputs, outputs)
    )
    return _HARNESS_TEMPLATE.format(
        code=code,
        fn_name=fn_name,
        calls=call_lines,
    )


def run_tests(
    code: str,
    fn_name: str,
    inputs: list,
    outputs: list,
    timeout: int = 5,
) -> float:
    """
    Execute generated code against all test cases in an isolated subprocess.

    Returns partial-credit score: (passed tests) / (total tests).
    Returns 0.0 on any execution error, timeout, or missing function.
    """
    if not code or not has_function(code, fn_name):
        return 0.0

    script = _build_script(code, fn_name, inputs, outputs)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Parse only our sentinel lines to ignore any stray print() in generated code
        results = [
            int(line[len("__R__:"):])
            for line in proc.stdout.splitlines()
            if line.startswith("__R__:")
        ]
        if not results:
            return 0.0
        return sum(results) / len(inputs)   # partial credit over ALL test cases

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
