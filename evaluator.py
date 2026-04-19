"""
evaluator.py — C++ code extraction, compilation, and sandboxed test execution.

Execution model (Codeforces stdin/stdout problems):
    inputs[i]  = raw string piped to the compiled binary's stdin
    outputs[i] = expected string on stdout

For each test case:
    1. Run the compiled binary with inputs[i] on stdin.
    2. Compare stdout against outputs[i] (whitespace-normalised per line).

Fitness = (test cases passed) / (test cases evaluated) — partial credit.
"""

import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

import config


# ---------------------------------------------------------------------------
# Code extraction  (C++ fenced blocks, then any block, then raw text)
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """
    Pull C++ code out of an LLM response.

    Priority:
        1. ```cpp … ``` or ```c++ … ```
        2. Any ``` … ``` block
        3. Raw response text (assume entire reply is code)
    """
    # C++ tagged block
    m = re.search(r"```(?:cpp|c\+\+)\s*\n([\s\S]*?)```", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Any fenced block
    m = re.search(r"```\s*\n?([\s\S]*?)```", response)
    if m:
        return m.group(1).strip()

    # No fences — assume the whole response is code
    return response.strip()


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> list[str]:
    """Strip trailing whitespace from every line; drop trailing blank lines."""
    lines = text.rstrip("\n").split("\n")
    return [line.rstrip() for line in lines]


def _outputs_match(actual: str, expected: str) -> bool:
    return _normalise(actual) == _normalise(expected)


# ---------------------------------------------------------------------------
# g++ compilation
# ---------------------------------------------------------------------------

def _compile(code: str) -> tuple[Optional[str], str]:
    """
    Write code to a temp .cpp file and compile with g++.

    Returns:
        (binary_path, "")          on success
        (None,        error_msg)   on failure
    """
    src_fd, src_path = tempfile.mkstemp(suffix=".cpp")
    bin_path = src_path[:-4]   # strip .cpp
    try:
        with os.fdopen(src_fd, "w") as f:
            f.write(code)

        result = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-o", bin_path, src_path],
            capture_output=True,
            text=True,
            timeout=config.COMPILE_TIMEOUT,
        )

        if result.returncode != 0:
            return None, result.stderr[:500]

        return bin_path, ""

    except subprocess.TimeoutExpired:
        return None, "compilation timed out"
    except FileNotFoundError:
        return None, "g++ not found on PATH"
    except Exception as exc:
        return None, str(exc)
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test-case runner
# ---------------------------------------------------------------------------

def _run_one(bin_path: str, stdin_str: str, timeout: int) -> Optional[str]:
    """Run the compiled binary with stdin_str; return stdout or None on failure."""
    try:
        result = subprocess.run(
            [bin_path],
            input=stdin_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tests(
    code: str,
    inputs: list,
    outputs: list,
    timeout: int = 5,
) -> float:
    """
    Compile the C++ code and run it against up to MAX_TEST_CASES test cases.

    Returns partial-credit score: (passed) / (evaluated).
    Returns 0.0 on empty code, compilation failure, or all timeouts.
    """
    if not code or not code.strip():
        return 0.0

    bin_path, err = _compile(code)
    if bin_path is None:
        return 0.0

    try:
        n = min(len(inputs), config.MAX_TEST_CASES)
        passed = 0
        for i in range(n):
            actual = _run_one(bin_path, inputs[i], timeout)
            if actual is not None and _outputs_match(actual, outputs[i]):
                passed += 1

        return passed / n if n > 0 else 0.0

    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass
