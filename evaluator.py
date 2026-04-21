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
import tempfile
import time
from typing import Optional

import config

# Max chars to keep from stdout/expected per test case (prevents huge JSONL lines)
_MAX_IO_LEN = 500


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
    _success = False
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

        _success = True
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
        if not _success:
            try:
                os.unlink(bin_path)
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

def run_tests_verbose(
    code: str,
    inputs: list,
    outputs: list,
    timeout: int = 2,
) -> dict:
    """
    Compile + run all test cases, returning a DETAILED dict for logging.

    Returned dict schema:
        {
          "score":          float,       # passed / total
          "reason":         str,         # "empty_code" | "compile_fail" | "no_tests"
                                         # | "perfect" | "partial_pass"
                                         # | "all_fail" | "all_fail_early_stop"
          "compile": {
              "success":    bool,
              "error":      str | None,  # g++ stderr on failure
              "time_s":     float,
          },
          "test_cases": [
              {
                "index":    int,
                "passed":   bool,
                "actual":   str | None,  # truncated to _MAX_IO_LEN
                "expected": str,         # truncated to _MAX_IO_LEN
                "time_s":   float,
                "timeout":  bool,
              }, ...
          ],
          "tests_total":    int,         # total available
          "tests_run":      int,         # how many actually executed (early-stop cuts)
          "early_stopped":  bool,
          "total_test_time_s": float,
        }

    Early stopping: stop after EARLY_STOP_FAILURES consecutive failures.
    Score denominator is always total test count (fair penalty for bad solutions).
    """
    result: dict = {
        "score": 0.0,
        "reason": None,
        "compile": {"success": False, "error": None, "time_s": 0.0},
        "test_cases": [],
        "tests_total": len(inputs),
        "tests_run": 0,
        "early_stopped": False,
        "total_test_time_s": 0.0,
    }

    if not code or not code.strip():
        result["reason"] = "empty_code"
        return result

    # --- compile ---
    t0 = time.monotonic()
    bin_path, err = _compile(code)
    result["compile"]["time_s"] = round(time.monotonic() - t0, 3)

    if bin_path is None:
        result["reason"] = "compile_fail"
        result["compile"]["error"] = err[:500] if err else "unknown"
        return result

    result["compile"]["success"] = True

    # --- run tests ---
    try:
        n = len(inputs)
        if n == 0:
            result["reason"] = "no_tests"
            return result

        passed = 0
        consecutive_fails = 0
        t_tests_start = time.monotonic()

        for i in range(n):
            t_tc = time.monotonic()
            actual = _run_one(bin_path, inputs[i], timeout)
            tc_time = round(time.monotonic() - t_tc, 3)

            is_pass = actual is not None and _outputs_match(actual, outputs[i])

            result["test_cases"].append({
                "index":    i,
                "passed":   is_pass,
                "actual":   (actual[:_MAX_IO_LEN] if actual is not None else None),
                "expected": outputs[i][:_MAX_IO_LEN] if isinstance(outputs[i], str) else str(outputs[i])[:_MAX_IO_LEN],
                "time_s":   tc_time,
                "timeout":  actual is None,
            })
            result["tests_run"] = i + 1

            if is_pass:
                passed += 1
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails >= config.EARLY_STOP_FAILURES:
                    result["early_stopped"] = True
                    break

        result["total_test_time_s"] = round(time.monotonic() - t_tests_start, 3)
        result["score"] = passed / n

        if passed == n:
            result["reason"] = "perfect"
        elif passed == 0:
            result["reason"] = "all_fail_early_stop" if result["early_stopped"] else "all_fail"
        else:
            result["reason"] = "partial_pass"

        return result

    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def run_tests(
    code: str,
    inputs: list,
    outputs: list,
    timeout: int = 2,
) -> float:
    """Thin wrapper — returns only the float score (backward compat)."""
    return run_tests_verbose(code, inputs, outputs, timeout)["score"]
