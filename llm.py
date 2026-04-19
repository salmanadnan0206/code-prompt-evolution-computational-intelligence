"""
llm.py — All LLM calls via Claude Code CLI (claude -p).

Uses the Claude.ai subscription (Max plan) instead of the Anthropic API SDK,
so no per-token charges are incurred beyond the subscription cost.

Two models:
    TARGET_MODEL    — generates code (fitness evaluation)
    OPTIMIZER_MODEL — crossover and mutation meta-operations

NOTE: The CLI does not expose a --temperature flag. Code generation therefore
      runs at the model's default temperature instead of the configured 0.0.
      Fitness scores may vary slightly between runs as a result.
"""

import json
import re
import subprocess
import time

import config
from cost_tracker import tracker


def _call(
    phase: str,
    model: str,
    system: str,
    user: str,
    retries: int = 4,
) -> str:
    """
    Invoke `claude -p` in non-interactive JSON mode.

    Handles:
      - Rate-limit errors  → exponential back-off (30 s base)
      - Transient errors   → short back-off (5 s base)
      - Timeouts           → retry up to `retries` times
      - Auth errors        → raises RuntimeError immediately (no retry)

    Logs token usage and cost to the global CostTracker.
    Returns the assistant text, or "" after all retries are exhausted.
    """
    cmd = [
        "claude",
        "-p", user,
        "--output-format", "json",
        "--model", model,
        "--max-turns", "1",
        "--tools", "",          # disable all built-in tools; pure text generation
        "--system-prompt", system,
    ]

    for attempt in range(retries):
        proc = None
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.CLAUDE_CLI_TIMEOUT,
            )

            stdout = proc.stdout.strip()

            # Empty stdout usually means a hard startup failure
            if not stdout:
                stderr = proc.stderr.strip()
                _check_auth_error(stderr)
                if attempt < retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  [empty-output] stderr={stderr!r} | retry in {wait}s …")
                    time.sleep(wait)
                    continue
                return ""

            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                raw = stdout[:120]
                if attempt < retries - 1:
                    print(f"  [json-parse-error] raw={raw!r} | retry in 5s …")
                    time.sleep(5)
                    continue
                print(f"  [json-parse-error] giving up | raw={raw!r}")
                return ""

            if data.get("is_error"):
                result_msg = str(data.get("result", ""))
                api_status = data.get("api_error_status")

                _check_auth_error(result_msg)  # raises immediately if auth issue

                # Rate limit: HTTP 429 or message contains "rate"/"limit"
                is_rate_limit = (
                    api_status == 429
                    or "rate" in result_msg.lower()
                    or "too many" in result_msg.lower()
                )
                if is_rate_limit and attempt < retries - 1:
                    wait = 2 ** attempt * 30
                    print(f"  [rate-limit] waiting {wait}s …")
                    time.sleep(wait)
                    continue

                if attempt < retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  [error] {result_msg!r} | retry in {wait}s …")
                    time.sleep(wait)
                    continue

                print(f"  [error] {result_msg}")
                return ""

            # --- success path ---
            result_text = data.get("result", "")

            # Extract per-model token stats (most accurate source)
            model_usage = data.get("modelUsage", {}).get(model, {})
            input_tokens  = model_usage.get("inputTokens", 0)
            output_tokens = model_usage.get("outputTokens", 0)
            cost_usd      = model_usage.get("costUSD", data.get("total_cost_usd", 0.0))

            tracker.record(
                phase=phase,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

            return result_text

        except subprocess.TimeoutExpired:
            print(f"  [timeout] attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                time.sleep(10)

        except RuntimeError:
            raise  # auth errors bubble up immediately

        except FileNotFoundError:
            raise RuntimeError(
                "`claude` command not found on PATH. "
                "Install Claude Code: https://claude.ai/code"
            )

    return ""


def _check_auth_error(msg: str) -> None:
    """Raise RuntimeError immediately if `msg` signals an authentication failure."""
    lower = msg.lower()
    if "not logged in" in lower or "please run /login" in lower:
        raise RuntimeError(
            "Claude CLI is not authenticated.\n"
            "Run:  claude auth login\n"
            "Then re-run this script."
        )


# ---------------------------------------------------------------------------
# Code generation  (fitness evaluation)
# ---------------------------------------------------------------------------

def generate_code(system_prompt: str, question: str) -> str:
    """
    Ask the target LLM to solve a Codeforces problem with a complete C++ program.

    Arguments:
        system_prompt — the GA chromosome (instruction being evolved)
        question      — the problem statement text

    Returns the raw LLM response (code extraction happens in evaluator.py).
    """
    user_msg = (
        f"{question}\n\n"
        "Write a complete, compilable C++ program that reads from stdin "
        "and writes the answer to stdout."
    )
    return _call(
        phase="fitness",
        model=config.TARGET_MODEL,
        system=system_prompt,
        user=user_msg,
    )


# ---------------------------------------------------------------------------
# Crossover  (LLM-mediated merge of two parent prompts)
# ---------------------------------------------------------------------------

_CROSSOVER_SYSTEM = (
    "You merge coding-instruction prompts. Output ONLY the merged prompt — "
    "no commentary, no labels, no markdown."
)

_CROSSOVER_USER = """\
Combine these two coding-instruction prompts into one coherent prompt.
Preserve the strongest strategies from each parent (e.g. edge-case thinking
from one, algorithm identification from the other).
Do not simply concatenate — produce a unified instruction paragraph.

--- Parent A ---
{parent_a}

--- Parent B ---
{parent_b}
"""


def crossover(prompt_a: str, prompt_b: str) -> str:
    """Merge two parent prompts into one offspring via the optimizer LLM."""
    return _call(
        phase="crossover",
        model=config.OPTIMIZER_MODEL,
        system=_CROSSOVER_SYSTEM,
        user=_CROSSOVER_USER.format(parent_a=prompt_a, parent_b=prompt_b),
    ).strip()


# ---------------------------------------------------------------------------
# Mutation  (three independent operators)
# ---------------------------------------------------------------------------

_MUT_SYSTEM = (
    "You edit coding-instruction prompts. Output ONLY the modified prompt — "
    "no commentary, no labels, no markdown."
)


def mutate_inject(prompt: str) -> str:
    """Add one new instructional strategy to the prompt."""
    user = (
        "Add one new, specific coding strategy to this instruction prompt. "
        "Examples of strategies: 'validate your output against the examples', "
        "'consider time complexity', 'check for off-by-one errors'. "
        "Output only the modified prompt.\n\n"
        f"Prompt:\n{prompt}"
    )
    return _call(
        phase="mutate_inject",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM,
        user=user,
    ).strip()


def mutate_delete(prompt: str) -> str:
    """Remove one sentence from the prompt to make it leaner."""
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    if len(sentences) <= 1:
        return prompt  # nothing to delete
    user = (
        "Remove one sentence from this instruction prompt — the one that "
        "contributes the least to code-generation quality. Keep everything "
        "else verbatim. Output only the modified prompt.\n\n"
        f"Prompt:\n{prompt}"
    )
    return _call(
        phase="mutate_delete",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM,
        user=user,
    ).strip()


def mutate_rephrase(prompt: str) -> str:
    """Rephrase the prompt without changing its meaning."""
    user = (
        "Rephrase this instruction prompt using different words, but keep "
        "the same meaning and all the same strategies. "
        "Output only the rephrased prompt.\n\n"
        f"Prompt:\n{prompt}"
    )
    return _call(
        phase="mutate_rephrase",
        model=config.OPTIMIZER_MODEL,
        system=_MUT_SYSTEM,
        user=user,
    ).strip()
