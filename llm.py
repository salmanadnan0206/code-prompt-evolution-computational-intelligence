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
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import config
from cost_tracker import tracker

# Set by main.py to enable persistent error logging for CLI retry/backoff events.
_error_log_path: Path | None = None
_error_log_lock = threading.Lock()


def set_error_log_path(path: Path) -> None:
    """Point llm module at a file for persistent CLI-error logs."""
    global _error_log_path
    _error_log_path = path
    path.parent.mkdir(parents=True, exist_ok=True)


def _log_err(msg: str) -> None:
    """Emit an LLM retry/error line to both stdout and (if set) the error log."""
    line = f"[{time.strftime('%H:%M:%S')}] [LLM_ERR] {msg}"
    print(line, flush=True)
    if _error_log_path is not None:
        with _error_log_lock:
            with _error_log_path.open("a") as f:
                f.write(line + "\n")


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
    # Exhaustive disallow list — the `--tools ""` form doesn't actually disable tools.
    # Even with --disallowed-tools, Claude keeps TRYING tools and eating turns. So
    # we set --max-turns high enough that after several denied tool attempts, Claude
    # gives up and responds with text. Empirically, num_turns=2 hits max_turns=3,
    # so CLI counts user+assistant differently — bumping to 12 for safety.
    _DISALLOWED_TOOLS = (
        "Read,Write,Edit,MultiEdit,NotebookEdit,Glob,Grep,LS,"
        "Bash,BashOutput,KillBash,WebFetch,WebSearch,"
        "Task,TodoWrite,ExitPlanMode,Agent,Skill"
    )
    cmd = [
        "claude",
        "-p", user,
        "--output-format", "json",
        "--model", model,
        "--max-turns", "12",
        "--disallowed-tools", _DISALLOWED_TOOLS,
        "--permission-mode", "plan",   # read-only mode, prevents side-effects
        "--system-prompt", system,
    ]

    for attempt in range(retries):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.CLAUDE_CLI_TIMEOUT,
            )

            stdout = proc.stdout.strip()

            # Empty stdout usually means CLI startup failure or silent throttling
            if not stdout:
                stderr = proc.stderr.strip()[:500]
                returncode = proc.returncode
                _check_auth_error(stderr)
                if attempt < retries - 1:
                    wait = 2 ** attempt * 5 + random.uniform(0, 3)
                    _log_err(
                        f"empty-output phase={phase} attempt={attempt+1}/{retries}  "
                        f"returncode={returncode}  stderr={stderr!r}  "
                        f"stdout_raw={proc.stdout[:200]!r} | retry in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                _log_err(
                    f"empty-output phase={phase} GIVING UP after {retries} attempts  "
                    f"returncode={returncode}  stderr={stderr!r}"
                )
                return ""

            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                raw = stdout[:120]
                if attempt < retries - 1:
                    wait = 5 + random.uniform(0, 2)
                    _log_err(f"json-parse-error phase={phase} attempt={attempt+1}/{retries}  raw={raw!r}  retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                _log_err(f"json-parse-error phase={phase} GIVING UP  raw={raw!r}")
                return ""

            if data.get("is_error"):
                result_msg = str(data.get("result", ""))
                api_status = data.get("api_error_status")
                stderr_tail = proc.stderr.strip()[-300:] if proc.stderr else ""

                # === DIAGNOSTIC DUMP ===
                # On every is_error, dump the full JSON keys + values so we can
                # finally see what the CLI is actually returning.
                raw_dump = json.dumps({k: (v if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__)
                                       for k, v in data.items()})[:800]
                _log_err(
                    f"is_error_dump phase={phase} attempt={attempt+1}/{retries}  "
                    f"api_status={api_status}  stderr_tail={stderr_tail!r}  "
                    f"data={raw_dump}"
                )

                _check_auth_error(result_msg)  # raises immediately if auth issue

                # Rate limit: HTTP 429 or message contains "rate"/"limit"
                is_rate_limit = (
                    api_status == 429
                    or "rate" in result_msg.lower()
                    or "too many" in result_msg.lower()
                )
                if is_rate_limit and attempt < retries - 1:
                    wait = 2 ** attempt * 30 + random.uniform(0, 10)
                    _log_err(f"rate-limit phase={phase} attempt={attempt+1}/{retries}  waiting {wait:.1f}s  status={api_status}")
                    time.sleep(wait)
                    continue

                if attempt < retries - 1:
                    wait = 2 ** attempt * 5 + random.uniform(0, 3)
                    _log_err(f"error phase={phase} attempt={attempt+1}/{retries}  msg={result_msg!r}  retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue

                _log_err(f"error phase={phase} GIVING UP  msg={result_msg!r}")
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
            wait = 10 + random.uniform(0, 5)
            _log_err(f"timeout phase={phase} attempt={attempt+1}/{retries}  (subprocess exceeded {config.CLAUDE_CLI_TIMEOUT}s)  retry in {wait:.1f}s")
            if attempt < retries - 1:
                time.sleep(wait)

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

# Anti-tool directive appended to every chromosome system prompt.
# The chromosome itself is kept pure (for GA integrity) — this directive is only
# added at call-time, so evolved prompts don't accumulate this text.
_ANTI_TOOL_DIRECTIVE = (
    "\n\n---\nCRITICAL OPERATIONAL CONSTRAINT (do not ignore):\n"
    "You are running in a headless evaluation harness with NO filesystem access "
    "and NO tool availability. Do NOT call Write, Edit, Bash, Read, Glob, Grep, "
    "or any other tool — every tool call will be denied and waste your turns. "
    "Respond with the C++ source code as plain text in your message. Nothing else."
)


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
        "and writes the answer to stdout.\n\n"
        "IMPORTANT: Respond with the C++ source code directly in a single "
        "message. Do NOT attempt to use any tools (no Write, Edit, Bash, etc.). "
        "Output the code inline as text only."
    )
    return _call(
        phase="fitness",
        model=config.TARGET_MODEL,
        system=system_prompt + _ANTI_TOOL_DIRECTIVE,
        user=user_msg,
    )


# ---------------------------------------------------------------------------
# Crossover  (LLM-mediated merge of two parent prompts)
# ---------------------------------------------------------------------------

_CROSSOVER_SYSTEM = (
    "You merge coding-instruction prompts. Output ONLY the merged prompt — "
    "no commentary, no labels, no markdown. "
    "Do NOT use any tools. Respond with the merged prompt as plain text only."
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
        system=_CROSSOVER_SYSTEM + _ANTI_TOOL_DIRECTIVE,
        user=_CROSSOVER_USER.format(parent_a=prompt_a, parent_b=prompt_b),
    ).strip()


# ---------------------------------------------------------------------------
# Mutation  (three independent operators)
# ---------------------------------------------------------------------------

_MUT_SYSTEM = (
    "You edit coding-instruction prompts. Output ONLY the modified prompt — "
    "no commentary, no labels, no markdown. "
    "Do NOT use any tools. Respond with the modified prompt as plain text only."
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
        system=_MUT_SYSTEM + _ANTI_TOOL_DIRECTIVE,
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
        system=_MUT_SYSTEM + _ANTI_TOOL_DIRECTIVE,
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
        system=_MUT_SYSTEM + _ANTI_TOOL_DIRECTIVE,
        user=user,
    ).strip()
