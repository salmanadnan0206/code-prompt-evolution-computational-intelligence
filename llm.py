"""
llm.py — All Claude API calls: code generation, crossover, and mutation.

Two models are used:
    TARGET_MODEL    — generates code (cheap, deterministic, temperature 0)
    OPTIMIZER_MODEL — performs crossover and mutation (creative, temperature 0.7)

Every call is logged to the CostTracker singleton (see cost_tracker.py).
"""

import re
import time
import anthropic
import config
from cost_tracker import tracker


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazy-initialise the Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _call(
    phase: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str,
    user: str,
    retries: int = 3,
) -> str:
    """
    Call the Claude API with automatic retry on transient errors.
    Logs token usage to the global CostTracker.
    Returns the assistant's text response.
    """
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # Log tokens to cost tracker
            tracker.record(
                phase=phase,
                model=model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
            return resp.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  [rate-limit] waiting {wait}s …")
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == retries - 1:
                print(f"  [api-error] {e}")
                return ""
            time.sleep(2)
    return ""


# ---------------------------------------------------------------------------
# Code generation  (fitness evaluation)
# ---------------------------------------------------------------------------

def generate_code(system_prompt: str, question: str, fn_name: str) -> str:
    """
    Ask the target LLM to solve a coding problem.

    Arguments:
        system_prompt — the GA chromosome (instruction being evolved)
        question      — the APPS problem text
        fn_name       — name the function must have

    Returns the raw LLM response (code extraction happens in evaluator.py).
    """
    user_msg = (
        f"{question}\n\n"
        f"Write a Python function named `{fn_name}`."
    )
    return _call(
        phase="fitness",
        model=config.TARGET_MODEL,
        temperature=config.TARGET_TEMPERATURE,
        max_tokens=config.MAX_CODE_TOKENS,
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
        temperature=config.OPTIMIZER_TEMPERATURE,
        max_tokens=config.MAX_OPERATOR_TOKENS,
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
        temperature=config.OPTIMIZER_TEMPERATURE,
        max_tokens=config.MAX_OPERATOR_TOKENS,
        system=_MUT_SYSTEM,
        user=user,
    ).strip()


def mutate_delete(prompt: str) -> str:
    """Remove one sentence from the prompt to make it leaner."""
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    if len(sentences) <= 1:
        return prompt  # nothing to delete — keep as-is
    user = (
        "Remove one sentence from this instruction prompt — the one that "
        "contributes the least to code-generation quality. Keep everything "
        "else verbatim. Output only the modified prompt.\n\n"
        f"Prompt:\n{prompt}"
    )
    return _call(
        phase="mutate_delete",
        model=config.OPTIMIZER_MODEL,
        temperature=config.OPTIMIZER_TEMPERATURE,
        max_tokens=config.MAX_OPERATOR_TOKENS,
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
        temperature=config.OPTIMIZER_TEMPERATURE,
        max_tokens=config.MAX_OPERATOR_TOKENS,
        system=_MUT_SYSTEM,
        user=user,
    ).strip()
