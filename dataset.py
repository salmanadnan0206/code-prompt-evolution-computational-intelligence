"""
dataset.py — Load and split APPS Codewars introductory fn_name problems.

Why Codewars only?
  • 2,346 introductory problems from codewars.com, all standalone functions.
  • Consistent execution format:
      inputs[i]  = Python list of arguments already parsed (e.g. [1] or [2, "x"])
      outputs[i] = [return_value]  — the result wrapped in a single-element list
  • LeetCode-style problems use class Solution, which complicates extraction.
"""

import json
import random
import sys
from pathlib import Path


def load_problems(apps_train_dir: str | Path) -> list[dict]:
    """
    Return every Codewars introductory fn_name problem from the APPS train set.

    Each problem dict has:
        id        – folder name (e.g. "3715")
        question  – full problem text
        fn_name   – function the LLM must write
        inputs    – list of test-case argument lists
        outputs   – list of [expected_return_value]
    """
    sys.set_int_max_str_digits(1_000_000)   # needed for a handful of problems
    problems = []
    train_dir = Path(apps_train_dir)

    for prob_dir in train_dir.iterdir():
        meta_f = prob_dir / "metadata.json"
        io_f   = prob_dir / "input_output.json"
        q_f    = prob_dir / "question.txt"

        if not (meta_f.exists() and io_f.exists() and q_f.exists()):
            continue

        try:
            meta = json.loads(meta_f.read_text())
            io   = json.loads(io_f.read_text())
        except Exception:
            continue   # skip malformed JSON

        # Keep only: introductory difficulty, fn_name style, Codewars source
        if (
            meta.get("difficulty") != "introductory"
            or "fn_name" not in io
            or "codewars.com" not in meta.get("url", "")
            or not io.get("inputs")
            or not io.get("outputs")
        ):
            continue

        problems.append({
            "id":       prob_dir.name,
            "question": q_f.read_text().strip(),
            "fn_name":  io["fn_name"],
            "inputs":   io["inputs"],
            "outputs":  io["outputs"],
        })

    return problems


def split(
    problems: list[dict],
    evolution_size: int,
    holdout_size: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Shuffle and split into:
        evolution — used for fitness evaluation every generation
        holdout   — unseen until final evaluation
        rest      — remaining problems (not used during training)
    """
    rng = random.Random(seed)
    shuffled = problems.copy()
    rng.shuffle(shuffled)

    evolution = shuffled[:evolution_size]
    holdout   = shuffled[evolution_size : evolution_size + holdout_size]
    rest      = shuffled[evolution_size + holdout_size :]

    return evolution, holdout, rest
