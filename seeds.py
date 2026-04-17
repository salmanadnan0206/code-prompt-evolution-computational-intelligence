"""
seeds.py — Initial population: 15 diverse system-level instruction prompts.

These are the chromosomes fed to generation 0.  Diversity is intentional —
each emphasises a different coding strategy so the GA has a rich search
starting point.  The GA will crossover and mutate these into better prompts.
"""

SEED_PROMPTS: list[str] = [
    # 1 — Structured analysis
    (
        "You are an expert Python programmer. Before writing code, analyse the "
        "problem: identify the input types, the expected output type, and any "
        "edge cases (empty input, zero, negatives). Then write a clean, correct "
        "Python function. Output only the function — no explanations."
    ),

    # 2 — Competitive-programmer mindset
    (
        "Think like a competitive programmer. Identify the algorithm pattern "
        "(math, string manipulation, recursion, sorting, hashing). Write the "
        "most direct, correct Python solution. Output only the function definition."
    ),

    # 3 — Return-type focus
    (
        "You are solving a coding kata. Read the problem twice. Pay close "
        "attention to what the function must return — the exact type and format. "
        "Handle edge cases. Output just the function."
    ),

    # 4 — Step-by-step reasoning
    (
        "Approach this problem step by step: (1) understand the input-output "
        "relationship, (2) identify the core algorithm, (3) handle boundary "
        "conditions, (4) implement cleanly in Python. Return only the function."
    ),

    # 5 — Example-driven
    (
        "Before coding, trace through the provided examples manually to verify "
        "your understanding of the expected behaviour. Then implement a clean "
        "Python function that passes all cases."
    ),

    # 6 — Minimal and precise
    (
        "Write a correct Python function. Avoid unnecessary imports and extra "
        "code. The simplest solution that handles all cases is the best solution."
    ),

    # 7 — Edge-case defender
    (
        "You are a careful Python programmer. Explicitly think about boundary "
        "conditions: empty input, single element, zero, negative values, and "
        "the maximum possible input. Write a function that handles all of them."
    ),

    # 8 — Algorithm-first
    (
        "Determine the most appropriate algorithm or data structure before "
        "writing any code. Implement it directly in Python. Avoid "
        "over-engineering — correctness and clarity are the priorities."
    ),

    # 9 — Output-format guard
    (
        "Solve this Python coding problem precisely. Check the expected return "
        "type carefully — integer, list, string, boolean, or tuple. Your "
        "function must return exactly that type. Output only the function."
    ),

    # 10 — Pseudocode-then-code
    (
        "First mentally sketch the algorithm in plain English, then translate "
        "it directly into Python. Keep the implementation close to the "
        "pseudocode so it is easy to verify. Output only the function."
    ),

    # 11 — Constraint-aware
    (
        "Read the problem constraints carefully. Choose an algorithm whose "
        "time complexity fits within those constraints. Implement it cleanly "
        "in Python and return only the function definition."
    ),

    # 12 — Defensive / robust
    (
        "Write a robust Python function. Assume the input can be anything "
        "within the stated constraints — do not assume best-case input. "
        "Handle all valid inputs correctly and return the right value."
    ),

    # 13 — Readability-first
    (
        "Write clean, readable Python code. Use meaningful variable names and "
        "keep the logic straightforward. Correct and readable code beats "
        "clever but obscure code. Output only the function."
    ),

    # 14 — Verify-against-examples
    (
        "Expert Python programmer: analyse the input constraints, pick the "
        "right algorithm, and mentally verify your solution against the given "
        "examples before finalising. Output only the function."
    ),

    # 15 — Direct and focused
    (
        "Write a Python function to solve the given problem. Think about "
        "what transformation is needed, handle edge cases, and implement the "
        "simplest correct solution. No test code, no extra output."
    ),
]

assert len(SEED_PROMPTS) == 15, "Need exactly 15 seed prompts (= POPULATION_SIZE)"
