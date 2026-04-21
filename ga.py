"""
ga.py — Genetic Algorithm for evolving system-level instruction prompts.

Pipeline summary (one generation):
    ┌──────────────────────────────────────────────────────────┐
    │  1. SELECTION    — tournament (size 3), pick 2 parents   │
    │  2. CROSSOVER    — LLM merges parents (rate 0.8)         │
    │  3. MUTATION     — inject / delete / rephrase (indep.)   │
    │  4. FITNESS      — run code on 100 APPS problems         │
    │  5. REPLACEMENT  — generational with top-2 elitism       │
    └──────────────────────────────────────────────────────────┘

Selection     : tournament (size 3) — balanced exploration vs exploitation
Crossover     : LLM-mediated merge — preserves semantic coherence
Mutation      : 3 independent types — inject(0.25), delete(0.15), rephrase(0.10)
Elitism       : top 2 — prevents losing the best prompt discovered so far
Replacement   : generational — entire population (except elites) is replaced
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import config
import evaluator
import llm

# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------
# Two log streams per run:
#   1. progress.log           — human-readable timeline (INFO + DEBUG lines)
#   2. evaluations_genNN.jsonl — one JSON line per problem evaluation
# ---------------------------------------------------------------------------
_progress_log: Path | None = None
_eval_log_dir: Path | None = None
_log_lock = threading.Lock()
_eval_log_lock = threading.Lock()


def _log(msg: str, level: str = "INFO") -> None:
    """Append a timestamped, level-tagged line to progress.log. Thread-safe."""
    line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if _progress_log is not None:
            with _progress_log.open("a") as f:
                f.write(line + "\n")


def _debug(msg: str) -> None:
    """Shortcut for DEBUG-level log entries."""
    _log(msg, level="DEBUG")


def _log_evaluation(entry: dict) -> None:
    """
    Append one evaluation record to evaluations_gen{NN}.jsonl.
    Entry is a pre-built dict; one JSON object per line.
    Thread-safe.
    """
    if _eval_log_dir is None:
        return
    gen = entry.get("generation", 0)
    path = _eval_log_dir / f"evaluations_gen{gen:02d}.jsonl"
    line = json.dumps(entry, ensure_ascii=False)
    with _eval_log_lock:
        with path.open("a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One chromosome = one system-level instruction prompt."""
    prompt: str
    fitness: float = -1.0
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_a: str = ""      # uid of parent A (empty for seeds)
    parent_b: str = ""      # uid of parent B
    operators: str = ""     # operators applied, e.g. "crossover+inject"


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _eval_one_problem(
    gen: int,
    individual: Individual,
    prob: dict,
    idx: int,
    total: int,
) -> float:
    """
    Evaluate one problem for one individual. Runs in a thread.
    Logs a full JSONL record to evaluations_gen{NN}.jsonl.

    Startup jitter: random 0-2s sleep staggers subprocess launches so that
    multiple workers don't all spawn `claude -p` at the exact same instant.
    Addresses CLI contention that previously caused ~19% empty-response failures.
    """
    time.sleep(random.uniform(0, 2))

    _debug(f"ind={individual.uid}  prob={idx+1}/{total}  id={prob['id']}  BEGIN (calling Claude)")

    # --- Claude call ---
    t_llm = time.monotonic()
    response = llm.generate_code(
        system_prompt=individual.prompt,
        question=prob["question"],
    )
    llm_time = round(time.monotonic() - t_llm, 3)
    _debug(f"ind={individual.uid}  prob={idx+1}/{total}  id={prob['id']}  LLM done ({llm_time}s, {len(response)} chars)")

    # --- extract code ---
    code = evaluator.extract_code(response)
    del response

    # --- run tests (verbose) ---
    if not code or not code.strip():
        test_result = {
            "score": 0.0,
            "reason": "empty_response",
            "compile": {"success": False, "error": None, "time_s": 0.0},
            "test_cases": [],
            "tests_total": len(prob.get("inputs", [])),
            "tests_run": 0,
            "early_stopped": False,
            "total_test_time_s": 0.0,
        }
        _debug(f"ind={individual.uid}  prob={idx+1}/{total}  id={prob['id']}  empty response — skipping compile/tests")
    else:
        _debug(f"ind={individual.uid}  prob={idx+1}/{total}  id={prob['id']}  compiling + testing ({len(code)} chars of code)")
        test_result = evaluator.run_tests_verbose(
            code=code,
            inputs=prob["inputs"],
            outputs=prob["outputs"],
            timeout=config.EXEC_TIMEOUT,
        )

    score = test_result["score"]
    reason = test_result["reason"]

    # --- write full JSONL record ---
    entry = {
        "ts":                   time.strftime("%H:%M:%S"),
        "generation":           gen,
        "individual_uid":       individual.uid,
        "individual_operators": individual.operators or "seed",
        "system_prompt":        individual.prompt,
        "problem_id":           prob["id"],
        "problem_rating":       prob.get("rating"),
        "problem_index":        idx,
        "llm_call_time_s":      llm_time,
        "extracted_code":       code,
        "compile":              test_result["compile"],
        "test_cases":           test_result["test_cases"],
        "tests_total":          test_result["tests_total"],
        "tests_run":            test_result["tests_run"],
        "early_stopped":        test_result["early_stopped"],
        "total_test_time_s":    test_result["total_test_time_s"],
        "score":                score,
        "score_reason":         reason,
    }
    _log_evaluation(entry)
    del entry, code

    _log(
        f"ind={individual.uid}  prob={idx+1}/{total}  id={prob['id']}  "
        f"score={score:.2f}  reason={reason}  llm={llm_time}s"
    )
    return score


def compute_fitness(gen: int, individual: Individual, problems: list[dict]) -> float:
    """
    Evaluate a prompt on the evolution pool using parallel threads.

    PARALLEL_EVALS problems are evaluated simultaneously per individual.
    Each thread: LLM generates C++ → extract → compile with g++ → run tests → log JSONL.

    Fitness = mean score across all problems.
    """
    _debug(f"compute_fitness BEGIN  gen={gen}  ind={individual.uid}  problems={len(problems)}  workers={config.PARALLEL_EVALS}")
    t0 = time.monotonic()
    scores = [0.0] * len(problems)

    with ThreadPoolExecutor(max_workers=config.PARALLEL_EVALS) as executor:
        futures = {
            executor.submit(_eval_one_problem, gen, individual, prob, i, len(problems)): i
            for i, prob in enumerate(problems)
        }
        _debug(f"compute_fitness  gen={gen}  ind={individual.uid}  submitted {len(futures)} futures")

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                scores[idx] = future.result()
            except Exception as exc:
                _log(f"ind={individual.uid}  prob={idx+1}  ERROR: {exc}", level="ERROR")
                scores[idx] = 0.0
            completed += 1
            if completed % 10 == 0:
                _debug(f"compute_fitness  gen={gen}  ind={individual.uid}  progress {completed}/{len(problems)}")

    fitness = sum(scores) / len(scores)
    elapsed = round(time.monotonic() - t0, 1)
    _debug(f"compute_fitness END  gen={gen}  ind={individual.uid}  fitness={fitness:.4f}  elapsed={elapsed}s")
    return fitness


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_select(population: list[Individual], rng: random.Random) -> Individual:
    """Pick TOURNAMENT_SIZE random individuals, return the fittest."""
    contestants = rng.sample(population, config.TOURNAMENT_SIZE)
    return max(contestants, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# Crossover + mutation → one offspring
# ---------------------------------------------------------------------------

def breed(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Individual:
    """
    Produce one offspring from two parents.

    1. Crossover (probability CROSSOVER_RATE):
        The optimizer LLM merges the two parent prompts.
        If skipped, the offspring copies a random parent's prompt.

    2. Mutation (three independent rolls per offspring):
        - inject   (0.25) — LLM adds a new instructional strategy
        - delete   (0.15) — LLM removes the weakest sentence
        - rephrase (0.10) — LLM rewrites without changing meaning

    Returns an unevaluated Individual (fitness = -1).
    """
    ops = []
    _debug(f"breed BEGIN  parents={parent_a.uid}+{parent_b.uid}")

    # --- crossover ---
    if rng.random() < config.CROSSOVER_RATE:
        _debug(f"breed  calling crossover LLM")
        result = llm.crossover(parent_a.prompt, parent_b.prompt)
        prompt = result if result.strip() else rng.choice([parent_a.prompt, parent_b.prompt])
        ops.append("crossover")
        _debug(f"breed  crossover done (prompt={len(prompt)} chars)")
    else:
        prompt = rng.choice([parent_a.prompt, parent_b.prompt])
        ops.append("copy")
        _debug(f"breed  copy (skipped crossover)")

    # --- mutations (independent — all three can fire) ---
    if rng.random() < config.MUT_INJECT_PROB:
        _debug(f"breed  calling mutate_inject")
        result = llm.mutate_inject(prompt)
        prompt = result if result.strip() else prompt
        ops.append("inject")
    if rng.random() < config.MUT_DELETE_PROB:
        _debug(f"breed  calling mutate_delete")
        result = llm.mutate_delete(prompt)
        prompt = result if result.strip() else prompt
        ops.append("delete")
    if rng.random() < config.MUT_REPHRASE_PROB:
        _debug(f"breed  calling mutate_rephrase")
        result = llm.mutate_rephrase(prompt)
        prompt = result if result.strip() else prompt
        ops.append("rephrase")

    _debug(f"breed END  ops={'+'.join(ops)}  final_prompt={len(prompt)} chars")

    return Individual(
        prompt=prompt,
        parent_a=parent_a.uid,
        parent_b=parent_b.uid,
        operators="+".join(ops),
    )


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run(
    seed_prompts: list[str],
    problems: list[dict],
    run_dir: Path | None = None,
    on_generation=None,
) -> tuple[Individual, list[dict]]:
    """
    Run the full genetic algorithm.

    Args:
        seed_prompts   — initial population prompts (len == POPULATION_SIZE)
        problems       — evolution pool (100 APPS problems)
        run_dir        — results directory; progress.log is written here
        on_generation  — optional callback(gen: int, population, stats)
                        called after each generation for checkpointing

    Returns:
        (best_individual, history)
    """
    global _progress_log, _eval_log_dir
    if run_dir is not None:
        _progress_log = run_dir / "progress.log"
        _eval_log_dir = run_dir
        _debug(f"ga.run  logging to {_progress_log}")
        _debug(f"ga.run  evaluations JSONL dir = {_eval_log_dir}")

    rng = random.Random(config.RANDOM_SEED)
    history: list[dict] = []

    # --- initialise population ---
    _debug(f"ga.run  creating initial population of {len(seed_prompts)} individuals")
    population = [Individual(prompt=p, operators="seed") for p in seed_prompts]
    _log(f"[gen 0] evaluating {len(population)} seed individuals …")
    for seed_idx, ind in enumerate(population):
        _debug(f"[gen 0] evaluating seed {seed_idx+1}/{len(population)}  uid={ind.uid}")
        ind.fitness = compute_fitness(0, ind, problems)
        _log(f"  SEED DONE  {ind.uid}  fitness={ind.fitness:.4f}")

    for gen in range(1, config.GENERATIONS + 1):
        # --- record stats ---
        population.sort(key=lambda x: x.fitness, reverse=True)
        fits = [x.fitness for x in population]
        stats = {
            "generation": gen - 1,
            "best": max(fits),
            "mean": sum(fits) / len(fits),
            "worst": min(fits),
            "best_uid": population[0].uid,
            "best_prompt": population[0].prompt,
        }
        history.append(stats)
        _log(
            f"[gen {gen-1}] STATS  best={stats['best']:.4f}  "
            f"mean={stats['mean']:.4f}  worst={stats['worst']:.4f}"
        )

        if on_generation:
            _debug(f"[gen {gen-1}] saving checkpoint")
            on_generation(gen - 1, population, stats)

        # --- elitism: carry over top ELITISM_COUNT ---
        new_pop = population[: config.ELITISM_COUNT]
        _debug(f"[gen {gen}] carrying {config.ELITISM_COUNT} elites: {[e.uid for e in new_pop]}")

        # --- breed offspring to fill remaining slots ---
        _log(f"[gen {gen}] breeding {config.POPULATION_SIZE - config.ELITISM_COUNT} offspring …")
        while len(new_pop) < config.POPULATION_SIZE:
            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)
            _debug(f"[gen {gen}] breed #{len(new_pop) - config.ELITISM_COUNT + 1}  parents={p1.uid}+{p2.uid}")
            child = breed(p1, p2, rng)
            new_pop.append(child)

        # --- evaluate only the new offspring ---
        _log(f"[gen {gen}] evaluating {config.POPULATION_SIZE - config.ELITISM_COUNT} offspring …")
        for off_idx, ind in enumerate(new_pop[config.ELITISM_COUNT:]):
            _debug(f"[gen {gen}] evaluating offspring {off_idx+1}/{config.POPULATION_SIZE - config.ELITISM_COUNT}  uid={ind.uid}  ops={ind.operators}")
            ind.fitness = compute_fitness(gen, ind, problems)
            _log(f"  OFFSPRING DONE  {ind.uid}  fitness={ind.fitness:.4f}  ops={ind.operators}")

        population = new_pop

    # --- final stats ---
    population.sort(key=lambda x: x.fitness, reverse=True)
    fits = [x.fitness for x in population]
    history.append({
        "generation": config.GENERATIONS,
        "best": max(fits),
        "mean": sum(fits) / len(fits),
        "worst": min(fits),
        "best_uid": population[0].uid,
        "best_prompt": population[0].prompt,
    })
    _log(
        f"[gen {config.GENERATIONS}] FINAL  best={max(fits):.4f}  "
        f"mean={sum(fits)/len(fits):.4f}"
    )

    return population[0], history
