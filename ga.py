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

import random
import uuid
from dataclasses import dataclass, field

import config
import evaluator
import llm


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

def compute_fitness(individual: Individual, problems: list[dict]) -> float:
    """
    Evaluate a prompt on the evolution pool.

    For each problem:
        1. LLM generates code using the candidate prompt as system instruction.
        2. Code is extracted and executed against all test cases.
        3. Score = fraction of test cases passed  (partial credit).

    Fitness = mean score across all problems.
    """
    total = 0.0
    for prob in problems:
        response = llm.generate_code(
            system_prompt=individual.prompt,
            question=prob["question"],
            fn_name=prob["fn_name"],
        )
        code = evaluator.extract_code(response)
        score = evaluator.run_tests(
            code=code,
            fn_name=prob["fn_name"],
            inputs=prob["inputs"],
            outputs=prob["outputs"],
            timeout=config.EXEC_TIMEOUT,
        )
        total += score
    return total / len(problems)


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

    # --- crossover ---
    if rng.random() < config.CROSSOVER_RATE:
        prompt = llm.crossover(parent_a.prompt, parent_b.prompt)
        ops.append("crossover")
    else:
        prompt = rng.choice([parent_a.prompt, parent_b.prompt])
        ops.append("copy")

    # --- mutations (independent — all three can fire) ---
    if rng.random() < config.MUT_INJECT_PROB:
        prompt = llm.mutate_inject(prompt)
        ops.append("inject")
    if rng.random() < config.MUT_DELETE_PROB:
        prompt = llm.mutate_delete(prompt)
        ops.append("delete")
    if rng.random() < config.MUT_REPHRASE_PROB:
        prompt = llm.mutate_rephrase(prompt)
        ops.append("rephrase")

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
    on_generation=None,
) -> tuple[Individual, list[dict]]:
    """
    Run the full genetic algorithm.

    Args:
        seed_prompts   — initial population prompts (len == POPULATION_SIZE)
        problems       — evolution pool (100 APPS problems)
        on_generation  — optional callback(gen: int, population, stats)
                         called after each generation for checkpointing

    Returns:
        (best_individual, history)
    """
    rng = random.Random(config.RANDOM_SEED)
    history: list[dict] = []

    # --- initialise population ---
    population = [Individual(prompt=p) for p in seed_prompts]
    print(f"[gen 0] evaluating {len(population)} seed individuals …")
    for ind in population:
        ind.fitness = compute_fitness(ind, problems)
        print(f"  {ind.uid}  fitness={ind.fitness:.4f}")

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
        print(
            f"[gen {gen-1}] best={stats['best']:.4f}  "
            f"mean={stats['mean']:.4f}  worst={stats['worst']:.4f}"
        )

        if on_generation:
            on_generation(gen - 1, population, stats)

        # --- elitism: carry over top ELITISM_COUNT ---
        new_pop = population[: config.ELITISM_COUNT]

        # --- breed offspring to fill remaining slots ---
        while len(new_pop) < config.POPULATION_SIZE:
            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)
            child = breed(p1, p2, rng)
            new_pop.append(child)

        # --- evaluate only the new offspring ---
        print(f"[gen {gen}] evaluating {config.POPULATION_SIZE - config.ELITISM_COUNT} offspring …")
        for ind in new_pop[config.ELITISM_COUNT :]:
            ind.fitness = compute_fitness(ind, problems)
            print(f"  {ind.uid}  fitness={ind.fitness:.4f}  ops={ind.operators}")

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
    print(
        f"[gen {config.GENERATIONS}] FINAL  best={max(fits):.4f}  "
        f"mean={sum(fits)/len(fits):.4f}"
    )

    return population[0], history
