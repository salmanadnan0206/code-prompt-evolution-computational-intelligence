"""
main.py — Entry point.  Load data → run GA → save results → save cost report.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python main.py

Results are saved to Code/results/run_<timestamp>/
Cost reports are saved to Costs/cost_<timestamp>.json  (new file each run)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import dataset
import evaluator
import ga
import llm
from cost_tracker import tracker
from seeds import SEED_PROMPTS


def save_checkpoint(run_dir: Path, gen: int, population, stats):
    """Save population state after each generation for crash recovery."""
    gen_file = run_dir / f"generation_{gen:02d}.json"
    data = {
        "generation": gen,
        "stats": stats,
        "population": [
            {
                "uid": ind.uid,
                "prompt": ind.prompt,
                "fitness": ind.fitness,
                "parent_a": ind.parent_a,
                "parent_b": ind.parent_b,
                "operators": ind.operators,
            }
            for ind in population
        ],
    }
    gen_file.write_text(json.dumps(data, indent=2))


def evaluate_on_holdout(best_prompt: str, holdout: list[dict]) -> float:
    """Evaluate the best evolved prompt on the held-out test set."""
    total = 0.0
    for i, prob in enumerate(holdout):
        response = llm.generate_code(best_prompt, prob["question"], prob["fn_name"])
        code = evaluator.extract_code(response)
        score = evaluator.run_tests(
            code, prob["fn_name"], prob["inputs"], prob["outputs"], config.EXEC_TIMEOUT,
        )
        total += score
        if (i + 1) % 50 == 0:
            print(f"  holdout progress: {i+1}/{len(holdout)}")
    return total / len(holdout)


def main():
    # --- pre-flight checks ---
    if not config.ANTHROPIC_API_KEY:
        print("ERROR: set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    if not config.APPS_TRAIN.exists():
        print(f"ERROR: APPS train directory not found: {config.APPS_TRAIN}")
        sys.exit(1)

    # --- load and split dataset ---
    print("Loading APPS Codewars introductory problems …")
    all_problems = dataset.load_problems(config.APPS_TRAIN)
    print(f"  Loaded {len(all_problems)} problems")

    if len(all_problems) < config.EVOLUTION_SIZE + config.HOLDOUT_SIZE:
        print("ERROR: not enough problems after filtering.")
        sys.exit(1)

    evolution_pool, holdout, rest = dataset.split(
        all_problems, config.EVOLUTION_SIZE, config.HOLDOUT_SIZE, config.RANDOM_SEED,
    )
    print(f"  Evolution pool : {len(evolution_pool)} problems")
    print(f"  Held-out test  : {len(holdout)} problems")
    print(f"  Remaining      : {len(rest)} problems")

    # --- create run directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    (run_dir / "config.json").write_text(json.dumps({
        "population_size":  config.POPULATION_SIZE,
        "generations":      config.GENERATIONS,
        "crossover_rate":   config.CROSSOVER_RATE,
        "tournament_size":  config.TOURNAMENT_SIZE,
        "elitism_count":    config.ELITISM_COUNT,
        "mut_inject":       config.MUT_INJECT_PROB,
        "mut_delete":       config.MUT_DELETE_PROB,
        "mut_rephrase":     config.MUT_REPHRASE_PROB,
        "evolution_size":   config.EVOLUTION_SIZE,
        "holdout_size":     config.HOLDOUT_SIZE,
        "target_model":     config.TARGET_MODEL,
        "optimizer_model":  config.OPTIMIZER_MODEL,
        "random_seed":      config.RANDOM_SEED,
    }, indent=2))

    # save evolution problem IDs for reproducibility
    (run_dir / "evolution_ids.json").write_text(
        json.dumps([p["id"] for p in evolution_pool])
    )

    print(f"\nResults will be saved to: {run_dir}\n")

    # --- run GA ---
    start = time.time()
    best, history = ga.run(
        seed_prompts=SEED_PROMPTS,
        problems=evolution_pool,
        on_generation=lambda gen, pop, stats: save_checkpoint(run_dir, gen, pop, stats),
    )
    elapsed = time.time() - start

    # --- save results ---
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "best_prompt.txt").write_text(best.prompt)

    print(f"\n{'='*60}")
    print(f"GA complete in {elapsed/60:.1f} minutes")
    print(f"Best fitness (evolution pool): {best.fitness:.4f}")
    print(f"Best prompt:\n{best.prompt}")
    print(f"{'='*60}\n")

    # --- evaluate best prompt on held-out set ---
    print(f"Evaluating best prompt on {len(holdout)} held-out problems …")
    holdout_fitness = evaluate_on_holdout(best.prompt, holdout)
    print(f"Held-out fitness: {holdout_fitness:.4f}")

    # --- save final summary ---
    summary = {
        "elapsed_minutes": round(elapsed / 60, 1),
        "best_prompt": best.prompt,
        "best_fitness_evolution": best.fitness,
        "best_fitness_holdout": holdout_fitness,
        "best_uid": best.uid,
        "total_generations": config.GENERATIONS,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- save cost report to ../Costs/ ---
    cost_path = tracker.flush(extra={
        "run_dir": str(run_dir),
        "best_fitness_evolution": best.fitness,
        "best_fitness_holdout": holdout_fitness,
        "generations": config.GENERATIONS,
        "population_size": config.POPULATION_SIZE,
        "evolution_size": config.EVOLUTION_SIZE,
    })
    print(f"\nCost report saved to: {cost_path}")
    print(f"Total cost this run:  ${tracker.total():.2f}")
    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
