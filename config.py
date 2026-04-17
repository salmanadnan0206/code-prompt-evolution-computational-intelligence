"""
config.py — All hyperparameters and paths in one place.
Edit here; everything else reads from this module.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the current working directory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
APPS_TRAIN   = ROOT / "Dataset Examples" / "APPS" / "train"
RESULTS_DIR  = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Only Codewars introductory fn_name problems are used (consistent format).
EVOLUTION_SIZE = 100   # fixed pool for fitness evaluation during GA
HOLDOUT_SIZE   = 200   # held-out for final within-distribution test
RANDOM_SEED    = 42

# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
POPULATION_SIZE   = 15
GENERATIONS       = 20
ELITISM_COUNT     = 2   # top-N individuals carried unchanged to next generation
TOURNAMENT_SIZE   = 3   # contestants per tournament selection draw

CROSSOVER_RATE      = 0.8   # probability crossover is applied to each offspring

# Independent per-individual mutation probabilities (can all apply at once)
MUT_INJECT_PROB   = 0.25   # add a new instructional strategy
MUT_DELETE_PROB   = 0.15   # drop a random sentence
MUT_REPHRASE_PROB = 0.10   # rewrite without changing meaning

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model being optimised — generates code, evaluated for fitness.
TARGET_MODEL       = "claude-haiku-4-5-20251001"
TARGET_TEMPERATURE = 0.0   # deterministic for repeatable fitness scores

# Model used for crossover / mutation meta-prompts — needs creativity.
OPTIMIZER_MODEL       = "claude-sonnet-4-6"
OPTIMIZER_TEMPERATURE = 0.7

MAX_CODE_TOKENS     = 1024
MAX_OPERATOR_TOKENS = 600

# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------
EXEC_TIMEOUT = 5   # seconds per subprocess evaluation
