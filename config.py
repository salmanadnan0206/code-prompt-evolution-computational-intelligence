"""
config.py — All hyperparameters and paths in one place.
Edit here; everything else reads from this module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
APPS_TRAIN   = ROOT / "Dataset Examples" / "APPS" / "train"
APPS_TEST    = ROOT / "Dataset Examples" / "APPS" / "test"
RESULTS_DIR  = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# 300 pre-selected Codeforces problems (stdin/stdout, full C++ programs).
# Distribution: 100×(800|1000), 100×(1100|1200|1300), 65×(1400|1500), 35×(1600|1700)
EVOLUTION_SIZE  = 100   # fixed pool for fitness evaluation during GA
HOLDOUT_SIZE    = 200   # held-out for final within-distribution test
RANDOM_SEED     = 42
MAX_TEST_CASES  = 10    # cap per problem for speed (first N test cases used)

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
# LLM  (Claude Code CLI — uses Claude.ai subscription, no API key required)
# ---------------------------------------------------------------------------
# Model being optimised — generates code, evaluated for fitness.
TARGET_MODEL = "claude-haiku-4-5-20251001"

# Model used for crossover / mutation meta-prompts — needs creativity.
OPTIMIZER_MODEL = "claude-sonnet-4-6"

# NOTE: The `claude -p` CLI does not expose a --temperature flag.
#       TARGET_TEMPERATURE (0.0) and OPTIMIZER_TEMPERATURE (0.7) are retained
#       below as documentation, but they have no effect on CLI calls.
TARGET_TEMPERATURE    = 0.0
OPTIMIZER_TEMPERATURE = 0.7

MAX_CODE_TOKENS     = 1024   # kept for documentation; not enforced by CLI
MAX_OPERATOR_TOKENS = 600    # kept for documentation; not enforced by CLI

# Seconds to wait for a single `claude -p` subprocess before timing out.
CLAUDE_CLI_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Code execution  (g++ compilation + binary runs)
# ---------------------------------------------------------------------------
EXEC_TIMEOUT    = 5    # seconds per test-case binary run
COMPILE_TIMEOUT = 30   # seconds allowed for g++ compilation
