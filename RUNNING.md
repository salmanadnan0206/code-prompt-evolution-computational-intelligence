# RUNNING.md — How to start, monitor, inspect, and stop a GA run

This is a complete operational runbook. Every command is explained flag-by-flag.

---

## 1. What this program does (one paragraph)

A Genetic Algorithm evolves **system-level instruction prompts** for Claude Haiku.
Each candidate prompt (a "chromosome") is tested on 100 Codeforces problems —
Haiku generates C++ for each problem, g++ compiles it, the binary runs against
the problem's test cases, and the score = (tests passed) / (total tests).
Higher-scoring prompts survive and are bred (crossover + mutation) by Claude
Sonnet to produce the next generation.

---

## 2. Prerequisites

- Claude Code CLI installed and authenticated (`claude auth login`)
- `g++` with `-std=c++17` support
- Python 3.11+ (uses `sys.set_int_max_str_digits`)

Check everything is in place:

```bash
which claude && which g++ && python3 --version
```

Expected: three paths printed, no errors.

---

## 3. Starting a fresh run

Open a terminal (call it **Terminal 1**). Run:

```bash
cd "/home/rolex/Salman Adnan/HU Files/8th Semester/Computational Intelligence/Project/Code" && python3 -u main.py 2>&1 | tee run.log
```

**Breakdown of the command:**

| Part | Meaning |
|---|---|
| `cd "..."` | Change to the Code directory (quoted because path has spaces) |
| `python3` | Python 3 interpreter |
| `-u` | **Unbuffered output** — print statements flush immediately instead of sitting in memory. Without this you'd see nothing for hours because Python buffers stdout in non-interactive mode. |
| `main.py` | The entry script |
| `2>&1` | Redirect stderr (file descriptor 2) into stdout (fd 1). Merges error messages into the main output stream. |
| `\|` | Pipe — feed left side's output into right side's input |
| `tee` | T-junction: reads stdin, writes to BOTH the terminal AND a file |
| `run.log` | File that `tee` saves to (in the Code/ directory) |

**Effect:** you see live output in the terminal AND a permanent copy is saved in `run.log`.

### What you'll see in Terminal 1 immediately:

```
[MAIN] ==== START ====
[MAIN] pre-flight: checking claude CLI
[MAIN] pre-flight: claude CLI OK
[MAIN] pre-flight: dataset dirs exist
[MAIN] loading 300 pre-selected Codeforces problems …
[MAIN] loaded 300 problems
[MAIN] splitting into evolution / holdout
[MAIN]   Evolution pool : 100 problems
[MAIN]   Held-out test  : 200 problems
[MAIN] run directory created: .../results/run_20260421_120000
[MAIN] ==== STARTING GA ====
[HH:MM:SS] [INFO] [gen 0] evaluating 15 seed individuals …
[HH:MM:SS] [DEBUG] ga.run  creating initial population of 15 individuals
[HH:MM:SS] [DEBUG] [gen 0] evaluating seed 1/15  uid=abc12345
[HH:MM:SS] [INFO] ind=abc12345  prob=3/100  id=train/2179  score=1.00  reason=perfect  llm=15.3s
...
```

Log levels:
- `[INFO]` — high-level events (one per completed problem, one per generation boundary)
- `[DEBUG]` — fine-grained progress (entering/exiting functions, sub-steps)
- `[ERROR]` — errors (rare, recoverable)

---

## 4. Finding the active run directory (Terminal 2)

Open a SECOND terminal. Get the path of the current run:

```bash
ls -dt "/home/rolex/Salman Adnan/HU Files/8th Semester/Computational Intelligence/Project/Code/results/run_"* | head -1
```

**Breakdown:**
- `ls -dt` — `-d` list directories as entries (not their contents), `-t` sort by modification time
- `run_"*` — glob pattern matching all run folders
- `head -1` — first (newest) one

Save this in a shell variable so you don't retype it:

```bash
RUN=$(ls -dt "/home/rolex/Salman Adnan/HU Files/8th Semester/Computational Intelligence/Project/Code/results/run_"* | head -1)
echo "$RUN"
```

Now every command below uses `$RUN`.

---

## 5. Monitoring the run (Terminal 2)

All of these are **read-only** — they do NOT affect the running program. Run any of them whenever.

### 5.1 Live human-readable timeline

```bash
tail -f "$RUN/progress.log"
```

**Breakdown:**
- `tail` — show the end of a file
- `-f` — follow mode: keeps file open and streams new lines as they're appended
- Stop watching with `Ctrl+C` (stops only the `tail`, not the main run)

### 5.2 Count completed evaluations

```bash
wc -l "$RUN/evaluations_gen00.jsonl"
```

**Breakdown:**
- `wc` = word count, `-l` flag = count LINES
- Since every evaluation is exactly one line of JSON, line count = completed evaluations

### 5.3 Rich dashboard (rate, ETA, score breakdown)

```bash
python3 /tmp/monitor_ga.py
```

Prints per-generation counts, throughput (calls/min), ETA for gen 0 completion, and breakdown of score reasons for the last 100 calls.

### 5.4 Check main process is still alive

```bash
ps aux | grep "python3 -u main.py" | grep -v grep
```

**Breakdown:**
- `ps aux` — list ALL processes on the system
- `grep "python3 -u main.py"` — filter for our process
- `grep -v grep` — remove the grep command itself from the output (else it finds its own command line)

If empty output → the run crashed or finished.

### 5.5 Count active Claude subprocesses

```bash
ps --ppid $(pgrep -f "python3 -u main.py") -o comm= | grep -c claude
```

**Breakdown:**
- `pgrep -f "python3 -u main.py"` — find the main.py PID
- `ps --ppid <pid>` — list children of that PID
- `-o comm=` — print only the command name (no header)
- `grep -c claude` — count lines containing "claude"

With `PARALLEL_EVALS=10`, this should oscillate between 1 and 10 during active evaluation.

---

## 6. Inspecting the 50 evaluations (smoke test)

Wait until `wc -l` shows at least 50. Then inspect with `jq` (a JSON query tool).

### 6.1 Score breakdown (reasons)

```bash
cat "$RUN/evaluations_gen00.jsonl" | jq -r '.score_reason' | sort | uniq -c | sort -rn
```

**Breakdown:**
- `cat ... | jq -r '.score_reason'` — extract `score_reason` field from each line (`-r` = raw, no quotes)
- `sort | uniq -c` — sort then count duplicates (`-c` prefixes each with count)
- `sort -rn` — sort the counts in reverse numeric order

**Possible score_reason values:**
| Reason | Meaning |
|---|---|
| `perfect` | all tests passed — score = 1.0 |
| `partial_pass` | some passed, some failed — 0 < score < 1.0 |
| `all_fail` | all tests ran, all failed — score = 0.0 |
| `all_fail_early_stop` | first 3 tests failed → stopped — score = 0.0 |
| `compile_fail` | g++ rejected the code — score = 0.0 |
| `empty_code` | Claude returned nothing/garbage — score = 0.0 |
| `empty_response` | same as above (logged from ga.py path) |
| `no_tests` | problem had no test cases (shouldn't happen — pre-filtered) |

### 6.2 Average score per seed

```bash
jq -r '"\(.individual_uid) \(.score)"' "$RUN/evaluations_gen00.jsonl" | awk '{s[$1]+=$2;n[$1]++}END{for(k in s)printf "%s  avg=%.3f  count=%d\n",k,s[k]/n[k],n[k]}' | sort -k2 -r
```

**Breakdown:**
- `jq -r` outputs `"<uid> <score>"` per line
- `awk` accumulates sum and count per uid, then prints `uid avg=X count=Y`
- `sort -k2 -r` sorts by column 2 (avg) descending

Tells you which seed prompts are doing best so far.

### 6.3 Pretty-print one full evaluation

```bash
head -1 "$RUN/evaluations_gen00.jsonl" | jq .
```

Shows: timestamp, generation, uid, operators, full system prompt, problem ID, rating, LLM time, extracted C++ code, compile result, per-test-case breakdown, final score + reason.

### 6.4 Look at a failure

```bash
jq -c 'select(.score == 0.0)' "$RUN/evaluations_gen00.jsonl" | head -1 | jq .
```

**Breakdown:**
- `jq -c 'select(.score == 0.0)'` — filter to score=0 entries, compact output (one per line)
- `head -1` — first match
- `jq .` — pretty-print

Read the `compile.error`, `test_cases[].actual` vs `expected`, and `extracted_code` to understand why it scored 0.

### 6.5 Get the generated code for one problem

```bash
jq -r 'select(.problem_id == "train/0002") | .extracted_code' "$RUN/evaluations_gen00.jsonl"
```

Prints the raw C++ each seed produced for that problem.

### 6.6 Find compile failures and their error messages

```bash
jq -c 'select(.score_reason == "compile_fail") | {uid: .individual_uid, prob: .problem_id, err: (.compile.error[0:200])}' "$RUN/evaluations_gen00.jsonl" | head -10
```

First 10 compile errors with a 200-char excerpt of the g++ error.

### 6.7 Find slowest LLM calls

```bash
jq -c 'select(.llm_call_time_s > 30) | {uid: .individual_uid, prob: .problem_id, t: .llm_call_time_s}' "$RUN/evaluations_gen00.jsonl"
```

Calls that took >30s to return from Claude.

---

## 7. Decision point — continue or abort?

After inspecting the first 50, decide:

### 7.1 Output looks good → do nothing

The run keeps going automatically. You can close Terminal 2 (monitoring) — the main run stays alive as long as Terminal 1 is open.

If you want to disconnect Terminal 1 without killing the run, use `screen` or `tmux` (beyond this document's scope), OR the run must stay attached.

### 7.2 Output looks wrong → stop the run

**In Terminal 1, press `Ctrl+C`.**

This sends `SIGINT` to `python3`. The `ThreadPoolExecutor` context manager will attempt to shut down cleanly, but any mid-flight `claude -p` subprocess continues until it returns.

If processes linger after Ctrl+C:

```bash
pkill -f "claude -p"
pkill -f "python3 -u main.py"
```

**`pkill -f <pattern>`:**
- `pkill` — kill processes by name
- `-f` — match against the full command line (not just the process name)
- `<pattern>` — substring to match

Confirm all clean:

```bash
ps aux | grep -E "main\.py|claude -p" | grep -v grep | wc -l
```

Should print `0`.

---

## 8. Output file structure

After a run (or during it), the run directory contains:

```
results/run_20260421_120000/
├── config.json              Snapshot of hyperparameters for this run
├── evolution_ids.json       List of 100 problem IDs used for fitness
├── progress.log             Human-readable timeline (INFO + DEBUG + ERROR)
├── calls.jsonl              Per-call cost record (model, tokens, $)
├── evaluations_gen00.jsonl  Per-evaluation records for generation 0
├── evaluations_gen01.jsonl  Per-evaluation records for generation 1
├── ...
├── evaluations_gen20.jsonl  Per-evaluation records for generation 20
├── generation_00.json       Population snapshot after gen 0 (for crash recovery)
├── generation_01.json       Population snapshot after gen 1
├── ...
├── history.json             Per-generation stats (best/mean/worst fitness)
├── best_prompt.txt          Plain text of the best evolved prompt
└── summary.json             Final summary (elapsed, best fitness, holdout score)
```

Separately, at `Code/../Costs/cost_20260421_120000.json` — final cost report.

---

## 9. Fields in evaluations_genNN.jsonl

Every line is one JSON object with these keys:

| Key | Type | Description |
|---|---|---|
| `ts` | string | `HH:MM:SS` timestamp |
| `generation` | int | Generation number (0 = seeds) |
| `individual_uid` | string | 8-char hex unique ID of the prompt being evaluated |
| `individual_operators` | string | `"seed"` for gen 0, else `"crossover+inject"` etc. |
| `system_prompt` | string | The full chromosome (prompt being evolved) |
| `problem_id` | string | `"train/0002"` or `"test/0197"` format |
| `problem_rating` | int | Codeforces difficulty rating (800–1700) |
| `problem_index` | int | Index within the 100-problem evolution pool |
| `llm_call_time_s` | float | Time for Haiku to respond |
| `extracted_code` | string | C++ code pulled from the LLM response |
| `compile.success` | bool | Did g++ succeed? |
| `compile.error` | string\|null | First 500 chars of g++ stderr if failed |
| `compile.time_s` | float | g++ wall time |
| `test_cases` | list | Per-test-case results (see below) |
| `tests_total` | int | Number of available test cases |
| `tests_run` | int | Number actually executed (early stopping reduces this) |
| `early_stopped` | bool | True if run_tests hit the 3-consecutive-fail threshold |
| `total_test_time_s` | float | Wall time running all test cases |
| `score` | float | (tests passed) / tests_total |
| `score_reason` | string | One of the 8 reasons above |

Each entry in `test_cases` contains:

| Key | Type | Description |
|---|---|---|
| `index` | int | Test case number |
| `passed` | bool | Matched expected output? |
| `actual` | string\|null | Program's stdout (truncated to 500 chars, null if timeout) |
| `expected` | string | Problem's expected output (truncated to 500 chars) |
| `time_s` | float | Binary execution time |
| `timeout` | bool | Did it hit the 2s limit? |

---

## 10. Troubleshooting

### Nothing appears in Terminal 1

You forgot the `-u` flag. Python is buffering. Kill with Ctrl+C, restart with `python3 -u main.py`.

### "claude: command not found"

Run `claude auth login` to set up. If still broken, `which claude` returns nothing → install Claude Code CLI first.

### "Loaded X problems" where X < 300

Some problem folders are missing `question.txt` or `input_output.json`, or have empty test cases. Warnings are printed above the count. Usually harmless.

### Run crashes mid-evaluation

Latest `generation_XX.json` is on disk. Current config has no resume feature — restarting = fresh run. (Resume feature can be added; ask if needed.)

### All scores are 0.00 for one seed

That seed's prompt is generating consistently bad C++ (probably compile failures). Use command 6.6 to see the g++ errors. Normal GA behavior — that prompt gets eliminated by selection.

### All scores are 0.00 for ALL seeds

Something is systematically wrong. Check:
1. `jq '.compile.success' "$RUN/evaluations_gen00.jsonl" | sort | uniq -c` — if all false, g++ can't compile anything → check g++ installation or dataset format
2. `jq '.extracted_code | length' "$RUN/evaluations_gen00.jsonl" | sort | uniq -c` — if all 0, Claude is returning empty → check CLI auth

---

## 11. Quick-reference cheat sheet

```bash
# START
cd "/home/rolex/Salman Adnan/HU Files/8th Semester/Computational Intelligence/Project/Code"
python3 -u main.py 2>&1 | tee run.log

# ACTIVE RUN PATH
RUN=$(ls -dt /home/rolex/Salman\ Adnan/HU\ Files/8th\ Semester/Computational\ Intelligence/Project/Code/results/run_* | head -1)

# MONITOR
tail -f "$RUN/progress.log"                          # live log
wc -l "$RUN/evaluations_gen00.jsonl"                 # eval count
python3 /tmp/monitor_ga.py                           # dashboard

# INSPECT (after 50 calls)
cat "$RUN/evaluations_gen00.jsonl" | jq -r '.score_reason' | sort | uniq -c   # reasons
head -1 "$RUN/evaluations_gen00.jsonl" | jq .                                 # one full entry
jq -c 'select(.score == 0)' "$RUN/evaluations_gen00.jsonl" | head -1 | jq .   # a failure

# STOP
# In Terminal 1: Ctrl+C
pkill -f "claude -p"                                 # clean up stragglers
```
