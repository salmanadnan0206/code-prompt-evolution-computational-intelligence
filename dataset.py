"""
dataset.py — Load and split 300 pre-selected Codeforces problems.

Problem format (stdin / stdout — no fn_name):
    inputs[i]  = raw string piped to the program's stdin
    outputs[i] = expected string on stdout

Rating distribution (300 total):
    Tier 1 — 800  rating : 50 problems (29 train + 21 test)
    Tier 1 — 1000 rating : 50 problems (10 train + 40 test)
    Tier 2 — 1100 rating : 33 problems (14 train + 19 test)
    Tier 2 — 1200 rating : 33 problems (23 train + 10 test)
    Tier 2 — 1300 rating : 34 problems (23 train + 11 test)
    Tier 3 — 1400 rating : 32 problems (27 train +  5 test)
    Tier 3 — 1500 rating : 33 problems (33 train +  0 test)
    Tier 4 — 1600 rating : 17 problems (17 train +  0 test)
    Tier 4 — 1700 rating : 18 problems (18 train +  0 test)
"""

import json
import random
import sys
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Pre-selected problem IDs  (split → rating → [folder_id, …])
# ---------------------------------------------------------------------------
_SELECTED: dict[str, dict[int, list[str]]] = {
    "train": {
        800:  ["0002","0003","0009","0015","0026","0038","0044","0055","0060",
               "0066","0076","0081","0082","0083","0105","0113","1607","1671",
               "2094","2179","2193","2197","2207","2249","2346","2363","2368",
               "2383","2386"],
        1000: ["0033","0048","0054","0071","0088","0092","0095","0121","1606",
               "2394"],
        1100: ["0012","0086","0094","0100","0103","0117","2004","2160","2203",
               "2334","2352","2366","2372","2381"],
        1200: ["0021","0022","0030","0035","0036","0045","0052","0091","0098",
               "0099","2011","2013","2027","2029","2040","2099","2105","2224",
               "2227","2371","2378","2392","2395"],
        1300: ["0004","0010","0019","0024","0039","0047","0069","0078","0090",
               "0107","0108","0110","2056","2115","2151","2171","2195","2198",
               "2211","2311","2320","2339","2393"],
        1400: ["0005","0008","0013","0018","0031","0032","0043","0046","0057",
               "0063","0072","0093","0111","0116","0118","2012","2017","2057",
               "2063","2064","2071","2080","2096","2188","2230","2245","2390"],
        1500: ["0020","0028","0059","0064","0065","0070","0085","0106","2021",
               "2025","2033","2043","2045","2051","2053","2065","2079","2086",
               "2087","2104","2110","2130","2142","2206","2216","2246","2248",
               "2285","2317","2335","2337","2347","2362"],
        1600: ["0001","0056","0062","0067","0089","0097","1605","2002","2006",
               "2010","2019","2023","2054","2069","2075","2089","2093"],
        1700: ["0041","0042","0050","0051","0068","0114","0120","2168","2001",
               "2008","2039","2046","2076","2083","2095","2098","2138","2148"],
    },
    "test": {
        800:  ["0017","0021","0050","0067","0083","0087","0124","0131","0145",
               "0157","0163","0168","0171","0184","0202","0216","0225","0233",
               "0271","0276","0282"],
        1000: ["0020","0076","0095","0102","0107","0146","0172","0177","0185",
               "0187","0195","0198","0204","0207","0215","0224","0244","0259",
               "0273","0310","0315","0333","0338","0347","0354","0390","0395",
               "0397","0398","0405","0412","0419","0426","0433","0437","0438",
               "0443","0446","0448","0481"],
        1100: ["0015","0016","0037","0046","0057","0061","0065","0070","0101",
               "0113","0116","0121","0158","0175","0189","0231","0234","0263",
               "0269"],
        1200: ["0008","0019","0034","0041","0060","0093","0098","0114","0125",
               "0128"],
        1300: ["0000","0001","0005","0032","0038","0088","0089","0092","0103",
               "0104","0108"],
        1400: ["0025","0027","0059","0075","0077"],
    },
}


def load_problems() -> list[dict]:
    """
    Load all 300 pre-selected Codeforces problems.

    Each returned dict contains:
        id       – "<split>/<folder>"  (e.g. "train/0002")
        question – full problem statement text
        inputs   – list[str]  (stdin strings, one per test case)
        outputs  – list[str]  (expected stdout strings)
        rating   – int  (Codeforces difficulty rating)
    """
    sys.set_int_max_str_digits(1_000_000)
    problems: list[dict] = []

    split_dirs = {"train": config.APPS_TRAIN, "test": config.APPS_TEST}

    for split, ratings in _SELECTED.items():
        base = split_dirs[split]
        for rating, folders in ratings.items():
            for folder in folders:
                prob_dir = base / folder
                q_f  = prob_dir / "question.txt"
                io_f = prob_dir / "input_output.json"

                if not (q_f.exists() and io_f.exists()):
                    print(f"  [warn] missing files for {split}/{folder}, skipping")
                    continue

                try:
                    io = json.loads(io_f.read_text())
                except Exception:
                    print(f"  [warn] bad JSON for {split}/{folder}, skipping")
                    continue

                inputs  = io.get("inputs", [])
                outputs = io.get("outputs", [])

                if not inputs or not outputs:
                    print(f"  [warn] empty test cases for {split}/{folder}, skipping")
                    continue

                problems.append({
                    "id":       f"{split}/{folder}",
                    "question": q_f.read_text().strip(),
                    "inputs":   inputs,
                    "outputs":  outputs,
                    "rating":   rating,
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
        evolution — used for fitness evaluation every generation (100)
        holdout   — unseen until final evaluation (200)
        rest      — any remainder (empty when len(problems) == 300)
    """
    rng = random.Random(seed)
    shuffled = problems.copy()
    rng.shuffle(shuffled)

    evolution = shuffled[:evolution_size]
    holdout   = shuffled[evolution_size : evolution_size + holdout_size]
    rest      = shuffled[evolution_size + holdout_size :]

    return evolution, holdout, rest
