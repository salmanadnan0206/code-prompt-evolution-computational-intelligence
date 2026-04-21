"""
cost_tracker.py — Track every LLM call's token usage and estimated cost.

Every time main.py runs, a new JSON file is saved to ../Costs/ with a
complete breakdown: per-call tokens, per-phase subtotals, and grand total.

When running via the Claude Code CLI (subscription mode), costs reported are
the API-equivalent estimates returned by the CLI — no actual charges are made
beyond the flat subscription fee.

Pricing source: https://docs.anthropic.com/en/docs/about-claude/pricing
                 (fetched April 2026)
"""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Official Anthropic pricing (USD per 1 million tokens) — April 2026
# ---------------------------------------------------------------------------
PRICING = {
    # model_id: (input_$/MTok, output_$/MTok)
    "claude-haiku-4-5-20251001": (1.00,  5.00),
    "claude-haiku-3-5-20241022": (0.80,  4.00),
    "claude-haiku-3-20240307":   (0.25,  1.25),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-sonnet-4-5-20250514":(3.00, 15.00),
    "claude-sonnet-4-20250514":  (3.00, 15.00),
    "claude-opus-4-6":           (5.00, 25.00),

    # Batch API (50 % discount) — keys suffixed with :batch
    "claude-haiku-4-5-20251001:batch": (0.50,  2.50),
    "claude-sonnet-4-6:batch":         (1.50,  7.50),
}

COSTS_DIR = Path(__file__).parent / "Costs"


# ---------------------------------------------------------------------------
# Single-call record
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    phase: str          # "fitness", "crossover", "mutate_inject", etc.
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None  # when set, used directly instead of pricing table

    @property
    def input_cost(self) -> float:
        if self.cost_usd is not None:
            return 0.0  # cost rolled into total_cost via cost_usd
        rate_in, _ = PRICING.get(self.model, (0, 0))
        return self.input_tokens * rate_in / 1_000_000

    @property
    def output_cost(self) -> float:
        if self.cost_usd is not None:
            return 0.0
        _, rate_out = PRICING.get(self.model, (0, 0))
        return self.output_tokens * rate_out / 1_000_000

    @property
    def total_cost(self) -> float:
        if self.cost_usd is not None:
            return self.cost_usd
        return self.input_cost + self.output_cost


# ---------------------------------------------------------------------------
# Run-level tracker
# ---------------------------------------------------------------------------
class CostTracker:
    """Streams each call to a JSONL file immediately; keeps only running totals in memory."""

    def __init__(self):
        self.start_time: float = time.time()
        self._jsonl_path: Path | None = None
        self._lock = threading.Lock()
        # running totals only — no per-call list kept in memory
        self._total_calls: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._phase_totals: dict = {}
        self._model_totals: dict = {}

    def set_log_path(self, path: Path) -> None:
        """Point the tracker at a JSONL file for streaming writes."""
        self._jsonl_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, phase: str, model: str,
                input_tokens: int = 0, output_tokens: int = 0,
                cost_usd: float | None = None):
        """
        Log one LLM call — written to disk immediately, not kept in memory.
        Thread-safe: protected by internal lock.
        """
        rec = CallRecord(
            phase=phase, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        call_cost = rec.total_cost
        del rec  # no longer needed — totals extracted

        with self._lock:
            # update running totals
            self._total_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += call_cost

            for bucket, key in [(self._phase_totals, phase), (self._model_totals, model)]:
                if key not in bucket:
                    bucket[key] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}
                bucket[key]["calls"] += 1
                bucket[key]["input_tokens"] += input_tokens
                bucket[key]["output_tokens"] += output_tokens
                bucket[key]["total_cost_usd"] = round(bucket[key]["total_cost_usd"] + call_cost, 6)

            # stream to JSONL immediately, then free the dict
            if self._jsonl_path is not None:
                entry = {
                    "t": time.strftime("%H:%M:%S"),
                    "phase": phase, "model": model,
                    "in": input_tokens, "out": output_tokens,
                    "cost": round(call_cost, 6),
                    "total_calls": self._total_calls,
                    "running_cost": round(self._total_cost_usd, 6),
                }
                with self._jsonl_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                del entry  # written to disk — free immediately

    def total(self) -> float:
        return self._total_cost_usd

    # --- save to ../Costs/ ---
    def flush(self, extra: dict | None = None) -> Path:
        """Write a timestamped summary JSON to the Costs/ folder."""
        COSTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = COSTS_DIR / f"cost_{ts}.json"
        elapsed = time.time() - self.start_time

        report = {
            "timestamp":        ts,
            "elapsed_seconds":  round(elapsed, 1),
            "mode":             "claude-code-cli (subscription)",
            "note":             (
                "Costs shown are API-equivalent estimates from the CLI. "
                "No actual per-token charges on Max subscription."
            ),
            "total_llm_calls":           self._total_calls,
            "total_input_tokens":        self._total_input_tokens,
            "total_output_tokens":       self._total_output_tokens,
            "grand_total_estimated_usd": round(self._total_cost_usd, 6),
            "breakdown_by_phase":        self._phase_totals,
            "breakdown_by_model":        self._model_totals,
        }
        if extra:
            report["run_info"] = extra

        path.write_text(json.dumps(report, indent=2))
        return path


# ---------------------------------------------------------------------------
# Module-level singleton  (imported by llm.py)
# ---------------------------------------------------------------------------
tracker = CostTracker()
