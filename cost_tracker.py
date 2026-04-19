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
import time
from dataclasses import dataclass, field
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
    """Accumulates CallRecords and saves a full breakdown JSON on flush."""

    def __init__(self):
        self.calls: list[CallRecord] = []
        self.start_time: float = time.time()

    def record(self, phase: str, model: str,
                input_tokens: int = 0, output_tokens: int = 0,
                cost_usd: float | None = None):
        """
        Log one LLM call.

        When called from the CLI path, pass cost_usd (reported directly by the
        CLI JSON) along with token counts. When cost_usd is provided it takes
        precedence over pricing-table calculations.
        """
        self.calls.append(CallRecord(
            phase=phase,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ))

    # --- aggregation helpers ---
    def _by_phase(self) -> dict:
        phases: dict[str, dict] = {}
        for c in self.calls:
            if c.phase not in phases:
                phases[c.phase] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                }
            p = phases[c.phase]
            p["calls"]          += 1
            p["input_tokens"]   += c.input_tokens
            p["output_tokens"]  += c.output_tokens
            p["total_cost_usd"] += c.total_cost
        for p in phases.values():
            p["total_cost_usd"] = round(p["total_cost_usd"], 6)
        return phases

    def _by_model(self) -> dict:
        models: dict[str, dict] = {}
        for c in self.calls:
            if c.model not in models:
                models[c.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                }
            m = models[c.model]
            m["calls"]          += 1
            m["input_tokens"]   += c.input_tokens
            m["output_tokens"]  += c.output_tokens
            m["total_cost_usd"] += c.total_cost
        for m in models.values():
            m["total_cost_usd"] = round(m["total_cost_usd"], 6)
        return models

    def total(self) -> float:
        return sum(c.total_cost for c in self.calls)

    # --- save to ../Costs/ ---
    def flush(self, extra: dict | None = None) -> Path:
        """
        Write a timestamped JSON to the Costs/ folder.

        Returns the path of the saved file.
        """
        COSTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = COSTS_DIR / f"cost_{ts}.json"

        total_in  = sum(c.input_tokens for c in self.calls)
        total_out = sum(c.output_tokens for c in self.calls)
        elapsed   = time.time() - self.start_time

        report = {
            "timestamp":        ts,
            "elapsed_seconds":  round(elapsed, 1),
            "mode":             "claude-code-cli (subscription)",
            "note":             (
                "Costs shown are API-equivalent estimates from the CLI. "
                "No actual per-token charges on Max subscription."
            ),
            "total_llm_calls":     len(self.calls),
            "total_input_tokens":  total_in,
            "total_output_tokens": total_out,
            "grand_total_estimated_usd": round(self.total(), 6),
            "breakdown_by_phase": self._by_phase(),
            "breakdown_by_model": self._by_model(),
        }
        if extra:
            report["run_info"] = extra

        path.write_text(json.dumps(report, indent=2))
        return path


# ---------------------------------------------------------------------------
# Module-level singleton  (imported by llm.py)
# ---------------------------------------------------------------------------
tracker = CostTracker()
