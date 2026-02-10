"""Token usage tracker with persistent storage and budget monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path

# Opus 4.6 pricing (per million tokens)
PRICE_INPUT = 15.0
PRICE_OUTPUT = 75.0

DATA_PATH = Path(__file__).resolve().parent.parent / ".usage.json"


def _load() -> dict:
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return {
        "budget": 500.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "sessions": [],
    }


def _save(data: dict) -> None:
    DATA_PATH.write_text(json.dumps(data, indent=2))


def set_budget(amount: float) -> None:
    data = _load()
    data["budget"] = amount
    _save(data)


def record(input_tokens: int, output_tokens: int) -> dict:
    """Record a single API call's usage. Returns cost info."""
    cost_in = input_tokens * PRICE_INPUT / 1_000_000
    cost_out = output_tokens * PRICE_OUTPUT / 1_000_000
    cost = cost_in + cost_out

    data = _load()
    data["total_input_tokens"] += input_tokens
    data["total_output_tokens"] += output_tokens
    data["total_cost"] += cost
    data["sessions"].append({
        "time": datetime.now(timezone.utc).isoformat(),
        "input": input_tokens,
        "output": output_tokens,
        "cost": round(cost, 6),
    })
    _save(data)

    remaining = data["budget"] - data["total_cost"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "total_cost": data["total_cost"],
        "remaining": remaining,
        "budget": data["budget"],
    }


def summary() -> str:
    data = _load()
    remaining = data["budget"] - data["total_cost"]
    pct = (data["total_cost"] / data["budget"] * 100) if data["budget"] > 0 else 0
    calls = len(data["sessions"])

    bar_len = 20
    filled = int(pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    return (
        f"┌─── Token Usage ───────────────────────┐\n"
        f"│ Budget:    ${data['budget']:.2f}\n"
        f"│ Spent:     ${data['total_cost']:.4f} ({pct:.2f}%)\n"
        f"│ Remaining: ${remaining:.4f}\n"
        f"│ [{bar}] {pct:.1f}%\n"
        f"│ Input:  {data['total_input_tokens']:,} tokens\n"
        f"│ Output: {data['total_output_tokens']:,} tokens\n"
        f"│ Calls:  {calls}\n"
        f"└───────────────────────────────────────┘"
    )


def check_budget() -> bool:
    """Return True if budget still available."""
    data = _load()
    return data["total_cost"] < data["budget"]
