"""Static AWS instance price table for the `stats` command.

Prices in USD/hr. Update annually or when AWS adjusts pricing.
Source: https://instances.vantage.sh — verified entries below.
"""
from __future__ import annotations

PRICES_USD_PER_HOUR: dict[tuple[str, str], float] = {
    ("g6.xlarge", "od"): 0.8048,    # verified 2026-05-09
    ("g6.xlarge", "spot"): 0.30,    # representative; varies by AZ/time
    ("g6e.xlarge", "od"): 1.861,    # verified 2026-05-09
    ("g6e.xlarge", "spot"): 0.55,
    ("g6.2xlarge", "od"): 0.9776,   # verified 2026-05-09
    ("g6e.2xlarge", "od"): 2.2424,  # verified 2026-05-09
    ("g5.xlarge", "od"): 1.006,     # verified 2026-05-09
}


def hourly_price(instance_type: str, *, spot: bool) -> float:
    """Best-effort lookup. Returns 0.0 if unknown — `stats` will note 'unknown' in output."""
    return PRICES_USD_PER_HOUR.get((instance_type, "spot" if spot else "od"), 0.0)
