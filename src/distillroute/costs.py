"""The cost model. Every price comes from data/prices.json, which records its source and date.

LLM cost: estimated tokens times the listed price. Tokens are estimated at 4 characters per token,
plus 4 tokens of overhead per chat message. The prompt holds the full label list, so the label list
is most of the bill.

Student cost: measured CPU seconds times the per-second price of a 2-core cloud runner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .teacher import build_request

PRICES = Path(__file__).resolve().parents[2] / "data" / "prices.json"


def load_prices(path: Path = PRICES) -> dict:
    return json.loads(path.read_text())


def est_tokens(text: str, chars_per_token: int = 4) -> int:
    return math.ceil(len(text) / chars_per_token)


def request_tokens(req: dict, chars_per_token: int = 4) -> int:
    return sum(est_tokens(m["content"], chars_per_token) + 4 for m in req["messages"])


def llm_cost_per_10k(texts: list[str], labels: list[str], prices: dict) -> dict:
    """Mean estimated tokens per ticket and the cost of 10,000 LLM calls."""
    p = prices["llm"]
    cpt = prices["chars_per_token"]
    tin = [request_tokens(build_request(t, labels, p["model"]), cpt) for t in texts]
    tout = [est_tokens(y, cpt) + 1 for y in labels]
    mean_in = sum(tin) / len(tin)
    mean_out = sum(tout) / len(tout)
    usd = 10_000 * (mean_in * p["input_usd_per_million"] + mean_out * p["output_usd_per_million"]) / 1e6
    return {"mean_input_tokens": mean_in, "mean_output_tokens": mean_out, "usd_per_10k": usd}


def compute_cost(seconds: float, prices: dict) -> float:
    return seconds * prices["compute"]["usd_per_minute"] / 60


def student_cost_per_10k(seconds_per_ticket: float, prices: dict) -> float:
    return compute_cost(10_000 * seconds_per_ticket, prices)
