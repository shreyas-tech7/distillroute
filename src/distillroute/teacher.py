"""The teacher step: an LLM labels training texts, through a cache keyed by request hash.

Every request is a canonical JSON object (model, label list, ticket text, temperature 0). Its
SHA-256 names a file in the fixture folder. That file holds the raw response bytes. Replay reads
the same bytes back, so a replayed run is byte for byte the same as the recorded one.

Record mode sends the request to an OpenAI-compatible endpoint with a key from an environment
variable. Only the ticket text and the label list leave the machine. Without a key, or when the
host is blocked, record mode stops and says so. The rest of the pipeline then trains on gold.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "llama-3.1-8b-instant"),
    "openrouter": (
        "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.1-8b-instruct:free",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
        "gemini-3.1-flash-lite",
    ),
}

SYSTEM = "You route customer support tickets. Reply with exactly one label from the list and nothing else."


class RecordBlocked(RuntimeError):
    """Record mode cannot run: no key, a blocked host or a refused request."""


def build_request(text: str, labels: list[str], model: str) -> dict:
    prompt = "Labels:\n" + "\n".join(labels) + f"\n\nTicket: {text}\nLabel:"
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }


def request_hash(req: dict) -> str:
    canonical = json.dumps(req, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_label(raw: bytes, labels: list[str]) -> str | None:
    """Pull the label out of a chat completion. Returns None when the answer is not a known label."""
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    answer = content.strip().strip("`\"'.").strip().lower()
    by_lower = {c.lower(): c for c in labels}
    if answer in by_lower:
        return by_lower[answer]
    hits = [c for c in labels if c.lower() in answer]
    return max(hits, key=len) if hits else None


class Cache:
    """Raw response bytes stored as <fixtures>/<sha256>.json. Nothing rewrites a stored file."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def path(self, key: str) -> Path:
        return self.folder / f"{key}.json"

    def get(self, key: str) -> bytes | None:
        p = self.path(key)
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, raw: bytes) -> None:
        p = self.path(key)
        if p.exists():
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)


def http_send(provider: str, req: dict, timeout: float = 30.0, tries: int = 5) -> bytes:
    url, env, _ = PROVIDERS[provider]
    key = os.environ.get(env, "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(req).encode()
    wait = 2.0
    for _ in range(tries):
        try:
            r = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310 (fixed https URL from PROVIDERS)
            with urllib.request.urlopen(r, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(wait)
                wait *= 2
                continue
            raise RecordBlocked(f"{provider} answered HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RecordBlocked(f"{provider} unreachable: {e}") from e
    raise RecordBlocked(f"{provider} kept answering 429 after {tries} tries")


class Teacher:
    """Label texts through the cache. In replay mode a cache miss is an error, never a live call."""

    def __init__(self, labels: list[str], cache: Cache, model: str, send=None, record: bool = False):
        self.labels = labels
        self.cache = cache
        self.model = model
        self.send = send
        self.record = record
        self.calls = 0
        self.hits = 0

    def raw(self, text: str) -> bytes:
        req = build_request(text, self.labels, self.model)
        key = request_hash(req)
        got = self.cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        if not self.record or self.send is None:
            raise KeyError(f"no recorded teacher label for request {key[:12]} and record mode is off")
        got = self.send(req)
        self.calls += 1
        self.cache.put(key, got)
        return got

    def label(self, text: str) -> str | None:
        return parse_label(self.raw(text), self.labels)


def fake_response(label: str) -> bytes:
    """The shape of an OpenAI-compatible chat completion, used for test fixtures only."""
    body = {
        "id": "fixture",
        "object": "chat.completion",
        "model": "fixture-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": label}, "finish_reason": "stop"}
        ],
    }
    return json.dumps(body, sort_keys=True).encode()
