# Demo

A real terminal transcript from the build machine on 2026-11-21. No sign-in is needed. Open [index.html](index.html) from disk for the results page.

```text
$ distillroute hashes
ok   banking77/train.csv
ok   banking77/test.csv
ok   banking77/categories.json
ok   banking77/LICENSE
ok   synthetic/tickets.csv

$ distillroute demo
synthetic tfidf: accuracy 0.858 macro-F1 0.856, threshold 0.61 sends 10% to the LLM, hybrid accuracy between 0.833 and 0.933
synthetic embed-hash: accuracy 0.850 macro-F1 0.852, threshold 0.54 sends 7% to the LLM, hybrid accuracy between 0.842 and 0.908
LLM only: not run: record mode blocked (no provider key, provider hosts refused)

$ distillroute record --provider groq --budget 3
{"date": "2026-11-21", "provider": "groq", "model": "llama-3.1-8b-instant", "requested": 3, "result": "blocked", "error": "groq unreachable: <urlopen error Tunnel connection failed: 403 Forbidden (agent proxy: no rule or network setting allows this host)>", "calls": 0}
(exit code 2)

$ distillroute readme --check
generated files are up to date
```
