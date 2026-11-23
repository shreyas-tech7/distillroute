# Demo

A real terminal transcript from the build machine on 2026-11-23. No sign-in is needed. Open [index.html](index.html) from disk for the results page.

```text
$ distillroute hashes
ok   banking77/train.csv
ok   banking77/test.csv
ok   banking77/categories.json
ok   banking77/LICENSE
ok   synthetic/tickets.csv

$ distillroute demo
synthetic tfidf: accuracy 0.858 macro-F1 0.856, threshold 0.77 sends 13% to the LLM, hybrid accuracy between 0.833 and 0.967
synthetic embed-hash: accuracy 0.850 macro-F1 0.852, threshold 0.66 sends 11% to the LLM, hybrid accuracy between 0.833 and 0.942
LLM only: not run: record mode blocked (no provider key, provider hosts refused)

$ distillroute record --provider groq --budget 3 --dev-budget 0
{"date": "2026-11-23", "provider": "groq", "model": "llama-3.1-8b-instant", "seed": 0, "requested": 3, "result": "blocked", "error": "groq: GROQ_API_KEY is not set", "calls": 0}
(exit code 2)

$ distillroute readme --check
generated files are up to date
```
