# distillroute project context

## What it is

distillroute trains a small numpy classifier on LLM labels to route support tickets. The classifier answers the tickets it is sure about. The rest go to the LLM. It measures cost, latency and accuracy for LLM only, each student and the hybrid.

## Status

Release candidate v0.1.0-rc1. Record mode is blocked on the build machine (no key, provider hosts refused), so the students train on gold labels. FastEmbed could not download, so the embedding student uses a hash embedder.

## Stack

Python 3.11+, numpy only at runtime. pytest and ruff for development. FastEmbed is an optional extra (`embed`).

## Commands

- `./init.sh`: venv, install, lint, link check, offline tests.
- `distillroute run | curve | timings | readme [--check] | record --provider P | demo | hashes | synth`.

## Layout

- `src/distillroute/data.py`: loaders, hashes, splits, overlap drop.
- `src/distillroute/synth.py`: the 600 synthetic tickets.
- `src/distillroute/students.py`: TF-IDF, hash embedder, FastEmbed wrapper, softmax regression.
- `src/distillroute/teacher.py`: request hash cache, replay, record mode.
- `src/distillroute/router.py`: threshold choice on dev, hybrid bounds.
- `src/distillroute/metrics.py`: accuracy, macro-F1, Wilson and bootstrap intervals.
- `src/distillroute/costs.py`: price table and token estimates.
- `src/distillroute/bench.py`: the pipeline, the curve and the timings.
- `src/distillroute/readme.py`: README blocks and `docs/index.html` from results.
- `data/`: BANKING77 (CC BY 4.0), the synthetic CSV, `SOURCES.json`, `prices.json`.
- `fixtures/teacher/`: recorded teacher responses by request hash (empty until record mode runs).
- `tests/fixtures/teacher/`: fake recorded responses for tests only.
- `results/`: the result files every README number comes from.

## Rules

- Never train, tune or pick thresholds on the test split.
- Never type numbers into the README or the docs blocks. Run `distillroute readme`.
- Never edit a recorded fixture by hand.
- Tests run offline. `tests/conftest.py` blocks network sockets.
- Writing: active voice, plain words, no em dashes, no dash punctuation, no semicolons.

## Skills that fit

superpowers test-driven-development, systematic-debugging, verification-before-completion, requesting-code-review.

## Do not read

`.venv/`, `logs/`, `data/banking77/*.csv` (large), `docs/build/BUILD_PROMPT.md` unless you need the brief.
