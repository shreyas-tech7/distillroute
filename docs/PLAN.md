# Plan

The brief in `docs/build/BUILD_PROMPT.md` is the spec. This plan turns its build order into steps. Each step ends with its tests green and `./init.sh` passing.

1. Initializer. Memory files, `init.sh`, MIT license, `.gitignore`, `.env.example`, gitleaks config, saved brief. Data loader for BANKING77 with license, source URL and SHA-256 hashes in `data/SOURCES.json`. Split test: dev from train, no test text in train or dev.
2. Synthetic domain. 600 tickets, 12 categories, seeded templates written from scratch, a share of vague tickets.
3. TF-IDF student. Tokenizer, TF-IDF in numpy, softmax regression with Adam. Tests on separable data and on the synthetic domain.
4. Embedding student. FastEmbed wrapper behind an optional extra, hash embedder fallback used in tests.
5. Metrics. Accuracy with Wilson intervals, macro-F1 with a seeded bootstrap, agreement.
6. Teacher. Canonical request, SHA-256 key, raw byte cache, replay that never calls out, record mode with retries on 429 and a clear blocked state.
7. Hybrid threshold. Sweep on dev, pick the lowest threshold that meets the kept accuracy target, apply on test, report bounds when the LLM did not run.
8. Learning curve. 100, 250, 500, 800 and all labels, three seeds, both students. A noise sweep as the teacher noise stand-in.
9. Report. `distillroute readme` writes README blocks and a static `docs/index.html` with SVG charts. `--check` in CI.
10. README, study-later kit, release check, hand-off.

Assumptions (also in `DECISIONS.md`):

- Record mode may be blocked. Then the students train on gold labels and the hybrid shows bounds.
- Timings use the whole training split on 2 pinned cores, the worst case.
- Prices come from public pages read on the build day.
