# Blockers

| What | Exact error | What I did instead | Manual step |
| --- | --- | --- | --- |
| Record mode, Groq | 2026-11-21 probe: `groq unreachable: Tunnel connection failed: 403 Forbidden (agent proxy: no rule or network setting allows this host)`. Since 2026-11-23 record mode stops before any request when the key is missing: `groq: GROQ_API_KEY is not set` | Students train on gold labels. README says the teacher did not run. | Add `GROQ_API_KEY` and allow `api.groq.com`, then `distillroute record --provider groq` |
| Record mode, OpenRouter | same proxy 403 for `openrouter.ai` | as above | as above with `OPENROUTER_API_KEY` |
| Record mode, Gemini | 2026-11-21 probe: `gemini answered HTTP 400` (host reachable, no key). Now `gemini: GEMINI_API_KEY is not set` | as above | add `GEMINI_API_KEY`, then `distillroute record --provider gemini` |
| LLM-only baseline | no teacher, so no LLM answers | cost estimated from tokens and the price table, hybrid shows bounds | run record mode on the 300 ticket test sample |
| FastEmbed model | `ValueError: Could not load model BAAI/bge-small-en-v1.5 from any source.` (HF CONNECT 403, model bucket 403) | hash embedder student, labeled as not a learned embedding | `pip install -e ".[embed]"`, `DISTILLROUTE_FASTEMBED=1 distillroute run` where HF is reachable |
| Skill installs (`npx skills add ...`) | denied by the session's permission system | used the skills already in the session and followed Ponytail and caveman rules by hand | install the toolkit skills in a normal Claude Code session |
| osv-scanner and semgrep | `api.osv.dev` and `semgrep.dev` proxy 403 | pip-audit and `ruff check --select S` | none |
| gh CLI, PR, tags, repo settings | the proxy refuses GraphQL and tag pushes | the coordinator opens the PR and sets About and topics. Values are in `docs/GITHUB_ABOUT.md` and `docs/topics.txt` | see `docs/HUMAN_STEPS.md` |

## Bugs found and fixed

- Live bug, score 5 x 3 = 15: the official BANKING77 train file shares 7 texts with the test file. Found by the split test. Fixed in `data.banking77()`.
- Live bug, score 5 x 2 = 10: curve points lost their training size because the score dict's `n` overwrote it. Found when the report tried to draw the curve. Fixed in `bench.curve()`, covered by `test_curve_comes_from_real_runs`.
- Live bug, score 5 x 3 = 15: recorded teacher labels could never replay, because `record` and `run` hashed requests with different model names and seeds. Found by the code review. Fixed with per-model recording folders and matching samples.
- Live bug, score 4 x 3 = 12: timings were not 2-core numbers, because numpy's BLAS threads were started before the pinning. Found by the outside check. Fixed in `bench.pin_cores()` and `bench.timings()`.
- Live bug, score 3 x 3 = 9: the threshold rule passed on luck with a small dev sample, and test kept accuracy missed the 95% target. Found by the code review. Fixed with the Wilson lower bound.
- Live bug, score 3 x 2 = 6: the hybrid lower bound for macro-F1 was above the student alone. Found by the outside check. Fixed in `router.hybrid_predictions()`.
