# Blockers

| What | Exact error | What I did instead | Manual step |
| --- | --- | --- | --- |
| Record mode, Groq | `groq unreachable: Tunnel connection failed: 403 Forbidden (agent proxy: no rule or network setting allows this host)` | Students train on gold labels. README says the teacher did not run. | Add `GROQ_API_KEY` and allow `api.groq.com`, then `distillroute record --provider groq` |
| Record mode, OpenRouter | same proxy 403 for `openrouter.ai` | as above | as above with `OPENROUTER_API_KEY` |
| Record mode, Gemini | `gemini answered HTTP 400` (host reachable, no key) | as above | add `GEMINI_API_KEY`, then `distillroute record --provider gemini` |
| LLM-only baseline | no teacher, so no LLM answers | cost estimated from tokens and the price table, hybrid shows bounds | run record mode on the 300 ticket test sample |
| FastEmbed model | `ValueError: Could not load model BAAI/bge-small-en-v1.5 from any source.` (HF CONNECT 403, model bucket 403) | hash embedder student, labeled as not a learned embedding | `pip install -e ".[embed]"`, `DISTILLROUTE_FASTEMBED=1 distillroute run` where HF is reachable |
| Skill installs (`npx skills add ...`) | denied by the session's permission system | used the skills already in the session and followed Ponytail and caveman rules by hand | install the toolkit skills in a normal Claude Code session |
| osv-scanner and semgrep | `api.osv.dev` and `semgrep.dev` proxy 403 | pip-audit and `ruff check --select S` | none |
| gh CLI, PR, tags, repo settings | the proxy refuses GraphQL and tag pushes | the coordinator opens the PR and sets About and topics. Values are in `docs/GITHUB_ABOUT.md` and `docs/topics.txt` | see `docs/HUMAN_STEPS.md` |

## Bugs found and fixed

- Live bug, score 5 x 3 = 15: the official BANKING77 train file shares 7 texts with the test file. Found by the split test. Fixed in `data.banking77()`.
- Live bug, score 5 x 2 = 10: curve points lost their training size because the score dict's `n` overwrote it. Found when the report tried to draw the curve. Fixed in `bench.curve()`, covered by `test_curve_comes_from_real_runs`.
