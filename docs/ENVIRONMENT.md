# Environment

Probes run on 2026-11-21 from the build machine with `curl -sS -o /dev/null -w '%{http_code}' -m 10 <url>`.

| Host | Result | Used for |
| --- | --- | --- |
| https://raw.githubusercontent.com | 301 (reachable) | BANKING77 download |
| https://huggingface.co | proxy refused (CONNECT 403) | FastEmbed model, BANKING77 mirror |
| https://storage.googleapis.com | 400 on the root, 403 on the qdrant-fastembed model files | FastEmbed model bucket |
| https://api.groq.com | proxy refused (CONNECT 403) | record mode |
| https://openrouter.ai | proxy refused (CONNECT 403) | record mode |
| https://generativelanguage.googleapis.com | reachable, HTTP 400 without a key | record mode |
| https://api.osv.dev | proxy refused (CONNECT 403) | osv-scanner (skipped, pip-audit used) |
| https://semgrep.dev | proxy refused (CONNECT 403) | semgrep (skipped, ruff S rules used) |
| https://pypi.org/pypi/distillroute/json | 404 (name free) | name check |
| https://registry.npmjs.org/distillroute | 404 (name free) | name check |

No provider key variable was set (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`).

Machine: Intel Xeon Processor @ 2.10GHz, 4 cores, 15 GB RAM, Python 3.11 in `.venv` (VM Python 3.13). Timings pin cores 0 and 1.
