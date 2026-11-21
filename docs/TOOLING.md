# Tooling

| Tool | Source | Version | Why | Check |
| --- | --- | --- | --- | --- |
| superpowers skills | already in the session | session copy | plan, TDD, debugging, verification, review | used, no install needed |
| ponytail, caveman, find-skills, humanizer | `npx skills add` | none | brief toolkit | install denied by the permission system. Rules followed by hand |
| gitleaks | `go install github.com/zricethezav/gitleaks/v8@latest` | v8.30.1 | secret scan before commits and on history | installed |
| actionlint | `go install github.com/rhysd/actionlint/cmd/actionlint@latest` | v1.7.12 | workflow lint | installed, passes |
| pip-audit | `uv tool install pip-audit` | latest on 2026-11-21 | dependency check | installed |
| pre-commit | `uv tool install pre-commit` | latest on 2026-11-21 | gitleaks hook config | installed |
| ruff | `.venv` dev dependency | from `pyproject.toml` | lint, format, `--select S` security rules | passes |
| graphify, cisco-ai-skill-scanner | `uv tool install` | not installed | resume runs on big codebases, skill vetting | skipped: under 30 source files, no new skills installed |
| GITHUB_REPOS.md list | portfolio list | none | brief rule 10 | none fit a numpy classifier. Hermes Agent, OmniRoute, LlamaFirewall and the marketing skills are agent or security tools. Skipped |

From `GITHUB_REPOS.md`, one line each:

- obra/superpowers: used (already in the session).
- ponytail, caveman, find-skills, humanizer, rtk: install denied, rules followed by hand.
- graphify: not needed, the codebase is small.
- litellm and OmniRoute: need a provider key, so they are a manual step. Record mode talks to OpenAI-compatible endpoints directly with the standard library.
- cheahjs/free-llm-api-resources: the free provider list behind the `groq`, `openrouter` and `gemini` choices in `teacher.py`.
- gitleaks: used. osv-scanner and semgrep: hosts blocked, pip-audit and ruff S rules used.
- PurpleLlama, Prompt Guard, garak, mcp-scanner, crawl4ai, playwright-mcp, hermes-agent: do not fit a numpy classifier with no web UI and no agent. Skipped.
- coreyhaines31/marketingskills: not installed (installs denied). README written by hand to the brief's writing rules.
