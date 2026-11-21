# Handover

Done: everything but the release check and hand-off (F15). Branch `claude/project-thread-we8eq4`.

Next, in order:

1. `./init.sh` and `.venv/bin/distillroute readme --check`.
2. Release check from `docs/build/BUILD_PROMPT.md` section 11. Log results in `docs/RELEASE_CHECK.md`.
3. Write the release SHA into `CHANGELOG.md` and `docs/HUMAN_STEPS.md`, push to origin and backup, write the bundle, the report and the queue row.

Open decisions: none. Heavy jobs: `distillroute run`, `distillroute curve` (about 5 minutes), `distillroute timings`.
