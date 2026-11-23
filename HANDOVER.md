# Handover

Done: all features F01 to F15, including the release check. Branch `claude/project-thread-we8eq4`.

Next, in order:

1. `./init.sh` and `.venv/bin/distillroute readme --check`.
2. With a provider key: `distillroute record --provider gemini --seed 0`, then `distillroute run` and `distillroute readme`. The run picks up the recording by model folder.
3. With Hugging Face reachable: `pip install -e ".[embed]"` and `DISTILLROUTE_FASTEMBED=1 distillroute run`.

Open decisions: none. Heavy jobs: `distillroute run` (about 2 minutes on 2 cores), `distillroute run --dataset banking77 --budget 100000 --dev-budget all`, `distillroute curve` (about 5 minutes), `distillroute timings`.
