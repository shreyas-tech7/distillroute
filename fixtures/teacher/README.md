# Recorded teacher labels

`distillroute record --provider <name> --seed <n>` writes one file per request here, under `<dataset>/<model>/<sha256 of the request>.json`, plus `<dataset>/<model>/model.txt` with the model name. It labels the training and dev samples that `distillroute run` uses with the same seed and budgets. Each file holds the raw response bytes, and `run` replays them byte for byte. No folder for a model means the teacher step did not run, and the run trains on gold labels and says so. A folder that misses some texts of the sample stops the run with an error.

No real recordings exist yet. Record mode was blocked on the build machine. See `results/record-status.json`.
