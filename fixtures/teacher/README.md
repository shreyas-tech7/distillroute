# Recorded teacher labels

`distillroute record --provider <name>` writes one file per request here, under `<dataset>/<sha256 of the request>.json`. Each file holds the raw response bytes. `distillroute run` replays them byte for byte. A missing file means the teacher step did not run, and the run trains on gold labels and says so.

No real recordings exist yet. Record mode was blocked on the build machine. See `results/record-status.json`.
