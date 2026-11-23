# Release check

Run on 2026-11-21 and finished on 2026-11-23, before the release candidate.

| Step | Result |
| --- | --- |
| Ponytail review | The ponytail skill could not install (permission system). A fresh reviewer subagent did a least code review of the branch diff instead. Findings and fixes below. |
| Security audit | Skipped by the brief. No sign-in, personal data or payments. |
| `gitleaks git -v .` | no leaks found |
| `pip-audit` on the venv | no known vulnerabilities (the local `distillroute` package itself is not on PyPI, so it is skipped) |
| `ruff check --select S` | passes. Five findings were false positives and carry a reason: four S311 for seeded sampling, one S310 for a fixed https URL |
| `actionlint` | passes |
| `scripts/check_links.py` | 0 problems. External links are listed as not checked |
| Humanizer | not installed (permission system). README and docs were checked by hand and by `test_writing_rules` |
| Outside check | a fresh subagent cloned the repo, ran `./init.sh` and checked each Done-means item. Results below |
| `git status` clean, flags match | yes, see `feature_list.json` |

## Code review findings and fixes

A fresh reviewer subagent read the whole branch. Every must-fix item and most should-fix items were fixed on 2026-11-23.

| Finding | Fix | Test |
| --- | --- | --- |
| Recordings could never replay. `record` hashed requests with the provider's model and seed 0, `run` with the price-table model and the run seed. A partial cache fell back to gold. | Recordings live under `fixtures/teacher/<dataset>/<model>/` with `model.txt`. `run` replays whatever model folder covers its sample. `record` takes `--seed`, `--budget` and `--dev-budget` and labels exactly the samples `run` uses. A partial recording is an error. | `test_run_trains_on_recorded_teacher_labels`, `test_record_and_run_agree_on_the_requests`, `test_partial_recording_is_an_error` |
| The "800 labels" claim also used 1,498 gold dev labels. | Dev is budgeted too (300 by default, `--dev-budget all` for the full split) and labeled from the same source as training. Tables say how many labels of each kind. | `test_run_trains_on_recorded_teacher_labels` checks the label count |
| The threshold rule used point accuracy, so a small dev sample passed the 95% target by luck. On test the kept accuracy fell to 93.2%. | Rule uses the Wilson lower bound and returns `met`. With 300 dev labels no threshold passes on BANKING77, so the hybrid sends everything to the LLM, and the README says so. | `test_threshold_rule_uses_the_lower_bound` |
| The headline mixed rows and said the target held. | The headline names one row and says whether the kept tickets reach the target on test. | `readme --check` |
| Unparsed teacher answers became a fake class. | Those rows are dropped and counted. | `test_unparsed_dev_answer_is_dropped` |
| Redirects could carry the Authorization header to another host, and `Retry-After` was ignored. | No-redirect opener, `Retry-After` honored, no request at all without a key. | `test_redirects_are_refused`, `test_http_send_honors_retry_after`, `test_http_send_refused_host_is_blocked` |
| Cost used p50 time, and one-time costs were not stated. | Cost uses mean time per ticket. The cost note states the labeling and training costs it leaves out. | `readme --check` |
| The curve meta named the embedder the run picked, not the one the curve used. Display names had no fallback. The dataset count had a typed 7. | Fixed in `bench.curve` and `readme.py`. | `readme --check` |
| `bench.yml` skipped the seed 1 and full-label runs. | Added. | actionlint |

Left as is: the bootstrap macro-F1 interval sits off-center for 77 classes, because resamples miss rare classes. Noted in DECISIONS.md.

## Outside check

A fresh subagent cloned the branch and checked each Done-means item. It found two more bugs, both fixed on 2026-11-23.

| Done-means | Result |
| --- | --- |
| 1. Offline tests with recorded labels and the hash embedder | pass |
| 2. Each student under 2 min and 2 GB on 2 cores | the limits held, but the old numbers were not 2-core numbers. `sched_setaffinity(0)` ran after numpy started its BLAS threads, so those threads kept all cores. Fixed: every thread is pinned, the timing child is pinned before exec, and BLAS threads are capped at 2. The new timings are in `results/timings.json` and match the checker's `taskset -c 0,1` run. |
| 3. README table for LLM-only, students and hybrid | partial, as disclosed: the LLM-only row is not run, its cost is an estimate, and the hybrid shows bounds |
| 4. Learning curve from real runs | pass, four points recomputed to 6 decimals |
| 5. Cache by request hash, byte for byte replay | pass after the replay fix above |
| 6. Dataset license and attribution | pass |
| 7. Part 1 quality bar | pass. The CI badge points at a run that exists only after the PR opens. |
| Extra bug | the hybrid's "lower" macro-F1 used a placeholder label that added no false positives, so it came out above the student alone. Fixed: each sent ticket gets a wrong real class. `test_macro_f1_lower_bound_is_not_above_the_student` |
| Test leak | none found |
