# Decisions

One line per choice: the options weighed and the reason.

- 2026-11-21 Name: `distillroute`. PyPI and npm both answered 404 for `distillroute` (and for `labelcheap`) on 2026-11-21. It matches the repo name, so no rename step.
- 2026-11-21 Branch: push to `claude/project-thread-we8eq4`, not `claude/build-distillroute`. This session's git rules allow only that branch (brief rule 11).
- 2026-11-21 Working copy: the clone this session already had on the allowed branch, instead of a fresh clone. Same remote, less setup.
- 2026-11-21 Data: commit the four BANKING77 files (about 1 MB) under `data/banking77/` with the CC BY 4.0 license file. Options: download at test time, or commit. Committing keeps tests and CI offline and pins the bytes. CC BY 4.0 allows redistribution with attribution.
- 2026-11-21 Train/test overlap: the official train file repeats 7 test texts. Drop them from training before the dev split. Options: keep them (leaks test), drop test rows (breaks the official test set), drop train rows. Dropping train rows keeps the official test intact.
- 2026-11-21 Dev split: 15% of each intent from the training file, seed 0. The brief bans the test split for tuning.
- 2026-11-21 Record mode blocked, so students train on gold labels (brief fallback). The README says this shows training, not distillation.
- 2026-11-21 Hybrid accuracy without a teacher: report two bounds (LLM always wrong, LLM always right) instead of guessing an LLM accuracy. The truth lies between them.
- 2026-11-21 Simulated teacher noise: swap 10%, 20% and 30% of labels for a random wrong intent to show how label noise shows up. Labeled as a simulation everywhere. A real teacher's errors cluster on similar intents, so this is a rough guide.
- 2026-11-21 Threshold rule: the lowest confidence where the student's kept accuracy on dev reaches 95%. Options: maximize accuracy minus a cost weight (needs an LLM accuracy we do not have), or a fixed share. The kept-accuracy target needs no LLM numbers and reads as a business rule.
- 2026-11-21 Embedding student: FastEmbed could not download (Hugging Face and the model bucket refused). The brief fallback is TF-IDF only. I kept a second student that uses a hash embedder (hashed character 3 to 5 grams) and labeled it everywhere as not a learned embedding, because it exercises the same code path and gives a second point of comparison.
- 2026-11-21 Regression: full-batch Adam, 300 steps, L2 1e-4. Tuned on dev with 800 labels: L2 1e-3 lost about 11 points, 1e-5 and below lost 1 to 8 points on TF-IDF. More steps changed nothing.
- 2026-11-21 TF-IDF: word unigrams plus bigrams, min DF 2, top 6000 by DF, sublinear TF, smooth IDF, L2 rows. Dense float32 is small enough (peak memory in `results/timings.json`).
- 2026-11-21 Intervals: Wilson for accuracy, 300 resample seeded bootstrap for macro-F1. Wilson is exact enough and cheap. Macro-F1 has no simple closed form.
- 2026-11-21 Price table: `gemini-3.1-flash-lite` at $0.25 in and $1.50 out per million tokens, read from Google's pricing page on 2026-11-21. Groq's pricing page now redirects to its homepage and a third-party page listed Llama 3.1 8B as contact sales, so no Groq number could be read. Compute: GitHub's Linux 2-core runner at $0.006 per minute.
- 2026-11-21 Tokens: estimate at 4 characters per token plus 4 per message. No tokenizer download is possible here.
- 2026-11-21 Timings: a fresh child process per student, pinned to cores 0 and 1 with `sched_setaffinity`, trained on the whole training split (the worst case, larger than the 800 budget).
- 2026-11-21 Synthetic domain: about 15% vague tickets drawn from a shared pool, so the domain is not trivially separable. Without them both students scored 100%.
- 2026-11-21 Test fixtures: fake recorded responses for 40 synthetic texts live in `tests/fixtures/teacher/` with model name `fixture-model`, so tests prove replay without pretending a real teacher ran.
- 2026-11-23 Threshold rule changed to the Wilson lower bound of kept accuracy on dev at or above 95%. The point rule passed by luck on a small dev sample and missed the target on test. Cost: with 300 dev labels no BANKING77 threshold passes, so the hybrid sends everything to the LLM. That is the honest answer for that budget, and the full-label run shows where the rule starts to pay.
- 2026-11-23 Dev labels count toward the label budget (300 by default) and come from the same source as the training labels. Options: free gold dev labels (overstates what 800 labels buy) or a share of the 800 (fewer training labels). A separate dev budget keeps the training budget comparable to the curve.
- 2026-11-23 Recordings are stored per model under `fixtures/teacher/<dataset>/<model>/` with `model.txt`. `run` uses the first model folder that covers its whole sample, and a partial cover is an error.
- 2026-11-23 Unparsed teacher answers are kept in the cache, so replay matches what the teacher said, and dropped from training and tuning, and counted.
- 2026-11-23 Hybrid lower bound: each ticket sent to the LLM gets a wrong real class (the student's own wrong guess, else the next class). A placeholder class added no false positives, so macro-F1 came out too high.
- 2026-11-23 Timings: every thread is pinned, the child is pinned before exec, and BLAS threads are capped at 2, because numpy starts threads at import.
- 2026-11-23 Cost per ticket uses mean time, not p50, because a bill adds up every ticket. p50 stays as the latency column.
- 2026-11-23 The macro-F1 bootstrap interval sits off-center with 77 classes because resamples miss rare classes. Left as is and noted, since a fix (a stratified bootstrap) changes the method for little gain.
