# Walkthrough

A tour of every file. One idea per sentence.

## The idea

A support team gets tickets. Each ticket needs a queue. An LLM can pick the queue. That costs money on every ticket. A small model can learn from the LLM's answers. Then the small model handles the easy tickets. The LLM handles only the hard ones.

## Data

- `data/banking77/train.csv` and `test.csv` are the public BANKING77 files. They hold bank customer questions and their intent.
- `data/banking77/categories.json` lists the 77 intents.
- `data/banking77/LICENSE` is the CC BY 4.0 license that comes with the data.
- `data/synthetic/tickets.csv` holds 600 made-up shop tickets. `synth.py` wrote them.
- `data/SOURCES.json` records where each file came from, its license and its SHA-256 hash.
- `data/prices.json` holds the comparison prices, with their source pages and the day they were read.

## Code in `src/distillroute/`

- `data.py` loads the files and checks the hashes. It drops 7 training texts that also sit in the test file. It cuts a development split out of the training file. The test file is never touched for training.
- `synth.py` fills sentence templates with words like "blender" or "Denver". A seed makes it repeatable. Some tickets are vague on purpose.
- `students.py` holds the two students. TF-IDF turns words into weighted counts. The hash embedder turns letter groups into a fixed size vector. Softmax regression learns one weight per feature and intent. It is written in plain numpy, so you can read the math.
- `metrics.py` scores predictions. Accuracy counts hits. Macro-F1 scores each intent alone, then averages. The intervals say how sure the number is.
- `teacher.py` talks to the LLM. It builds one exact request per ticket. The SHA-256 of that request names a file. The file keeps the raw answer bytes. Replay reads the same bytes back.
- `router.py` decides who answers. If the student is confident enough, it answers. If not, the LLM answers. The threshold comes from the development split.
- `costs.py` turns token counts and compute seconds into dollars.
- `bench.py` runs the whole pipeline. It also runs the learning curve, the noise sweep and the timings.
- `readme.py` writes the tables in the README and the page `docs/index.html` from the results files.
- `cli.py` is the `distillroute` command.

## Results

- `results/banking77-b800-seed0.json` is the main run.
- `results/banking77-b800-seed1.json` repeats it with another seed.
- `results/banking77-b8498-seed0.json` uses every training label.
- `results/synthetic-b360-seed0.json` is the synthetic domain.
- `results/curve-banking77.json` holds the learning curve and the noise sweep.
- `results/timings.json` holds train time and memory on 2 cores.
- `results/record-status.json` says what happened when record mode tried each provider.

## Tests in `tests/`

- `conftest.py` blocks every network connection during tests.
- `test_data.py` checks hashes, sizes and that no test text leaks into training.
- `test_students.py` checks the features and the regression.
- `test_metrics.py` checks the scores against hand worked values.
- `test_teacher.py` checks the cache, replay and parsing.
- `test_router.py` checks the threshold rule and that it only sees dev labels.
- `test_results.py` checks the results meet the brief and match the README.
- `test_repo.py` checks the README layout, writing rules and shipped files.
- `tests/fixtures/teacher/synthetic/fixture-model/` holds fake recorded answers for 40 synthetic training tickets and 20 dev tickets, for tests only. `scripts/make_test_fixtures.py` writes them.

## The headline

<!-- gen:headline -->
With 1100 labels (800 to train, 300 to set the threshold), the best student, Hash embedding + logistic regression, gets 65.2% accuracy on the BANKING77 test split. No confidence threshold passes the 95% rule on dev, so its hybrid sends every ticket to the LLM. With all 8498 training labels and 1498 dev labels, TF-IDF + logistic regression gets 86.0%. Its hybrid sends 37% of tickets to the LLM, and the tickets it keeps are 95.7% right on test, so it meets the 95% target.
<!-- /gen:headline -->

## Other files

- `init.sh` sets up and tests everything in one command.
- `.github/workflows/ci.yml` runs `init.sh` on each push. `bench.yml` runs the heavy jobs by hand.
- `scripts/check_links.py` checks every link in the docs.
- `scripts/make_social_preview.py` draws `docs/social-preview.png`.
- `DECISIONS.md`, `BLOCKERS.md` and `PROGRESS.md` record what happened and why.
