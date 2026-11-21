# Interview questions and answers

Each answer points at the code and the results file it rests on. The numbers sit in generated blocks, written by `distillroute readme` from `results/`.

**1. What does distillroute do, in one breath?**

It trains a small classifier on labels from an LLM, lets the classifier answer the tickets it is sure about, and sends the rest to the LLM. Then it measures accuracy, macro-F1, cost per 10,000 tickets, latency and the share sent to the LLM. The pipeline is `bench.run()` in `src/distillroute/bench.py`.

**2. What did the main run find?**

<!-- gen:headline -->
With 800 labels, the best student gets 65.2% accuracy on the BANKING77 test split, and the hybrid has to send 75% of tickets to the LLM to keep the student's share at the 95% target. With all 8498 training labels, TF-IDF gets 86.0% and the hybrid sends 30%.
<!-- /gen:headline -->

**3. When does distillation pay off?**

When the task is narrow, the label set is fixed and you have enough labels per class. The learning curve in `results/curve-banking77.json` shows accuracy still climbing at 800 labels for 77 intents, about 10 per intent. With the full training split the student keeps most tickets. So it pays off once the label budget covers the classes well, and the LLM bill is large enough that a one time labeling cost wins back quickly.

**4. How does the threshold trade cost against accuracy?**

A lower threshold lets the student answer more tickets. That saves LLM calls, but the student makes more mistakes. `router.pick_threshold()` takes the lowest threshold whose kept accuracy on the dev split reaches 95%. The dev sweep is drawn in `docs/index.html`.

<!-- gen:hybrid -->
| Student first | Threshold (from dev) | Kept accuracy on dev | Kept share on test | Kept accuracy on test (95% CI) |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.70 | 95.2% | 9.7% | 284/298 = 95.3% (92.3% to 97.2%) |
| Hash embedding + logistic regression | 0.52 | 95.2% | 25.4% | 729/782 = 93.2% (91.2% to 94.8%) |
<!-- /gen:hybrid -->

**5. Why macro-F1 with 77 classes?**

Accuracy weighs each ticket the same, so the common intents dominate it. Macro-F1 scores each intent alone and averages them, so a student that ignores rare intents scores low. `metrics.macro_f1()` and the test `test_accuracy_and_macro_f1_known_values` show a case where one class is never predicted: accuracy still looks fine, and macro-F1 drops below one half.

**6. How does label noise from the teacher show up?**

As a steady drop in test accuracy as the share of wrong labels grows. The table comes from `bench.label_noise()`, which swaps labels for a random wrong intent. A real teacher confuses similar intents, which hurts more on those intents and less overall.

<!-- gen:noise -->
| Student | Wrong labels | Accuracy, mean | Macro-F1, mean |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 0% | 60.2% | 58.9% |
| TF-IDF + logistic regression | 10% | 56.3% | 54.9% |
| TF-IDF + logistic regression | 20% | 52.1% | 50.9% |
| TF-IDF + logistic regression | 30% | 46.8% | 45.2% |
| Hash embedding + logistic regression | 0% | 65.4% | 64.4% |
| Hash embedding + logistic regression | 10% | 61.2% | 60.0% |
| Hash embedding + logistic regression | 20% | 57.0% | 56.0% |
| Hash embedding + logistic regression | 30% | 51.7% | 49.9% |

Training sets of 800 with a share of labels swapped for a random wrong intent, seeds [1, 2, 3]. This simulates a noisy teacher. It is not a real teacher. File `results/curve-banking77.json`.
<!-- /gen:noise -->

**7. Why did the teacher not run?**

The build machine had no provider key, and the proxy refused the provider hosts. `results/record-status.json` holds the exact errors. The brief's fallback is to train on gold labels and say so. The README says the result shows training, not distillation.

**8. How do you know the test set did not leak?**

The test file is never passed to training, tuning or threshold choice. `test_dev_comes_from_train_and_test_never_leaks` checks it. The official train file repeats 7 test texts, and `data.drop_overlap()` removes them. `test_threshold_is_chosen_on_dev_not_test` spies on the threshold call and checks it only saw dev labels.

**9. How does the cache replay byte for byte?**

`teacher.build_request()` makes one canonical request. `request_hash()` takes the SHA-256 of its sorted JSON. `Cache` stores the raw response bytes under that hash and never rewrites a file. `test_replay_is_byte_for_byte` compares replayed bytes to the files.

**10. How is the cost computed?**

<!-- gen:cost -->
LLM price: `gemini-3.1-flash-lite` at $0.25 per million input tokens and $1.5 per million output tokens (paid tier, Standard, text input), read on 2026-11-21 from https://ai.google.dev/gemini-api/docs/pricing. Tokens are estimated at 4 characters per token. One call has about 464 input tokens, and the 77 label names are most of them. Student price: the GitHub-hosted Linux 2-core (x64) runner at $0.006 per minute, read on 2026-11-21 from https://docs.github.com/en/billing/reference/actions-runner-pricing, times the measured p50 time per ticket. Labeling the 800 training texts once costs about $0.10 at the same LLM price. These are comparison prices, not a bill.
<!-- /gen:cost -->

**11. Why is the hybrid accuracy a range?**

Without LLM answers I cannot score the tickets the student sends away. So `router.hybrid_predictions()` scores them both ways: all wrong and all right. The real hybrid sits between the two bounds. Record mode would close the gap.

**12. Why numpy and not scikit-learn?**

The brief asked for visible math. `SoftmaxRegression` is about 40 lines: a softmax, a gradient and Adam updates. It also keeps the install to one dependency.

**13. Why is the embedding student a hash embedder?**

FastEmbed could not download its model here. The hash embedder hashes letter groups into 1024 buckets. It is not a learned embedding, and the README says so. With the `embed` extra and network access, `FastEmbedder` drops in.

**14. How fast and small are the students?**

`bench.timeit()` trains each student on the whole training split in a fresh process pinned to 2 cores and reads peak memory from `resource`. `results/timings.json` holds the numbers, and `test_timing_under_budget` checks the 2 minute and 2 GB limits.

**15. What would you build next?**

Active learning, so the teacher labels the texts the student is least sure of first. Then calibration of the confidence on dev, and a per intent threshold. They are in `docs/PROPOSALS.md`.
