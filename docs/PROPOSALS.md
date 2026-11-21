# Proposals

Ideas that are not in the brief. Shreyas picks what gets built.

1. Active learning (the brief's stretch item): send the student's least confident training texts to the teacher first, and compare the curve with random sampling.
2. Calibrate the student's confidence with temperature scaling on the dev split, so the threshold means the same thing across budgets.
3. A per intent threshold, because some intents are easy and some are not.
4. A shorter teacher prompt: a label shortlist from the student instead of all 77 label names. The label list is most of the token bill.
5. A drift check: score the student weekly on a fresh labeled sample and raise the threshold when accuracy drops.
