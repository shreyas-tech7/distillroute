# Contributing

Thanks for looking. This is a small benchmark, so the rules are short.

1. Run `./init.sh` before and after your change. It installs, lints, checks links and runs every test offline.
2. Write a test that fails first. Name the broken case in its docstring.
3. Never type a number into the README. Change the code or the results, then run `distillroute readme`.
4. Never train, tune or pick a threshold on the test split. A test checks the splits.
5. Never edit a recorded teacher fixture by hand. Record it again instead.
6. New ideas go in `docs/PROPOSALS.md` first.

Writing rules for docs: active voice, plain words, no em dashes, no dashes as punctuation and no semicolons in prose. A test checks the last three.
