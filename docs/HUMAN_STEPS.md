# Human steps

Steps that only Shreyas can do, in order.

Release commit: `ab15e2806f782dbcb2a41a74c0f83ad5e79d59fa` (the same SHA is in CHANGELOG.md).

1. Open a pull request from `claude/project-thread-we8eq4` into `main`. Read the diff. Merge with a merge commit, so the release commit stays in the history of `main`.
2. Create the tag on the release commit: `git fetch origin`, then `git tag -a v0.1.0-rc1 ab15e2806f782dbcb2a41a74c0f83ad5e79d59fa -m "Release candidate 1"`, then `git push origin v0.1.0-rc1`. Or create a pre-release with that tag on GitHub.
3. Run the secret scan once more: `gitleaks git -v .`
4. Fill in the About box from `docs/GITHUB_ABOUT.md` and the topics from `docs/topics.txt`, if the coordinator could not.
5. Upload `docs/social-preview.png` under Settings, General, Social preview.
6. Make the repo public. Done 2026-11-26: the repo is public.
7. Turn on Pages from `docs/` if you want the results page online. It opens from disk without it.
8. Pin the repo or list it in the profile README.
9. Publish to PyPI with trusted publishing, if you want the package there. The name `distillroute` was free on 2026-11-21.
10. Create a free provider key (Groq, OpenRouter or Google AI Studio). Add it as an environment variable or a network secret, allow the host, then run:
    `distillroute record --provider groq` (about 800 calls), then `distillroute run`, `distillroute curve` and `distillroute readme`. Also record the 300 ticket LLM baseline sample (see `docs/PROPOSALS.md` and `BLOCKERS.md`).
11. Where Hugging Face is reachable, run `pip install -e ".[embed]"` and `DISTILLROUTE_FASTEMBED=1 distillroute run` for the real embedding student.
12. Install the toolkit skills (ponytail, caveman, humanizer) in a normal session, if you want them for later work.
13. Decide whether `docs/build/BUILD_PROMPT.md` stays public.
