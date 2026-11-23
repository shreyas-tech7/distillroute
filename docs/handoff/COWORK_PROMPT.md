# Prompt for Claude Cowork

Paste the text below into Claude Cowork.

---

You are helping Shreyas finish the browser steps for the private GitHub repo shreyas-tech7/distillroute. Do not ask questions. Do not stop without a result.

Do these steps, all free and all reversible:

1. Open the repo's Actions page and check that the latest CI run on the default branch passed. Write down the run link and its result.
2. Check the About box. If it is empty, fill it in with the description in `docs/GITHUB_ABOUT.md` and the topics in `docs/topics.txt`.
3. Upload `docs/social-preview.png` under Settings, General, Social preview.
4. Check that a pre-release named v0.1.0-rc1 exists. If not, write that down. Do not create tags yourself.

Do not make the repo public. Do not publish a package. Do not enter any key. Do not sign in for Shreyas. Do not pay for anything.

When you finish, write the next prompt for Claude Code. It must hold: the context (repo, branch, what each step above found), what is complete, what is still open (record mode with a free key, the FastEmbed student, making the repo public), and this resume launcher: "Read docs/build/BUILD_PROMPT.md and follow it. Resume from PROGRESS.md on branch claude/project-thread-we8eq4. Do not ask questions. Do not stop until every Done-means item is true or blocked with a fallback."
