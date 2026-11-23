# Changelog

## v0.1.0-rc1 (2026-11-23)

Release commit: ab15e2806f782dbcb2a41a74c0f83ad5e79d59fa

- BANKING77 loader with license record and hashes, a dev split from train, and a drop of the 7 train texts that repeat in test.
- A synthetic domain of 600 small-business tickets in 12 categories.
- Two numpy students: TF-IDF plus softmax regression, and an embedding student (hash embedder fallback, FastEmbed optional).
- A teacher cache keyed by request hash that replays byte for byte, and a record mode for Groq, OpenRouter and Gemini.
- A hybrid router with a threshold picked on a budgeted dev sample by the Wilson lower bound of kept accuracy, with bounds when the LLM did not run.
- Recordings stored per teacher model, and a `record` that labels exactly the samples `run` uses.
- Metrics with Wilson and bootstrap intervals, a dated cost model, a learning curve, a label noise sweep and 2 core timings.
- A README and a static results page written from the results files.
