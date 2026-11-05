# Style guide: Python Libraries study material

This repo is stage 2 of the Gradient Atlas pathway (Python → **Python libraries** → Machine learning → Deep learning → Generative AI).
Readers know core Python (stage 1) and nothing about NumPy, pandas, Matplotlib, Seaborn or SciPy yet.

## How the repo works

- Chapters are written as Markdown in `src/NN_slug.md`. **Never edit the `.ipynb` files by hand.**
- `python tools/build_notebooks.py` turns every `src/*.md` into `NN_slug.ipynb` in the repo root and executes it, so the notebooks are committed **with outputs**.
- In a source file, every fenced ` ```python ` block becomes a code cell. Text between code blocks becomes a markdown cell; each `## ` heading also starts a new markdown cell.
- Data lives in `data/` and is produced by `tools/make_data.py` (seeded, so it is identical on every run). Notebooks read it with `pd.read_csv("data/orders.csv")`, since they run from the repo root.

## Versions

numpy 2.x, pandas 3.x, matplotlib 3.x, seaborn 0.13+, scipy 1.x. Be aware of pandas 3 behaviour:
- Copy-on-Write is always on: chained assignment (`df["a"][0] = 1`) never modifies `df`. Teach `df.loc[row, "a"] = ...`.
- Text columns load as the `str` dtype, not `object`.
- No `inplace=True` in examples; reassign instead.

## Chapter shape

Every chapter follows the same layout:

1. `# Chapter N: Title` followed by a short paragraph: what you will be able to do by the end, and which earlier chapters it builds on (for example "builds on Chapter 2").
2. A **setup** code cell with the imports for the chapter (and `rng = np.random.default_rng(42)` if randomness is used).
3. Numbered sections, `## N.1 Title`, `## N.2 Title` and so on (6 to 10 per chapter). Use `### ` for sub-points if needed.
   Each section: the idea in plain words first (2–5 sentences, why it matters), then small code cells that show it, then one sentence on what the output shows.
4. Cross-reference earlier material by number: "see 3.2", "Chapter 5". The Gradient Atlas site turns these into links and uses them to plan reading routes, so add them wherever a concept was introduced earlier.
5. `## N.x Common mistakes`: a Markdown table with columns `Mistake | What happens | Fix` (4–6 rows).
6. `## N.x Cheat sheet`: a Markdown table with columns `Task | Code` (8–15 rows). Code goes in backticks.
7. `## N.x Exercises`: 3–5 exercises, each followed by a solution in a code cell that starts with `# Solution`.

## Writing

- Plain, friendly English. Short sentences. Address the reader as "you".
- Explain why before how. A single sentence of motivation beats a paragraph of theory.
- Keep code cells short (usually under 12 lines) and make each one print or display something.
- Use the shared datasets in `data/` for anything realistic; use tiny hand-written arrays for mechanics.
- Seed all randomness. No network access, no downloads, no files written outside a temporary folder.
- Use `assert` cells to prove a claim when helpful ("the vectorised result matches the loop").
- Plots: always label the axes and give a title. Call `plt.show()` at the end of plotting cells.
- All content must be original. Do not copy text or code from other courses or repos.
