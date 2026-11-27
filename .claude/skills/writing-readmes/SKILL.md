---
name: writing-readmes
description: Use when writing, rewriting or reviewing a project's README.md, or when a README is long, stale, buries its quick start, has placeholder commands or hand-typed badges, or makes performance claims without evidence. Applies to libraries, CLIs, apps, ML/data projects and research code.
---

# Writing READMEs

## Overview

A README is the project's landing page, not its manual. The most-starred projects (React, FastAPI,
Streamlit, Ollama, LangChain, Transformers) all answer three questions in the first screen: **what is
it, does it work, how do I run it**. Everything else is linked.

Core principle: **every line is checked against the repository.** Commands are run, numbers are read
from artifacts, links resolve, badges are live. A README that cannot be verified is marketing.

## The structure (in this order)

| # | Part | Its job | Budget |
|---|------|---------|--------|
| 1 | **Title block**: logo (optional), name as H1, one-line promise in italics or bold | Say what it is in one line a stranger understands | 3–6 lines |
| 2 | **Live badges**: CI status, package version from the registry, licence (+ downloads or demo link if real) | Show it is alive. Only badges that update themselves | 1 line, 3–5 badges |
| 3 | **What and who**: one plain sentence, then 3–5 bold-led bullets of what sets it apart | Earn the next scroll. Claims here must be proven below | ≤ 8 lines |
| 4 | **Proof**: a screenshot or GIF of it working, or a headline result table vs a named baseline | Show, don't assert. Numbers come from a file in the repo | ≤ 15 lines |
| 5 | **Quick start**: install + the smallest run that produces visible output, with that output | A stranger succeeds in under a minute by copy-paste | ≤ 20 lines, inside the first 60 |
| 6 | **Usage / features**: the 2–4 most common tasks, each a heading plus one short example | Cover the main paths, not every option | ≤ 60 lines |
| 7 | **How it works**: a short diagram or 5–8 bullets; deep detail goes to `docs/` | Let an expert judge the approach | ≤ 25 lines |
| 8 | **Documentation**: links to `docs/` pages (reference, architecture, results, experiment log) | Move depth out of the landing page | link list |
| 9 | **Limitations / status**: maturity, known gaps, data licences, what it does not do | Prevent misuse; builds trust | ≤ 10 lines |
| 10 | **Contributing, citation, licence** | Close the loop | short |

Total landing page: about 120–250 lines. A project with more to say gets a `docs/` folder, never a
longer README.

## Evidence rules

Each rule closes a gap that a first draft usually leaves open.

- **Repository URL.** Read it with `git remote get-url origin` and use it; never `<your-repo-url>`.
- **Package name and version.** Read them from `pyproject.toml`, `package.json` or `Cargo.toml`. Badges
  point at the registry (`img.shields.io/pypi/v/<name>`), so they never need editing. If the project has
  no CI, no registry release and no remote yet, it gets no badges until it does.
- **Commands.** Run every quick-start command in a clean environment and paste the real output.
  If a command cannot be run here (needs GPU, credentials), say so next to it.
- **Numbers.** Copy metrics from the evaluation artifact (`metrics.json`, `results/*.json`), name
  the baseline, the dataset and the test split, and link the file. If the README shows a large
  results table, generate it with a script and mark the block (`<!-- RESULTS:START -->`).
- **Claims.** Every adjective in part 3 ("fast", "accurate", "beats X") points to part 4 or a doc.
  If it cannot be pointed at, cut it or soften it to what the evidence shows.
- **Honesty.** State what the comparison does not show (different data, not significant, small
  sample). Readers trust a README that names its limits.

## Long projects: split, don't grow

When the draft passes about 250 lines, move these to `docs/` and link them from part 8:
API or CLI reference tables → `docs/reference.md`; architecture and data flow → `docs/architecture.md`;
full results and benchmarks → `docs/results.md`; experiment or decision log → `docs/experiments.md`;
data sources and licences → `docs/data.md`. The README keeps a one-table summary of each.

## Workflow

1. Read the code, the package metadata, the CI config, the licence and any results files.
   Note the facts you will cite and where each lives.
2. Run the install and quick-start commands. Capture real output.
3. Write parts 1–10 in order, filling each from the facts gathered.
4. Run the checker and fix every error:
   `python .claude/skills/writing-readmes/scripts/check_readme.py README.md`
   It flags placeholders (code blocks included), broken relative links and images, unbalanced fences,
   a missing H1, a quick start outside the first screen, hand-typed version badges and length. YAML front
   matter does not count toward the first screen. Check the `docs/` pages with `--page`, which keeps only the
   link, placeholder and fence checks.
5. Re-read the first screen as a stranger: can you say what it is, whether it works and how to run it?

## Exemplars

| Project | Borrow |
|---|---|
| React | Three bold-led bullets that define the project; short README that links to docs |
| FastAPI | Live badges; "Documentation / Source" links up top; a performance section with evidence |
| Ollama | Install commands per platform in the first screen; quick start before explanation |
| Streamlit | A "what people build" gallery as proof; plain-language promise |
| Transformers | Task-oriented usage sections; a citation block for research use |

## Common mistakes

| Mistake | Fix |
|---|---|
| Placeholder in the first command (`git clone <your-repo-url>`) | Read the remote with git; the checker fails on placeholders |
| Static `version-0.3.1` badge | Registry badge that updates itself |
| Full API reference, maths and file tree inline | One example per task; the rest in `docs/` |
| Benchmark at the bottom, claims at the top | Proof directly under the what-and-who block |
| Numbers typed by hand | Copy from the artifact, link it, or generate the block |
| "State of the art", "blazing fast" with no comparison | Name the baseline and the number, or drop the phrase |
| Example output invented | Run it and paste the real output |
