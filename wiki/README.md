# AI Studies Wiki

An experiment in LLM-maintained research knowledge bases — just to see how it works.

## What this is

This is a personal wiki for AI/ML research papers, maintained by LLM agents. Source papers live in `raw/` (not tracked in git). The wiki is a layer of structured, interlinked markdown summaries, concept pages, and synthesis documents that the LLM builds incrementally as papers are ingested.

The wiki is designed to be read in [Obsidian](https://obsidian.md) — wiki links and frontmatter work natively there. But it's just markdown files, so any editor works.

## Structure

- `papers/` — one summary page per source paper
- `concepts/` — pages for cross-cutting ideas (attention, RLHF, memory architectures, etc.)
- `synthesis/` — comparisons, literature reviews, open questions
- `index.md` — catalog of all wiki pages
- `log.md` — changelog

## How it works

1. A paper PDF goes into `raw/`.
2. An LLM agent reads the paper and creates/updates wiki pages.
3. The index and log are updated.
4. Over time, concept and synthesis pages accumulate, reflecting everything read so far.

The conventions the LLM follows are documented in `../CLAUDE.md`.

## Status

- Papers ingested: 4 / 17
- Concept pages: 5
- Synthesis pages: 0
- Last updated: 2026-04-08
