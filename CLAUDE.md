# AI Studies Wiki — Conventions

You are maintaining a personal AI-studies knowledge base. The wiki lives in `wiki/` and is backed by source papers in `raw/`. You own the wiki layer entirely — you create pages, update them, and keep everything consistent. The human reads the wiki; you write it.

## Directory structure

```
wiki/
  papers/     — one page per source paper. Filename matches raw/ stem (lowercase).
  concepts/   — one page per cross-cutting concept. Kebab-case filenames.
  synthesis/  — longer-form pages: comparisons, reviews, open questions.
  index.md    — content catalog. Updated on every ingest.
  log.md      — append-only changelog.
  README.md   — project description (rarely changes).
```

## Naming conventions

- Paper pages: `{raw-file-stem-lowercased}.md`. Example: `raw/TRM.pdf` → `papers/trm.md`.
- Concept pages: `{descriptive-kebab-case}.md`. Example: `concepts/attention-mechanisms.md`.
- Synthesis pages: `{descriptive-kebab-case}.md`. Example: `synthesis/rl-fine-tuning-landscape.md`.
- All filenames are lowercase, no spaces, hyphens not underscores.

## Frontmatter

Every wiki page has YAML frontmatter. Required fields by type:

**paper**: `type`, `source`, `title`, `authors`, `year`, `tags`, `ingested`
**concept**: `type`, `tags`, `papers`, `created`, `updated`
**synthesis**: `type`, `tags`, `papers`, `concepts`, `created`, `updated`

- `tags` is a free-form array. Use lowercase hyphenated tags. Check existing tags in `index.md` before inventing new ones.
- `papers` and `concepts` are arrays of page stems (no path, no extension) linking to relevant pages.

## Page templates

### Paper page (wiki/papers/)

```markdown
---
type: paper
source: raw/FILENAME.pdf
title: "Full Paper Title"
authors: [Author1, Author2]
year: 2024
tags: [tag1, tag2]
ingested: 2026-04-08
---

# Paper Title

## One-line summary
[Single sentence capturing the paper's contribution]

## Key contributions
- [Numbered list of 3-5 main contributions]

## Core ideas
[2-4 paragraphs distilling the main technical content]

## Methods
[Key architectural details, training setup, evaluation approach]

## Results
[Main quantitative results with context]

## Limitations noted by authors
[What the paper itself acknowledges as weaknesses or future work]

## Connections
- Relates to [[concept-name]] — explanation
- Extended by [[paper-name]] — explanation

## Open questions
[1-3 questions this paper raises but doesn't answer]

## PyTorch implementation sketch
[Concise, runnable PyTorch code illustrating the paper's core method. Focus on the novel component (e.g. the loss function, the sampling procedure, the training loop). Keep it minimal — enough to convey the idea, not a full reproduction.]
```

### Concept page (wiki/concepts/)

```markdown
---
type: concept
tags: [tag1, tag2]
papers: [paper1, paper2]
created: 2026-04-08
updated: 2026-04-08
---

# Concept Name

## Definition
[1-2 sentences: what is this concept]

## Why it matters
[1-2 sentences: why this concept is important]

## How it works
[Technical explanation at a level a practitioner could implement]

## Variants and extensions
[Sub-approaches, architectural variants, key modifications]

## Key papers
- [[paper1]] — what it contributed to this concept
- [[paper2]] — what it contributed to this concept

## Current state
[Where this concept stands given all papers ingested so far]
```

### Synthesis page (wiki/synthesis/)

```markdown
---
type: synthesis
tags: [tag1, tag2]
papers: [paper1, paper2]
concepts: [concept1]
created: 2026-04-08
updated: 2026-04-08
---

# Synthesis Title

[Free-form structure — let the content determine the sections.
 Use tables for comparisons, sections by theme for reviews,
 numbered lists for open questions.]

## References
- [[paper1]] — brief note on relevance
- [[paper2]] — brief note on relevance
```

## Cross-referencing

Use Obsidian-style wiki links: `[[page-stem]]`. Do not include the directory path or extension — just the stem.

When you create or update a page:
1. Add `[[links]]` to related pages within the body text.
2. Add relevant page stems to the `papers` or `concepts` frontmatter arrays.
3. Check whether linked-to pages need a reciprocal link or back-reference added.

Every page in the wiki must appear in `wiki/index.md`.

## Ingest workflow

When the user asks you to ingest a paper from `raw/`:

1. Read the PDF (or converted `.md`) from `raw/`. If the content is garbled, missing sections, or the conversion is clearly incomplete, stop and tell the user before proceeding. Do not ingest from a corrupted or partial source.
2. Discuss key takeaways with the user before writing anything. Get their read on what matters.
3. Create the paper page in `wiki/papers/` following the template.
4. For each significant concept the paper introduces or engages with:
   - If a concept page exists, update it with the new information.
   - If no concept page exists, create one.
5. Update `wiki/index.md` to include the new paper (and any new concepts).
6. Append an entry to `wiki/log.md`.
7. Ask the user if they want to explore connections or generate a synthesis page.

Ingest one paper at a time. Do not batch unless the user explicitly asks.

## Query workflow

When the user asks a question:

1. Read `wiki/index.md` to find relevant pages.
2. Read the relevant pages (paper, concept, synthesis as needed).
3. If existing pages don't cover the question well enough, read the relevant raw PDFs.
4. Synthesize an answer with citations (wiki links to pages you drew from).
5. If the answer is substantial and reusable, offer to file it as a new page (usually in `synthesis/`).

## Lint workflow

When the user asks for a health check (or you notice issues during normal work):

- Check for pages not listed in `index.md`.
- Check for wiki links `[[x]]` where `x` has no corresponding page.
- Check for concept pages whose `papers` list is stale (a related paper was ingested but not added).
- Check for contradictions between pages. If found, flag — don't silently resolve.
- Suggest concepts that deserve their own page based on frequency of mentions.
- Suggest synthesis pages that could be valuable given current content.

Report findings to the user before making changes.

## What NOT to do

- Do not modify files in `raw/`. Ever.
- Do not delete wiki pages without explicit user instruction.
- Do not invent content you didn't read in a source. If unsure, say so.
- Do not create empty stub pages. A page should have real content before it's filed.
- Do not over-format. The wiki should read like good technical notes, not a corporate document.
- Do not update `index.md` or `log.md` retroactively. New entries go at the top (index) or bottom (log).
- Do not use HTML in markdown files.
- Do not create directories other than `papers/`, `concepts/`, and `synthesis/` without discussing with the user first.
