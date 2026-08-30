# DEMO.md — recording guide and technique reference

A cue sheet for demonstrating the project: what to click, what to say, and the method
behind each thing on screen. README is the pitch, DOCUMENTATION the design write-up,
CONTEXT the engineering narrative; this one is for the camera.

---

## Before you record

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000>. Check the startup banner — it states the graph size and
whether Ollama was found. **Either mode is fine to demo**; deterministic mode is arguably
the better story, because it proves the model is not doing the reasoning.

Reset to a clean state between takes with the **Start over** button in the rail, or
`POST /api/reset`.

---

## Running order

| # | Do this | Say this |
|---|---|---|
| 1 | Load the **Python dev, no ML** persona, hit **Build my path** | "One sentence in. The plan is computed, not written." |
| 2 | Point at the milestone phases and the linking lines | "Those lines are prerequisite edges. The order is a topological sort — it is a guarantee, not a preference." |
| 3 | Click a skill chip → the **inspector** → click through to a neighbour | "The curriculum is a real graph. This is the same data the planner walks." |
| 4 | **Why this?** on a mid-path course | "Three questions answered from edges: why it is here, why it is *here* and not earlier, and what breaks if you skip it." |
| 5 | **Mark complete** | "Everything recomputes. Courses that only existed to reach that skill disappear." |
| 6 | Phase pills → focus one milestone | "Same plan, filtered." |
| 7 | Rail → **Routes**, switch to *Free first* | "Five valid routes to the same skills. The graph fixes what and in what order; a strategy only changes which course." |
| 8 | Rail → **Compare**, enter "I want to fine-tune LLMs" | "76% overlap — the second goal costs one extra course, not a second curriculum." |
| 9 | Rail → **Insights**, all three tabs | "Leverage, the critical path, and where the curriculum itself is thin." |
| 10 | Type "I want to be a web developer" | "It refuses. Most tools guess." |

---

## Technique behind each feature

### Intake

| On screen | Method |
|---|---|
| Free-text goal → skill IDs | **Constrained information extraction.** The LLM receives the closed vocabulary of 134 skill IDs and may answer only with those; anything invented is dropped by post-validation. |
| The deterministic fallback | **Four scored layers:** curated phrase patterns → alias lookup (803 aliases) → fuzzy matching (`difflib`, 0.82 cutoff) for typos → **TF-IDF cosine similarity** for paraphrase. |
| Refusing an unclear goal | **Confidence thresholds with an explicit reject option.** Above `ACCEPT` it plans; between thresholds it returns ranked candidates; below both it reports out-of-domain. |
| Ticking a known skill | **Downward transitive closure** — one tick credits everything beneath it in the graph. |
| Heard of / can use / can teach | **Graded self-assessment**, feeding a refresher bias into the course scorer. |
| Pasting a course history | The same resolver, applied line by line; unmatched lines are reported, never credited. |

### Building the path — the recommendation method

| Step | Method |
|---|---|
| What you still need | **Transitive prerequisite closure** of the targets, traversal stopping at skills you already have, then a **set difference**. That is the gap. |
| The order | **Kahn's topological sort**, with a deterministic tie-break (category → graph depth → ID). This is the structural guarantee. |
| Choosing the courses | **Greedy set cover** — the standard approximation to an NP-hard problem — guided by a **weighted linear scoring function**: +5 per newly covered skill, −6 for a prerequisite the plan never teaches, −1.5 per level mismatch, +2 per interest tag, small hours penalty. |
| Keeping it consistent | A **fixpoint loop**: if a chosen course assumes something the plan does not cover, that assumption joins the gap and the path is rebuilt. |
| Milestone phases | **Run-length segmentation** by skill category over the ordered path, then greedy merging to land between three and five phases. |

> No embeddings and no vector search. Similarity answers "what is related to my goal"; the
> real question is "what am I ready for", which is reachability, not distance.

### The "Why this?" modal

| Section | Method |
|---|---|
| The prerequisite chain | **Breadth-first search over reversed edges**, from the missing skill up to a goal skill. BFS because the shortest chain is the most readable one. |
| The sentence | **Template-based generation** from that trace — not model output. |
| "Why here and not earlier" | **Longest-path dynamic programming** on the DAG for the earliest slot, a backward pass for the latest; the difference is slack. |
| "What if you skip it" | **Forward reachability** over the dependents index: which skills go unlearned, which courses are stranded, whether a goal skill is lost. |
| The plain-words box | Optional LLM rewording, labelled as such. The trace is always shown alongside. |

### The rail panels

| Panel | Method |
|---|---|
| Insights → Leverage | **Downstream reachability count** per course — how many skills and courses it unblocks. |
| Insights → Critical path | **Critical Path Method (CPM)** with hours as durations: earliest start, latest start, slack. Zero slack means the whole plan waits for it. |
| Insights → Graph health | Aggregate queries over the graph: single-source skills, fan-out, depth distribution. |
| Routes | **The same scorer, re-weighted five ways.** Every route is equally valid by construction. |
| Compare goals | **Set operations** over two computed paths, plus a third build over the union to get marginal cost. |

### Interactions

| Action | Method |
|---|---|
| Mark complete | **Full recomputation** from the updated skill set — no incremental patching, so state cannot drift. |
| Feedback (too hard / too easy) | **Candidate exclusion + replan**; alternatives come from an inverted index with a multi-key sort. |
| Already know this | Skills marked possessed, then the gap is recomputed. |
| Readiness for any course | The same gap machinery pointed at one course's prerequisites instead of a goal. |
| Time budget | **Cumulative walk** over the ordered path against hours-per-week and an optional deadline. |
| Your pace | **Ratio estimation** — actual hours over estimated, applied to the remainder. |
| Refreshers | **Staleness × graph position** — a skill resurfaces only when something that directly requires it is next. |
| Chat | **Intent classification by keyword matching** with slot-filling from path data. Computed first, optionally reworded. |

### Under the hood

| Concern | Method |
|---|---|
| Data validation at load | **DFS three-colour cycle detection** for the DAG, alias uniqueness, full course coverage. All fail loudly. |
| Ollama safety | One wrapper, **20-second hard timeout**, never raises, 60-second backoff after a failure. |
| Persistence | One JSON file, **atomic write** via temp file and `os.replace`. |
| Frontend | React 18 UMD + Babel standalone, no build step. Prerequisite links are **SVG béziers drawn from measured DOM positions**. |
| Performance | Everything recomputes per request — path build **0.3 ms**, goal parsing **4 ms** — so there is no cache to go stale. |

---

## Questions you are likely to be asked

**"What is the AI/ML technique here?"**
Knowledge representation as a DAG, classical graph algorithms for the planning, greedy
approximation for set cover, TF-IDF information retrieval for language understanding, and
an LLM confined to constrained extraction and grounded rewording.

**"Why not just ask the LLM for a learning path?"**
Three reasons. Ordering is unenforced — nothing structurally prevents transformers before
backpropagation. Explanations are post-hoc — asking a model to justify its own output
produces the same confidence whether the item was right or hallucinated. And state changes
require regeneration rather than incremental update. The graph fixes all three by
construction.

**"What happens if Ollama is not running?"**
Every feature still works. Paths, ordering, explanations and replanning are identical;
only the phrasing is plainer. The banner says which mode you are in.

**"Is the recommendation optimal?"**
No, and deliberately. Minimum set cover is NP-hard; this uses the standard greedy
approximation, which is near-optimal on a catalogue this size and is fully inspectable —
every score is arithmetic you can read.

**"How do you know it is correct?"**
Seven test suites. The persona suite asserts the invariants directly: no course before its
prerequisites, every gap skill covered, every goal reached, no redundant course. It is what
caught a real ordering bug when the catalogue grew.

**"What are the limitations?"**
The catalogue is hand-authored and single-domain. 27 of 134 skills are still taught by only
one course, so "too hard" cannot always swap. Course hours are nominal. And two
prerequisite edges are pedagogically arguable — `cnns → transformers` and
`deep-learning-fundamentals → feature-engineering` — which is exactly why the curriculum
lives in data rather than in a prompt.

---

## Numbers worth quoting

- **134 skills, 208 prerequisite edges, 239 courses, max depth 20**, eight categories.
- Maths specialisation **unblocks 19 courses and 25 skills**; the final RAG course unblocks
  nothing.
- **16 of 20 courses (339h) are forced sequence**, against 393h studied one at a time.
- RAG and fine-tuning share **76% of their courses** — the second goal costs 1 course
  and 14h, not 369h.
- Logging 45h against a 60h course moves the remaining estimate from **333h to 250h**.
- Path build **0.3 ms**, goal parsing **4 ms** — 8× faster than before the graph doubled,
  after indexing.
