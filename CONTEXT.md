# CONTEXT.md — what was built, and how

A handover document for the AI-Powered Personalized Learning Path Recommender.
README.md is the pitch, DOCUMENTATION.md is the design write-up; this file is the
engineering narrative: what exists, why each decision was made, what broke along the way,
and where to pick it up.

---

## 1. The one idea everything else defends

A learning path is a **prerequisite graph problem**, not a ranking problem and not a
text-generation problem.

```
free-text goal ──[LLM, validated | keyword resolver]──▶ target skill ids
                                                             │
                 learner profile (possessed skills) ─────────┤
                                                             ▼
                        transitive prerequisite closure, stopping at what you know
                                                             ▼
                                      GAP: exactly the skills between you and the goal
                                                             ▼
                              topological sort (Kahn, deterministic tie-break)
                                                             ▼
                          greedy set-cover course selection (weighted scoring)
                                                             ▼
                                grouping into 3–5 named milestone phases
                                                             ▼
                       explanations read directly off the prerequisite edges
```

The language model never chooses a course, never orders anything, and never decides why
something is necessary. It does two jobs at the edges — free text → skill ids from a fixed
vocabulary, and rewording a sentence the graph already produced — plus grounded chat, which
also computes its answer deterministically first and only asks the model to rephrase it.

Every consequence in the app follows from that: paths are reproducible, explanations cannot
disagree with the plan (both derive from the same edges), and killing Ollama changes only
phrasing.

---

## 2. Current state

| | |
|---|---|
| Skills / edges / max depth | **134 / 208 / 20** |
| Courses / providers / free | **239 / 23 / 34** |
| Courses per skill | **2.1** (27 skills still single-source) |
| Categories | 8 |
| API endpoints | 25 |
| Test suites | 7, all passing |
| Lines | engine 1,905 · main 1,160 · frontend 2,299 · tests 1,000 |
| Commits | 17 |
| Dependencies | fastapi, uvicorn, requests |

Run it:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Ollama is optional. Without it the app runs in deterministic mode and the startup banner
says so.

---

## 3. How it was built, in order

The first five steps were the original brief, each committed separately.

**1. Data first (`73bebe1`).** `data/skills.json` and `data/courses.json`, hand-authored,
with the DAG validated at load. Data before code because every guarantee downstream rests
on the graph being well formed.

**2. Engine, tested headless (`ee1a4c2`).** `engine.py` plus `scripts/test_personas.py`,
which runs three personas and asserts the invariants — no course before its prerequisites,
every gap skill covered, every goal reached, no redundant course. The paths were eyeballed
before any UI existed, which is the reason the UI never had to debug the engine.

**3. API with fallbacks (`fa1e38e`).** FastAPI over the engine, learner state in one JSON
file, and `ask_ollama()` as the single choke point for every model call. Verified twice:
with Ollama unreachable and against a live model.

**4. Frontend (`50e12b8`, redesigned `8283fcd`).** One `static/index.html`: React 18 UMD +
Babel standalone + Tailwind, no build step. Roadmap first, then dashboard, then chat.

**5. Docs (`eb6fa32`).** README and DOCUMENTATION.

Then, in response to review:

**6. Rough edges (`410ced3`).** Skill-graph inspector wired to an endpoint that had no UI,
goal changes that keep progress, the path-complete end state actually exercised.

**7. The goal parser (`2220507`, `982399a`).** The biggest correctness fix in the project —
see §5.

**8. Eleven features (`fdb5859` → `e9d7146`).** Graph analytics and learner modelling,
engine → API → UI, each layer tested before the next.

**9. Graph expansion (`3ffb989`).** 62 → 134 skills, 75 → 239 courses, with a performance
budget and an 8× parser speedup.

---

## 4. Architecture

```
main.py            FastAPI: HTTP, learner state, the two LLM calls + fallbacks
  │
  ├── engine.py    the entire recommender. No HTTP client, no knowledge Ollama exists
  │     ├── load_graph()          DAG validation, alias uniqueness, coverage checks
  │     ├── resolve_goal()        four-layer goal parser that may return nothing
  │     ├── gap_analysis()        transitive closure minus possessed
  │     ├── topo_order()          Kahn with a documented tie-break
  │     ├── select_courses()      greedy set cover + weighted scoring + strategies
  │     ├── build_path()          milestones, card links, fixpoint over course prereqs
  │     ├── explain()             structured trace from prerequisite edges
  │     ├── replan()              remaining path + progress from current state
  │     ├── skip_impact() leverage_ranking() slot_analysis() critical_path()
  │     ├── readiness() alternative_routes() compare_goals() graph_health()
  │     └── split_self_assessment() velocity_report() budget_forecast()
  │         refresher_prompts() parse_history()
  │
  ├── data/skills.json      134 skills: id, name, description, requires[], aliases[]
  ├── data/courses.json     239 courses: teaches[], requires[], level, hours, tags[]
  ├── learners.json         the only persistence (gitignored, atomic writes)
  ├── vocab_gaps.json       terms the graph could not name (gitignored)
  └── static/index.html     the whole frontend, one file
```

**The load-bearing split** is that `engine.py` has no HTTP client and no idea Ollama
exists. The API layer can only hand it validated skill ids. That is what makes "the model
never decides anything" a structural claim rather than a promise.

**No caching anywhere.** Every request recomputes the path from the stored profile —
milliseconds of graph traversal — so there is no cache to invalidate and no way for stored
state to drift from the graph. `scripts/benchmark.py` exists to keep that affordable.

---

## 5. The goal parser: the project's sharpest edge

The original `keyword_match_targets()` ended with `if not hits: hits = DEFAULT_TARGETS`.
Written to be "robust", it meant a typo, an out-of-domain request, gibberish and an empty
string all produced a confident twelve-course plan for generic supervised learning. The
system was never *unsure*; it was wrong with the same certainty it was right — in a project
whose entire claim is never being confidently wrong.

`resolve_goal()` replaced it with four scored layers, and permission to return nothing:

| Layer | Technique | Catches |
|---|---|---|
| 1 | curated goal phrases | "chat with my docs" → rag |
| 2 | alias / name / id lookup (~1,500 aliases) | "pinecone", "vLLM", "kubernetes" |
| 3 | fuzzy match, `difflib` at 0.82 | "pytourch", "recomender", "kubernets" |
| 4 | TF-IDF cosine over name + description + aliases | "make computers understand pictures" |

Two thresholds turn scores into behaviour: above `ACCEPT_SCORE` it plans; between
`SUGGEST_SCORE` and `ACCEPT_SCORE` it returns ranked candidates and the UI asks; below
both it reports `out_of_domain`. Every match carries its method and evidence, so the
interface shows *how* a goal was read.

Two matching rules were needed for real phrasings:

- **Stem-based matching** — "chatbots over our internal docs" was resolving to *Prompt
  Engineering* because the phrase `"chatbot over"` did not match the plural.
- **Longest match wins on the stem span** — "vector databases" was also crediting SQL via
  the shorter alias "database", and Linear Algebra via "vectors".

The LLM path is held to the same standard: catalogue ids only, an explicit instruction to
return `[]` for anything outside AI/ML, and post-validation that drops anything invented.

**The vocabulary gap log** (`vocab_gaps.json`, `GET /api/vocab-gaps`) records what the
parser could not name, plus what the learner *meant* when they clarified — "typed X, chose
Y" is an alias waiting to be written. It paid for itself twice within minutes: 4-letter
plurals ("llms", "apis") were not being stemmed, and "chatbot" was in a goal pattern but in
no skill's aliases.

---

## 6. Feature inventory

**Core loop** — intake (goal + graded self-assessment + optional history paste) → path →
"Why this?" → mark complete → replan → feedback (too hard / too easy / already know) →
grounded chat.

**Graph analytics** (only answerable because it is a graph):

- **Skip risk** — `explain()` run backwards. Walks forward from what a course teaches to
  the skills that would go unlearned and the courses stranded. Verdict: goal-critical,
  blocking, or safe.
- **Leverage** — courses ranked by how much of the plan they unblock.
- **Slot windows** — earliest/latest slot each course could occupy, and its slack.
- **Critical path** — the zero-slack chain, in **hours** (CPM), with parallel vs sequential
  totals.
- **Readiness** — for any course in the catalogue: how many skills away, and the prep.
- **Alternative routes** — five selection strategies over the same required skills.
- **Goal comparison** — overlap and the marginal cost of a second goal.
- **Graph health** — fragile single-source skills, fan-out, depth, coverage.

**Learner modelling** (no machine learning — arithmetic over what the learner did):

- **Graded self-assessment** — heard of / can use / can teach; only can_use+ counts as
  known, heard_of earns a shorter refresher course.
- **Observed pace** — log actual hours, and every later forecast uses your ratio.
- **Time budgets** — hours per week and an optional deadline; names the milestone at risk.
- **Refreshers** — review timed by graph position, not a generic schedule.
- **History import** — paste completed courses; unmatched lines are reported, not credited.

**Clarification** — when the parser is unsure, the UI asks with ranked candidates instead
of planning something wrong.

---

## 7. Testing

Seven suites, all runnable without Ollama:

| Suite | Covers |
|---|---|
| `test_personas.py` | three personas, path invariants |
| `test_goal_parsing.py` | the resolver, including refusing to guess |
| `test_analytics.py` | skip risk, leverage, slack, critical path, routes, comparison |
| `test_learner_model.py` | self-assessment, pace, budgets, refreshers, import |
| `test_api.py` | every core endpoint end to end |
| `test_features_api.py` | the analytics and learner-modelling endpoints |
| `benchmark.py` | per-operation performance budget |

The persona suite is the one that matters most: it asserts the invariants the whole pitch
rests on, and it is what caught the ordering bug in §8.

---

## 8. Bugs found and fixed (the honest list)

**Ordering bug, caught by the persona suite after the graph grew.** `NLP with Transformers`
was scheduled for a learner without `python-oop`, which the plan never taught. The scorer
only *penalised* an unmet prerequisite (−6), and a course covering two skills (+10) could
still win. Penalties cannot protect an invariant: selection now filters those candidates
out, and `build_path` iterates to a fixpoint — if the only course teaching a skill assumes
something uncovered, that assumption joins the gap and the path rebuilds.

**"Shortest" made paths longer.** Penalising raw hours made the greedy selector pick many
small courses: 395h against a 393h baseline, under a label promising the shortest path.
Fixed by scoring hours *per newly covered skill* → 389h.

**The critical path was empty.** Slack was measured in slot indices, where zero is
unreachable — a 20-course plan is never a 20-long chain. Real CPM with hours as durations
gives the true answer.

**Dishonest feedback.** `too_hard` on a skill taught by only one course silently
re-selected the same course and claimed it had swapped.

**Goal changes discarded progress.** Re-aiming created a new learner; now `keep_progress`
carries completed courses and learned skills across.

**Three browser-only frontend bugs.** Babel's automatic JSX runtime emits an `import` that
kills a classic `<script>`; measuring card geometry in an unconditional layout effect looped
forever (React #185); an open dropdown was painted over by the next card because both are
`position: relative` siblings.

**Eight ambiguous aliases** after the expansion ("random forest" claimed by both
ensemble-methods and tree-models). Alias uniqueness is now enforced at load time.

**Ollama partial presence.** A daemon that is up but missing the model 404s while looking
healthy on `/api/tags`. Reachability was redefined to include model presence, and
`ask_ollama` now backs off 60s after a failure — intake went from 2,000 ms to 4 ms in
deterministic mode.

The first three are worth dwelling on: they produced confident, plausible, wrong numbers.
That is exactly the failure this project exists to avoid, appearing in its own analytics
rather than in a model.

---

## 9. Performance

`scripts/benchmark.py` sets a budget per operation, because the graph is meant to keep
growing and nothing is cached.

| Operation | Median |
|---|---|
| `build_path` | 0.3 ms |
| `replan` | 0.4 ms |
| `explain` / `skip_impact` / `slot_analysis` | < 0.1 ms |
| `alternative_routes` (5 paths) | 1.8 ms |
| `compare_goals` (3 paths) | 1.0 ms |
| `resolve_goal` | 4 ms |
| cold `load_graph` | 1.4 ms |

The expansion pushed goal parsing to 32 ms — exact matching regex-tested all ~1,500 aliases
and fuzzy matching compared every word against all of them. Two indexes built once per
graph (aliases by first token; phrase keys by length with a character-overlap prefilter)
took it to 4 ms and made the cost flat rather than linear in catalogue size.

---

## 10. Decisions worth knowing about

**Topological tie-break defaults to `"phase"`, not `"depth"`.** The brief said depth then
alphabetically; that is implemented and available, but pure depth ordering interleaves
independent branches (Docker between NumPy and Pandas) and produces incoherent milestone
names. The default adds a category term, with identical prerequisite guarantees.

**No embeddings, no vector store.** Similarity answers "what is related to my goal"; the
real question is "what am I ready for", which is reachability, not distance. At this scale
lexical matching over curated aliases is faster, has no dependencies, and can name the
phrase that produced each match.

**No database.** Recomputation is milliseconds, so `learners.json` with atomic writes is
enough.

**Greedy set cover, not exact.** Minimum set cover is NP-hard; greedy is the standard
approximation and near-optimal on this catalogue.

**Hand-authored data.** The graph is the product. `GET /api/graph/health` and
`/api/vocab-gaps` exist so growing it is driven by evidence rather than guesswork.

**Debatable edges, flagged rather than hidden.** `cnns → transformers` follows the brief's
spine but is pedagogically arguable and adds ~2 courses to LLM paths;
`deep-learning-fundamentals → feature-engineering` pulls classical ML into deep-learning
goals. Both are one-line edits in `skills.json` — which is the argument for keeping the
curriculum in data rather than in a prompt.

---

## 11. Where to pick it up

- **Hard vs soft prerequisite edges.** Every edge is currently blocking. Adding
  `recommended` edges would allow a fast track and a thorough route for one goal, with the
  trade-off stated.
- **`scripts/expand_graph.py`** — LLM proposes nodes and edges, the existing validator
  rejects bad ones, a human accepts. Same "model proposes, graph verifies" discipline.
- **The 27 single-source skills** listed by `/api/graph/health` — each is a place where
  "too hard" cannot swap anything.
- **Job description → skill gap**, which now has honest OOV handling underneath it.
- **A rendered DAG view.** Deliberately skipped (no graph library); the skill inspector
  walks the graph node by node instead.

---

## 12. Demo path

1. Load the **Python dev, no ML** persona → **Build my path**.
2. Click a skill chip → the graph inspector → walk to a neighbour.
3. **Why this?** on a mid-path course → prerequisite chain, *why here and not earlier*,
   *what if you skip it*.
4. **Mark complete** → watch the plan shrink, with the delta in the toast.
5. Phase pills → focus one milestone.
6. Rail → **Routes** (switch to Free first), **Compare** (add fine-tuning: 76% overlap),
   **Insights** (leverage, critical path, graph health).
7. Type "I want to be a web developer" → it declines instead of guessing.

Reset between takes with the **Start over** button or `POST /api/reset`.
