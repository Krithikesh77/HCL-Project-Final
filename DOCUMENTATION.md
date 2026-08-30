# Design Documentation

AI-Powered Personalized Learning Path Recommender — domain: AI / ML engineering.

---

## 1. Problem understanding

A learner arrives with a sentence — *"I want to build RAG chatbots"* — and a partial,
usually unarticulated set of existing skills. They want to know what to study, **in what
order**, and why.

The naive framing is a recommendation problem: score courses against the goal and return
the top ten. That framing is wrong, and it fails in a way learners feel immediately. The
top ten courses for "RAG chatbots" are all *about* RAG. None of them are the courses the
learner can actually start on Monday. A relevance ranking has no notion of readiness.

The correct framing is a **dependency resolution problem**. Skills form a partial order:
you cannot usefully learn vector databases before embeddings, and you cannot learn
embeddings before transformers. A learning path is therefore:

> a topological ordering of the transitive prerequisite closure of the goal, minus what
> the learner already knows, covered by concrete courses.

This is the same shape of problem as a package manager resolving dependencies, and it
should be solved the same way — by traversing a graph, not by asking a model to imagine
an ordering.

### Why not just let an LLM write the path?

An LLM will happily produce a 12-week plan for any goal. Three things go wrong:

1. **Ordering is unenforced.** Nothing structurally prevents "Week 3: Transformers,
   Week 7: Backpropagation". It usually gets the famous orderings right and quietly gets
   the unfamiliar ones wrong.
2. **Explanations are post-hoc.** Asking "why is this course here?" asks the model to
   justify its own previous output. It cannot say "it isn't, that was a mistake" — the
   justification is generated with the same confidence whether the item was correct or
   hallucinated. The explanation carries no information about correctness.
3. **State changes require regeneration.** "I finished course 3" means re-prompting and
   hoping the new plan is consistent with the old one. There is no incremental update.

The graph fixes all three by construction, and it makes the explanation a *read* rather
than a *generation*:

```
explain("c052") ─▶ walk reverse edges from vector-databases up to the goal
                ─▶ "rag requires vector-databases, which requires embeddings"
```

If that chain does not exist, the course is not in the path in the first place. The
explanation cannot disagree with the plan, because it is derived from the same edges the
plan was derived from. **This grounded-explanation property is the entire point of the
project**, and it is why the LLM is deliberately confined to two jobs at the edges.

---

## 2. Solution approach

### The pipeline

```
1. goal text ──▶ target skill ids          LLM (constrained vocabulary) | keyword fallback
2. profile   ──▶ possessed skill ids       + downward closure of the graph
3. gap       = closure(targets) - possessed          transitive closure
4. order     = topological sort of the gap subgraph  Kahn's algorithm
5. courses   = greedy set cover over the ordered skills
6. phases    = contiguous chunks of the ordered path, named by skill category
7. why       = reverse-edge traversal from a taught skill up to a target
```

### Step 2 in detail: downward closure of the profile

If a learner ticks "scikit-learn basics", the system credits them with everything below
it in the graph — pandas, NumPy, Python, statistics, probability, calculus, linear
algebra, data visualisation. Eleven skills from six ticks. This is what turns the
fine-tuning persona's path from 20 courses into 12, and it means the intake form does not
have to enumerate a whole CV.

### Step 1 in detail: resolving a goal, and refusing to

The first version of the parser ended with `if not hits: hits = DEFAULT_TARGETS`.
It was written to be robust - "never fail, always return a target" - and that was
the wrong instinct. A typo, an out-of-domain request, gibberish and an empty
string all became a confident twelve-course plan for generic supervised learning.
The system was never *unsure*; it was wrong with the same certainty it was right.

`resolve_goal()` now runs four scored layers and is allowed to return nothing:

| Layer | Technique | Catches |
|---|---|---|
| 1 | curated goal phrases | "chat with my docs" -> rag |
| 2 | alias / name / id lookup over 803 curated aliases | "pinecone", "vLLM", "kubernetes" |
| 3 | fuzzy match (`difflib`, cutoff 0.82) | "pytourch", "recomender", "kubernets" |
| 4 | TF-IDF cosine over name + description + aliases | "make computers understand pictures" |

Two thresholds turn scores into behaviour. Above `ACCEPT_SCORE` it plans. Between
`SUGGEST_SCORE` and `ACCEPT_SCORE` it returns ranked candidates and the UI asks
which one the learner meant. Below both it reports `out_of_domain` and says the
graph does not cover this. Every match carries its method and evidence, so the
interface can show *how* a goal was read rather than asserting a result.

Two matching rules earn their keep. Matching happens on **stems**, so "chatbots
over our internal docs" still hits the phrase "chatbot over". And **longest match
wins on the stem span**, so "vector databases" credits Vector Databases without
also crediting SQL through the shorter alias "database" or Linear Algebra through
"vectors" - whichever phrase explains the most words owns that span.

The LLM path is held to the same standard: it may only answer with ids from the
catalogue, it is told to return `[]` for anything outside AI/ML, and an empty or
invalid answer falls through to the resolver rather than to a default.

What the parser deliberately does *not* do is fake coverage, and reinforcement learning
is the worked example. The graph originally had no RL node, so "I want to do
reinforcement learning" offered RLHF as a low-confidence candidate and nothing more -
RLHF is an LLM fine-tuning technique, not RL, and mapping one onto the other would have
been the same sin as the default target, one level further down. The refusal was logged
as a vocabulary gap, the gap justified building the RL branch, and the goal now resolves
to Reinforcement Learning Fundamentals - still not to RLHF, which the test suite asserts.
Refuse, record, then fix the content: that loop is the point, not the refusal on its own.

### The vocabulary gap log

Terms that resolve to nothing are appended to `vocab_gaps.json` along with the
goal, the candidates offered and - when the learner clarifies - what they
actually meant. `GET /api/vocab-gaps` ranks them. That last field is the useful
one: "learner typed X, meant Y" is exactly an alias waiting to be written, and
the ranked unknown terms are a content backlog generated by real use instead of
guesswork. Two aliases and a stemming fix in this build came straight out of
reading that log after ten minutes of testing.

### Step 3 in detail: closure that stops at what you know

The traversal does **not** expand through possessed skills. If you already have PyTorch,
its prerequisites are not re-derived and not re-taught. The gap is exactly the frontier
between what you have and what your goal needs.

### Step 4 in detail: deterministic ordering

Kahn's algorithm produces *a* valid order; which one depends on how you break ties among
simultaneously-unblocked nodes. Arbitrary tie-breaking means the same learner sees a
different path on Tuesday, which destroys trust. Two documented rules are implemented:

- `tie_break="depth"` — `(graph depth, skill id)`, the literal reading of "depth then
  alphabetically".
- `tie_break="phase"` *(default)* — `(skill category, graph depth, skill id)`.

Both are fully deterministic. The default adds the category term because pure depth
ordering, while valid, interleaves unrelated branches — Docker landing between NumPy and
Pandas — which produces incoherent milestone names like *Foundations → Production &
MLOps → Data & Statistics → Core ML → Production & MLOps II*. The category term keeps
each branch contiguous and yields *Foundations → Data & Statistics → Core ML →
Production & MLOps*, with identical prerequisite guarantees.

### Step 5 in detail: course selection as set cover

Walking the ordered skills, each still-uncovered skill picks the highest-scoring course
that teaches it; that course also covers any other gap skills it teaches, which is what
keeps paths short. The score is pure Python arithmetic:

| Term | Weight | Purpose |
|---|---|---|
| newly covered gap skills | +5 each | prefer courses that cover more of the gap (set-cover greed) |
| skills taught but not needed | −1 each | penalise irrelevant breadth |
| prerequisites scheduled later in the path | −2 each | discourage ordering friction |
| prerequisites the path never covers | −6 each | hard penalty: extra baggage the learner would have to acquire elsewhere |
| level distance from the skill's intrinsic level | −1.5 each step | a beginner skill should get a beginner course |
| course level equals learner level | +1 | mild personalisation |
| interest tag overlap | +2 per match, capped | tag-based personalisation |
| hours | −0.05 per hour | prefer the shorter of two equivalent courses |

Ties break on course id, so selection is reproducible. This is where the competing
courses in the catalogue matter: the RAG learner gets *Building and Evaluating RAG
Applications* rather than *Production RAG*, because the latter additionally requires
model-evaluation, which that learner's path never covers (−6).

### Step 7 in detail: the explanation trace

`explain()` returns structured data, not prose:

- for each skill the course teaches: the **shortest reverse-edge chain** from that skill
  up to one of the learner's goal skills, plus the sentence rendered from it;
- each of the course's own prerequisites, tagged with where it came from — already known,
  or taught by a specific earlier course in this path;
- every later course this one unblocks, and via which skill;
- position and milestone.

The frontend renders the chain as breadcrumbs (`Retrieval-Augmented Generation (your
goal) → requires → Vector Databases → requires → Embeddings`). The optional LLM rewrite
sits in a separate box, explicitly labelled *"reworded by ollama"* versus *"straight from
the graph"*, so a viewer can always tell which text is derived and which is generated.

---

## 3. System architecture

```
                    ┌──────────────────────────────────────────────┐
   browser          │  static/index.html                           │
   localhost:8000   │  React 18 + Babel standalone + Tailwind, CDN │
                    │  ┌──────────┬──────────────┬──────────────┐  │
                    │  │ Intake / │   ROADMAP    │  Dashboard   │  │
                    │  │   Chat   │  (timeline + │  (progress,  │  │
                    │  │          │   SVG prereq │  milestones, │  │
                    │  │          │   links)     │  next action)│  │
                    │  └──────────┴──────────────┴──────────────┘  │
                    └───────────────────┬──────────────────────────┘
                                        │  fetch (JSON)
                    ┌───────────────────▼──────────────────────────┐
                    │  main.py  -  FastAPI                         │
                    │   /api/intake  /api/plan  /api/explain/{id}  │
                    │   /api/complete  /api/feedback  /api/chat    │
                    │   /api/learner  /api/health  /api/skills     │
                    └───┬───────────────────────────────┬──────────┘
                        │                               │
            ┌───────────▼──────────┐        ┌───────────▼───────────────┐
            │  engine.py           │        │  ask_ollama()             │
            │  (no model, ever)    │        │  20s hard timeout         │
            │                      │        │  never raises             │
            │  load_graph          │        │                           │
            │  gap_analysis        │        │  (a) goal -> skill ids    │
            │  topo_order          │        │  (b) reword explanation   │
            │  select_courses      │        └───────┬───────────────────┘
            │  build_path          │                │ fails / absent
            │  explain             │                ▼
            │  replan              │        ┌───────────────────────────┐
            └───────┬──────────────┘        │  deterministic fallbacks  │
                    │                       │  keyword_match_targets()  │
        ┌───────────▼─────────────┐         │  trace["summary"]         │
        │ data/skills.json  (DAG) │         │  canned_answer()          │
        │ data/courses.json       │         └───────────────────────────┘
        │ learners.json  (state)  │
        └─────────────────────────┘                 ┌──────────────────┐
                                                    │ Ollama (optional)│
                                                    │ localhost:11434  │
                                                    │ llama3.2:3b      │
                                                    └──────────────────┘
```

The critical property of this diagram is the **left/right split under FastAPI**: the
engine column never talks to the model column. Nothing in a path, an ordering or an
explanation flows through the right-hand side. Deleting Ollama from the diagram entirely
leaves the system functional.

### State

A single `learners.json` file: `{version, active, learners: {id: profile}}`. A profile
holds the goal, level, known skills, possessed skills (the closure), interests, target
skills, completed courses, rejected courses and an event history. Writes go to a temp
file and `os.replace()`, so a crash mid-write cannot corrupt it. There is no database,
no ORM and no migration story — deliberately, for a system whose whole job is to be
runnable in one command.

Every read endpoint recomputes the path from the stored profile rather than caching it.
Recomputation is a few milliseconds of graph traversal, so there is no cache to
invalidate and no way for stored path state to drift from the graph.

---

## 4. AI/ML techniques used

**Knowledge representation.** The domain is modelled as a directed acyclic graph of 134
skills with 208 prerequisite edges (max depth 20), plus a course catalogue that maps
courses onto skills they teach and skills they presuppose. This is a hand-authored
knowledge base, and it is the reason the system can make guarantees a statistical model
cannot.

**Graph algorithms.**
- DFS three-colour cycle detection for DAG validation at load time, which names the
  offending cycle rather than failing obscurely.
- Transitive closure (frontier expansion) for gap analysis, with early stopping at
  possessed nodes.
- Kahn's topological sort with an explicit, documented tie-break for stability.
- Reverse-edge BFS for shortest goal-justification chains in explanations.
- Longest-path depth computation (memoised) used for ordering and level heuristics.

**Greedy set cover.** Course selection is the classic greedy approximation to minimum set
cover, extended with a domain-specific cost function (prerequisite satisfaction, level
fit, interest overlap, duration). Exact minimum set cover is NP-hard; greedy is the right
engineering call and produces near-minimal paths on this catalogue.

**Content-based personalisation without embeddings.** Interest matching is tag overlap
plus substring matching over title and description; level matching compares the course
level against the skill's intrinsic level (derived from graph depth) and the learner's
stated level. No FAISS, no sentence-transformers, no torch — for a 75-course catalogue,
lexical matching over curated tags is both faster and more debuggable than a vector index.

**Information retrieval without embeddings.** Goal parsing uses a TF-IDF vector
space model over the skill catalogue (log-scaled term frequency, smoothed inverse
document frequency, L2-normalised, cosine similarity) plus character-level fuzzy
matching for typos - about 40 lines of pure Python, no vector database and no
model. For a 62-document corpus this beats an embedding index on latency,
dependencies and explainability: every match can name the phrase that produced it.

**Constrained LLM information extraction.** The one place the model does real work is
mapping free text onto a closed vocabulary. The prompt supplies the full list of 62 valid
skill ids, demands a JSON array, and the response is parsed, filtered against the graph
and normalised (targets implied by a deeper target are dropped, since the closure will
pull them in anyway). An unusable response is indistinguishable from an absent one:
both fall back.

**Grounded generation.** The chat and explanation endpoints implement a
compute-then-phrase pattern: a deterministic answer is produced first, then handed to the
model with instructions to reword and add nothing. This bounds hallucination to phrasing
rather than facts, and it means the fallback path is not a degraded second implementation
— it is the same answer, unrewritten.

**Graceful degradation as a design constraint.** Every model call is wrapped in a single
function with a hard timeout that returns `None` on any failure. Reachability means "the
daemon answers *and* the configured model is pulled", because a running daemon without
the model 404s on every generate and would otherwise look healthy.

---

## 5. Key features and workflows

### Intake → path

The learner writes a sentence, picks a level, and ticks known skills (or loads one of
three demo personas). The sentence goes to the LLM with the closed skill vocabulary;
the fallback is phrase matching. The ticked skills are closed downward. The gap is
computed, ordered, covered with courses, and grouped into 3–5 milestone phases named
after the dominant skill category in each contiguous chunk.

### The roadmap

A vertical timeline grouped by phase, on a warm off-white canvas with each card tinted by
its milestone colour — mint for Foundations, sky for Data & Statistics, lilac for Core ML,
blush for Deep Learning, peach for LLM Systems, sage for Production & MLOps. A row of
filter pills above the timeline narrows it to a single phase, which is useful both for
long paths and for talking through one phase at a time on camera. Each card shows title,
provider, hours, level, the skills it adds to the profile, and how many courses it needs
and unblocks. Prerequisite
links are drawn as SVG bezier curves in a left gutter, measured from real card positions
after layout; long-range links bow out further so they do not overlap short ones.
Hovering a card highlights its incoming and outgoing edges and dims unrelated cards —
the graph structure becomes visible without a graph library.

### "Why this?"

Opens the trace: the prerequisite chain as breadcrumbs from goal to missing skill, the
course's own prerequisites annotated with where each one comes from, the later courses it
unblocks, and a plain-words paragraph labelled by provenance.

### Mark complete → replan

Completing a course adds its taught skills to the profile, and the entire path is
recomputed from scratch. Because other courses may have existed only to reach those
skills, the plan can shrink by more than one course — the UI reports the delta explicitly
("Gained Linear Algebra, Calculus for ML, Probability. 1 course dropped out of the plan,
60h saved. 19 courses left.").

### Feedback → swap or skip

- `too_hard` / `too_easy` reject the course, nudge the learner's level down or up, and
  replan. If the skill is taught by exactly one course, the system says so instead of
  pretending to swap.
- `already_know` marks the taught skills possessed, which removes them and anything that
  existed only to reach them.

### Skill inspector: walking the graph

Any skill chip — on a course card, in the dashboard, in a "Why this?" chain — opens that
node: its prerequisites and dependents (each annotated with whether you already have it or
still need it), the courses that teach it, and which of those is scheduled in your path.
Every related skill in the inspector is itself clickable, so the graph can be explored in
either direction. This is served by `GET /api/graph/skill/{id}`, which annotates the node
against the active learner's current path.

### Changing your goal without losing progress

Because learner state is a *set of possessed skills* rather than a rendered plan, a new
goal does not mean starting over. `POST /api/intake` with `keep_progress` re-aims the
existing learner: completed courses, learned skills, feedback and history all survive, and
the gap for the new target is computed from where they now stand. A learner who finished
the RAG path and then asks for fine-tuning gets a much shorter path than a newcomer would,
automatically — no special-casing, just set arithmetic on the graph.

### The finished path

When the gap empties, `replan()` reports `complete`, progress reads 100%, and there is no
next action. The roadmap shows a completion card, the dashboard zeroes out, and chat
answers from what was achieved rather than what is left. `scripts/test_api.py` drives a
learner all the way to this state by completing every course in turn, and separately
covers a learner whose stated skills already satisfy their goal — an empty gap from the
very first plan.

### Grounded chat

Questions are answered from path data: what to start with, how long it will take at ten
hours a week, why a named course is present (which routes to the same graph trace), how
the ordering works, progress. With Ollama present the deterministic answer is reworded;
without it, it is served as-is.

---

## 5b. Graph analytics: questions a ranked list cannot answer

Everything in this section is a query over the same DAG, with no model involved.

**Skip risk** (`skip_impact`) is `explain()` run backwards. It walks reverse edges from
what a course teaches to collect every gap skill that would go unlearned, then finds the
later courses that require any of them. The verdict is *goal-critical* when a target skill
is among the casualties, *blocking* when only later courses are, *safe* for a leaf. It
also reports whether another course teaches the same skills, so the honest advice is often
"swap, don't skip".

**Leverage** (`leverage_ranking`) counts downstream gap skills and transitively waiting
courses per course. It exists because the argument for foundations is much easier to make
with a number: on the RAG path the maths specialisation unblocks 19 courses and 25 skills,
while the final RAG course unblocks nothing.

**Slot windows and the critical path** (`slot_analysis`, `critical_path`) answer *why
position 11 and not 3*. Earliest slot is the longest dependency chain ending at a course;
latest is bounded by what it blocks. Criticality, though, is a question about **time**, not
positions - a first attempt computed slack from slot indices, which can never be zero
because a 20-course plan is never a single 20-long chain. Proper CPM with hours as
durations gives the real answer: 16 of 20 courses and 339h are forced sequence, against
393h if studied strictly one at a time.

**Readiness** (`readiness`) runs the gap machinery against a single course's prerequisites
rather than a goal, so it works for any course in the catalogue.

**Alternative routes** (`alternative_routes`) re-weights the same scorer five ways. The
graph still fixes what must be learned and in what order, so every route is equally valid -
they differ in length, price and style. The honest caveat ships with the feature: real
choice exists at only 16 of 27 skills on the RAG path, because the rest are taught by
exactly one course.

**Goal comparison** (`compare_goals`) computes the number that actually informs a career
decision - the overlap. Two goals sharing a prerequisite spine cost far less together than
apart: RAG and fine-tuning share 76% of their courses, so the second costs 1 course and
14h rather than 369h.

**Graph health** (`graph_health`) is the maintainer's console: 27 of 134 skills are taught
by exactly one course, which is precisely where "too hard" feedback cannot swap anything
and where every route is identical.

## 5c. Learner modelling without machine learning

**Graded self-assessment.** Binary "do you know pandas" overstates half the answers, so
the intake asks for *heard of / can use / can teach*. Only can_use and above become
possessed; heard_of is not a skill you have, but it changes the plan - `select_courses`
takes a refresh set and biases those picks toward shorter, lower-level courses. Python
drops from a 30h course to a 22h one. Where the only alternative is a multi-skill bundle
the pick correctly does not change, because coverage still outweighs length.

**Observed pace.** `POST /api/complete` accepts the hours a course actually took;
`velocity_report` divides actual by estimated and re-forecasts. A learner running 1.46x
sees 574h remaining rather than 393h. With nothing timed the factor is exactly 1.0 and the
report says so, rather than implying precision it does not have.

**Time budgets.** `budget_forecast` walks the path accumulating hours against
hours-per-week and an optional deadline, marks where the budget runs out and names the
first milestone at risk. It takes the pace factor, so the two compose.

**Refreshers.** Review timing driven by graph position rather than a generic repetition
schedule: a skill resurfaces only when a course that directly requires it is coming up.

**History import.** `parse_history` reuses the goal resolver wholesale - unknown words in
a learning history are the same problem as unknown words in a goal - and keeps the same
discipline, reporting unmatched lines instead of crediting them.

## 5d. Growing the graph, and keeping it fast

The catalogue went from 62 skills / 75 courses to **134 skills, 208 edges and 239
courses** across eight categories, adding a full reinforcement-learning branch, an AI
safety and governance branch, and depth in data engineering, classical ML, generative
vision and LLM systems. Courses per skill went from 1.4 to 2.1, and skills taught by
exactly one course - the fragile ones - from 38 of 62 to 27 of 134.

Three things made that safe rather than risky:

**The validator did the reviewing.** Adding 72 skills failed the load immediately with a
list of skills no course taught, then again on eight aliases claimed by two skills each
("random forest" wanted by both ensemble methods and tree models, "orchestration" by both
Kubernetes and multi-agent systems). Alias uniqueness is now enforced at load time
alongside the DAG check, because an ambiguous alias silently degrades goal parsing rather
than breaking anything visibly.

**A real ordering bug surfaced, and was fixed structurally.** With more courses competing,
the persona suite caught `NLP with Transformers` being scheduled for a learner who did not
have `python-oop`, which the plan never taught. The old scorer only *penalised* an unmet
prerequisite (-6) and a course covering two skills (+10) could still win. Penalties cannot
protect an invariant, so selection now filters those candidates out entirely, and
`build_path` iterates to a fixpoint: if the only course teaching a skill assumes something
the plan does not cover, that assumption is added to the gap and the path is rebuilt. The
guarantee is now structural in the course layer as well as the skill layer.

**Performance was measured, not assumed.** `scripts/benchmark.py` sets a budget per
operation. The expansion pushed goal parsing to 32 ms, because exact matching tested all
~1,500 aliases with a regex each and fuzzy matching compared every word against all of
them. Two indexes built once per graph - aliases bucketed by first token, and phrase keys
bucketed by length with a character-overlap prefilter - took that to **4 ms**, an 8x
improvement that also makes the cost flat rather than linear in catalogue size. Everything
else was already sub-millisecond and stayed there.

One demo-visible latency fix came out of the same measurement: a running Ollama daemon
with the model missing 404s in about two seconds, and the app was paying that on every
request. `ask_ollama` now backs off for 60 seconds after a failure and re-probes
afterwards, which took intake from 2,000 ms to 4 ms in deterministic mode while still
picking up a model pulled mid-session.

## 6. Challenges faced

**Keeping the LLM out of the reasoning loop while still using it for language.** The
temptation is to let the model "help" with ordering or selection, which would silently
destroy the property the project exists to demonstrate. The resolution was architectural:
the engine module has no HTTP client and no knowledge that Ollama exists, and the API
layer can only pass it validated skill ids. The compute-then-phrase pattern lets the model
improve readability without ever being upstream of a decision.

**Deterministic ordering versus coherent milestones.** The first implementation broke
topological ties by `(depth, id)` exactly as specified. It was correct and produced
useless phase names, because valid orderings interleave independent branches. Rather than
patching the presentation layer, the tie-break became a parameter with a category-aware
default — same guarantees, same determinism, coherent phases. Both modes are kept and
documented, since the change is a curriculum-design opinion rather than a correctness fix.

**Making course selection non-trivial without over-engineering it.** Early scoring was
coverage-only, which always picked the biggest bundle — a 60-hour maths specialisation to
teach one skill. Adding an hours penalty overcorrected into fragmentation. The balance
that works weights new coverage heavily, penalises unmet prerequisites much more heavily
when the path will never cover them, and uses hours only as a tie-breaker.

**Ollama absence is easy; Ollama *partial presence* is not.** The daemon running without
the configured model pulled returns HTTP 404 on every generate while looking perfectly
healthy on `/api/tags` — the app reported "LLM mode" and then fell back on every request.
Reachability was redefined to include model presence, and the startup banner now
distinguishes the three states and prints the models that *are* available. A related
discovery: a cold model load on this machine took 73 seconds, far beyond the 20-second
cap, so the first request after starting Ollama legitimately falls back. That is
documented rather than papered over by raising the timeout.

**Honest feedback handling.** `too_hard` on a course whose skill has no alternative
originally re-selected the same course and reported "Replaced with …" — the exclusion
silently failed because the fallback candidate list ignored the ban. Caught by the API
test. The fix was to check for a genuine alternative *before* mutating any state and
report the situation plainly when there is none.

**No build step, real framework.** Three browser-only issues, none visible from the
Python side: recent Babel standalone defaults to the automatic JSX runtime and emits an
`import` statement that a classic `<script>` cannot execute (fixed with classic-runtime
pragmas and a pinned major version); measuring card geometry in an unconditional layout
effect that calls `setState` loops forever (React error #185, fixed by keying the effect
to the path); and an open dropdown was painted over by the next card because both are
`position: relative` siblings, so DOM order wins (fixed by lifting the card while its menu
is open). All three were found by driving the actual page, which is the argument for
testing the UI rather than trusting that it renders.

**Two features that had to be re-derived after their first version lied.** The `fastest`
strategy penalised raw hours per course, which made the greedy selector pick many small
courses and pushed the total *up* - 395h against a 393h baseline, under a label promising
the shortest path. What actually shortens a path is hours per newly covered skill, so
bundles teaching three things at once are rewarded; it now lands at 389h. Separately, the
critical path measured slack in slot indices and reported that zero courses were critical,
because a 20-course plan is never a single 20-long chain. Both bugs produced confident,
plausible, wrong numbers - which is exactly the failure mode this project exists to avoid,
appearing in my own analytics rather than in a model.

**Modelling judgement calls worth flagging.** The `cnns → transformers` edge follows the
requested spine but is pedagogically debatable, and it adds roughly two courses to every
LLM-oriented path. Similarly, `deep-learning-fundamentals → feature-engineering` pulls the
whole classical-ML branch into deep-learning goals. Both are defensible curriculum
positions, and both are one-line edits in `skills.json` — which is itself the argument for
keeping the curriculum in data rather than in a prompt.
