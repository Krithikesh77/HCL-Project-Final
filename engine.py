"""
Prerequisite-graph engine for the learning path recommender.

The core claim of this project: a learning path is a GRAPH problem, not a
ranking problem and not a text-generation problem.  Everything in this file is
deterministic pure Python:

    targets  -> transitive prerequisite closure -> minus what you already know
             -> topological sort -> course selection -> milestones -> explanation

No LLM is involved in any of it.  The LLM (see main.py) only ever (a) turns
free-text goals into target skill ids and (b) rewrites an explanation that was
already computed here.  If the LLM is unavailable, every function below still
produces exactly the same path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LEVELS = {"beginner": 0, "intermediate": 1, "advanced": 2}
LEVEL_NAMES = ["beginner", "intermediate", "advanced"]


class GraphError(RuntimeError):
    """Raised loudly at load time when the skill data is not a usable DAG."""


# ---------------------------------------------------------------------------
# Skill categories.  Used only to name milestones ("Foundations", "Core ML"...).
# Every skill must appear in exactly one bucket; that is checked at load time.
# ---------------------------------------------------------------------------
CATEGORIES: List[Tuple[str, Set[str]]] = [
    ("Foundations", {
        "python", "python-oop", "command-line", "git-version-control", "sql",
        "math-linear-algebra", "math-calculus", "probability",
        "algorithms", "software-testing", "dev-environments", "debugging-profiling",
        "math-optimization", "information-theory",
    }),
    ("Data & Statistics", {
        "numpy", "pandas", "data-cleaning", "data-viz", "data-pipelines",
        "statistics", "hypothesis-testing", "data-versioning",
        "data-modeling", "big-data-spark", "streaming-data", "data-quality",
        "bayesian-statistics", "causal-inference", "data-labeling", "synthetic-data",
    }),
    ("Core Machine Learning", {
        "sklearn-basics", "supervised-learning", "unsupervised-learning",
        "model-evaluation", "regularization", "feature-engineering",
        "hyperparameter-tuning", "ensemble-methods", "gradient-boosting",
        "imbalanced-data", "time-series", "recommender-systems",
        "linear-models", "tree-models", "svm-kernels", "dimensionality-reduction",
        "anomaly-detection", "model-interpretability", "model-calibration", "automl",
        "learning-to-rank", "active-learning",
    }),
    ("Deep Learning", {
        "deep-learning-fundamentals", "backpropagation", "pytorch",
        "training-optimization", "cnns", "computer-vision", "rnns",
        "distributed-training", "model-quantization",
        "transfer-learning", "data-augmentation", "object-detection",
        "image-segmentation", "autoencoders", "gans", "diffusion-models",
        "self-supervised", "graph-neural-networks", "speech-audio",
        "multimodal-models",
    }),
    ("Reinforcement Learning", {
        "rl-fundamentals", "bandits", "rl-environments", "q-learning", "deep-rl",
        "policy-gradients", "actor-critic", "offline-rl",
    }),
    ("LLM Systems", {
        "nlp-fundamentals", "tokenization", "attention-mechanism", "transformers",
        "llm-fundamentals", "prompt-engineering", "embeddings", "vector-databases",
        "semantic-search", "rag", "llm-agents", "llm-evaluation", "llm-finetuning",
        "peft-lora", "rlhf", "llm-serving",
        "structured-outputs", "chunking-strategies", "reranking", "rag-evaluation",
        "agent-memory", "multi-agent-systems", "llm-observability",
        "prompt-injection-defense", "long-context", "model-merging",
        "inference-optimization", "llm-data-curation", "llm-cost-optimization",
    }),
    ("Production & MLOps", {
        "docker", "rest-apis", "cloud-fundamentals", "model-deployment",
        "experiment-tracking", "ci-cd", "mlops-fundamentals", "model-monitoring",
        "ml-system-design",
        "kubernetes", "infrastructure-as-code", "gpu-infrastructure", "model-registry",
        "feature-store", "progressive-delivery", "online-experimentation",
        "ml-cost-optimization", "data-privacy", "edge-deployment",
    }),
    ("AI Safety & Governance", {
        "ai-ethics", "bias-fairness", "model-documentation", "adversarial-robustness",
        "privacy-preserving-ml", "ai-governance",
    }),
]
CATEGORY_ORDER = [name for name, _ in CATEGORIES]
SKILL_CATEGORY: Dict[str, str] = {
    sid: name for name, members in CATEGORIES for sid in members
}


# ---------------------------------------------------------------------------
# Goal text -> target skills, deterministic keyword fallback (no LLM).
# main.py tries Ollama first and falls back to this; the tests use this directly.
# Order matters: more specific phrases first.
# ---------------------------------------------------------------------------
GOAL_PATTERNS: List[Tuple[Sequence[str], List[str]]] = [
    (("rag", "retrieval augmented", "retrieval-augmented", "chat with my docs",
      "chatbot over", "document q", "knowledge base"), ["rag"]),
    (("fine-tune", "fine tune", "finetune", "finetuning", "fine-tuning", "lora",
      "qlora", "peft", "instruction tun"), ["llm-finetuning", "peft-lora"]),
    (("rlhf", "dpo", "alignment", "preference tun", "reward model"), ["rlhf"]),
    (("agent", "tool use", "tool-calling", "autonomous"), ["llm-agents"]),
    (("semantic search", "vector search", "similarity search"), ["semantic-search"]),
    (("vector database", "vector db", "pinecone", "weaviate", "chroma", "faiss"),
     ["vector-databases"]),
    (("embedding",), ["embeddings"]),
    (("prompt engineer", "prompting"), ["prompt-engineering"]),
    (("serve", "serving", "inference server", "vllm", "latency", "throughput"),
     ["llm-serving"]),
    (("quantiz", "distill", "on-device", "edge deploy"), ["model-quantization"]),
    (("evaluate llm", "llm eval", "hallucination", "groundedness"), ["llm-evaluation"]),
    (("llm", "large language model", "gpt", "genai", "generative ai", "chatbot"),
     ["llm-fundamentals", "prompt-engineering"]),
    (("transformer", "bert", "attention"), ["transformers"]),
    (("nlp", "natural language", "text classification", "sentiment"),
     ["nlp-fundamentals", "transformers"]),
    (("computer vision", "image classif", "object detection", "segmentation",
      "cnn", "convolutional"), ["computer-vision"]),
    (("deep learning", "neural net", "pytorch", "tensorflow"),
     ["deep-learning-fundamentals", "pytorch"]),
    (("distributed training", "multi-gpu", "multi gpu"), ["distributed-training"]),
    (("mlops", "ml platform", "productioniz", "production ml", "deploy",
      "deployment", "ship models"), ["model-deployment", "mlops-fundamentals"]),
    (("monitor", "drift"), ["model-monitoring"]),
    (("experiment track", "mlflow", "weights and biases", "wandb"),
     ["experiment-tracking"]),
    (("data engineer", "etl", "elt", "pipeline", "airflow", "dbt"), ["data-pipelines"]),
    (("time series", "forecast", "demand predict"), ["time-series"]),
    (("recommend", "recsys", "personaliz"), ["recommender-systems"]),
    (("kaggle", "tabular", "xgboost", "lightgbm", "boosting"), ["gradient-boosting"]),
    (("a/b test", "ab test", "experiment design", "hypothesis test"),
     ["hypothesis-testing"]),
    (("ml engineer", "machine learning engineer", "mle"),
     ["supervised-learning", "model-evaluation", "feature-engineering",
      "model-deployment", "mlops-fundamentals"]),
    (("ml system design", "system design"), ["ml-system-design"]),
    (("data scientist", "data science"),
     ["supervised-learning", "model-evaluation", "statistics", "data-viz"]),
    (("data analyst", "analytics", "business intelligence"), ["sql", "data-viz"]),
    (("machine learning", "ml ", " ml", "predictive model"),
     ["supervised-learning", "model-evaluation"]),
]

DEFAULT_TARGETS = ["supervised-learning", "model-evaluation"]


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
class LearningGraph:
    def __init__(self, skills: List[dict], courses: List[dict]):
        self.skill_list = skills
        self.course_list = courses
        self.skills: Dict[str, dict] = {}
        self.courses: Dict[str, dict] = {}

        for s in skills:
            if s["id"] in self.skills:
                raise GraphError(f"duplicate skill id: {s['id']}")
            self.skills[s["id"]] = s
        for c in courses:
            if c["id"] in self.courses:
                raise GraphError(f"duplicate course id: {c['id']}")
            self.courses[c["id"]] = c

        # skill -> skills that directly require it (reverse edges)
        self.dependents: Dict[str, Set[str]] = {sid: set() for sid in self.skills}
        for s in skills:
            for r in s["requires"]:
                if r not in self.skills:
                    raise GraphError(f"skill '{s['id']}' requires unknown skill '{r}'")
                self.dependents[r].add(s["id"])

        # skill -> course ids that teach it
        self.courses_teaching: Dict[str, List[str]] = {sid: [] for sid in self.skills}
        for c in courses:
            for t in c["teaches"]:
                if t not in self.skills:
                    raise GraphError(f"course '{c['id']}' teaches unknown skill '{t}'")
                self.courses_teaching[t].append(c["id"])
            for r in c["requires"]:
                if r not in self.skills:
                    raise GraphError(f"course '{c['id']}' requires unknown skill '{r}'")
            if set(c["teaches"]) & set(c["requires"]):
                raise GraphError(f"course '{c['id']}' requires a skill it also teaches")

        self._depth_cache: Dict[str, int] = {}
        self._assert_dag()
        self._assert_unique_aliases()
        self._assert_categories()

        self.uncovered_skills = sorted(
            sid for sid, cs in self.courses_teaching.items() if not cs
        )

    # -- validation ---------------------------------------------------------
    def _assert_dag(self) -> None:
        """Depth-first cycle detection.  Fails loudly, naming the cycle."""
        WHITE, GREY, BLACK = 0, 1, 2
        color = {sid: WHITE for sid in self.skills}

        def visit(node: str, stack: List[str]) -> None:
            color[node] = GREY
            for req in sorted(self.skills[node]["requires"]):
                if color[req] == GREY:
                    cycle = stack[stack.index(req):] if req in stack else [req]
                    raise GraphError(
                        "skills.json is not a DAG - prerequisite cycle detected: "
                        + " -> ".join(cycle + [node, req])
                    )
                if color[req] == WHITE:
                    visit(req, stack + [node])
            color[node] = BLACK

        for sid in sorted(self.skills):
            if color[sid] == WHITE:
                visit(sid, [sid])

    def _assert_unique_aliases(self) -> None:
        """An alias claimed by two skills would make goal parsing ambiguous."""
        owner: Dict[str, str] = {}
        for sid, skill in self.skills.items():
            for alias in skill.get("aliases", []):
                if alias in owner and owner[alias] != sid:
                    raise GraphError(
                        f"alias '{alias}' is claimed by both '{owner[alias]}' and '{sid}'")
                owner[alias] = sid

    def _assert_categories(self) -> None:
        missing = sorted(set(self.skills) - set(SKILL_CATEGORY))
        if missing:
            raise GraphError(f"skills missing a milestone category: {missing}")

    # -- basic graph queries ------------------------------------------------
    def name(self, sid: str) -> str:
        return self.skills[sid]["name"] if sid in self.skills else sid

    def depth(self, sid: str) -> int:
        """Longest prerequisite chain below a skill.  Used for stable ordering."""
        if sid in self._depth_cache:
            return self._depth_cache[sid]
        reqs = self.skills[sid]["requires"]
        d = 0 if not reqs else 1 + max(self.depth(r) for r in reqs)
        self._depth_cache[sid] = d
        return d

    def prereq_closure(self, skills: Iterable[str], stop_at: Iterable[str] = ()) -> Set[str]:
        """All transitive prerequisites of `skills` (not including `skills`).

        Traversal does not expand through anything in `stop_at`: if a learner
        already knows PyTorch we assume its prerequisites are satisfied too.
        """
        stop = set(stop_at)
        seen: Set[str] = set()
        frontier = [s for s in skills if s in self.skills]
        while frontier:
            node = frontier.pop()
            for req in self.skills[node]["requires"]:
                if req in seen or req in stop:
                    continue
                seen.add(req)
                frontier.append(req)
        return seen

    def expand_possessed(self, known: Iterable[str]) -> Set[str]:
        """Downward closure: knowing a skill implies knowing its prerequisites."""
        known = {k for k in known if k in self.skills}
        return known | self.prereq_closure(known)

    def category(self, sid: str) -> str:
        return SKILL_CATEGORY.get(sid, "Foundations")

    def course_category(self, course_id: str) -> str:
        cats = [self.category(t) for t in self.courses[course_id]["teaches"]]
        best, best_count = cats[0], 0
        for cat in cats:
            count = cats.count(cat)
            # ties resolve to the more advanced category
            if count > best_count or (
                count == best_count and CATEGORY_ORDER.index(cat) > CATEGORY_ORDER.index(best)
            ):
                best, best_count = cat, count
        return best


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_GRAPH: Optional[LearningGraph] = None


def load_graph(data_dir: str = DATA_DIR) -> LearningGraph:
    """Load skills.json + courses.json and validate the graph. Raises GraphError."""
    with open(os.path.join(data_dir, "skills.json"), "r", encoding="utf-8") as fh:
        skills = json.load(fh)["skills"]
    with open(os.path.join(data_dir, "courses.json"), "r", encoding="utf-8") as fh:
        courses = json.load(fh)["courses"]
    graph = LearningGraph(skills, courses)
    if graph.uncovered_skills:
        raise GraphError(
            "every skill must be taught by at least one course; uncovered: "
            + ", ".join(graph.uncovered_skills)
        )
    return graph


def get_graph() -> LearningGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = load_graph()
    return _GRAPH


def _g(graph: Optional[LearningGraph]) -> LearningGraph:
    return graph if graph is not None else get_graph()


# ---------------------------------------------------------------------------
# 1. Goal text -> target skills (deterministic fallback for the LLM)
#
# Four layers, cheapest first, each one scored so the caller can tell how sure
# the parser is:
#
#   1. curated goal patterns   ("chat with my docs" -> rag)
#   2. exact alias / name / id ("pinecone" -> vector-databases)
#   3. fuzzy match             ("reinforcment lerning" -> rlhf, typo-tolerant)
#   4. TF-IDF cosine           ("make computers understand pictures" -> computer-vision)
#
# Crucially it may return NOTHING. An earlier version defaulted to a generic
# ML-engineer goal whenever it understood nothing, which turned "I want to be a
# web developer", a typo and even an empty string into a confident 12-course
# plan for a goal the learner never expressed. Refusing to guess and asking
# instead is the honest behaviour, and it is what ACCEPT/SUGGEST encode below.
# ---------------------------------------------------------------------------
ACCEPT_SCORE = 0.62   # at or above this, plan without asking
SUGGEST_SCORE = 0.22  # at or above this, offer as a clarification candidate

_STOPWORDS = {
    "a", "an", "the", "i", "im", "me", "my", "we", "you", "your", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "but", "so", "as", "is", "am", "are", "be",
    "been", "want", "wanna", "wants", "would", "like", "need", "get", "getting", "got",
    "become", "becoming", "learn", "learning", "learnt", "study", "studying", "build",
    "building", "make", "making", "do", "doing", "work", "working", "use", "using",
    "know", "knowing", "understand", "understanding", "able", "good", "better", "best",
    "master", "mastering", "start", "starting", "begin", "how", "what", "can", "should",
    "into", "from", "that", "this", "it", "its", "some", "more", "very", "really",
    "career", "job", "role", "help", "please", "want_to", "own", "new", "up",
}


def _norm_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#/.\- ]+", " ", (text or "").lower())).strip()


def _stem(token: str) -> str:
    """Crude suffix stripping - enough to tie 'embeddings' to 'embedding'."""
    for suf in ("ing", "ers", "er", "es"):
        if len(token) > 4 and token.endswith(suf):
            return token[: -len(suf)]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]          # llms -> llm, apis -> api, cnns -> cnn
    return token


def _stem_phrase(phrase: str) -> str:
    return " ".join(_stem(t) for t in _norm_text(phrase).split())


def _tokens(text: str, keep_stopwords: bool = False) -> List[str]:
    out = []
    for tok in _norm_text(text).split():
        if len(tok) < 2:
            continue
        if not keep_stopwords and tok in _STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


def _vocab(g: LearningGraph) -> dict:
    """Phrase index + TF-IDF model over the skill catalogue, built once."""
    cached = getattr(g, "_vocab_cache", None)
    if cached is not None:
        return cached

    import math
    from collections import Counter

    phrases: Dict[str, str] = {}
    docs: Dict[str, List[str]] = {}
    for sid, skill in g.skills.items():
        surface = [skill["name"], sid.replace("-", " ")] + list(skill.get("aliases", []))
        for p in surface:
            key = _norm_text(p)
            if key and key not in phrases:
                phrases[key] = sid
        docs[sid] = _tokens(" ".join(
            [skill["name"], skill.get("description", ""), g.category(sid)]
            + list(skill.get("aliases", []))))

    n_docs = len(docs) or 1
    df: Counter = Counter()
    for toks in docs.values():
        df.update(set(toks))
    idf = {t: math.log(1 + n_docs / (1 + c)) for t, c in df.items()}

    vectors: Dict[str, Dict[str, float]] = {}
    for sid, toks in docs.items():
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors[sid] = {t: v / norm for t, v in vec.items()}

    # Fuzzy matching compares a word against every phrase key, which grows with
    # the catalogue. Two cheap indexes keep that bounded: candidates must be
    # within a couple of characters in length (difflib cannot reach 0.82
    # similarity otherwise) and must share at least one character.
    by_length: Dict[int, List[str]] = {}
    charsets: Dict[str, frozenset] = {}
    # A phrase can only match if its first word appears in the goal, so index by
    # that: exact matching then tests a handful of candidates instead of all of
    # them, which is what keeps parsing flat as the catalogue grows.
    by_first: Dict[str, List[str]] = {}
    for key in phrases:
        by_length.setdefault(len(key), []).append(key)
        charsets[key] = frozenset(key)
        by_first.setdefault(_stem(key.split()[0]), []).append(key)

    cache = {"phrases": phrases, "idf": idf, "vectors": vectors,
             "known_tokens": set(df), "by_length": by_length, "charsets": charsets,
             "by_first": by_first,
             "math": math, "Counter": Counter}
    setattr(g, "_vocab_cache", cache)
    return cache


def _add(hits: Dict[str, dict], sid: str, score: float, method: str, evidence: str) -> None:
    prev = hits.get(sid)
    if prev is None or score > prev["score"]:
        hits[sid] = {"skill_id": sid, "score": round(score, 3),
                     "method": method, "evidence": evidence}


def resolve_goal(goal: str, graph: Optional[LearningGraph] = None,
                 max_candidates: int = 4) -> dict:
    """Map free text onto skill ids, reporting how confident the mapping is.

    Returns targets only when the best match clears ACCEPT_SCORE. Otherwise the
    caller gets candidates to ask about, or out_of_domain when nothing in the
    catalogue is even close.
    """
    g = _g(graph)
    voc = _vocab(g)
    math_, Counter = voc["math"], voc["Counter"]
    text = _norm_text(goal)
    padded = " " + text + " "
    # a stemmed copy of the goal, so "chatbots over our docs" still matches the
    # phrase "chatbot over" and "embeddings" matches the alias "embedding"
    padded_stem = " " + " ".join(_stem(t) for t in text.split()) + " "
    hits: Dict[str, dict] = {}

    def _matches(phrase: str) -> bool:
        if not phrase:
            return False
        if " " in phrase:
            return phrase in padded or _stem_phrase(phrase) in padded_stem
        if re.search(r"\b" + re.escape(phrase) + r"s?\b", padded):
            return True
        return re.search(r"\b" + re.escape(_stem(phrase)) + r"\b", padded_stem) is not None

    # --- layer 1: curated goal patterns -----------------------------------
    for patterns, targets in GOAL_PATTERNS:
        for p in patterns:
            if _matches(p):
                for t in targets:
                    if t in g.skills:
                        _add(hits, t, 1.0, "pattern", f'goal phrase "{p.strip()}"')
                break

    # --- layer 2: exact alias / name / id ---------------------------------
    # Longest match wins, compared on stems so that "vector databases" credits
    # Vector Databases only - not SQL via "database", nor Linear Algebra via
    # "vectors". Whichever phrase explains the most words owns that span.
    goal_stems = {_stem(t) for t in text.split()}
    candidate_phrases = {k for stem in goal_stems for k in voc["by_first"].get(stem, ())}
    matched = [(p, _stem_phrase(p), voc["phrases"][p]) for p in candidate_phrases if _matches(p)]
    matched.sort(key=lambda m: m[0])
    covered: List[str] = []
    for phrase, stem, sid in matched:
        if any(stem != other and stem in other for _, other, _ in matched):
            continue
        covered.append(stem)
        _add(hits, sid, 0.95 if " " in phrase else 0.85, "alias", f'"{phrase}"')

    # --- layer 3: fuzzy match, for typos ----------------------------------
    import difflib
    goal_tokens = _tokens(text, keep_stopwords=False)
    raw_tokens = [t for t in _norm_text(goal).split() if t not in _STOPWORDS and len(t) > 3]
    grams = raw_tokens + [" ".join(raw_tokens[i:i + 2]) for i in range(len(raw_tokens) - 1)]
    by_length, charsets = voc["by_length"], voc["charsets"]
    for gram in grams:
        # do not re-explain words an exact phrase already accounted for:
        # "databases" inside "vector databases" is not evidence for SQL
        if any(_stem_phrase(gram) in span for span in covered):
            continue
        # only phrases of a similar length can clear the 0.82 cutoff
        span = max(1, round(len(gram) * 0.22))
        gram_chars = frozenset(gram)
        candidates = [k for n in range(len(gram) - span, len(gram) + span + 1)
                      for k in by_length.get(n, ())
                      if charsets[k] & gram_chars]
        if not candidates:
            continue
        near = difflib.get_close_matches(gram, candidates, n=1, cutoff=0.82)
        if near:
            ratio = difflib.SequenceMatcher(None, gram, near[0]).ratio()
            sid = voc["phrases"][near[0]]
            _add(hits, sid, ratio * 0.85, "fuzzy", f'"{gram}" looks like "{near[0]}"')

    # --- layer 4: TF-IDF cosine, for paraphrase ---------------------------
    if goal_tokens:
        tf = Counter(goal_tokens)
        qv = {t: (1 + math_.log(c)) * voc["idf"].get(t, 0.0) for t, c in tf.items()}
        qnorm = math_.sqrt(sum(v * v for v in qv.values())) or 1.0
        qv = {t: v / qnorm for t, v in qv.items()}
        for sid, vec in voc["vectors"].items():
            cos = sum(w * vec.get(t, 0.0) for t, w in qv.items())
            if cos > 0.12:
                shared = sorted((t for t in qv if t in vec),
                                key=lambda t: -qv[t] * vec[t])[:3]
                _add(hits, sid, min(cos * 1.4, 0.6), "semantic",
                     "wording overlaps " + ", ".join(shared))

    ranked = sorted(hits.values(), key=lambda h: (-h["score"], h["skill_id"]))
    best = ranked[0]["score"] if ranked else 0.0

    # terms the catalogue has never heard of - the vocabulary gap log
    unknown = [t for t in goal_tokens if t not in voc["known_tokens"]]

    if best >= ACCEPT_SCORE:
        keep = [h for h in ranked if h["score"] >= max(ACCEPT_SCORE, best * 0.75)]
        targets = normalize_targets([h["skill_id"] for h in keep], graph=g)
        return {"goal": goal, "targets": targets, "confidence": "high" if best >= 0.9 else "medium",
                "matches": keep[:max_candidates], "candidates": [],
                "unknown_terms": unknown, "out_of_domain": False,
                "method": keep[0]["method"] if keep else "none"}

    candidates = [h for h in ranked if h["score"] >= SUGGEST_SCORE][:max_candidates]
    return {"goal": goal, "targets": [], "confidence": "low" if candidates else "none",
            "matches": [], "candidates": candidates, "unknown_terms": unknown,
            "out_of_domain": not candidates,
            "method": candidates[0]["method"] if candidates else "none"}


def keyword_match_targets(goal: str, graph: Optional[LearningGraph] = None) -> List[str]:
    """Deterministic goal parser. Returns [] when it does not understand."""
    return resolve_goal(goal, graph=graph)["targets"]


def normalize_targets(targets: Iterable[str], graph: Optional[LearningGraph] = None,
                      limit: int = 6) -> List[str]:
    """Drop unknown ids, drop targets implied by a deeper target, cap the count."""
    g = _g(graph)
    valid = {t for t in targets if t in g.skills}
    if not valid:
        return []   # never silently substitute a goal the learner did not ask for
    # a target that is a prerequisite of another target is redundant
    implied = g.prereq_closure(valid)
    minimal = valid - implied
    ordered = sorted(minimal, key=lambda s: (-g.depth(s), s))
    return ordered[:limit]


# ---------------------------------------------------------------------------
# 2 + 3. Gap analysis: transitive closure minus what the learner has
# ---------------------------------------------------------------------------
def gap_analysis(target_skills: Iterable[str], possessed_skills: Iterable[str],
                 graph: Optional[LearningGraph] = None) -> Set[str]:
    """Skills the learner still needs: closure(targets) - possessed."""
    g = _g(graph)
    targets = [t for t in target_skills if t in g.skills]
    possessed = g.expand_possessed(possessed_skills)
    needed = set(targets) | g.prereq_closure(targets, stop_at=possessed)
    return needed - possessed


# ---------------------------------------------------------------------------
# 4. Deterministic topological sort of the gap subgraph
# ---------------------------------------------------------------------------
def topo_order(skill_subgraph: Iterable[str], tie_break: str = "phase",
               graph: Optional[LearningGraph] = None) -> List[str]:
    """Kahn's algorithm over the induced subgraph, with a deterministic tie-break.

    Two orderings are available; both are stable across runs (never arbitrary):
      "depth"  -> (graph depth, skill id)
      "phase"  -> (category, graph depth, skill id)   [default]

    "phase" adds one term in front so that when several skills are simultaneously
    unblocked, foundations come before data work, which comes before core ML, and
    so on.  Without it a valid order can interleave Docker between NumPy and
    Pandas: still correct, but it shreds the milestone grouping in the UI.
    """
    g = _g(graph)
    nodes = {s for s in skill_subgraph if s in g.skills}

    if tie_break == "depth":
        def key(s: str):
            return (g.depth(s), s)
    else:
        def key(s: str):
            return (CATEGORY_ORDER.index(g.category(s)), g.depth(s), s)

    indeg = {n: len({r for r in g.skills[n]["requires"] if r in nodes}) for n in nodes}
    ready = sorted((n for n in nodes if indeg[n] == 0), key=key)
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        newly = []
        for dep in g.dependents[node]:
            if dep in indeg:
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    newly.append(dep)
        if newly:
            ready.extend(newly)
            ready.sort(key=key)
    if len(order) != len(nodes):
        raise GraphError("cycle detected while ordering the gap subgraph")
    return order


# ---------------------------------------------------------------------------
# 5. Course selection: greedy set-cover walking the topological order
# ---------------------------------------------------------------------------
def _skill_target_level(g: LearningGraph, sid: str) -> int:
    d = g.depth(sid)
    return 0 if d <= 3 else (1 if d <= 10 else 2)


def _interest_bonus(course: dict, interests: Sequence[str]) -> float:
    if not interests:
        return 0.0
    haystack = " ".join(
        [course["title"], course["description"], " ".join(course.get("tags", []))]
    ).lower()
    matched = sum(1 for i in interests if i and i.lower().strip() in haystack)
    return min(matched, 2) * 2.0


# Selection strategies. The graph fixes WHAT you must learn and in what order;
# these change only WHICH course covers each skill, by re-weighting the same
# scorer. Every strategy therefore produces an equally valid path - which is
# the point of showing them side by side.
STRATEGIES: Dict[str, dict] = {
    "balanced": {
        "label": "Balanced",
        "blurb": "The default trade-off between length, level fit and interests.",
        "hours": -0.05, "free": 0.0, "projects": 0.0, "level_fit": -1.5, "breadth": -1.0,
    },
    "fastest": {
        "label": "Shortest",
        "blurb": "Fewest hours to the same skills, even if the courses are terse.",
        # penalising raw hours backfires: it picks many small courses and the
        # total goes UP. What actually shortens a path is hours per skill
        # covered, so bundles that teach three things at once are rewarded.
        "hours": -0.02, "free": 0.0, "projects": 0.0, "level_fit": -1.0,
        "breadth": 0.0, "efficiency": -0.9,
    },
    "free_first": {
        "label": "Free first",
        "blurb": "Prefers free material wherever it covers the same skill.",
        "hours": -0.05, "free": 7.0, "projects": 0.0, "level_fit": -1.5, "breadth": -1.0,
    },
    "hands_on": {
        "label": "Project-led",
        "blurb": "Prefers project and practice-heavy courses over lecture courses.",
        "hours": -0.03, "free": 0.0, "projects": 6.0, "level_fit": -1.0, "breadth": -0.5,
    },
    "thorough": {
        "label": "Thorough",
        "blurb": "Tolerates longer, deeper courses that cover more ground at once.",
        "hours": 0.03, "free": 0.0, "projects": 0.0, "level_fit": -0.8, "breadth": 0.5,
    },
}
FREE_TAGS = {"free"}
PROJECT_TAGS = {"projects", "hands-on", "practical"}


def score_course(course: dict, skill: str, needed: Set[str], uncovered: Set[str],
                 satisfied: Set[str], level: str, interests: Sequence[str],
                 graph: Optional[LearningGraph] = None,
                 strategy: str = "balanced", refresher: bool = False) -> float:
    """Tag overlap + simple heuristics. Pure Python, no embeddings anywhere."""
    g = _g(graph)
    w = STRATEGIES.get(strategy, STRATEGIES["balanced"])
    learner_level = LEVELS.get(level, 0)
    teaches = set(course["teaches"])

    new_cover = teaches & uncovered
    redundant = teaches - needed - satisfied

    unmet = set(course["requires"]) - satisfied
    unmet_soft = unmet & needed          # scheduled later in the path: mild
    unmet_hard = unmet - needed          # extra baggage the plan never covers

    course_level = LEVELS.get(course["level"], 0)
    fit = _skill_target_level(g, skill)

    tags = {t.lower() for t in course.get("tags", [])}

    score = 0.0
    score += 5.0 * len(new_cover)
    score += w["breadth"] * len(redundant)
    score -= 2.0 * len(unmet_soft)
    score -= 6.0 * len(unmet_hard)          # prerequisites the plan never covers
    score += w["level_fit"] * abs(course_level - fit)
    score += 1.0 if course_level == learner_level else 0.0
    score += _interest_bonus(course, interests)
    score += w["hours"] * course["hours"]
    if w.get("efficiency"):   # hours per newly covered skill, not raw hours
        score += w["efficiency"] * (course["hours"] / max(1, len(new_cover)))
    if refresher:
        # the learner has met this material before: they need the short course
        # that reminds them, not the comprehensive one that teaches it cold
        score -= 0.22 * course["hours"]
        score -= 1.0 * LEVELS.get(course["level"], 0)
    score += w["free"] if tags & FREE_TAGS else 0.0
    score += w["projects"] if tags & PROJECT_TAGS else 0.0
    return score


def select_courses(ordered_skills: Sequence[str], level: str = "beginner",
                   interests: Sequence[str] = (), possessed: Iterable[str] = (),
                   exclude: Iterable[str] = (), strategy: str = "balanced",
                   refresh: Iterable[str] = (),
                   graph: Optional[LearningGraph] = None) -> List[dict]:
    """Cover every skill in `ordered_skills` with as few sensible courses as possible.

    Walks the topological order, and for each still-uncovered skill picks the
    highest scoring course that teaches it.  A course that also teaches later
    skills covers them at the same time, which is what keeps the path short.
    """
    g = _g(graph)
    needed = set(ordered_skills)
    uncovered = set(ordered_skills)
    banned = set(exclude)
    refresh_set = set(refresh)
    satisfied = g.expand_possessed(possessed)
    chosen: List[dict] = []

    for skill in ordered_skills:
        if skill not in uncovered:
            continue
        candidates = [g.courses[cid] for cid in g.courses_teaching[skill]
                      if cid not in banned]
        if not candidates:  # every option banned: fall back to the full list
            candidates = [g.courses[cid] for cid in g.courses_teaching[skill]]
        # A course whose prerequisites the plan never teaches would break the
        # ordering guarantee, so it is filtered out rather than merely penalised.
        # Only if every option has that problem is one allowed through - and
        # build_path then pulls the missing prerequisite into the plan.
        clean = [c for c in candidates
                 if not (set(c["requires"]) - satisfied - needed)]
        pool = clean or candidates
        scored = sorted(
            pool,
            key=lambda c: (-score_course(c, skill, needed, uncovered, satisfied,
                                         level, interests, graph=g, strategy=strategy,
                                         refresher=skill in refresh_set), c["id"]),
        )
        best = scored[0]
        covers = sorted(set(best["teaches"]) & uncovered,
                        key=lambda s: ordered_skills.index(s))
        uncovered -= set(best["teaches"])
        satisfied |= set(best["teaches"])
        chosen.append({
            "course": best,
            "covers": covers,
            "trigger_skill": skill,
            # prerequisites this course needs that nothing in the plan supplies
            "unmet": sorted(set(best["requires"]) - satisfied - needed),
        })
    return chosen


# ---------------------------------------------------------------------------
# 6. Milestones: contiguous chunks of the ordered path, named by category
# ---------------------------------------------------------------------------
def _chunk_into_phases(g: LearningGraph, course_ids: List[str],
                       min_phases: int = 3, max_phases: int = 5) -> List[List[int]]:
    if not course_ids:
        return []
    cats = [g.course_category(cid) for cid in course_ids]
    chunks: List[List[int]] = []
    current = [0]
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            chunks.append(current)
            current = [i]
        else:
            current.append(i)
    chunks.append(current)

    # merge tiny chunks into the smaller neighbour, then cap the phase count
    def merge_smallest() -> None:
        idx = min(range(len(chunks)), key=lambda i: (len(chunks[i]), i))
        if idx == 0:
            target = 1
        elif idx == len(chunks) - 1:
            target = idx - 1
        else:
            target = idx - 1 if len(chunks[idx - 1]) <= len(chunks[idx + 1]) else idx + 1
        lo, hi = min(idx, target), max(idx, target)
        chunks[lo] = chunks[lo] + chunks[hi]
        del chunks[hi]

    while len(chunks) > 1 and (len(chunks) > max_phases or
                               any(len(c) < 2 for c in chunks) and len(chunks) > min_phases):
        merge_smallest()

    # if we ended up with too few phases and have material, split the biggest
    while len(chunks) < min_phases and any(len(c) >= 4 for c in chunks):
        idx = max(range(len(chunks)), key=lambda i: (len(chunks[i]), -i))
        half = len(chunks[idx]) // 2
        big = chunks[idx]
        chunks[idx:idx + 1] = [big[:half], big[half:]]
    return chunks


def _phase_name(g: LearningGraph, course_ids: List[str], used: Set[str]) -> str:
    cats = [g.course_category(cid) for cid in course_ids]
    best = max(sorted(set(cats), key=CATEGORY_ORDER.index),
               key=lambda c: (cats.count(c), CATEGORY_ORDER.index(c)))
    name = best
    if name in used:
        name = f"{best} II"
        n = 2
        while name in used:
            n += 1
            name = f"{best} {'I' * n}"
    return name


# ---------------------------------------------------------------------------
# build_path: the whole pipeline in one call
# ---------------------------------------------------------------------------
def build_path(target_skills: Iterable[str], possessed_skills: Iterable[str] = (),
               level: str = "beginner", interests: Sequence[str] = (),
               exclude: Iterable[str] = (), strategy: str = "balanced",
               refresh: Iterable[str] = (),
               graph: Optional[LearningGraph] = None) -> dict:
    """targets + profile -> ordered, milestone-grouped, fully explained path."""
    g = _g(graph)
    targets = normalize_targets(target_skills, graph=g)
    possessed = g.expand_possessed(possessed_skills)

    needed = gap_analysis(targets, possessed, graph=g)
    ordered = topo_order(needed, graph=g)
    refresh_set = {s for s in refresh if s in g.skills}
    # Skill prerequisites come from the graph, but courses have prerequisites of
    # their own. If the only course teaching a skill assumes something the plan
    # does not cover, that assumption becomes part of the gap and the path is
    # rebuilt - repeated until the plan teaches everything its own courses need.
    for _ in range(5):
        selection = select_courses(ordered, level=level, interests=interests,
                                   possessed=possessed, exclude=exclude,
                                   strategy=strategy, refresh=refresh_set, graph=g)
        extra: Set[str] = set()
        for item in selection:
            extra.update(s for s in item.get("unmet", ()) if s in g.skills)
        if not extra:
            break
        needed |= g.prereq_closure(extra, stop_at=possessed) | extra
        needed -= possessed
        ordered = topo_order(needed, graph=g)

    course_ids = [item["course"]["id"] for item in selection]
    position = {cid: i for i, cid in enumerate(course_ids)}

    # prerequisite edges between the cards themselves (used to draw links)
    provider_of: Dict[str, str] = {}
    for item in selection:
        for t in item["course"]["teaches"]:
            provider_of.setdefault(t, item["course"]["id"])

    entries: List[dict] = []
    for i, item in enumerate(selection):
        course = item["course"]
        depends_on = sorted(
            {provider_of[r] for r in course["requires"]
             if r in provider_of and position.get(provider_of[r], 10 ** 6) < i},
            key=lambda cid: position[cid],
        )
        entries.append({
            # flagged when the learner said they had "heard of" the skill: this
            # is the short reminder course, not the comprehensive one
            "refresher": item["trigger_skill"] in refresh_set,
            "order": i,
            "course_id": course["id"],
            "title": course["title"],
            "provider": course["provider"],
            "level": course["level"],
            "hours": course["hours"],
            "url": course["url"],
            "description": course["description"],
            "tags": course.get("tags", []),
            "covers": item["covers"],
            "covers_named": [g.name(s) for s in item["covers"]],
            "teaches": course["teaches"],
            "requires": course["requires"],
            "depends_on": depends_on,
            "prereqs_already_known": sorted(s for s in course["requires"] if s in possessed),
            "milestone": None,
        })

    # forward links: which later cards this card unblocks
    for entry in entries:
        unlocks = []
        for later in entries[entry["order"] + 1:]:
            if set(entry["teaches"]) & set(later["requires"]):
                unlocks.append(later["course_id"])
        entry["unlocks"] = unlocks

    milestones: List[dict] = []
    used_names: Set[str] = set()
    for chunk in _chunk_into_phases(g, course_ids):
        chunk_ids = [course_ids[i] for i in chunk]
        name = _phase_name(g, chunk_ids, used_names)
        used_names.add(name)
        chunk_entries = [entries[i] for i in chunk]
        for entry in chunk_entries:
            entry["milestone"] = name
        milestones.append({
            "name": name,
            "index": len(milestones),
            "course_ids": chunk_ids,
            "hours": sum(e["hours"] for e in chunk_entries),
            "skills": sorted({s for e in chunk_entries for s in e["covers"]}),
            "skill_names": [g.name(s) for s in
                            sorted({s for e in chunk_entries for s in e["covers"]})],
        })

    return {
        "targets": targets,
        "targets_named": [g.name(t) for t in targets],
        "possessed": sorted(possessed),
        "possessed_named": [g.name(s) for s in sorted(possessed)],
        "gap_skills": ordered,
        "gap_skills_named": [g.name(s) for s in ordered],
        "courses": entries,
        "milestones": milestones,
        "total_courses": len(entries),
        "total_hours": sum(e["hours"] for e in entries),
        "skills_needed": len(ordered),
        "skills_covered": sorted({s for e in entries for s in e["covers"]}),
        "level": level,
        "interests": list(interests),
    }


# ---------------------------------------------------------------------------
# 7. Grounded explanation, read straight off the graph
# ---------------------------------------------------------------------------
def _chain_to_target(g: LearningGraph, skill: str, targets: Set[str],
                     scope: Set[str]) -> List[str]:
    """Shortest upward chain skill -> ... -> some target, following reverse edges."""
    if skill in targets:
        return [skill]
    seen = {skill}
    queue: List[List[str]] = [[skill]]
    while queue:
        path = queue.pop(0)
        for nxt in sorted(g.dependents[path[-1]]):
            if nxt in seen or (nxt not in scope and nxt not in targets):
                continue
            seen.add(nxt)
            new_path = path + [nxt]
            if nxt in targets:
                return new_path
            queue.append(new_path)
    return [skill]


def explain(course_id: str, path_state: dict,
            graph: Optional[LearningGraph] = None) -> dict:
    """Structured trace: why this course, what made it necessary, what it unblocks.

    Every sentence here is derived from prerequisite edges in the graph. No
    language model produced any of this reasoning.
    """
    g = _g(graph)
    if course_id not in g.courses:
        raise KeyError(course_id)
    course = g.courses[course_id]
    entries = {e["course_id"]: e for e in path_state.get("courses", [])}
    entry = entries.get(course_id)
    targets = set(path_state.get("targets", []))
    scope = set(path_state.get("gap_skills", [])) | targets
    possessed = set(path_state.get("possessed", []))
    covers = entry["covers"] if entry else sorted(set(course["teaches"]) & scope)

    teaches_trace = []
    sentences: List[str] = []
    for skill in covers:
        chain = _chain_to_target(g, skill, targets, scope)
        edges = [
            {"from": chain[i], "to": chain[i + 1],
             "from_name": g.name(chain[i]), "to_name": g.name(chain[i + 1])}
            for i in range(len(chain) - 1)
        ]
        if len(chain) == 1:
            sentence = (f"{g.name(skill)} is one of your goal skills, and this "
                        f"course teaches it directly.")
        else:
            # chain is [skill, ..., target]; read it back down from the goal
            hops = ", which requires ".join(g.name(s) for s in reversed(chain[:-1]))
            sentence = (f"Your goal {g.name(chain[-1])} requires {hops} - and you do not "
                        f"have {g.name(skill)} yet. This course teaches it.")
        teaches_trace.append({
            "skill": skill,
            "skill_name": g.name(skill),
            "description": g.skills[skill]["description"],
            "chain": chain,
            "chain_named": [g.name(s) for s in chain],
            "edges": edges,
            "serves_target": chain[-1],
            "serves_target_name": g.name(chain[-1]),
            "sentence": sentence,
        })
        sentences.append(sentence)

    # prerequisites of this course and where they came from
    prereq_sources = []
    for req in course["requires"]:
        source = None
        if entry:
            for cid in entry["depends_on"]:
                if req in g.courses[cid]["teaches"]:
                    source = {"type": "course", "course_id": cid,
                              "title": g.courses[cid]["title"]}
                    break
        if source is None and req in possessed:
            source = {"type": "already_known"}
        prereq_sources.append({
            "skill": req, "skill_name": g.name(req),
            "source": source or {"type": "unmet"},
        })
    met_earlier = [p for p in prereq_sources if p["source"]["type"] == "course"]
    known = [p for p in prereq_sources if p["source"]["type"] == "already_known"]
    if known:
        sentences.append(
            "You already have " + ", ".join(p["skill_name"] for p in known)
            + ", which this course assumes.")
    if met_earlier:
        sentences.append(
            "It is placed here because "
            + ", ".join(f"{p['skill_name']} comes from {p['source']['title']}"
                        for p in met_earlier) + " earlier in your path.")

    unlocks = []
    if entry:
        for cid in entry["unlocks"]:
            later = g.courses[cid]
            via = sorted(set(course["teaches"]) & set(later["requires"]))
            unlocks.append({"course_id": cid, "title": later["title"],
                            "via": via, "via_named": [g.name(s) for s in via]})
        if unlocks:
            sentences.append(
                f"Finishing it unblocks {len(unlocks)} later course"
                + ("s" if len(unlocks) > 1 else "") + ": "
                + ", ".join(u["title"] for u in unlocks[:3])
                + (" and others." if len(unlocks) > 3 else "."))

    return {
        "course_id": course_id,
        "title": course["title"],
        "provider": course["provider"],
        "hours": course["hours"],
        "level": course["level"],
        "url": course["url"],
        "milestone": entry["milestone"] if entry else None,
        "position": {"index": (entry["order"] + 1) if entry else None,
                     "total": path_state.get("total_courses")},
        "teaches": teaches_trace,
        "prerequisites": prereq_sources,
        "unlocks": unlocks,
        "sentences": sentences,
        "summary": " ".join(sentences),
        "grounded_in": "prerequisite graph traversal (no LLM reasoning)",
    }


# ---------------------------------------------------------------------------
# 8. Replanning from the learner's current state
# ---------------------------------------------------------------------------
def replan(learner: dict, graph: Optional[LearningGraph] = None) -> dict:
    """Recompute the remaining path from whatever the learner possesses now."""
    g = _g(graph)
    path = build_path(
        learner.get("target_skills", []),
        learner.get("possessed", []),
        level=learner.get("experience_level", "beginner"),
        interests=learner.get("interests", []),
        exclude=learner.get("rejected_courses", []),
        strategy=learner.get("strategy", "balanced"),
        refresh=learner.get("refresh_skills", []),
        graph=g,
    )
    completed = [cid for cid in learner.get("completed_courses", []) if cid in g.courses]
    all_targets = normalize_targets(learner.get("target_skills", []), graph=g)
    original_gap = gap_analysis(all_targets, learner.get("known_skills", []), graph=g)
    remaining_gap = set(path["gap_skills"])
    done_skills = original_gap - remaining_gap

    path["progress"] = {
        "skills_total": len(original_gap),
        "skills_done": len(done_skills),
        "skills_remaining": len(remaining_gap),
        "percent": round(100 * len(done_skills) / len(original_gap)) if original_gap else 100,
        "courses_completed": len(completed),
        "hours_remaining": path["total_hours"],
        "completed_courses": [
            {"course_id": cid, "title": g.courses[cid]["title"],
             "provider": g.courses[cid]["provider"], "hours": g.courses[cid]["hours"]}
            for cid in completed
        ],
        "hours_done": sum(g.courses[cid]["hours"] for cid in completed),
        "done_skills": sorted(done_skills),
        "done_skills_named": [g.name(s) for s in sorted(done_skills)],
    }
    path["next_action"] = path["courses"][0] if path["courses"] else None
    path["complete"] = not path["courses"]
    return path


# ---------------------------------------------------------------------------
# Small helpers used by the API layer
# ---------------------------------------------------------------------------
def alternative_courses(course_id: str, skills: Iterable[str], direction: str = "easier",
                        graph: Optional[LearningGraph] = None) -> List[dict]:
    """Other courses covering the same skills, sorted easier-first or harder-first."""
    g = _g(graph)
    skills = set(skills) or set(g.courses[course_id]["teaches"])
    cands = {cid for s in skills if s in g.courses_teaching for cid in g.courses_teaching[s]}
    cands.discard(course_id)
    base = LEVELS.get(g.courses[course_id]["level"], 0)
    def key(cid: str):
        lv = LEVELS.get(g.courses[cid]["level"], 0)
        delta = (lv - base) if direction == "harder" else (base - lv)
        overlap = len(set(g.courses[cid]["teaches"]) & skills)
        return (-overlap, -delta, g.courses[cid]["hours"], cid)
    return [g.courses[cid] for cid in sorted(cands, key=key)]


def search_skills(query: str, graph: Optional[LearningGraph] = None) -> List[dict]:
    g = _g(graph)
    q = (query or "").lower().strip()
    out = []
    for sid, s in g.skills.items():
        if not q or q in sid or q in s["name"].lower() or q in s["description"].lower():
            out.append({"id": sid, "name": s["name"], "description": s["description"],
                        "depth": g.depth(sid), "category": g.category(sid)})
    return sorted(out, key=lambda s: (s["depth"], s["name"]))


def graph_stats(graph: Optional[LearningGraph] = None) -> dict:
    g = _g(graph)
    return {
        "skills": len(g.skills),
        "courses": len(g.courses),
        "edges": sum(len(s["requires"]) for s in g.skill_list),
        "max_depth": max(g.depth(s) for s in g.skills),
        "categories": CATEGORY_ORDER,
    }


# ---------------------------------------------------------------------------
# Graph analytics
#
# Everything below asks a question about the *structure* of a plan rather than
# its contents, which is only answerable because the curriculum is a graph:
# what breaks if you skip this, what does finishing it open up, where in the
# order is it actually allowed to sit, and how far are you from being ready for
# something that is not even in your plan.
# ---------------------------------------------------------------------------
def skip_impact(course_id: str, path_state: dict,
                graph: Optional[LearningGraph] = None) -> dict:
    """What breaks if this course is skipped: the inverse of explain()."""
    g = _g(graph)
    entry = next((c for c in path_state["courses"] if c["course_id"] == course_id), None)
    if entry is None:
        raise KeyError(f"{course_id} is not in this path")

    gap = set(path_state["gap_skills"])
    targets = set(path_state["targets"])
    covered_here = set(entry["covers"])

    # skills that would go unlearned: what it teaches, plus everything in the
    # gap that transitively depends on those skills
    lost = set(covered_here)
    frontier = list(covered_here)
    while frontier:
        sid = frontier.pop()
        for dep in g.dependents.get(sid, ()):  # reverse edges
            if dep in gap and dep not in lost:
                lost.add(dep)
                frontier.append(dep)

    # later courses that need any of it
    blocked = []
    for other in path_state["courses"]:
        if other["course_id"] == course_id or other["order"] <= entry["order"]:
            continue
        via = sorted(set(other["requires"]) & lost)
        if via:
            blocked.append({"course_id": other["course_id"], "title": other["title"],
                            "order": other["order"] + 1, "hours": other["hours"],
                            "via": via, "via_named": [g.name(v) for v in via]})

    at_risk = sorted(targets & lost)
    alternatives = [
        {"id": cid, "title": g.courses[cid]["title"], "provider": g.courses[cid]["provider"],
         "hours": g.courses[cid]["hours"], "level": g.courses[cid]["level"]}
        for sid in sorted(covered_here) for cid in g.courses_teaching.get(sid, [])
        if cid != course_id
    ]
    seen, unique_alts = set(), []
    for alt in alternatives:
        if alt["id"] not in seen:
            seen.add(alt["id"])
            unique_alts.append(alt)

    if at_risk:
        verdict = "goal-critical"
        sentence = (f"Skipping {entry['title']} costs you "
                    f"{', '.join(g.name(s) for s in sorted(lost))} - including "
                    f"{', '.join(g.name(t) for t in at_risk)}, which is your goal. "
                    "There is no route to your goal that goes around it.")
    elif blocked:
        verdict = "blocking"
        hours = sum(b["hours"] for b in blocked)
        sentence = (f"Skipping {entry['title']} strands {len(blocked)} later course"
                    f"{'s' if len(blocked) > 1 else ''} ({hours}h) that need "
                    f"{', '.join(g.name(s) for s in sorted(lost - covered_here)) or 'what it teaches'}.")
    else:
        verdict = "safe"
        sentence = (f"{entry['title']} is a leaf in your plan: nothing later depends on "
                    f"it, so skipping it only costs you "
                    f"{', '.join(entry['covers_named'])} itself.")

    return {
        "course_id": course_id, "title": entry["title"], "verdict": verdict,
        "sentence": sentence,
        "skills_lost": sorted(lost),
        "skills_lost_named": [g.name(s) for s in sorted(lost)],
        "downstream_only": sorted(lost - covered_here),
        "blocked_courses": blocked,
        "hours_blocked": sum(b["hours"] for b in blocked) + entry["hours"],
        "targets_at_risk": at_risk,
        "targets_at_risk_named": [g.name(t) for t in at_risk],
        "alternatives": unique_alts,
        "grounded_in": "prerequisite edges, walked forwards from what this course teaches",
    }


def leverage_ranking(path_state: dict, graph: Optional[LearningGraph] = None) -> List[dict]:
    """Rank courses by how much of the rest of the plan they unblock.

    A dull NumPy course can outrank an exciting RAG one because sixteen things
    sit downstream of it. That argument is much easier to make with a number.
    """
    g = _g(graph)
    gap = set(path_state["gap_skills"])
    by_id = {c["course_id"]: c for c in path_state["courses"]}
    out = []
    for entry in path_state["courses"]:
        downstream = set()
        frontier = list(entry["covers"])
        while frontier:
            sid = frontier.pop()
            for dep in g.dependents.get(sid, ()):
                if dep in gap and dep not in downstream:
                    downstream.add(dep)
                    frontier.append(dep)
        # courses that transitively wait on this one
        waiting, seen = set(), list(entry["unlocks"])
        while seen:
            cid = seen.pop()
            if cid in waiting or cid not in by_id:
                continue
            waiting.add(cid)
            seen.extend(by_id[cid]["unlocks"])
        out.append({
            "course_id": entry["course_id"], "title": entry["title"],
            "order": entry["order"] + 1, "hours": entry["hours"],
            "milestone": entry["milestone"],
            "unlocks_skills": len(downstream), "unlocks_courses": len(waiting),
            "unlocked_hours": sum(by_id[c]["hours"] for c in waiting),
            "teaches": entry["covers_named"],
            # hours are the price, unlocked work is the return
            "score": round((len(downstream) * 2 + len(waiting) * 3) / max(entry["hours"], 1), 3),
        })
    return sorted(out, key=lambda r: (-r["unlocks_courses"], -r["unlocks_skills"],
                                      r["course_id"]))


def slot_analysis(path_state: dict, graph: Optional[LearningGraph] = None) -> dict:
    """Earliest and latest slot each course could occupy, and its slack.

    The plan is one valid linearisation of a partial order; this says how much
    freedom each course actually had. Zero slack means it is on the critical
    path and the whole plan waits for it.
    """
    _g(graph)
    entries = sorted(path_state["courses"], key=lambda c: c["order"])
    by_id = {c["course_id"]: c for c in entries}
    n = len(entries)

    earliest: Dict[str, int] = {}
    for entry in entries:  # already topologically ordered
        deps = [earliest[d] for d in entry["depends_on"] if d in earliest]
        earliest[entry["course_id"]] = (max(deps) + 1) if deps else 0

    latest: Dict[str, int] = {}
    for entry in reversed(entries):
        outs = [latest[u] for u in entry["unlocks"] if u in latest]
        latest[entry["course_id"]] = (min(outs) - 1) if outs else n - 1

    # Slot indices alone can never produce zero slack, because a 20-course plan
    # is never a single 20-long chain. Criticality is a question about time, so
    # it is computed the way CPM does it: hours as durations, assuming anything
    # independent could be studied in parallel.
    est: Dict[str, int] = {}   # earliest start, in hours
    for entry in entries:
        finishes = [est[d] + by_id[d]["hours"] for d in entry["depends_on"] if d in est]
        est[entry["course_id"]] = max(finishes) if finishes else 0
    project_hours = max((est[c["course_id"]] + c["hours"] for c in entries), default=0)

    lst: Dict[str, int] = {}   # latest start, in hours
    for entry in reversed(entries):
        cid = entry["course_id"]
        starts = [lst[u] for u in entry["unlocks"] if u in lst]
        lst[cid] = (min(starts) if starts else project_hours) - entry["hours"]

    out = {}
    for entry in entries:
        cid = entry["course_id"]
        slack = latest[cid] - earliest[cid]
        hours_slack = lst[cid] - est[cid]
        blockers = [{"course_id": d, "title": by_id[d]["title"],
                     "order": by_id[d]["order"] + 1} for d in entry["depends_on"] if d in by_id]
        waiters = [{"course_id": u, "title": by_id[u]["title"],
                    "order": by_id[u]["order"] + 1} for u in entry["unlocks"] if u in by_id]
        if hours_slack == 0:
            note = ("On the critical path: it cannot move. Everything after it waits, "
                    "so an hour lost here is an hour lost overall.")
        elif not blockers:
            note = (f"Nothing feeds it, so it could start immediately, but it must land "
                    f"by step {latest[cid] + 1} to keep {len(waiters)} later course"
                    f"{'s' if len(waiters) != 1 else ''} unblocked.")
        else:
            note = (f"It cannot start before step {earliest[cid] + 1} because "
                    + ", ".join(b["title"] for b in blockers[:3])
                    + f" must come first, and it cannot slip past step {latest[cid] + 1} "
                    + (f"without blocking {waiters[0]['title']}." if waiters
                       else "without pushing the finish out."))
        out[cid] = {
            "course_id": cid, "position": entry["order"] + 1,
            "earliest": earliest[cid] + 1, "latest": latest[cid] + 1,
            "slack": slack,                       # freedom in slots
            "hours_slack": hours_slack,           # freedom in time (CPM)
            "critical": hours_slack == 0,
            "earliest_start_hours": est[cid], "latest_start_hours": lst[cid],
            "blocked_by": blockers, "blocks": waiters, "note": note,
        }
    return {"courses": out,
            "project": {"hours_if_parallel": project_hours,
                        "hours_sequential": sum(c["hours"] for c in entries)}}


def critical_path(path_state: dict, graph: Optional[LearningGraph] = None) -> dict:
    """The chain with no slack: the true minimum duration of the plan."""
    analysis = slot_analysis(path_state, graph=graph)
    slots, project = analysis["courses"], analysis["project"]
    by_id = {c["course_id"]: c for c in path_state["courses"]}
    chain = [s for s in sorted(slots.values(), key=lambda s: s["position"]) if s["critical"]]
    hours = sum(by_id[s["course_id"]]["hours"] for s in chain)
    flexible = [s for s in slots.values() if not s["critical"]]
    return {
        "chain": [{"course_id": s["course_id"], "title": by_id[s["course_id"]]["title"],
                   "position": s["position"], "hours": by_id[s["course_id"]]["hours"]}
                  for s in chain],
        "length": len(chain),
        "hours": hours,
        "flexible_courses": len(flexible),
        "flexible_hours": sum(by_id[s["course_id"]]["hours"] for s in flexible),
        "hours_if_parallel": project["hours_if_parallel"],
        "hours_sequential": project["hours_sequential"],
        "sentence": (
            f"{len(chain)} of {len(by_id)} courses ({hours}h) sit on the critical chain - "
            f"nothing can shorten them. The other {len(flexible)} have slack and could be "
            f"taken in a different order, or alongside. Studied strictly one after another "
            f"the plan is {project['hours_sequential']}h; the dependencies themselves only "
            f"force {project['hours_if_parallel']}h of sequence."),
    }


def readiness(course_id: str, possessed_skills: Iterable[str],
              level: str = "beginner", graph: Optional[LearningGraph] = None) -> dict:
    """How far is this learner from being able to take an arbitrary course?

    Works for any course in the catalogue, in or out of the current plan - the
    question people actually have while browsing a course listing.
    """
    g = _g(graph)
    if course_id not in g.courses:
        raise KeyError(f"unknown course {course_id}")
    course = g.courses[course_id]
    possessed = g.expand_possessed(possessed_skills)

    missing = gap_analysis(course["requires"], possessed, graph=g)
    ordered = topo_order(missing, graph=g)
    prep = select_courses(ordered, level=level, graph=g,
                          possessed=possessed, exclude={course_id})
    return {
        "course_id": course_id, "title": course["title"], "provider": course["provider"],
        "hours": course["hours"], "level": course["level"],
        "ready": not missing,
        "missing_skills": ordered,
        "missing_skills_named": [g.name(s) for s in ordered],
        "have_already": sorted(set(course["requires"]) & possessed),
        "have_already_named": [g.name(s) for s in sorted(set(course["requires"]) & possessed)],
        "prep_courses": [{"course_id": c["course"]["id"], "title": c["course"]["title"],
                          "provider": c["course"]["provider"],
                          "hours": c["course"]["hours"],
                          "covers": c["covers"],
                          "covers_named": [g.name(s) for s in c["covers"]]} for c in prep],
        "prep_hours": sum(c["course"]["hours"] for c in prep),
        "sentence": (f"You are ready for {course['title']} now."
                     if not missing else
                     f"You are {len(ordered)} skill{'s' if len(ordered) != 1 else ''} "
                     f"away from {course['title']}: "
                     f"{', '.join(g.name(s) for s in ordered[:4])}"
                     f"{'...' if len(ordered) > 4 else ''}. That is "
                     f"{len(prep)} course{'s' if len(prep) != 1 else ''}, "
                     f"{sum(c['course']['hours'] for c in prep)}h of preparation."),
    }


def alternative_routes(target_skills: Iterable[str], possessed_skills: Iterable[str] = (),
                       level: str = "beginner", interests: Sequence[str] = (),
                       strategies: Sequence[str] = ("balanced", "fastest", "free_first",
                                                    "hands_on", "thorough"),
                       graph: Optional[LearningGraph] = None) -> dict:
    """Several equally valid routes to the same goal, and what each one costs.

    The graph decides what must be learned and in what order; a strategy only
    decides which course covers each skill. So every route here reaches the same
    skills - they differ in length, price and style, never in correctness.
    """
    g = _g(graph)
    targets = normalize_targets(target_skills, graph=g)
    possessed = g.expand_possessed(possessed_skills)

    # how much real choice exists: skills with more than one teaching course
    needed = gap_analysis(targets, possessed, graph=g)
    choice_points = sum(1 for sid in needed if len(g.courses_teaching.get(sid, [])) > 1)

    routes = []
    baseline_ids: Set[str] = set()
    for name in strategies:
        if name not in STRATEGIES:
            continue
        path = build_path(targets, possessed, level=level, interests=interests,
                          strategy=name, graph=g)
        ids = {c["course_id"] for c in path["courses"]}
        if name == strategies[0]:
            baseline_ids = ids
        tags_of = lambda c: {t.lower() for t in c["tags"]}  # noqa: E731
        routes.append({
            "strategy": name,
            "label": STRATEGIES[name]["label"],
            "blurb": STRATEGIES[name]["blurb"],
            "courses": path["total_courses"],
            "hours": path["total_hours"],
            "milestones": len(path["milestones"]),
            "free_courses": sum(1 for c in path["courses"] if tags_of(c) & FREE_TAGS),
            "project_courses": sum(1 for c in path["courses"] if tags_of(c) & PROJECT_TAGS),
            "avg_course_hours": round(path["total_hours"] / max(path["total_courses"], 1), 1),
            "course_ids": sorted(ids),
            "titles": [c["title"] for c in path["courses"]],
            "differs_from_baseline": sorted(ids ^ baseline_ids),
            "shared_with_baseline": len(ids & baseline_ids),
            "skills_covered": path["skills_covered"],
        })

    for route in routes:
        delta_h = route["hours"] - routes[0]["hours"]
        route["hours_vs_baseline"] = delta_h
        route["courses_vs_baseline"] = route["courses"] - routes[0]["courses"]

    return {
        "targets": targets,
        "targets_named": [g.name(t) for t in targets],
        "skills_required": len(needed),
        "choice_points": choice_points,
        "routes": routes,
        "sentence": (
            f"All {len(routes)} routes teach the same {len(needed)} skills in a valid "
            f"order - they differ only in which course covers each one. Real choice "
            f"exists at {choice_points} of {len(needed)} skills; the rest are taught by "
            f"exactly one course in the catalogue, so every route shares them."),
    }


def compare_goals(goal_a: Iterable[str], goal_b: Iterable[str],
                  possessed_skills: Iterable[str] = (), level: str = "beginner",
                  interests: Sequence[str] = (), labels: Sequence[str] = ("A", "B"),
                  graph: Optional[LearningGraph] = None) -> dict:
    """Two goals side by side, plus the marginal cost of wanting both.

    The interesting number is not what each costs alone - it is the overlap.
    Two goals that share their prerequisite spine cost far less together than
    apart, and that is a career decision the graph can actually inform.
    """
    g = _g(graph)
    possessed = g.expand_possessed(possessed_skills)
    a_targets = normalize_targets(goal_a, graph=g)
    b_targets = normalize_targets(goal_b, graph=g)

    path_a = build_path(a_targets, possessed, level=level, interests=interests, graph=g)
    path_b = build_path(b_targets, possessed, level=level, interests=interests, graph=g)
    both = build_path(set(a_targets) | set(b_targets), possessed,
                      level=level, interests=interests, graph=g)

    ids_a = {c["course_id"] for c in path_a["courses"]}
    ids_b = {c["course_id"] for c in path_b["courses"]}
    hours = {c["course_id"]: c["hours"] for c in
             path_a["courses"] + path_b["courses"] + both["courses"]}
    skills_a, skills_b = set(path_a["gap_skills"]), set(path_b["gap_skills"])

    shared = sorted(ids_a & ids_b)
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)

    def titles(ids):
        return [{"course_id": i, "title": g.courses[i]["title"],
                 "hours": g.courses[i]["hours"],
                 "provider": g.courses[i]["provider"]} for i in ids]

    marginal_courses = both["total_courses"] - path_a["total_courses"]
    marginal_hours = both["total_hours"] - path_a["total_hours"]
    overlap_pct = round(100 * len(shared) / max(len(ids_a | ids_b), 1))

    return {
        "labels": list(labels),
        "goal_a": {"targets": a_targets, "targets_named": [g.name(t) for t in a_targets],
                   "courses": path_a["total_courses"], "hours": path_a["total_hours"],
                   "skills": len(skills_a)},
        "goal_b": {"targets": b_targets, "targets_named": [g.name(t) for t in b_targets],
                   "courses": path_b["total_courses"], "hours": path_b["total_hours"],
                   "skills": len(skills_b)},
        "both": {"courses": both["total_courses"], "hours": both["total_hours"],
                 "skills": len(both["gap_skills"])},
        "shared_courses": titles(shared),
        "only_a": titles(only_a),
        "only_b": titles(only_b),
        "shared_hours": sum(hours.get(i, 0) for i in shared),
        "shared_skills": sorted(skills_a & skills_b),
        "shared_skills_named": [g.name(s) for s in sorted(skills_a & skills_b)],
        "overlap_percent": overlap_pct,
        "marginal": {
            "courses": marginal_courses, "hours": marginal_hours,
            "separately_courses": path_a["total_courses"] + path_b["total_courses"],
            "separately_hours": path_a["total_hours"] + path_b["total_hours"],
            "saved_hours": (path_a["total_hours"] + path_b["total_hours"]) - both["total_hours"],
        },
        "sentence": (
            f"{labels[0]} and {labels[1]} share {len(shared)} of "
            f"{len(ids_a | ids_b)} courses ({overlap_pct}% overlap). Once you have "
            f"{labels[0]}, adding {labels[1]} costs only {marginal_courses} more course"
            f"{'s' if marginal_courses != 1 else ''} and {marginal_hours}h - against "
            f"{path_b['total_hours']}h if you started it from scratch."),
    }


# ---------------------------------------------------------------------------
# Learner modelling: graded self-assessment, pace, and staleness
#
# None of this is machine learning. It is arithmetic over what the learner has
# actually done, which is both more honest and more explainable than a model.
# ---------------------------------------------------------------------------
SKILL_LEVELS = ["heard_of", "can_use", "can_teach"]


def split_self_assessment(skill_levels: Dict[str, str],
                          graph: Optional[LearningGraph] = None) -> dict:
    """Turn graded self-assessment into possessed skills plus refresher hints.

    Binary "do you know pandas" overstates half the answers. Only can_use and
    above count as possessed; heard_of is not a skill you have, but it does mean
    you need a refresher rather than a full course.
    """
    g = _g(graph)
    possessed, refresh = set(), set()
    for sid, lvl in (skill_levels or {}).items():
        if sid not in g.skills:
            continue
        if lvl in ("can_use", "can_teach"):
            possessed.add(sid)
        elif lvl == "heard_of":
            refresh.add(sid)
    return {
        "possessed": sorted(g.expand_possessed(possessed)),
        "refresh": sorted(refresh - g.expand_possessed(possessed)),
        "confident": sorted(s for s, l in (skill_levels or {}).items()
                            if l == "can_teach" and s in g.skills),
    }


def velocity_report(learner: dict, path_state: dict, hours_per_week: float = 10.0,
                    graph: Optional[LearningGraph] = None) -> dict:
    """Re-forecast the remaining path from the learner's own observed pace."""
    g = _g(graph)
    log = [e for e in learner.get("completion_log", []) if e.get("course_id") in g.courses]
    estimated = sum(e.get("estimated_hours", 0) for e in log)
    actual = sum(e.get("actual_hours") or e.get("estimated_hours", 0) for e in log)
    measured = [e for e in log if e.get("actual_hours")]

    factor = round(actual / estimated, 2) if estimated else 1.0
    remaining = path_state["total_hours"]
    adjusted = int(round(remaining * factor))
    hours_per_week = max(hours_per_week or 10.0, 0.5)

    if not measured:
        note = ("No timings logged yet, so this is the catalogue estimate. Log the hours "
                "a course actually took and every later forecast uses your own pace.")
    elif factor > 1.05:
        note = (f"You are running {factor}x the catalogue estimate across "
                f"{len(measured)} timed course{'s' if len(measured) != 1 else ''}, so the "
                f"honest remaining figure is {adjusted}h rather than {remaining}h.")
    elif factor < 0.95:
        note = (f"You are running {factor}x the catalogue estimate - faster than the "
                f"listings assume - so {adjusted}h is more realistic than {remaining}h.")
    else:
        note = f"You are tracking the catalogue estimates almost exactly ({factor}x)."

    return {
        "courses_timed": len(measured),
        "courses_completed": len(log),
        "estimated_hours": estimated,
        "actual_hours": actual,
        "pace_factor": factor,
        "remaining_hours_listed": remaining,
        "remaining_hours_adjusted": adjusted,
        "hours_per_week": hours_per_week,
        "weeks_listed": round(remaining / hours_per_week, 1),
        "weeks_adjusted": round(adjusted / hours_per_week, 1),
        "note": note,
    }


def budget_forecast(path_state: dict, hours_per_week: float = 10.0,
                    weeks_available: Optional[float] = None, pace_factor: float = 1.0,
                    graph: Optional[LearningGraph] = None) -> dict:
    """How far the plan gets inside a real time budget, and what slips past it."""
    g = _g(graph)
    hours_per_week = max(hours_per_week or 10.0, 0.5)
    budget = None if weeks_available is None else hours_per_week * weeks_available

    cumulative, schedule, reachable = 0.0, [], []
    for entry in path_state["courses"]:
        cost = entry["hours"] * max(pace_factor, 0.1)
        start_week = cumulative / hours_per_week
        cumulative += cost
        item = {
            "course_id": entry["course_id"], "title": entry["title"],
            "milestone": entry["milestone"], "hours": entry["hours"],
            "start_week": round(start_week, 1), "finish_week": round(cumulative / hours_per_week, 1),
            "within_budget": budget is None or cumulative <= budget,
        }
        schedule.append(item)
        if item["within_budget"]:
            reachable.append(item)

    done_milestones, cut = [], None
    for ms in path_state["milestones"]:
        ids = set(ms["course_ids"])
        if ids and ids.issubset({r["course_id"] for r in reachable}):
            done_milestones.append(ms["name"])
        elif cut is None and ids:
            cut = ms["name"]

    total_weeks = round(cumulative / hours_per_week, 1)
    if budget is None:
        sentence = (f"At {hours_per_week:g}h a week the whole plan takes about "
                    f"{total_weeks} weeks.")
    elif len(reachable) == len(schedule):
        sentence = (f"The whole plan fits: {total_weeks} weeks at {hours_per_week:g}h a "
                    f"week, inside your {weeks_available:g}-week budget.")
    else:
        missed = len(schedule) - len(reachable)
        sentence = (f"In {weeks_available:g} weeks at {hours_per_week:g}h a week you reach "
                    f"course {len(reachable)} of {len(schedule)}"
                    + (f", finishing {', '.join(done_milestones)}" if done_milestones else "")
                    + f". {missed} course{'s' if missed != 1 else ''} slip past the deadline"
                    + (f", starting with {cut}." if cut else "."))

    return {
        "hours_per_week": hours_per_week, "weeks_available": weeks_available,
        "pace_factor": pace_factor, "total_weeks": total_weeks,
        "budget_hours": budget, "schedule": schedule,
        "reachable_courses": len(reachable), "total_courses": len(schedule),
        "milestones_completed": done_milestones, "first_milestone_at_risk": cut,
        "sentence": sentence,
    }


def refresher_prompts(learner: dict, path_state: dict, stale_days: int = 45,
                      lookahead: int = 3, now: Optional[float] = None,
                      graph: Optional[LearningGraph] = None) -> List[dict]:
    """Skills learned a while ago that the next courses are about to lean on.

    Review timing driven by position in the graph rather than a generic
    repetition schedule: a skill only becomes worth refreshing when something
    that directly requires it is coming up.
    """
    import time as _time
    g = _g(graph)
    now = now if now is not None else _time.time()
    acquired: Dict[str, float] = learner.get("skill_acquired_at", {}) or {}
    if not acquired:
        return []

    out = []
    for entry in path_state["courses"][:lookahead]:
        for req in entry["requires"]:
            when = acquired.get(req)
            if when is None:
                continue
            age_days = (now - when) / 86400.0
            if age_days < stale_days:
                continue
            teachers = g.courses_teaching.get(req, [])
            shortest = min((g.courses[c] for c in teachers),
                           key=lambda c: c["hours"], default=None)
            out.append({
                "skill": req, "skill_name": g.name(req),
                "age_days": int(age_days),
                "needed_by": entry["course_id"], "needed_by_title": entry["title"],
                "position": entry["order"] + 1,
                "suggested_course": None if shortest is None else {
                    "course_id": shortest["id"], "title": shortest["title"],
                    "hours": shortest["hours"], "provider": shortest["provider"]},
                "sentence": (f"You learned {g.name(req)} {int(age_days // 7)} weeks ago and "
                             f"{entry['title']} builds directly on it."),
            })

    seen, unique = set(), []
    for item in sorted(out, key=lambda i: (-i["age_days"], i["skill"])):
        if item["skill"] not in seen:
            seen.add(item["skill"])
            unique.append(item)
    return unique


def parse_history(text: str, graph: Optional[LearningGraph] = None) -> dict:
    """Turn a pasted list of courses and certificates into profile skills.

    Same discipline as goal parsing: match against the catalogue, report the
    evidence, and say plainly what could not be matched rather than inventing
    credit for it.
    """
    import difflib
    g = _g(graph)
    titles = {_norm_text(c["title"]): cid for cid, c in g.courses.items()}
    title_keys = list(titles)

    lines = [ln.strip(" -*\t") for ln in re.split(r"[\n;|]+", text or "") if ln.strip(" -*\t")]
    matched_courses, matched_skills, unmatched = [], [], []

    for line in lines:
        norm = _norm_text(line)
        if not norm:
            continue

        cid = titles.get(norm)
        if cid is None:
            near = difflib.get_close_matches(norm, title_keys, n=1, cutoff=0.72)
            if near:
                cid = titles[near[0]]
        if cid is not None:
            course = g.courses[cid]
            matched_courses.append({
                "line": line, "course_id": cid, "title": course["title"],
                "provider": course["provider"], "hours": course["hours"],
                "teaches": course["teaches"],
                "teaches_named": [g.name(s) for s in course["teaches"]],
                "evidence": "course title",
            })
            continue

        resolved = resolve_goal(line, graph=g)
        if resolved["targets"] and resolved["confidence"] in ("high", "medium"):
            match = resolved["matches"][0] if resolved["matches"] else {}
            matched_skills.append({
                "line": line, "skills": resolved["targets"],
                "skills_named": [g.name(s) for s in resolved["targets"]],
                "evidence": match.get("evidence", "matched the skill catalogue"),
                "method": match.get("method", "resolver"),
            })
            continue

        unmatched.append({
            "line": line,
            "candidates": [dict(c, name=g.name(c["skill_id"]))
                           for c in resolved["candidates"][:3]],
        })

    skills: Set[str] = set()
    for item in matched_courses:
        skills.update(item["teaches"])
    for item in matched_skills:
        skills.update(item["skills"])
    closed = g.expand_possessed(skills)

    return {
        "lines_read": len(lines),
        "matched_courses": matched_courses,
        "matched_skills": matched_skills,
        "unmatched": unmatched,
        "skills_direct": sorted(skills),
        "skills_direct_named": [g.name(s) for s in sorted(skills)],
        "skills_with_prerequisites": sorted(closed),
        "implied": sorted(closed - skills),
        "implied_named": [g.name(s) for s in sorted(closed - skills)],
        "course_ids": [c["course_id"] for c in matched_courses],
        "sentence": (
            f"Read {len(lines)} line{'s' if len(lines) != 1 else ''}: matched "
            f"{len(matched_courses)} course{'s' if len(matched_courses) != 1 else ''} and "
            f"{len(matched_skills)} skill mention{'s' if len(matched_skills) != 1 else ''}, "
            f"giving {len(skills)} skills directly and {len(closed - skills)} more by "
            f"prerequisite."
            + (f" {len(unmatched)} line{'s' if len(unmatched) != 1 else ''} could not be "
               f"matched to anything in the catalogue." if unmatched else "")),
    }


def graph_health(graph: Optional[LearningGraph] = None) -> dict:
    """Maintainer's view: where this curriculum is thin or fragile."""
    g = _g(graph)
    single_source, uncovered = [], []
    for sid in sorted(g.skills):
        teachers = g.courses_teaching.get(sid, [])
        if not teachers:
            uncovered.append({"id": sid, "name": g.name(sid)})
        elif len(teachers) == 1:
            single_source.append({"id": sid, "name": g.name(sid),
                                  "only_course": g.courses[teachers[0]]["title"],
                                  "depth": g.depth(sid),
                                  "dependents": len(g.dependents.get(sid, ()))})

    roots = [{"id": s, "name": g.name(s)} for s in sorted(g.skills)
             if not g.skills[s]["requires"]]
    leaves = [{"id": s, "name": g.name(s)} for s in sorted(g.skills)
              if not g.dependents.get(s)]
    deepest = sorted(g.skills, key=lambda s: (-g.depth(s), s))[:5]
    per_category: Dict[str, int] = {}
    for sid in g.skills:
        per_category[g.category(sid)] = per_category.get(g.category(sid), 0) + 1
    fan_out = sorted(g.skills, key=lambda s: (-len(g.dependents.get(s, ())), s))[:5]

    return {
        "skills": len(g.skills), "courses": len(g.courses),
        "edges": sum(len(s["requires"]) for s in g.skill_list),
        "max_depth": max(g.depth(s) for s in g.skills),
        "avg_prereqs": round(sum(len(s["requires"]) for s in g.skill_list) / len(g.skills), 2),
        "skills_per_category": per_category,
        "uncovered_skills": uncovered,          # must be empty: load_graph enforces it
        "single_source_skills": single_source,  # fragile: one course is the only route
        "roots": roots, "leaves": leaves,
        "deepest_skills": [{"id": s, "name": g.name(s), "depth": g.depth(s)} for s in deepest],
        "highest_fan_out": [{"id": s, "name": g.name(s),
                             "unlocks": len(g.dependents.get(s, ()))} for s in fan_out],
        "courses_per_skill": round(
            sum(len(g.courses_teaching.get(s, [])) for s in g.skills) / len(g.skills), 2),
    }


if __name__ == "__main__":  # quick smoke test: python engine.py
    G = load_graph()
    print("graph ok:", graph_stats(G))
