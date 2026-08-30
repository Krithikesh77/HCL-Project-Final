"""added new line"""
"""
FastAPI layer for the AI-Powered Personalized Learning Path Recommender.

The path itself is computed entirely by engine.py (prerequisite graph traversal).
This file only:
  * keeps learner state in a single learners.json file,
  * exposes the engine over HTTP,
  * serves the single-file frontend,
  * and calls a local Ollama model for EXACTLY TWO things:
        (a) free-text goal  -> target skill ids
        (b) prettifying an explanation that the graph already produced
    plus one convenience use, grounded chat, which is also given a
    deterministic fallback.

Every Ollama call goes through ask_ollama(), which has a hard timeout and never
raises.  If Ollama is not running, every endpoint still returns a correct,
complete answer from the deterministic code paths.  Start the server and read
the banner line to see which mode you are in.

Run:  uvicorn main:app --reload      then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "learners.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = 20        # seconds, hard cap on every single call
OLLAMA_RETRY_AFTER = 60    # seconds to stay in fallback mode after a failure

APP_VERSION = "1.0.0"

GRAPH = engine.load_graph()  # raises loudly if skills.json is not a valid DAG

# Runtime status, refreshed on startup and after every call attempt.
OLLAMA_STATE: Dict[str, Any] = {
    "reachable": False,
    "model": OLLAMA_MODEL,
    "url": OLLAMA_URL,
    "checked_at": None,
    "last_error": None,
    "calls": 0,
    "fallbacks": 0,
}


# ---------------------------------------------------------------------------
# Ollama: one wrapper, one timeout, never raises
# ---------------------------------------------------------------------------
def probe_ollama() -> bool:
    """Cheap reachability check against /api/tags. Never raises.

    'reachable' means usable for generation: the daemon answers AND the
    configured model is actually pulled. A running daemon without the model
    would 404 on every generate, so it is treated as unavailable.
    """
    base = OLLAMA_URL.split("/api/")[0]
    try:
        resp = requests.get(f"{base}/api/tags", timeout=3)
        ok = resp.status_code == 200
        models = []
        if ok:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        present = OLLAMA_MODEL in models or any(
            m.split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in models)
        OLLAMA_STATE.update({
            "reachable": bool(ok and present),
            "daemon_up": ok,
            "checked_at": time.time(),
            "last_error": None if ok else f"HTTP {resp.status_code}",
            "models": models,
            "model_present": present,
        })
        return OLLAMA_STATE["reachable"]
    except Exception as exc:  # connection refused, DNS, timeout, anything
        OLLAMA_STATE.update({
            "reachable": False,
            "daemon_up": False,
            "checked_at": time.time(),
            "last_error": f"{type(exc).__name__}: {exc}",
            "models": [],
            "model_present": False,
        })
        return False


def ask_ollama(prompt: str, system: Optional[str] = None,
               temperature: float = 0.1, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Single entry point for the LLM. Returns None on any problem at all."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }
    if system:
        payload["system"] = system

    # Back off after a failure instead of paying the timeout on every request.
    # A daemon that is up but missing the model 404s in ~2s, which is invisible
    # in a test suite and very visible in the UI. Re-probe after the window so a
    # model pulled mid-session is still picked up.
    last = OLLAMA_STATE.get("checked_at") or 0
    if not OLLAMA_STATE["reachable"] and (time.time() - last) < OLLAMA_RETRY_AFTER:
        OLLAMA_STATE["fallbacks"] += 1
        return None
    if not OLLAMA_STATE["reachable"] and not probe_ollama():
        OLLAMA_STATE["fallbacks"] += 1
        return None

    try:
        OLLAMA_STATE["calls"] += 1
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        text = (resp.json().get("response") or "").strip()
        # reasoning models (qwen3 etc.) emit a <think> block; it is not the answer
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        OLLAMA_STATE["reachable"] = True
        OLLAMA_STATE["last_error"] = None
        return text or None
    except Exception as exc:
        OLLAMA_STATE["reachable"] = False
        OLLAMA_STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        OLLAMA_STATE["fallbacks"] += 1
        return None


# --- LLM use #1: goal text -> target skill ids ------------------------------
SKILL_MENU = "\n".join(
    f"- {sid}: {GRAPH.skills[sid]['name']}" for sid in sorted(GRAPH.skills)
)

EXTRACT_SYSTEM = (
    "You map a learner's goal onto skill ids from a fixed catalog. "
    "You never invent ids and you reply with JSON only."
)


def extract_target_skills(goal: str) -> Dict[str, Any]:
    """(a) LLM goal parsing, backed by the deterministic four-layer resolver.

    Neither path is allowed to invent a goal. The model may only answer with ids
    from the catalogue and is told to return [] for anything outside AI/ML; the
    resolver returns candidates instead of targets when it is unsure. If both
    come up empty the caller asks the learner rather than guessing.
    """
    resolved = engine.resolve_goal(goal, graph=GRAPH)
    prompt = (
        f"Learner goal: \"{goal}\"\n\n"
        f"Available skill ids:\n{SKILL_MENU}\n\n"
        "Pick the 1-4 skill ids that ARE the goal itself (the end state), not the "
        "prerequisites - prerequisites are computed separately by a graph.\n"
        "If the goal is not about AI/ML engineering at all, or you cannot tell what "
        "it means, reply with an empty array [] - do not guess.\n"
        'Otherwise reply with only a JSON array of ids, e.g. ["rag","llm-agents"].'
    )
    raw = ask_ollama(prompt, system=EXTRACT_SYSTEM, temperature=0.0)
    if raw:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                ids = json.loads(match.group(0))
                valid = [i for i in ids if isinstance(i, str) and i in GRAPH.skills]
                if valid:
                    return {
                        "targets": engine.normalize_targets(valid, graph=GRAPH),
                        "source": "ollama",
                        "confidence": "high",
                        "matches": [{"skill_id": v, "name": GRAPH.name(v), "score": 1.0,
                                     "method": "ollama", "evidence": "chosen by the model"}
                                    for v in valid],
                        "candidates": resolved["candidates"],
                        "unknown_terms": resolved["unknown_terms"],
                        "out_of_domain": False,
                        "raw": raw[:400],
                        "resolver_targets": resolved["targets"],
                    }
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "targets": resolved["targets"],
        "source": "resolver" if resolved["targets"] else "unresolved",
        "confidence": resolved["confidence"],
        "matches": [dict(m, name=GRAPH.name(m["skill_id"])) for m in resolved["matches"]],
        "candidates": [dict(c, name=GRAPH.name(c["skill_id"])) for c in resolved["candidates"]],
        "unknown_terms": resolved["unknown_terms"],
        "out_of_domain": resolved["out_of_domain"],
        "raw": raw[:400] if raw else None,
        "resolver_targets": resolved["targets"],
    }


# ---------------------------------------------------------------------------
# Vocabulary gap log: what the graph has never heard of is the content backlog
# ---------------------------------------------------------------------------
GAP_FILE = os.path.join(BASE_DIR, "vocab_gaps.json")


def record_gap(goal: str, extraction: Dict[str, Any],
               chosen: Optional[List[str]] = None) -> None:
    """Append a goal the parser could not fully understand. Never raises."""
    if not extraction.get("unknown_terms") and extraction.get("targets") and not chosen:
        return  # fully understood, nothing to learn
    try:
        log = []
        if os.path.exists(GAP_FILE):
            with open(GAP_FILE, "r", encoding="utf-8") as fh:
                log = json.load(fh)
        log.append({
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "unknown_terms": extraction.get("unknown_terms", []),
            "resolved": bool(extraction.get("targets")) or bool(chosen),
            "out_of_domain": extraction.get("out_of_domain", False),
            "candidates_offered": [c["skill_id"] for c in extraction.get("candidates", [])],
            "learner_chose": chosen or [],
        })
        tmp = GAP_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(log[-500:], fh, indent=2)
        os.replace(tmp, GAP_FILE)
    except Exception:
        pass  # a logging failure must never break an intake


# --- LLM use #2: prettify an explanation the graph already computed ----------
PRETTIFY_SYSTEM = (
    "You rewrite an already-computed explanation so it reads naturally. "
    "You never add facts, never invent prerequisites, never change the reasoning. "
    "2-4 short sentences, second person, plain text."
)
_pretty_cache: Dict[str, str] = {}


def prettify_explanation(trace: Dict[str, Any]) -> Dict[str, Any]:
    """(b) Cosmetic rewrite only. The trace is the source of truth either way."""
    key = trace["course_id"] + "|" + str(len(trace["sentences"]))
    if key in _pretty_cache:
        return {"text": _pretty_cache[key], "source": "ollama-cached"}

    facts = "\n".join(f"- {s}" for s in trace["sentences"])
    prompt = (
        f"Course: {trace['title']} ({trace['provider']}, {trace['hours']} hours)\n"
        f"Facts derived from the learner's prerequisite graph:\n{facts}\n\n"
        "Rewrite these facts as a short, friendly explanation of why this course "
        "is in this learner's path right now. Use only the facts above."
    )
    text = ask_ollama(prompt, system=PRETTIFY_SYSTEM, temperature=0.3)
    if text and 20 < len(text) < 1200:
        text = text.strip().strip('"')
        _pretty_cache[key] = text
        return {"text": text, "source": "ollama"}
    return {"text": trace["summary"], "source": "graph-fallback"}


# ---------------------------------------------------------------------------
# Persistence: one JSON file, no database
# ---------------------------------------------------------------------------
def blank_state() -> Dict[str, Any]:
    return {"version": APP_VERSION, "active": None, "learners": {}}


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return blank_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        state.setdefault("learners", {})
        state.setdefault("active", None)
        return state
    except (json.JSONDecodeError, OSError):
        return blank_state()


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_FILE)


def active_learner(required: bool = True) -> Optional[Dict[str, Any]]:
    state = load_state()
    learner = state["learners"].get(state.get("active") or "")
    if learner is None and required:
        raise HTTPException(status_code=404,
                            detail="No learner yet. POST /api/intake first.")
    return learner


def persist(learner: Dict[str, Any]) -> None:
    state = load_state()
    state["learners"][learner["id"]] = learner
    state["active"] = learner["id"]
    save_state(state)


def log_event(learner: Dict[str, Any], kind: str, detail: str) -> None:
    learner.setdefault("history", []).append(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "detail": detail}
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class IntakeRequest(BaseModel):
    goal: str = Field(..., min_length=2)
    experience_level: str = "beginner"
    known_skills: List[str] = []
    interests: List[str] = []
    # Re-aim an existing learner instead of starting a fresh one. Completed
    # courses, everything they have learned and their feedback history survive
    # the goal change; only the destination moves.
    keep_progress: bool = False
    # Set by the clarification step: the learner picked from the candidates we
    # offered, so skip parsing entirely and use exactly what they chose.
    target_skills: List[str] = []
    # graded self-assessment: {skill_id: heard_of | can_use | can_teach}.
    # heard_of does not count as possessed but does earn a refresher course.
    skill_levels: Dict[str, str] = {}
    strategy: str = "balanced"


class CompleteRequest(BaseModel):
    course_id: str
    # what it really took, so later forecasts can use this learner's own pace
    actual_hours: Optional[float] = None


class FeedbackRequest(BaseModel):
    course_id: str
    signal: str  # too_hard | too_easy | already_know


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Learning Path Recommender", version=APP_VERSION)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def startup_banner() -> None:
    """The line the README promises: says out loud whether Ollama was reachable."""
    stats = engine.graph_stats(GRAPH)
    say = lambda line: print(line, flush=True)  # noqa: E731
    say("=" * 76)
    say(f"  Learning Path Recommender v{APP_VERSION}")
    say(f"  Graph: {stats['skills']} skills, {stats['edges']} prerequisite edges, "
        f"{stats['courses']} courses, max depth {stats['max_depth']} (DAG validated)")
    if probe_ollama():
        say(f"  OLLAMA: REACHABLE at {OLLAMA_URL} using model {OLLAMA_MODEL}")
        say("  -> LLM mode: goals parsed by the model, explanations reworded by it.")
        say("     Paths, ordering and reasoning are still computed by the graph only.")
    elif OLLAMA_STATE.get("daemon_up"):
        say(f"  OLLAMA: daemon is up but model '{OLLAMA_MODEL}' is not pulled.")
        say(f"     Pull it:  ollama pull {OLLAMA_MODEL}")
        available = ", ".join(OLLAMA_STATE.get("models") or []) or "none"
        say(f"     Or use one you already have (available: {available}) via")
        say("       OLLAMA_MODEL=<name> uvicorn main:app")
        say("  -> Until then: DETERMINISTIC MODE. Everything works.")
    else:
        say(f"  OLLAMA: NOT REACHABLE ({OLLAMA_STATE['last_error']})")
        say("  -> DETERMINISTIC MODE: goals parsed by keyword matching, explanations")
        say("     read straight off the graph. Nothing is missing except nicer")
        say("     phrasing - this is a fully supported way to run the app.")
    say("  Open http://localhost:8000")
    say("=" * 76)


@app.on_event("startup")
def _on_startup() -> None:
    startup_banner()


@app.get("/")
def index() -> FileResponse:
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="static/index.html not built yet")
    return FileResponse(path)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "has_learner": active_learner(required=False) is not None,
        "graph": engine.graph_stats(GRAPH),
        "ollama": {
            "reachable": OLLAMA_STATE["reachable"],
            "model": OLLAMA_MODEL,
            "url": OLLAMA_URL,
            "last_error": OLLAMA_STATE["last_error"],
            "llm_calls": OLLAMA_STATE["calls"],
            "fallbacks_used": OLLAMA_STATE["fallbacks"],
            "mode": "llm-assisted" if OLLAMA_STATE["reachable"] else "deterministic",
        },
    }


@app.get("/api/skills")
def list_skills(q: str = "") -> Dict[str, Any]:
    """Catalog for the known-skills multiselect in the intake panel."""
    skills = engine.search_skills(q, graph=GRAPH)
    return {"skills": skills, "categories": engine.CATEGORY_ORDER}


@app.get("/api/personas")
def personas() -> Dict[str, Any]:
    """Demo shortcuts so the intake form can be filled in one click."""
    return {"personas": [
        {"name": "Complete beginner", "goal": "I want to become an ML engineer",
         "experience_level": "beginner", "known_skills": [], "interests": []},
        {"name": "Python dev, no ML", "goal": "I want to build RAG chatbots",
         "experience_level": "intermediate",
         "known_skills": ["python", "python-oop", "command-line",
                          "git-version-control", "sql"],
         "interests": ["llm", "rag"]},
        {"name": "Knows scikit-learn", "goal": "I want to fine-tune LLMs",
         "experience_level": "intermediate",
         "known_skills": ["sklearn-basics", "supervised-learning", "model-evaluation",
                          "pandas", "numpy", "statistics"],
         "interests": ["llm", "finetuning"]},
    ]}


@app.post("/api/intake")
def intake(req: IntakeRequest) -> Dict[str, Any]:
    """Goal text -> target skills (LLM or fallback) -> learner profile.

    With keep_progress the existing learner is re-aimed rather than replaced:
    a new goal recomputes the gap from everything they have already learned,
    which is the whole advantage of holding state as a set of skills.
    """
    chosen = [s for s in req.target_skills if s in GRAPH.skills]
    if chosen:
        # came back from the clarification step - use exactly what was picked
        extraction = {
            "targets": engine.normalize_targets(chosen, graph=GRAPH),
            "source": "clarified", "confidence": "high",
            "matches": [{"skill_id": c, "name": GRAPH.name(c), "score": 1.0,
                         "method": "clarified", "evidence": "you picked this"}
                        for c in chosen],
            "candidates": [], "unknown_terms": [], "out_of_domain": False,
        }
        record_gap(req.goal, engine.resolve_goal(req.goal, graph=GRAPH), chosen=chosen)
    else:
        extraction = extract_target_skills(req.goal)
        record_gap(req.goal, extraction)

    if not extraction["targets"]:
        # Refuse to invent a goal. Hand back what we half-understood so the UI
        # can ask, and say plainly when nothing in this domain is close.
        return {
            "status": "needs_clarification",
            "goal": req.goal,
            "out_of_domain": extraction["out_of_domain"],
            "candidates": extraction["candidates"],
            "unknown_terms": extraction["unknown_terms"],
            "message": (
                "This planner only covers AI / ML engineering, and nothing in the "
                "skill graph is close to that goal. Try naming what you want to "
                "build or the technique you want to learn."
                if extraction["out_of_domain"] else
                "I am not confident I understood that. Which of these did you mean?"),
        }

    known = [s for s in req.known_skills if s in GRAPH.skills]
    level = req.experience_level if req.experience_level in engine.LEVELS else "beginner"
    interests = [i.strip() for i in req.interests if i.strip()]
    strategy = req.strategy if req.strategy in engine.STRATEGIES else "balanced"

    # graded self-assessment, if the learner gave one, sits alongside the plain
    # checkbox list: can_use and above are possessed, heard_of earns a refresher
    graded = engine.split_self_assessment(req.skill_levels, graph=GRAPH)
    known = sorted(set(known) | set(graded["possessed"]))
    refresh_skills = graded["refresh"]

    existing = active_learner(required=False) if req.keep_progress else None
    if existing is not None:
        learner = existing
        previous_goal = learner["goal"]
        learner["goal"] = req.goal
        learner["experience_level"] = level
        learner["known_skills"] = sorted(set(learner["known_skills"]) | set(known))
        learner["possessed"] = sorted(
            GRAPH.expand_possessed(set(learner["possessed"]) | set(known)))
        learner["interests"] = interests or learner.get("interests", [])
        learner["target_skills"] = extraction["targets"]
        learner["target_source"] = extraction["source"]
        learner["strategy"] = strategy
        if req.skill_levels:
            learner.setdefault("skill_levels", {}).update(
                {k: v for k, v in req.skill_levels.items()
                 if k in GRAPH.skills and v in engine.SKILL_LEVELS})
            learner["refresh_skills"] = sorted(
                set(learner.get("refresh_skills", [])) | set(refresh_skills))
        known = learner["known_skills"]
        possessed = learner["possessed"]
        log_event(learner, "goal-change",
                  f'"{previous_goal}" -> "{req.goal}" via {extraction["source"]}, '
                  f'keeping {len(learner["completed_courses"])} completed courses')
    else:
        possessed = sorted(GRAPH.expand_possessed(known))
        learner = {
            "id": uuid.uuid4().hex[:12],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": req.goal,
            "experience_level": level,
            "known_skills": known,
            "possessed": possessed,
            "interests": interests,
            "target_skills": extraction["targets"],
            "target_source": extraction["source"],
            "strategy": strategy,
            "skill_levels": {k: v for k, v in (req.skill_levels or {}).items()
                             if k in GRAPH.skills and v in engine.SKILL_LEVELS},
            "refresh_skills": refresh_skills,
            "completed_courses": [],
            "rejected_courses": [],
            "completion_log": [],
            "skill_acquired_at": {},
            "history": [],
        }
        log_event(learner, "intake",
                  f"goal parsed via {extraction['source']} -> "
                  + ", ".join(learner["target_skills"]))
    persist(learner)

    return {
        "learner": learner,
        "targets_named": [GRAPH.name(t) for t in learner["target_skills"]],
        "implied_skills": sorted(set(possessed) - set(known)),
        "implied_skills_named": [GRAPH.name(s) for s in sorted(set(possessed) - set(known))],
        "status": "ok",
        "extraction": {
            "source": extraction["source"],
            "confidence": extraction["confidence"],
            "matches": extraction["matches"],          # how each target was read
            "unknown_terms": extraction["unknown_terms"],
            "resolver_targets": extraction.get("resolver_targets", []),
        },
        "ollama_used": extraction["source"] == "ollama",
        "kept_progress": existing is not None,
        "completed_carried_over": len(learner["completed_courses"]),
    }


@app.post("/api/plan")
def plan() -> Dict[str, Any]:
    """The full computed path: gap -> topo order -> courses -> milestones."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    return {"learner": learner, "path": path,
            "mode": "llm-assisted" if OLLAMA_STATE["reachable"] else "deterministic"}


@app.get("/api/explain/{course_id}")
def explain(course_id: str, pretty: bool = True) -> Dict[str, Any]:
    """Grounded explanation: the trace is computed from the graph, then optionally
    reworded by the LLM. The trace is always returned so the UI can show both."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    if course_id not in GRAPH.courses:
        raise HTTPException(status_code=404, detail=f"unknown course {course_id}")
    try:
        trace = engine.explain(course_id, path, graph=GRAPH)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown course {course_id}")

    narrative = {"text": trace["summary"], "source": "graph"}
    if pretty and OLLAMA_STATE["reachable"]:
        narrative = prettify_explanation(trace)
    return {"trace": trace, "narrative": narrative}


@app.post("/api/complete")
def complete(req: CompleteRequest) -> Dict[str, Any]:
    """Mark a course done: its skills become possessed and the path is recomputed."""
    learner = active_learner()
    if req.course_id not in GRAPH.courses:
        raise HTTPException(status_code=404, detail=f"unknown course {req.course_id}")
    course = GRAPH.courses[req.course_id]

    before = engine.replan(learner, graph=GRAPH)
    if req.course_id not in learner["completed_courses"]:
        learner["completed_courses"].append(req.course_id)
    learner["possessed"] = sorted(
        GRAPH.expand_possessed(set(learner["possessed"]) | set(course["teaches"]))
    )

    now = time.time()
    learner.setdefault("completion_log", []).append({
        "course_id": req.course_id, "at": now,
        "estimated_hours": course["hours"],
        "actual_hours": req.actual_hours,
    })
    acquired = learner.setdefault("skill_acquired_at", {})
    for sid in course["teaches"]:
        acquired.setdefault(sid, now)
    log_event(learner, "complete",
              f"{req.course_id} ({course['title']}) -> gained "
              + ", ".join(course["teaches"]))
    persist(learner)

    after = engine.replan(learner, graph=GRAPH)
    return {
        "learner": learner,
        "path": after,
        "delta": {
            "courses_before": before["total_courses"],
            "courses_after": after["total_courses"],
            "hours_before": before["total_hours"],
            "hours_after": after["total_hours"],
            "skills_gained": course["teaches"],
            "skills_gained_named": [GRAPH.name(s) for s in course["teaches"]],
            "removed_courses": [c["course_id"] for c in before["courses"]
                                if c["course_id"] not in
                                {x["course_id"] for x in after["courses"]}],
        },
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """too_hard / too_easy swap the course; already_know marks its skills possessed."""
    learner = active_learner()
    if req.course_id not in GRAPH.courses:
        raise HTTPException(status_code=404, detail=f"unknown course {req.course_id}")
    if req.signal not in {"too_hard", "too_easy", "already_know"}:
        raise HTTPException(status_code=400, detail="signal must be too_hard, "
                                                    "too_easy or already_know")
    course = GRAPH.courses[req.course_id]
    before = engine.replan(learner, graph=GRAPH)
    before_entry = next((c for c in before["courses"]
                         if c["course_id"] == req.course_id), None)
    covered = before_entry["covers"] if before_entry else course["teaches"]
    message = ""

    if req.signal == "already_know":
        learner["known_skills"] = sorted(set(learner["known_skills"]) | set(course["teaches"]))
        learner["possessed"] = sorted(
            GRAPH.expand_possessed(set(learner["possessed"]) | set(course["teaches"]))
        )
        message = ("Marked " + ", ".join(GRAPH.name(s) for s in course["teaches"])
                   + " as already known. Anything that only existed to reach those "
                     "skills has been dropped from your path.")
    else:
        # Only swap if the catalog actually has another course for these skills.
        # Some skills are taught by exactly one course; pretending otherwise would
        # re-select the same course and lie about it.
        direction = "harder" if req.signal == "too_easy" else "easier"
        alternatives = [
            c for c in engine.alternative_courses(req.course_id, covered,
                                                  direction=direction, graph=GRAPH)
            if set(c["teaches"]) & set(covered)
            and c["id"] not in learner["rejected_courses"]
            and c["id"] not in learner["completed_courses"]
        ]
        if not alternatives:
            return {
                "learner": learner, "path": before, "signal": req.signal,
                "swapped_to": None,
                "message": (f"{course['title']} is the only course in the catalog that "
                            f"teaches " + ", ".join(GRAPH.name(s) for s in covered)
                            + ". It stays in your path - but if you already know this "
                              "material, use 'Already know this' to skip it entirely."),
                "delta": {"courses_before": before["total_courses"],
                          "courses_after": before["total_courses"],
                          "hours_before": before["total_hours"],
                          "hours_after": before["total_hours"]},
            }
        if req.course_id not in learner["rejected_courses"]:
            learner["rejected_courses"].append(req.course_id)
        if req.signal == "too_easy":
            order = engine.LEVEL_NAMES
            idx = min(order.index(learner["experience_level"]) + 1, len(order) - 1)
            learner["experience_level"] = order[idx]
            message = (f"Swapped it out and raised your level to "
                       f"{learner['experience_level']} so later picks go deeper.")
        else:
            order = engine.LEVEL_NAMES
            idx = max(order.index(learner["experience_level"]) - 1, 0)
            learner["experience_level"] = order[idx]
            message = (f"Swapped it out and eased your level to "
                       f"{learner['experience_level']} so later picks are gentler.")

    log_event(learner, "feedback", f"{req.signal} on {req.course_id}")
    persist(learner)
    after = engine.replan(learner, graph=GRAPH)

    swapped_to = None
    if req.signal in {"too_hard", "too_easy"}:
        for entry in after["courses"]:
            if entry["course_id"] != req.course_id and set(entry["covers"]) & set(covered):
                swapped_to = entry
                break
        if swapped_to:
            message = (f"Replaced with {swapped_to['title']} ({swapped_to['provider']}, "
                       f"{swapped_to['level']}, {swapped_to['hours']}h). " + message)

    return {"learner": learner, "path": after, "message": message,
            "swapped_to": swapped_to, "signal": req.signal,
            "delta": {"courses_before": before["total_courses"],
                      "courses_after": after["total_courses"],
                      "hours_before": before["total_hours"],
                      "hours_after": after["total_hours"]}}


# --- grounded chat ----------------------------------------------------------
def path_context(learner: Dict[str, Any], path: Dict[str, Any]) -> str:
    lines = [
        f"Goal: {learner['goal']}",
        "Goal skills: " + ", ".join(GRAPH.name(t) for t in path["targets"]),
        f"Level: {learner['experience_level']}",
        f"Remaining: {path['total_courses']} courses, {path['total_hours']} hours",
        f"Progress: {path['progress']['percent']}% of required skills "
        f"({path['progress']['skills_done']}/{path['progress']['skills_total']})",
        "Milestones: " + ", ".join(f"{m['name']} ({len(m['course_ids'])} courses)"
                                   for m in path["milestones"]),
        "Next courses:",
    ]
    for entry in path["courses"][:8]:
        lines.append(f"  {entry['order'] + 1}. {entry['title']} ({entry['provider']}, "
                     f"{entry['hours']}h, {entry['level']}) teaches "
                     + ", ".join(entry["covers_named"]))
    if path["progress"]["completed_courses"]:
        lines.append("Completed: " + ", ".join(c["title"] for c in
                                               path["progress"]["completed_courses"]))
    return "\n".join(lines)


def canned_answer(message: str, learner: Dict[str, Any], path: Dict[str, Any]) -> str:
    """Deterministic Q&A over the path data. Used whenever Ollama is unavailable."""
    m = (message or "").lower()
    prog = path["progress"]
    nxt = path["next_action"]

    def course_named() -> Optional[Dict[str, Any]]:
        for entry in path["courses"]:
            title = entry["title"].lower()
            if title in m or entry["course_id"] in m:
                return entry
            words = [w for w in re.split(r"\W+", title) if len(w) > 5]
            if words and sum(1 for w in words if w in m) >= 2:
                return entry
        return None

    hit = course_named()
    if hit and any(k in m for k in ("why", "reason", "purpose", "need")):
        return engine.explain(hit["course_id"], path, graph=GRAPH)["summary"]

    if not path["courses"]:
        # nothing left to plan: answer from what was achieved instead
        return (f"You are done - every skill "
                + ", ".join(GRAPH.name(t) for t in path["targets"])
                + f" requires is in your profile. You completed "
                f"{prog['courses_completed']} courses and "
                f"{prog['hours_done']} hours to get here. Set a new goal and I will "
                "compute the gap from everything you now know.")

    if any(k in m for k in ("how long", "hours", "time", "duration", "finish", "weeks")):
        weeks = max(1, round(path["total_hours"] / 10))
        return (f"You have {path['total_hours']} hours left across "
                f"{path['total_courses']} courses. At about 10 hours a week that is "
                f"roughly {weeks} weeks. You have already logged "
                f"{prog['hours_done']} hours.")
    if any(k in m for k in ("next", "start", "begin", "first", "now")):
        if not nxt:
            return "Nothing left - you have covered every skill your goal requires."
        return (f"Start with {nxt['title']} ({nxt['provider']}, {nxt['hours']}h, "
                f"{nxt['level']}). It teaches " + ", ".join(nxt["covers_named"])
                + f", and it unblocks {len(nxt['unlocks'])} later course(s).")
    if any(k in m for k in ("skip", "already know", "know already")):
        return ("If you already know a course's material, hit 'Already know this' on "
                "its card. Those skills get marked as possessed and every course that "
                "only existed to reach them disappears from the path.")
    if any(k in m for k in ("progress", "how far", "percent", "done", "left", "remaining")):
        return (f"{prog['percent']}% of the required skills are covered "
                f"({prog['skills_done']} of {prog['skills_total']}). "
                f"{prog['courses_completed']} courses completed, "
                f"{path['total_courses']} to go ({path['total_hours']}h).")
    if any(k in m for k in ("milestone", "phase", "stage", "structure", "overview")):
        return "Your path has " + str(len(path["milestones"])) + " phases: " + "; ".join(
            f"{ms['name']} - {len(ms['course_ids'])} courses, {ms['hours']}h"
            for ms in path["milestones"])
    if any(k in m for k in ("goal", "target", "aiming", "outcome")):
        return ("Your goal skills are " + ", ".join(GRAPH.name(t) for t in path["targets"])
                + ". Everything in the path is a prerequisite of one of those, "
                  "computed by transitive closure over the skill graph.")
    if any(k in m for k in ("prereq", "prerequisite", "depend", "order", "why this order")):
        return ("The order is a topological sort of your gap subgraph: a course only "
                "appears once every skill it requires is either something you already "
                "have or something an earlier course in the path teaches.")
    if hit:
        return (f"{hit['title']} sits at position {hit['order'] + 1} of "
                f"{path['total_courses']} in the {hit['milestone']} phase. It teaches "
                + ", ".join(hit["covers_named"]) + f" over {hit['hours']} hours.")
    return (f"Here is where you stand: {prog['percent']}% of required skills covered, "
            f"{path['total_courses']} courses and {path['total_hours']} hours left, "
            f"aiming at " + ", ".join(GRAPH.name(t) for t in path["targets"]) + ". "
            "Ask me what to start next, how long it will take, or why a specific "
            "course is in your path.")


CHAT_SYSTEM = (
    "You are a learning coach. Answer ONLY from the learner's path data given to "
    "you. Never invent courses, hours or prerequisites. If the answer is not in the "
    "data, say so. Two or three sentences, plain text."
)


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    grounded = canned_answer(req.message, learner, path)

    if OLLAMA_STATE["reachable"]:
        prompt = (f"Learner's path data:\n{path_context(learner, path)}\n\n"
                  f"Learner asks: {req.message}\n\n"
                  f"A deterministic system already produced this correct answer:\n"
                  f"{grounded}\n\nRewrite it conversationally, adding nothing new.")
        text = ask_ollama(prompt, system=CHAT_SYSTEM, temperature=0.3)
        if text and 10 < len(text) < 1500:
            return {"reply": text.strip(), "source": "ollama", "grounded_answer": grounded}
    return {"reply": grounded, "source": "deterministic", "grounded_answer": grounded}


@app.get("/api/learner")
def get_learner() -> Dict[str, Any]:
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    return {"learner": learner, "path": path,
            "targets_named": [GRAPH.name(t) for t in learner["target_skills"]],
            "ollama": {"reachable": OLLAMA_STATE["reachable"],
                       "mode": "llm-assisted" if OLLAMA_STATE["reachable"]
                       else "deterministic"}}


# ---------------------------------------------------------------------------
# Graph analytics: questions only a prerequisite graph can answer
# ---------------------------------------------------------------------------
@app.get("/api/skip-impact/{course_id}")
def skip_impact(course_id: str) -> Dict[str, Any]:
    """What breaks if this course is skipped."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    try:
        return {"impact": engine.skip_impact(course_id, path, graph=GRAPH)}
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"{course_id} is not in the current path")


@app.get("/api/leverage")
def leverage() -> Dict[str, Any]:
    """Courses ranked by how much of the rest of the plan they unblock."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    return {"ranking": engine.leverage_ranking(path, graph=GRAPH)}


@app.get("/api/schedule")
def schedule() -> Dict[str, Any]:
    """Slot windows, slack and the critical chain."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    analysis = engine.slot_analysis(path, graph=GRAPH)
    return {"slots": analysis["courses"], "project": analysis["project"],
            "critical_path": engine.critical_path(path, graph=GRAPH)}


@app.get("/api/readiness/{course_id}")
def course_readiness(course_id: str) -> Dict[str, Any]:
    """How far this learner is from any course in the catalogue."""
    learner = active_learner(required=False)
    possessed = learner["possessed"] if learner else []
    level = learner["experience_level"] if learner else "beginner"
    try:
        return {"readiness": engine.readiness(course_id, possessed, level=level, graph=GRAPH)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown course {course_id}")


@app.get("/api/routes")
def routes() -> Dict[str, Any]:
    """Several equally valid routes to the same goal."""
    learner = active_learner()
    return {"current_strategy": learner.get("strategy", "balanced"),
            "strategies": engine.STRATEGIES,
            **engine.alternative_routes(learner["target_skills"], learner["possessed"],
                                        level=learner["experience_level"],
                                        interests=learner.get("interests", []),
                                        graph=GRAPH)}


class StrategyRequest(BaseModel):
    strategy: str


@app.post("/api/strategy")
def set_strategy(req: StrategyRequest) -> Dict[str, Any]:
    """Switch the learner onto one of the alternative routes."""
    learner = active_learner()
    if req.strategy not in engine.STRATEGIES:
        raise HTTPException(status_code=400, detail=f"unknown strategy {req.strategy}")
    before = engine.replan(learner, graph=GRAPH)
    learner["strategy"] = req.strategy
    log_event(learner, "strategy", f"switched to {req.strategy}")
    persist(learner)
    after = engine.replan(learner, graph=GRAPH)
    return {"learner": learner, "path": after,
            "message": (f"Switched to the {engine.STRATEGIES[req.strategy]['label']} route: "
                        f"{after['total_courses']} courses, {after['total_hours']}h "
                        f"({after['total_hours'] - before['total_hours']:+d}h)."),
            "delta": {"courses_before": before["total_courses"],
                      "courses_after": after["total_courses"],
                      "hours_before": before["total_hours"],
                      "hours_after": after["total_hours"]}}


class CompareRequest(BaseModel):
    goal_a: str = ""
    goal_b: str
    targets_a: List[str] = []
    targets_b: List[str] = []


@app.post("/api/compare")
def compare(req: CompareRequest) -> Dict[str, Any]:
    """Two goals side by side, plus the marginal cost of wanting both."""
    learner = active_learner(required=False)
    possessed = learner["possessed"] if learner else []
    level = learner["experience_level"] if learner else "beginner"

    a = req.targets_a or (learner["target_skills"] if learner and not req.goal_a
                          else engine.resolve_goal(req.goal_a, graph=GRAPH)["targets"])
    b = req.targets_b or engine.resolve_goal(req.goal_b, graph=GRAPH)["targets"]
    if not a or not b:
        which = "first" if not a else "second"
        return {"status": "needs_clarification",
                "message": f"I could not read the {which} goal well enough to compare it.",
                "candidates": engine.resolve_goal(req.goal_b if not b else req.goal_a,
                                                  graph=GRAPH)["candidates"]}

    label_a = ", ".join(GRAPH.name(t) for t in a)
    label_b = ", ".join(GRAPH.name(t) for t in b)
    return {"status": "ok",
            **engine.compare_goals(a, b, possessed, level=level,
                                   interests=learner.get("interests", []) if learner else [],
                                   labels=(label_a, label_b), graph=GRAPH)}


@app.get("/api/graph/health")
def graph_health() -> Dict[str, Any]:
    """Where the curriculum itself is thin: the maintainer's view."""
    return engine.graph_health(graph=GRAPH)


# ---------------------------------------------------------------------------
# Learner modelling: pace, budgets, refreshers, history import
# ---------------------------------------------------------------------------
@app.get("/api/forecast")
def forecast(hours_per_week: float = 10.0,
             weeks_available: Optional[float] = None) -> Dict[str, Any]:
    """Re-forecast from observed pace, and check it against a real time budget."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    velocity = engine.velocity_report(learner, path, hours_per_week=hours_per_week,
                                      graph=GRAPH)
    budget = engine.budget_forecast(path, hours_per_week=hours_per_week,
                                    weeks_available=weeks_available,
                                    pace_factor=velocity["pace_factor"], graph=GRAPH)
    return {"velocity": velocity, "budget": budget}


@app.get("/api/refreshers")
def refreshers(stale_days: int = 45) -> Dict[str, Any]:
    """Skills learned a while ago that the next courses are about to lean on."""
    learner = active_learner()
    path = engine.replan(learner, graph=GRAPH)
    return {"prompts": engine.refresher_prompts(learner, path, stale_days=stale_days,
                                                graph=GRAPH)}


class HistoryRequest(BaseModel):
    text: str
    apply: bool = False


@app.post("/api/import-history")
def import_history(req: HistoryRequest) -> Dict[str, Any]:
    """Parse a pasted list of completed courses, and optionally apply it."""
    parsed = engine.parse_history(req.text, graph=GRAPH)
    if not req.apply:
        return {"parsed": parsed, "applied": False}

    learner = active_learner()
    before = engine.replan(learner, graph=GRAPH)
    now = time.time()
    for cid in parsed["course_ids"]:
        if cid not in learner["completed_courses"]:
            learner["completed_courses"].append(cid)
            learner.setdefault("completion_log", []).append({
                "course_id": cid, "at": now,
                "estimated_hours": GRAPH.courses[cid]["hours"], "actual_hours": None,
                "imported": True})
    acquired = learner.setdefault("skill_acquired_at", {})
    for sid in parsed["skills_direct"]:
        acquired.setdefault(sid, now)
    learner["known_skills"] = sorted(set(learner["known_skills"])
                                     | set(parsed["skills_direct"]))
    learner["possessed"] = sorted(GRAPH.expand_possessed(
        set(learner["possessed"]) | set(parsed["skills_with_prerequisites"])))
    log_event(learner, "import",
              f"imported {len(parsed['course_ids'])} courses, "
              f"{len(parsed['skills_direct'])} skills")
    persist(learner)
    after = engine.replan(learner, graph=GRAPH)
    return {"parsed": parsed, "applied": True, "learner": learner, "path": after,
            "delta": {"courses_before": before["total_courses"],
                      "courses_after": after["total_courses"],
                      "hours_before": before["total_hours"],
                      "hours_after": after["total_hours"]}}


@app.get("/api/vocab-gaps")
def vocab_gaps(limit: int = 20) -> Dict[str, Any]:
    """What learners asked for that the graph could not name.

    This is the content backlog, generated by use rather than guesswork: terms
    ranked by how often they came up, plus the goals that had to be clarified
    or refused outright.
    """
    log: List[Dict[str, Any]] = []
    if os.path.exists(GAP_FILE):
        try:
            with open(GAP_FILE, "r", encoding="utf-8") as fh:
                log = json.load(fh)
        except (json.JSONDecodeError, OSError):
            log = []

    counts: Dict[str, int] = {}
    for entry in log:
        for term in entry.get("unknown_terms", []):
            counts[term] = counts.get(term, 0) + 1

    return {
        "total_logged": len(log),
        "unresolved_goals": sum(1 for e in log if not e.get("resolved")),
        "out_of_domain_goals": sum(1 for e in log if e.get("out_of_domain")),
        "clarified_goals": sum(1 for e in log if e.get("learner_chose")),
        "top_unknown_terms": [{"term": t, "count": c} for t, c in
                              sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]],
        "recent": list(reversed(log[-limit:])),
    }


@app.post("/api/reset")
def reset() -> Dict[str, Any]:
    """Clear the active learner. Handy when re-recording a demo."""
    state = load_state()
    state["active"] = None
    save_state(state)
    return {"status": "reset"}


@app.get("/api/graph/skill/{skill_id}")
def skill_detail(skill_id: str) -> Dict[str, Any]:
    """Inspect one node of the graph: prerequisites, dependents, courses.

    When a learner exists the node is annotated with their standing - possessed,
    still in the gap, or irrelevant to this goal - so the UI can let them walk
    the graph and see where they are in it.
    """
    if skill_id not in GRAPH.skills:
        raise HTTPException(status_code=404, detail=f"unknown skill {skill_id}")
    skill = GRAPH.skills[skill_id]
    learner = active_learner(required=False)

    possessed: set = set()
    gap: set = set()
    in_path_course = None
    if learner is not None:
        path = engine.replan(learner, graph=GRAPH)
        possessed = set(path["possessed"])
        gap = set(path["gap_skills"])
        for entry in path["courses"]:
            if skill_id in entry["teaches"]:
                in_path_course = {"course_id": entry["course_id"],
                                  "title": entry["title"],
                                  "order": entry["order"] + 1,
                                  "milestone": entry["milestone"]}
                break

    def annotate(sid: str) -> Dict[str, Any]:
        return {"id": sid, "name": GRAPH.name(sid),
                "possessed": sid in possessed, "in_gap": sid in gap}

    return {
        "id": skill_id,
        "name": skill["name"],
        "description": skill["description"],
        "depth": GRAPH.depth(skill_id),
        "category": GRAPH.category(skill_id),
        "requires": [annotate(r) for r in skill["requires"]],
        "unlocks": [annotate(d) for d in sorted(GRAPH.dependents[skill_id])],
        "taught_by": [{"id": c, "title": GRAPH.courses[c]["title"],
                       "provider": GRAPH.courses[c]["provider"],
                       "hours": GRAPH.courses[c]["hours"],
                       "level": GRAPH.courses[c]["level"],
                       "in_your_path": in_path_course is not None
                       and in_path_course["course_id"] == c}
                      for c in GRAPH.courses_teaching[skill_id]],
        "learner": None if learner is None else {
            "possessed": skill_id in possessed,
            "in_gap": skill_id in gap,
            "scheduled": in_path_course,
        },
    }


@app.exception_handler(engine.GraphError)
def graph_error_handler(_request, exc: engine.GraphError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"graph error: {exc}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
