"""
Performance benchmark for the engine.

The graph is meant to grow, so this exists to catch the growth that would make
it feel slow. Every path in the app recomputes from scratch on every request -
there is no cache to invalidate - so these numbers are the budget.

Run:  python scripts/benchmark.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402

BUDGET_MS = {          # what "still feels instant" means for each operation
    "load_graph": 250,
    "resolve_goal": 120,
    "build_path": 60,
    "replan": 80,
    "explain": 40,
    "skip_impact": 40,
    "leverage_ranking": 40,
    "slot_analysis": 40,
    "readiness": 80,
    "alternative_routes": 350,
    "compare_goals": 250,
    "graph_health": 60,
    "parse_history": 200,
}
FAILURES = []


def bench(label, fn, runs=20):
    fn()                                   # warm caches
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    med, worst = statistics.median(times), max(times)
    budget = BUDGET_MS[label]
    ok = med <= budget
    print(f"  [{'OK  ' if ok else 'SLOW'}] {label:20} median {med:7.2f} ms   "
          f"p100 {worst:7.2f} ms   budget {budget} ms")
    if not ok:
        FAILURES.append(f"{label}: {med:.1f}ms > {budget}ms")
    return med


t0 = time.perf_counter()
G = engine.load_graph()
load_ms = (time.perf_counter() - t0) * 1000
stats = engine.graph_stats(G)
print(f"graph: {stats['skills']} skills, {stats['edges']} edges, {stats['courses']} courses, "
      f"max depth {stats['max_depth']}, {len(stats['categories'])} categories")
print(f"cold load: {load_ms:.1f} ms\n")

POSSESSED = ["python", "python-oop", "sql"]
LEARNER = {"target_skills": ["rag"], "possessed": POSSESSED, "experience_level": "intermediate",
           "interests": ["llm"], "known_skills": POSSESSED, "completed_courses": [],
           "rejected_courses": []}
PATH = engine.build_path(["rag"], POSSESSED, level="intermediate", graph=G)
print(f"reference path: {PATH['total_courses']} courses, {PATH['total_hours']}h, "
      f"{len(PATH['gap_skills'])} skills\n")

bench("load_graph", lambda: engine.load_graph(), runs=5)
bench("resolve_goal", lambda: engine.resolve_goal(
    "I want to build agents that use retrieval and tools", graph=G))
bench("build_path", lambda: engine.build_path(["rag"], POSSESSED, level="intermediate", graph=G))
bench("replan", lambda: engine.replan(LEARNER, graph=G))
bench("explain", lambda: engine.explain(PATH["courses"][-1]["course_id"], PATH, graph=G))
bench("skip_impact", lambda: engine.skip_impact(PATH["courses"][1]["course_id"], PATH, graph=G))
bench("leverage_ranking", lambda: engine.leverage_ranking(PATH, graph=G))
bench("slot_analysis", lambda: engine.slot_analysis(PATH, graph=G))
bench("readiness", lambda: engine.readiness(PATH["courses"][-1]["course_id"], POSSESSED, graph=G))
bench("alternative_routes", lambda: engine.alternative_routes(
    ["rag"], POSSESSED, level="intermediate", graph=G), runs=8)
bench("compare_goals", lambda: engine.compare_goals(
    ["rag"], ["llm-finetuning"], POSSESSED, level="intermediate", graph=G), runs=8)
bench("graph_health", lambda: engine.graph_health(graph=G))
bench("parse_history", lambda: engine.parse_history(
    "Python for Everybody\nDocker for Machine Learning\nkubernetes\nreinforcement learning", graph=G))

print("\n== worst case: the deepest goal from a standing start")
t0 = time.perf_counter()
deep = engine.build_path(["llm-finetuning", "ml-system-design", "ai-governance"], [],
                         level="beginner", graph=G)
deep_ms = (time.perf_counter() - t0) * 1000
print(f"  {deep['total_courses']} courses, {deep['total_hours']}h, "
      f"{len(deep['gap_skills'])} skills in {deep_ms:.1f} ms")
if deep_ms > 150:
    FAILURES.append(f"worst-case build_path: {deep_ms:.1f}ms")
print(f"  [{'OK  ' if deep_ms <= 150 else 'SLOW'}] worst-case build under 150 ms")

print("\n" + ("ALL WITHIN BUDGET" if not FAILURES else f"OVER BUDGET: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
