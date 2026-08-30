"""
Graph-analytics suite: the questions that are only answerable because the
curriculum is a graph.

  skip_impact      what breaks if you skip this course
  leverage_ranking what finishing it opens up
  slot_analysis    where in the order it is actually allowed to sit
  critical_path    the chain with no slack
  readiness        how far you are from any course in the catalogue
  graph_health     where the curriculum itself is thin

Run:  python scripts/test_analytics.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402

G = engine.load_graph()
FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


PATH = engine.build_path(["rag"], ["python", "python-oop", "sql"],
                         level="intermediate", graph=G)
print(f"reference path: {PATH['total_courses']} courses, {PATH['total_hours']}h\n")

print("== skip_impact: what breaks if you skip it")
early = PATH["courses"][1]
late = PATH["courses"][-1]
for entry in (early, late):
    imp = engine.skip_impact(entry["course_id"], PATH, graph=G)
    print(f"  {imp['title'][:52]:54} [{imp['verdict']}]")
    print(f"    {imp['sentence'][:150]}")
check("early course is blocking or goal-critical",
      engine.skip_impact(early["course_id"], PATH, graph=G)["verdict"] in ("blocking", "goal-critical"))
last = engine.skip_impact(late["course_id"], PATH, graph=G)
check("the final course puts the goal at risk",
      last["verdict"] == "goal-critical" and last["targets_at_risk"] == ["rag"],
      str(last["targets_at_risk_named"]))
check("skipping a mid course strands later ones",
      len(engine.skip_impact(early["course_id"], PATH, graph=G)["blocked_courses"]) > 0)
check("impact is grounded in edges",
      "prerequisite edges" in last["grounded_in"])

print("\n== leverage_ranking: what a course unblocks")
lev = engine.leverage_ranking(PATH, graph=G)
for row in lev[:3]:
    print(f"  #{row['order']:<3} {row['title'][:44]:46} unlocks {row['unlocks_courses']:>2} courses "
          f"({row['unlocked_hours']:>3}h), {row['unlocks_skills']} skills")
print(f"  ...lowest: {lev[-1]['title'][:44]:46} unlocks {lev[-1]['unlocks_courses']} courses")
check("ranking covers every course", len(lev) == PATH["total_courses"])
check("the top course unblocks more than the last",
      lev[0]["unlocks_courses"] >= lev[-1]["unlocks_courses"])
check("a foundational course outranks the final one",
      lev[0]["order"] < PATH["total_courses"], f"top is step {lev[0]['order']}")

print("\n== slot_analysis: why position N and not earlier")
analysis = engine.slot_analysis(PATH, graph=G)
slots, project = analysis["courses"], analysis["project"]
mid = PATH["courses"][len(PATH["courses"]) // 2]
s = slots[mid["course_id"]]
print(f"  {mid['title'][:52]:54} position {s['position']}, "
      f"window {s['earliest']}-{s['latest']}, slack {s['slack']}")
print(f"    {s['note'][:150]}")
check("every course has a slot window", len(slots) == PATH["total_courses"])
check("no course is placed before its earliest slot",
      all(v["position"] >= v["earliest"] for v in slots.values()))
check("no course is placed after its latest slot",
      all(v["position"] <= v["latest"] for v in slots.values()))
check("slack is never negative", all(v["slack"] >= 0 for v in slots.values()))
check("a course with dependencies cannot start at step 1",
      all(v["earliest"] > 1 for v in slots.values() if v["blocked_by"]))

print("\n== critical_path")
cp = engine.critical_path(PATH, graph=G)
print(f"  {cp['sentence']}")
print("  chain: " + " -> ".join(c["title"][:22] for c in cp["chain"][:5]) + " ...")
check("critical path is non-empty", cp["length"] > 0)
check("critical + flexible = all courses",
      cp["length"] + cp["flexible_courses"] == PATH["total_courses"])
check("critical hours do not exceed total", cp["hours"] <= PATH["total_hours"])
check("parallel duration is no longer than sequential",
      cp["hours_if_parallel"] <= cp["hours_sequential"],
      f"{cp['hours_if_parallel']}h vs {cp['hours_sequential']}h")
check("courses with slack are the flexible ones",
      all(slots[c["course_id"]]["hours_slack"] > 0
          for c in PATH["courses"] if not slots[c["course_id"]]["critical"]))

print("\n== readiness for an arbitrary course")
hard = "c060"
for cid, possessed in [(hard, ["python"]), (hard, ["embeddings", "python", "rest-apis"])]:
    if cid not in G.courses:
        cid = PATH["courses"][-1]["course_id"]
    r = engine.readiness(cid, possessed, graph=G)
    print(f"  {r['title'][:46]:48} ready={r['ready']}  {r['sentence'][:96]}")
target_course = PATH["courses"][-1]["course_id"]
novice = engine.readiness(target_course, ["python"], graph=G)
expert = engine.readiness(target_course, PATH["gap_skills"] + PATH["possessed"], graph=G)
check("a novice is not ready for the last course", not novice["ready"])
check("prep courses are suggested", len(novice["prep_courses"]) > 0,
      f"{len(novice['prep_courses'])} courses, {novice['prep_hours']}h")
check("someone with every skill is ready", expert["ready"], expert["sentence"][:60])
check("the course itself is never its own prep",
      target_course not in [c["course_id"] for c in novice["prep_courses"]])

print("\n== graph_health")
h = engine.graph_health(graph=G)
print(f"  {h['skills']} skills / {h['courses']} courses / {h['edges']} edges, "
      f"max depth {h['max_depth']}, {h['courses_per_skill']} courses per skill")
print(f"  fragile (one teaching course only): {len(h['single_source_skills'])}")
print("  highest fan-out: " + ", ".join(
    f"{x['name']} ({x['unlocks']})" for x in h["highest_fan_out"][:3]))
check("no uncovered skills", h["uncovered_skills"] == [])
check("roots exist", len(h["roots"]) > 0, f"{len(h['roots'])} entry points")
check("fragile skills are reported", isinstance(h["single_source_skills"], list),
      f"{len(h['single_source_skills'])} of {h['skills']}")
check("every category is populated", all(v > 0 for v in h["skills_per_category"].values()))

print("\n== alternative_routes: same skills, different courses")
alt = engine.alternative_routes(["rag"], ["python", "python-oop", "sql"],
                                level="intermediate", graph=G)
print(f"  {alt['sentence'][:150]}")
for r in alt["routes"]:
    print(f"  {r['label']:12} {r['courses']:>3}c {r['hours']:>4}h ({r['hours_vs_baseline']:+4}h)"
          f"  free:{r['free_courses']} projects:{r['project_courses']}")
by_name = {r["strategy"]: r for r in alt["routes"]}
check("every strategy produced a route", len(alt["routes"]) == len(engine.STRATEGIES))
check("all routes cover the same skills",
      len({tuple(sorted(r["skills_covered"])) for r in alt["routes"]}) == 1,
      f"{len(alt['routes'][0]['skills_covered'])} skills each")
check("'Shortest' is not longer than balanced",
      by_name["fastest"]["hours"] <= by_name["balanced"]["hours"],
      f"{by_name['fastest']['hours']}h vs {by_name['balanced']['hours']}h")
check("'Free first' finds more free courses",
      by_name["free_first"]["free_courses"] > by_name["balanced"]["free_courses"],
      f"{by_name['free_first']['free_courses']} vs {by_name['balanced']['free_courses']}")
check("'Project-led' finds more project courses",
      by_name["hands_on"]["project_courses"] >= by_name["balanced"]["project_courses"])
check("'Thorough' uses longer courses on average",
      by_name["thorough"]["avg_course_hours"] >= by_name["balanced"]["avg_course_hours"])
check("routes genuinely differ",
      any(r["differs_from_baseline"] for r in alt["routes"][1:]))
check("choice points are reported honestly",
      0 < alt["choice_points"] < alt["skills_required"],
      f"{alt['choice_points']} of {alt['skills_required']} skills have a real choice")

print("\n== compare_goals: overlap and marginal cost")
cmp = engine.compare_goals(["rag"], ["llm-finetuning"], ["python", "python-oop", "sql"],
                           level="intermediate", labels=("RAG", "Fine-tuning"), graph=G)
print(f"  {cmp['sentence'][:190]}")
check("overlap is detected", len(cmp["shared_courses"]) > 0, f"{cmp['overlap_percent']}%")
check("each goal keeps something unique",
      len(cmp["only_a"]) > 0 and len(cmp["only_b"]) > 0)
check("doing both is cheaper than doing each separately",
      cmp["both"]["hours"] < cmp["marginal"]["separately_hours"],
      f"{cmp['both']['hours']}h vs {cmp['marginal']['separately_hours']}h")
check("marginal cost is less than starting the second goal cold",
      cmp["marginal"]["hours"] < cmp["goal_b"]["hours"],
      f"+{cmp['marginal']['hours']}h vs {cmp['goal_b']['hours']}h")
check("both-path covers at least as many skills as either alone",
      cmp["both"]["skills"] >= max(cmp["goal_a"]["skills"], cmp["goal_b"]["skills"]))

unrelated = engine.compare_goals(["rag"], ["time-series"], ["python"],
                                 level="beginner", graph=G)
print(f"  unrelated goals overlap: {unrelated['overlap_percent']}% "
      f"(vs {cmp['overlap_percent']}% for two LLM goals)")
check("unrelated goals overlap less than related ones",
      unrelated["overlap_percent"] < cmp["overlap_percent"])

print("\n== analytics are deterministic")
check("skip_impact stable",
      engine.skip_impact(early["course_id"], PATH, graph=G)
      == engine.skip_impact(early["course_id"], PATH, graph=G))
check("slot_analysis stable", engine.slot_analysis(PATH, graph=G)["courses"] == slots)

print("\n" + ("ALL ANALYTICS CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
