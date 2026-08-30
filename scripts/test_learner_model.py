"""
Learner-modelling suite: graded self-assessment, observed pace, time budgets
and staleness-aware refreshers.

None of this is machine learning - it is arithmetic over what the learner has
actually done, which is both more honest and easier to explain than a model.

Run:  python scripts/test_learner_model.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402

G = engine.load_graph()
FAILURES = []
NOW = time.time()
DAY = 86400.0


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


print("== split_self_assessment: 'heard of' is not 'I know it'")
graded = engine.split_self_assessment({
    "python": "can_teach", "pandas": "can_use", "statistics": "heard_of",
    "pytorch": "heard_of", "not-a-skill": "can_use",
}, graph=G)
print(f"  possessed: {len(graded['possessed'])} skills, refresh: {graded['refresh']}, "
      f"confident: {graded['confident']}")
check("can_use and can_teach count as possessed",
      "pandas" in graded["possessed"] and "python" in graded["possessed"])
check("heard_of does NOT count as possessed",
      "statistics" not in graded["possessed"] and "pytorch" not in graded["possessed"])
check("heard_of becomes a refresher hint",
      set(graded["refresh"]) == {"statistics", "pytorch"})
check("possession is closed downward", "numpy" in graded["possessed"],
      "pandas implies numpy")
check("unknown ids are ignored", "not-a-skill" not in graded["possessed"])

print("\n== refresher-biased selection picks the shorter course")
normal = engine.select_courses(["python"], graph=G)[0]["course"]
refresh = engine.select_courses(["python"], refresh=["python"], graph=G)[0]["course"]
print(f"  cold: {normal['title'][:38]:40} ({normal['hours']}h)")
print(f"  refresher: {refresh['title'][:38]:40} ({refresh['hours']}h)")
check("a refresher is no longer than the cold-start course",
      refresh["hours"] <= normal["hours"])
path_refresh = engine.build_path(["rag"], ["python", "python-oop", "sql"],
                                 level="intermediate",
                                 refresh=["pandas", "numpy", "statistics"], graph=G)
flagged = [c for c in path_refresh["courses"] if c["refresher"]]
check("refresher courses are flagged in the path", len(flagged) == 3,
      ", ".join(c["title"][:26] for c in flagged))

print("\n== velocity_report: forecast from the learner's own pace")
PATH = engine.build_path(["rag"], ["python", "python-oop", "sql"],
                         level="intermediate", graph=G)
slow = {"completion_log": [
    {"course_id": "c012", "estimated_hours": 10, "actual_hours": 15},
    {"course_id": "c013", "estimated_hours": 18, "actual_hours": 26},
]}
fast = {"completion_log": [
    {"course_id": "c012", "estimated_hours": 10, "actual_hours": 7},
]}
none = {"completion_log": [{"course_id": "c012", "estimated_hours": 10}]}
for name, learner in (("slow", slow), ("fast", fast), ("untimed", none)):
    v = engine.velocity_report(learner, PATH, hours_per_week=10, graph=G)
    print(f"  {name:8} pace {v['pace_factor']}x  {v['remaining_hours_listed']}h listed -> "
          f"{v['remaining_hours_adjusted']}h  ({v['weeks_adjusted']} weeks)")
v_slow = engine.velocity_report(slow, PATH, hours_per_week=10, graph=G)
v_fast = engine.velocity_report(fast, PATH, hours_per_week=10, graph=G)
v_none = engine.velocity_report(none, PATH, hours_per_week=10, graph=G)
check("a slow learner gets a longer forecast",
      v_slow["remaining_hours_adjusted"] > v_slow["remaining_hours_listed"],
      f"{v_slow['pace_factor']}x")
check("a fast learner gets a shorter forecast",
      v_fast["remaining_hours_adjusted"] < v_fast["remaining_hours_listed"])
check("no timings means no adjustment", v_none["pace_factor"] == 1.0)
check("untimed learners are told why", "No timings logged" in v_none["note"])
check("weeks follow hours per week",
      round(v_slow["remaining_hours_adjusted"] / 10, 1) == v_slow["weeks_adjusted"])

print("\n== budget_forecast: what fits in the time you actually have")
full = engine.budget_forecast(PATH, hours_per_week=10, graph=G)
tight = engine.budget_forecast(PATH, hours_per_week=5, weeks_available=12, graph=G)
roomy = engine.budget_forecast(PATH, hours_per_week=20, weeks_available=60, graph=G)
print(f"  no budget: {full['sentence']}")
print(f"  tight:     {tight['sentence'][:160]}")
print(f"  roomy:     {roomy['sentence'][:120]}")
check("a tight budget does not reach the end",
      tight["reachable_courses"] < tight["total_courses"])
check("a roomy budget fits everything",
      roomy["reachable_courses"] == roomy["total_courses"])
check("the milestone at risk is named",
      tight["first_milestone_at_risk"] is not None)
check("schedule weeks increase monotonically",
      all(a["finish_week"] <= b["finish_week"]
          for a, b in zip(tight["schedule"], tight["schedule"][1:])))
check("a slower pace pushes the finish out",
      engine.budget_forecast(PATH, hours_per_week=10, pace_factor=1.5,
                             graph=G)["total_weeks"] > full["total_weeks"])

print("\n== refresher_prompts: stale skills the next courses lean on")
# the first course is a foundation course with no prerequisites, so pick the
# earliest one that actually leans on something
upcoming = [c for c in PATH["courses"][:3] if c["requires"]]
learner = {"skill_acquired_at": {r: NOW - 90 * DAY
                                 for c in upcoming for r in c["requires"]}}
stale = engine.refresher_prompts(learner, PATH, stale_days=45, now=NOW, graph=G)
for item in stale[:3]:
    print(f"  {item['sentence'][:120]}")
check("stale prerequisites are surfaced", len(stale) > 0,
      f"{len(stale)} prompt(s)")
fresh_learner = {"skill_acquired_at": {r: NOW - 3 * DAY
                                       for c in upcoming for r in c["requires"]}}
check("recently learned skills are left alone",
      engine.refresher_prompts(fresh_learner, PATH, stale_days=45, now=NOW, graph=G) == [])
check("no history means no prompts",
      engine.refresher_prompts({}, PATH, now=NOW, graph=G) == [])
if stale:
    check("each prompt names the course that needs it",
          all(i["needed_by_title"] for i in stale))
    check("each prompt suggests the shortest course for the skill",
          all(i["suggested_course"] is None or
              i["suggested_course"]["hours"] == min(
                  G.courses[c]["hours"] for c in G.courses_teaching[i["skill"]])
              for i in stale))
check("prompts only look at the next few courses",
      all(i["position"] <= 3 for i in stale))

print("\n== parse_history: import what you have already done")
history = engine.parse_history("""Python for Everybody
NumPy: Numerical Computing in Python
- SQL and joins
docker
Advanced underwater basket weaving""", graph=G)
print(f"  {history['sentence'][:180]}")
for m in history["matched_courses"]:
    print(f"    course: {m['title'][:44]:46} -> {m['teaches_named']}")
for m in history["matched_skills"]:
    print(f"    skill : {m['line'][:44]:46} -> {m['skills_named']} ({m['evidence']})")
check("exact course titles are matched", len(history["matched_courses"]) >= 2)
check("loose skill mentions are matched", len(history["matched_skills"]) >= 2)
check("nonsense is reported, not credited", len(history["unmatched"]) == 1,
      history["unmatched"][0]["line"] if history["unmatched"] else "")
check("prerequisites are credited too",
      len(history["skills_with_prerequisites"]) > len(history["skills_direct"]),
      f"{len(history['skills_direct'])} direct -> {len(history['skills_with_prerequisites'])} total")
check("matched course ids come back for marking complete",
      all(c in G.courses for c in history["course_ids"]))

fuzzy = engine.parse_history("Python for Everybdy", graph=G)
check("a typo in a course title still matches",
      len(fuzzy["matched_courses"]) == 1,
      fuzzy["matched_courses"][0]["title"] if fuzzy["matched_courses"] else "no match")
check("empty input is handled", engine.parse_history("", graph=G)["lines_read"] == 0)

imported = engine.build_path(["rag"], history["skills_with_prerequisites"],
                             level="intermediate", graph=G)
cold = engine.build_path(["rag"], [], level="intermediate", graph=G)
print(f"  path after import: {imported['total_courses']} courses vs "
      f"{cold['total_courses']} from scratch")
check("importing history shortens the path",
      imported["total_courses"] < cold["total_courses"])

print("\n" + ("ALL LEARNER-MODEL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
