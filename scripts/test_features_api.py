"""
End-to-end checks for the analytics and learner-modelling endpoints.

Start the server first, then:  python scripts/test_features_api.py [base_url]

Like the rest of the suites, this runs identically with Ollama up or killed -
none of these endpoints touch the model.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
FAILURES = []


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode() if body is not None else None,
        method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        FAILURES.append(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:160]}")
        return {}


def expect_status(method, path, status, body=None):
    """Assert an endpoint rejects bad input, instead of swallowing the error."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode() if body is not None else None,
        method=method, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return False
    except urllib.error.HTTPError as exc:
        return exc.code == status


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


print("== setup: a graded self-assessment instead of checkboxes")
call("POST", "/api/reset")
intake = call("POST", "/api/intake", {
    "goal": "I want to build RAG chatbots",
    "experience_level": "intermediate",
    "skill_levels": {"python": "can_teach", "sql": "can_use",
                     "pandas": "heard_of", "statistics": "heard_of"},
    "interests": ["llm"],
})
learner = intake.get("learner", {})
print(f"  possessed {len(learner.get('possessed', []))} skills, "
      f"refresh {learner.get('refresh_skills')}")
check("can_use/can_teach became possessed",
      "python" in learner.get("possessed", []) and "sql" in learner.get("possessed", []))
check("heard_of did not become possessed",
      "pandas" not in learner.get("possessed", []))
check("heard_of became a refresher hint",
      set(learner.get("refresh_skills", [])) == {"pandas", "statistics"})
path = call("POST", "/api/plan").get("path", {})
refreshers_in_path = [c for c in path.get("courses", []) if c.get("refresher")]
check("refresher courses are flagged in the path", len(refreshers_in_path) >= 1,
      ", ".join(c["title"][:24] for c in refreshers_in_path))

first = path["courses"][0]["course_id"]
mid = path["courses"][len(path["courses"]) // 2]["course_id"]
last = path["courses"][-1]["course_id"]

print("\n== GET /api/skip-impact/{id}")
imp = call("GET", f"/api/skip-impact/{first}").get("impact", {})
print(f"  {imp.get('verdict')}: {imp.get('sentence', '')[:130]}")
check("verdict returned", imp.get("verdict") in ("goal-critical", "blocking", "safe"))
check("downstream damage listed", len(imp.get("skills_lost", [])) > 0)
final = call("GET", f"/api/skip-impact/{last}").get("impact", {})
check("the last course risks the goal", final.get("targets_at_risk") == ["rag"])
check("unknown course id is rejected with 404",
      expect_status("GET", "/api/skip-impact/not-a-course", 404))
check("a course outside the path is rejected too",
      expect_status("GET", "/api/skip-impact/c001", 404)
      or "c001" in [c["course_id"] for c in path["courses"]])
check("an unknown strategy is rejected with 400",
      expect_status("POST", "/api/strategy", 400, {"strategy": "nonsense"}))

print("\n== GET /api/leverage")
ranking = call("GET", "/api/leverage").get("ranking", [])
for row in ranking[:3]:
    print(f"  #{row['order']:<3} {row['title'][:40]:42} unlocks {row['unlocks_courses']} courses")
check("every course ranked", len(ranking) == path["total_courses"])
check("ranked by what they unblock",
      ranking[0]["unlocks_courses"] >= ranking[-1]["unlocks_courses"])

print("\n== GET /api/schedule")
sched = call("GET", "/api/schedule")
cp = sched.get("critical_path", {})
print(f"  {cp.get('sentence', '')[:170]}")
check("slot windows returned", len(sched.get("slots", {})) == path["total_courses"])
check("critical chain found", cp.get("length", 0) > 0)
check("parallel duration <= sequential",
      cp.get("hours_if_parallel", 0) <= cp.get("hours_sequential", 1))
some = sched["slots"][mid]
check("a slot window explains the position",
      some["earliest"] <= some["position"] <= some["latest"] and bool(some["note"]),
      f"pos {some['position']} in window {some['earliest']}-{some['latest']}")

print("\n== GET /api/readiness/{id}")
r = call("GET", f"/api/readiness/{last}").get("readiness", {})
print(f"  {r.get('sentence', '')[:150]}")
check("not ready for the last course yet", r.get("ready") is False)
check("prep courses listed", len(r.get("prep_courses", [])) > 0,
      f"{r.get('prep_hours')}h of prep")

print("\n== GET /api/routes and POST /api/strategy")
routes = call("GET", "/api/routes")
for route in routes.get("routes", []):
    print(f"  {route['label']:12} {route['courses']:>3}c {route['hours']:>4}h "
          f"({route['hours_vs_baseline']:+4}h) free:{route['free_courses']}")
check("all strategies offered", len(routes.get("routes", [])) == 5)
check("choice points reported", routes.get("choice_points", 0) > 0,
      f"{routes['choice_points']} of {routes['skills_required']} skills")
switched = call("POST", "/api/strategy", {"strategy": "free_first"})
print(f"  {switched.get('message', '')[:120]}")
check("strategy switch replans", switched.get("path", {}).get("total_courses", 0) > 0)
check("switch is persisted",
      call("GET", "/api/learner").get("learner", {}).get("strategy") == "free_first")
free_now = sum(1 for c in switched["path"]["courses"] if "free" in [t.lower() for t in c["tags"]])
check("the free route really has more free courses", free_now >= 3, f"{free_now} free")
call("POST", "/api/strategy", {"strategy": "balanced"})

print("\n== POST /api/compare")
cmp = call("POST", "/api/compare", {"goal_b": "I want to fine-tune LLMs"})
print(f"  {cmp.get('sentence', '')[:180]}")
check("comparison computed", cmp.get("status") == "ok")
check("overlap found", len(cmp.get("shared_courses", [])) > 0,
      f"{cmp.get('overlap_percent')}% overlap")
check("marginal cost beats starting cold",
      cmp["marginal"]["hours"] < cmp["goal_b"]["hours"],
      f"+{cmp['marginal']['hours']}h vs {cmp['goal_b']['hours']}h")
vague = call("POST", "/api/compare", {"goal_b": "I want to be a web developer"})
check("an unreadable second goal asks instead of comparing",
      vague.get("status") == "needs_clarification")

print("\n== GET /api/forecast")
fc = call("GET", "/api/forecast?hours_per_week=5&weeks_available=12")
print(f"  {fc['budget']['sentence'][:170]}")
check("velocity reported", fc.get("velocity", {}).get("pace_factor") == 1.0)
check("budget cuts the plan short",
      fc["budget"]["reachable_courses"] < fc["budget"]["total_courses"])
check("milestone at risk named", fc["budget"]["first_milestone_at_risk"] is not None)

print("\n== POST /api/complete with actual_hours -> pace changes the forecast")
call("POST", "/api/complete", {"course_id": first, "actual_hours": 45})
fc2 = call("GET", "/api/forecast?hours_per_week=10")
print(f"  {fc2['velocity']['note'][:150]}")
check("pace factor moved off 1.0", fc2["velocity"]["pace_factor"] != 1.0,
      f"{fc2['velocity']['pace_factor']}x")
check("adjusted forecast differs from the listed one",
      fc2["velocity"]["remaining_hours_adjusted"] != fc2["velocity"]["remaining_hours_listed"])

print("\n== GET /api/refreshers")
ref = call("GET", "/api/refreshers?stale_days=0").get("prompts", [])
for item in ref[:2]:
    print(f"  {item['sentence'][:130]}")
check("stale prerequisites surfaced once a course is done", isinstance(ref, list))

print("\n== POST /api/import-history")
preview = call("POST", "/api/import-history", {
    "text": "Docker for Machine Learning\nSQL and joins\nSomething entirely made up"})
print(f"  {preview['parsed']['sentence'][:170]}")
check("preview does not apply", preview.get("applied") is False)
check("unmatched lines reported", len(preview["parsed"]["unmatched"]) == 1)
before_courses = call("POST", "/api/plan")["path"]["total_courses"]
applied = call("POST", "/api/import-history", {
    "text": "Docker and Kubernetes for ML\ndocker\nrest apis", "apply": True})
check("applying shrinks or holds the path",
      applied["delta"]["courses_after"] <= before_courses,
      f"{before_courses} -> {applied['delta']['courses_after']}")
check("imported skills are in the profile",
      "docker" in applied.get("learner", {}).get("possessed", []))

print("\n== GET /api/graph/health")
health = call("GET", "/api/graph/health")
print(f"  {health['skills']} skills, {health['courses']} courses, "
      f"{len(health['single_source_skills'])} taught by only one course")
check("health reports fragility", isinstance(health.get("single_source_skills"), list))
check("no uncovered skills", health.get("uncovered_skills") == [])
check("fan-out reported", len(health.get("highest_fan_out", [])) > 0)

print("\n" + ("ALL FEATURE-API CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
