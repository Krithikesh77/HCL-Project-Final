"""
End-to-end smoke test for every API endpoint against a running server.

Start the server first, then:  python scripts/test_api.py [base_url]

Run it once with Ollama up and once with Ollama killed - the output should be
identical apart from the "source" fields, which flip from ollama to the
deterministic fallbacks.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
FAILURES = []


def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        FAILURES.append(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:200]}")
        return {}


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


print("== GET /api/health")
health = call("GET", "/api/health")
mode = health.get("ollama", {}).get("mode")
print(f"  mode: {mode}   ollama reachable: {health.get('ollama', {}).get('reachable')}")
check("graph loaded", health.get("graph", {}).get("skills", 0) > 40)

print("== POST /api/reset")
call("POST", "/api/reset")

print("== GET /api/skills")
skills = call("GET", "/api/skills")
check("skill catalog served", len(skills.get("skills", [])) > 40,
      f"{len(skills.get('skills', []))} skills")

print("== GET /api/personas")
personas = call("GET", "/api/personas")
check("personas served", len(personas.get("personas", [])) == 3)

print("== POST /api/intake")
intake = call("POST", "/api/intake", {
    "goal": "I want to build RAG chatbots",
    "experience_level": "intermediate",
    "known_skills": ["python", "python-oop", "command-line", "git-version-control", "sql"],
    "interests": ["llm", "rag"],
})
targets = intake.get("learner", {}).get("target_skills", [])
print(f"  targets: {targets}  (source: {intake.get('extraction', {}).get('source')})")
check("targets extracted", "rag" in targets)
check("known skills closed downward", len(intake.get("learner", {}).get("possessed", [])) >= 5)

print("== POST /api/plan")
plan = call("POST", "/api/plan").get("path", {})
courses = plan.get("courses", [])
print(f"  {plan.get('total_courses')} courses, {plan.get('total_hours')}h, "
      f"{len(plan.get('milestones', []))} milestones")
check("path built", len(courses) > 5)
check("milestones between 3 and 5", 3 <= len(plan.get("milestones", [])) <= 5)
check("progress block present", "percent" in plan.get("progress", {}))
check("next action present", plan.get("next_action") is not None)

first = courses[0]["course_id"]
mid = courses[len(courses) // 2]["course_id"]

print("== GET /api/explain/{course_id}")
exp = call("GET", f"/api/explain/{mid}")
trace = exp.get("trace", {})
print(f"  narrative source: {exp.get('narrative', {}).get('source')}")
print(f"  {trace.get('summary', '')[:160]}...")
check("explanation has prerequisite chains",
      any(t.get("edges") for t in trace.get("teaches", [])) or
      any(t.get("chain") for t in trace.get("teaches", [])))
check("explanation is graph-grounded", trace.get("grounded_in", "").startswith("prereq"))

print("== POST /api/chat")
for q in ["what should I start with?", "how long will this take?",
          f"why do I need {courses[len(courses) // 2]['title']}?"]:
    reply = call("POST", "/api/chat", {"message": q})
    print(f"  Q: {q}\n  A ({reply.get('source')}): {reply.get('reply', '')[:180]}")
    check(f"chat answered: {q[:24]}", len(reply.get("reply", "")) > 20)

print("== POST /api/complete")
done = call("POST", "/api/complete", {"course_id": first})
delta = done.get("delta", {})
print(f"  courses {delta.get('courses_before')} -> {delta.get('courses_after')}, "
      f"hours {delta.get('hours_before')} -> {delta.get('hours_after')}")
check("path shrank after completion",
      delta.get("courses_after", 99) < delta.get("courses_before", 0))
check("progress advanced", done.get("path", {}).get("progress", {}).get("percent", 0) > 0)

print("== POST /api/feedback (too_hard, course with alternatives)")
swappable, single = None, None
for entry in done.get("path", {}).get("courses", []):
    node = call("GET", f"/api/graph/skill/{entry['covers'][0]}")
    if len(node.get("taught_by", [])) > 1 and swappable is None:
        swappable = entry["course_id"]
    if len(node.get("taught_by", [])) == 1 and single is None:
        single = entry["course_id"]
fb = call("POST", "/api/feedback", {"course_id": swappable, "signal": "too_hard"})
print(f"  {fb.get('message', '')[:170]}")
check("course swapped out",
      swappable not in [c["course_id"] for c in fb.get("path", {}).get("courses", [])])
check("replacement named", fb.get("swapped_to") is not None)

print("== POST /api/feedback (too_hard, course with NO alternative)")
solo = call("POST", "/api/feedback", {"course_id": single, "signal": "too_hard"})
print(f"  {solo.get('message', '')[:170]}")
check("no-alternative case reported honestly",
      solo.get("swapped_to") is None and "only course" in solo.get("message", ""))
check("no-alternative case leaves path intact",
      single in [c["course_id"] for c in solo.get("path", {}).get("courses", [])])

print("== POST /api/feedback (already_know)")
target2 = fb.get("path", {}).get("courses", [])[0]["course_id"]
before_n = fb.get("path", {}).get("total_courses")
fb2 = call("POST", "/api/feedback", {"course_id": target2, "signal": "already_know"})
print(f"  {fb2.get('message', '')[:160]}")
check("path shrank after already_know",
      fb2.get("path", {}).get("total_courses", 99) < before_n)

print("== GET /api/learner")
learner = call("GET", "/api/learner")
check("state persisted", len(learner.get("learner", {}).get("history", [])) >= 3,
      f"{len(learner.get('learner', {}).get('history', []))} events")

print("== GET /api/graph/skill/{id}  (learner-annotated node, powers the inspector)")
node = call("GET", "/api/graph/skill/vector-databases")
check("skill node served", node.get("name") == "Vector Databases",
      f"requires {[r['id'] for r in node.get('requires', [])]}")
check("node annotated for this learner", node.get("learner") is not None,
      f"in_gap={node.get('learner', {}).get('in_gap')}")
check("prerequisites carry standing",
      all("possessed" in r and "in_gap" in r for r in node.get("requires", [])))
check("teaching courses flag the scheduled one",
      any(c.get("in_your_path") for c in node.get("taught_by", []))
      or node.get("learner", {}).get("scheduled") is None)

print("== POST /api/intake with keep_progress (re-aim, do not restart)")
before_state = call("GET", "/api/learner").get("learner", {})
done_before = len(before_state.get("completed_courses", []))
known_before = len(before_state.get("possessed", []))
re_aim = call("POST", "/api/intake", {
    "goal": "I want to fine-tune LLMs",
    "experience_level": before_state.get("experience_level", "intermediate"),
    "known_skills": [], "interests": [], "keep_progress": True,
})
after_state = re_aim.get("learner", {})
print(f"  goal now: {after_state.get('goal')}  targets: {after_state.get('target_skills')}")
check("same learner, not a new one", after_state.get("id") == before_state.get("id"))
check("completed courses survived",
      len(after_state.get("completed_courses", [])) == done_before, f"{done_before} kept")
check("learned skills survived",
      len(after_state.get("possessed", [])) >= known_before)
check("targets actually changed", after_state.get("target_skills") != before_state.get("target_skills"))
check("goal change recorded in history",
      any(h.get("kind") == "goal-change" for h in after_state.get("history", [])))
re_planned = call("POST", "/api/plan").get("path", {})
check("path rebuilt for the new goal", re_planned.get("total_courses", 0) > 0,
      f"{re_planned.get('total_courses')} courses, {re_planned.get('total_hours')}h")

print("== completing every course -> the path-complete end state")
guard = 0
path_now = re_planned
while path_now.get("courses") and guard < 60:
    guard += 1
    path_now = call("POST", "/api/complete",
                    {"course_id": path_now["courses"][0]["course_id"]}).get("path", {})
print(f"  exhausted after {guard} completions")
check("path reports complete", path_now.get("complete") is True)
check("no next action", path_now.get("next_action") is None)
check("no courses left", path_now.get("total_courses") == 0)
check("no hours left", path_now.get("total_hours") == 0)
check("no milestones left", path_now.get("milestones") == [])
check("progress reads 100%", path_now.get("progress", {}).get("percent") == 100,
      f"{path_now.get('progress', {}).get('skills_done')} skills acquired")
final_plan = call("POST", "/api/plan").get("path", {})
check("replanning a finished path is stable", final_plan.get("complete") is True)
done_chat = call("POST", "/api/chat", {"message": "how long will this take?"})
print(f"  chat on a finished path: {done_chat.get('reply', '')[:150]}")
check("chat handles a finished path", "done" in done_chat.get("reply", "").lower()
      or "complete" in done_chat.get("reply", "").lower())

print("== a learner who already knows everything")
call("POST", "/api/reset")
knows_all = call("POST", "/api/intake", {
    "goal": "I want to build RAG chatbots",
    "experience_level": "advanced",
    "known_skills": ["rag"], "interests": [],
})
empty = call("POST", "/api/plan").get("path", {})
check("empty gap handled", empty.get("total_courses") == 0 and empty.get("complete") is True,
      f"{empty.get('progress', {}).get('percent')}% with nothing to do")

print("\n" + ("ALL API CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
