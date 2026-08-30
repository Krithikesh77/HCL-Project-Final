"""
Headless correctness check for the graph engine - no server, no LLM, no network.

Runs three personas end to end and prints their computed paths, then asserts the
invariants that make the path trustworthy:

  * every prerequisite of every scheduled course is either already possessed or
    taught by an EARLIER course in the path (no forward references),
  * every skill in the gap is covered by exactly the courses claimed,
  * no course teaches something the learner already knows as its only value,
  * the goal skills are all reachable at the end.

Run:  python scripts/test_personas.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402

PERSONAS = [
    {
        "name": "Complete beginner",
        "goal": "I want to become an ML engineer",
        "experience_level": "beginner",
        "known_skills": [],
        "interests": [],
    },
    {
        "name": "Python developer, no ML",
        "goal": "I want to build RAG chatbots",
        "experience_level": "intermediate",
        "known_skills": ["python", "python-oop", "command-line", "git-version-control", "sql"],
        "interests": ["llm", "rag"],
    },
    {
        "name": "Knows scikit-learn",
        "goal": "I want to fine-tune LLMs",
        "experience_level": "intermediate",
        "known_skills": ["sklearn-basics", "supervised-learning", "model-evaluation",
                         "pandas", "numpy", "statistics"],
        "interests": ["llm", "finetuning"],
    },
]

BAR = "=" * 78


def check_invariants(g, path, persona):
    """Hard assertions - these are the properties a graph-based planner must have."""
    errors = []
    possessed = set(path["possessed"])
    seen_skills = set(possessed)
    for entry in path["courses"]:
        for req in entry["requires"]:
            if req not in seen_skills:
                errors.append(
                    f"  ORDERING BUG: {entry['course_id']} ({entry['title']}) requires "
                    f"{req} which is not known or taught earlier")
        seen_skills |= set(entry["teaches"])

    covered = {s for e in path["courses"] for s in e["covers"]}
    missing = set(path["gap_skills"]) - covered
    if missing:
        errors.append(f"  COVERAGE BUG: gap skills never taught: {sorted(missing)}")

    for target in path["targets"]:
        if target not in seen_skills:
            errors.append(f"  GOAL BUG: target skill {target} never reached")

    for entry in path["courses"]:
        if not entry["covers"]:
            errors.append(f"  REDUNDANCY BUG: {entry['course_id']} covers nothing new")

    order_index = {e["course_id"]: e["order"] for e in path["courses"]}
    for entry in path["courses"]:
        for dep in entry["depends_on"]:
            if order_index[dep] >= entry["order"]:
                errors.append(f"  LINK BUG: {entry['course_id']} depends on later {dep}")

    if errors:
        print("FAILED invariants for", persona["name"])
        print("\n".join(errors))
        return False
    return True


def run_persona(g, persona):
    print(BAR)
    print(f"PERSONA: {persona['name']}")
    print(f'GOAL   : "{persona["goal"]}"')
    print(f"LEVEL  : {persona['experience_level']}   "
          f"INTERESTS: {persona['interests'] or '-'}")
    print(f"KNOWS  : {', '.join(persona['known_skills']) or '(nothing)'}")
    print(BAR)

    # Step 1: goal text -> target skills (deterministic fallback path, no Ollama)
    targets = engine.keyword_match_targets(persona["goal"], graph=g)
    print("TARGET SKILLS (keyword fallback, no LLM): "
          + ", ".join(f"{t} [{g.name(t)}]" for t in targets))

    possessed = g.expand_possessed(persona["known_skills"])
    if persona["known_skills"]:
        implied = sorted(possessed - set(persona["known_skills"]))
        print(f"POSSESSED after downward closure: {len(possessed)} skills"
              + (f" (implied: {', '.join(implied)})" if implied else ""))

    # Steps 2-5: gap -> topo order -> courses -> milestones
    gap = engine.gap_analysis(targets, possessed, graph=g)
    ordered = engine.topo_order(gap, graph=g)
    print(f"GAP: {len(gap)} skills needed")
    print("TOPO ORDER: " + " -> ".join(ordered))

    path = engine.build_path(targets, possessed,
                             level=persona["experience_level"],
                             interests=persona["interests"], graph=g)

    print(f"\nPATH: {path['total_courses']} courses, {path['total_hours']} hours, "
          f"{len(path['milestones'])} milestones\n")
    for ms in path["milestones"]:
        print(f"  -- {ms['name']} ({len(ms['course_ids'])} courses, {ms['hours']}h)")
        for entry in path["courses"]:
            if entry["milestone"] != ms["name"]:
                continue
            deps = (" <- " + ", ".join(entry["depends_on"])) if entry["depends_on"] else ""
            print(f"     {entry['order'] + 1:>2}. [{entry['course_id']}] {entry['title']}"
                  f" ({entry['provider']}, {entry['hours']}h, {entry['level']}){deps}")
            print(f"         teaches: {', '.join(entry['covers_named'])}")
    print()

    # Step 6: grounded explanation for a mid-path course
    probe = path["courses"][len(path["courses"]) // 2] if path["courses"] else None
    if probe:
        trace = engine.explain(probe["course_id"], path, graph=g)
        print(f'  WHY "{trace["title"]}"? (read off the graph, no LLM)')
        for line in trace["sentences"]:
            print(f"     - {line}")
    print()

    # Step 7: replan after completing the first two courses
    learner = {
        "target_skills": targets,
        "known_skills": sorted(possessed),
        "possessed": sorted(possessed),
        "experience_level": persona["experience_level"],
        "interests": persona["interests"],
        "completed_courses": [],
        "rejected_courses": [],
    }
    for entry in path["courses"][:2]:
        learner["completed_courses"].append(entry["course_id"])
        learner["possessed"] = sorted(set(learner["possessed"]) | set(entry["teaches"]))
    replanned = engine.replan(learner, graph=g)
    done = ", ".join(learner["completed_courses"])
    print(f"  REPLAN after completing {done}: "
          f"{replanned['total_courses']} courses left "
          f"({replanned['total_hours']}h), progress "
          f"{replanned['progress']['percent']}% of skills")
    nxt = replanned["next_action"]
    print(f"  NEXT: {nxt['title']} ({nxt['provider']})" if nxt else "  NEXT: path complete")
    print()

    ok = check_invariants(g, path, persona)
    ok = check_invariants(g, replanned, persona) and ok
    print("INVARIANTS: " + ("all passed" if ok else "FAILED"))
    print()
    return ok


def main():
    g = engine.load_graph()
    stats = engine.graph_stats(g)
    print(f"Graph loaded and validated as a DAG: {stats['skills']} skills, "
          f"{stats['edges']} prerequisite edges, {stats['courses']} courses, "
          f"max depth {stats['max_depth']}.\n")
    results = [run_persona(g, p) for p in PERSONAS]
    print(BAR)
    print("RESULT: " + ("all personas passed" if all(results) else "FAILURES ABOVE"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
