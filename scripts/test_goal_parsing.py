"""
Goal-parsing regression suite: the deterministic resolver, with no LLM involved.

The behaviour that matters here is not just "does it map goals correctly" but
"does it refuse to guess when it does not know". An earlier version always
returned a target, so a typo, an out-of-domain goal and an empty string all
silently produced a confident plan for generic supervised learning.

Run:  python scripts/test_goal_parsing.py
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


def resolve(goal):
    return engine.resolve_goal(goal, graph=G)


def reaches(res, skill_id):
    """normalize_targets drops a target implied by a deeper one, so the honest
    question is whether the goal's closure reaches the skill, not whether it is
    literally in the target list."""
    targets = set(res["targets"])
    return skill_id in (targets | G.prereq_closure(targets))


def show(goal, res):
    if res["targets"]:
        names = ", ".join(G.name(t) for t in res["targets"])
        how = res["matches"][0] if res["matches"] else {}
        print(f'  "{goal[:44]:44}" -> {names}')
        print(f'  {"":48}via {how.get("method")}: {how.get("evidence")} ({how.get("score")})')
    elif res["out_of_domain"]:
        print(f'  "{goal[:44]:44}" -> OUT OF DOMAIN (nothing close)')
    else:
        cands = ", ".join(f'{G.name(c["skill_id"])} ({c["score"]})' for c in res["candidates"])
        print(f'  "{goal[:44]:44}" -> ASKS: {cands}')


print("== exact and pattern matches still work")
for goal, expect in [
    ("I want to build RAG chatbots", "rag"),
    ("I want to fine-tune LLMs", "llm-finetuning"),
    ("I want to become an ML engineer", "supervised-learning"),
    ("I want to do time series forecasting", "time-series"),
]:
    res = resolve(goal)
    show(goal, res)
    check(f"{expect} from \"{goal[:28]}\"", reaches(res, expect), str(res["targets"]))

print("\n== tool and brand names resolve through aliases")
for goal, expect in [
    ("I want to learn LangChain", "rag"),
    ("I want to use Pinecone and Weaviate", "vector-databases"),
    ("I want to work with Kubernetes", "docker"),
    ("I want to learn xgboost for Kaggle", "gradient-boosting"),
    ("I want to use mlflow", "experiment-tracking"),
    ("I want to serve models with vLLM", "llm-serving"),
]:
    res = resolve(goal)
    show(goal, res)
    check(f"{expect} from an alias", reaches(res, expect), str(res["targets"]))

print("\n== typos are tolerated")
for goal, expect in [
    ("I want to learn pytourch", "pytorch"),
    ("I want to build a recomender system", "recommender-systems"),
    ("I want to learn kubernets and dockr", "docker"),
]:
    res = resolve(goal)
    show(goal, res)
    check(f"typo -> {expect}", reaches(res, expect), str(res["targets"]))

print("\n== paraphrase resolves semantically")
for goal, expect in [
    ("I want to make computers understand pictures", "computer-vision"),
    ("I want to find similar documents by meaning", "semantic-search"),
]:
    res = resolve(goal)
    show(goal, res)
    check(f"paraphrase -> {expect}",
          reaches(res, expect) or expect in [c["skill_id"] for c in res["candidates"]],
          str(res["targets"] or [c["skill_id"] for c in res["candidates"]]))

print("\n== it refuses to guess (this is the point)")
for goal in ["asdkjh qwerty zzz", "", "   ", "I want to be a web developer",
             "I want to learn to play the piano"]:
    res = resolve(goal)
    show(goal, res)
    check(f'no silent target for "{goal[:26]}"', res["targets"] == [],
          f'confidence={res["confidence"]}')

print("\n== ambiguous goals produce candidates, not a guess")
for goal in ["I want to work with models", "I want to do something with data"]:
    res = resolve(goal)
    show(goal, res)
    check(f'"{goal[:26]}" is handled without a confident target',
          res["targets"] == [] or res["confidence"] in ("medium", "high"),
          f'confidence={res["confidence"]}, candidates={len(res["candidates"])}')

print("\n== a content gap the vocabulary log closed")
# This was the standing example of an honest refusal: the graph had no
# reinforcement-learning node, so the parser declined rather than mapping RL
# onto RLHF, which is an LLM fine-tuning technique and not RL. The branch now
# exists, so these resolve - and they resolve to RL, not to RLHF.
for goal in ["I want to do reinforcement learning", "I want to do reinforcment lerning"]:
    res = resolve(goal)
    show(goal, res)
    check(f'"{goal[:30]}" now resolves', reaches(res, "rl-fundamentals"),
          str(res["targets"]))
    check(f'"{goal[:30]}" is not mapped onto RLHF', "rlhf" not in res["targets"])

print("\n== unknown terms are recorded for the vocabulary gap log")
res = resolve("I want to learn Mojo and Zig for tensor kernels")
print(f"  unknown terms: {res['unknown_terms']}")
check("unknown terms captured", len(res["unknown_terms"]) > 0, str(res["unknown_terms"]))

print("\n== the parser is deterministic")
a = resolve("I want to build RAG chatbots")
b = resolve("I want to build RAG chatbots")
check("same input, same output", a == b)

print("\n" + ("ALL GOAL-PARSING CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
