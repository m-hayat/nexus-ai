"""
Nexus-AI Platform — Batch evaluator.

Run:
    python scripts/run_eval.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.agent_router import run_agent  # noqa: E402


THRESHOLD_ROUTING = 0.80
THRESHOLD_CITATION = 0.70
THRESHOLD_COVERAGE = 0.70
THRESHOLD_FAITHFULNESS = 0.75


async def evaluate_case(case: dict) -> dict:
    """Ekta case run kore metrics calculate kori."""
    start = time.time()

    try:
        state = await run_agent(
            query=case["query"],
            session_id=f"eval-{case['id']}",
            request_id=f"req-{case['id']}",
        )
    except Exception as exc:
        return {
            "id": case["id"],
            "error": str(exc),
            "passed": False,
            "duration": time.time() - start,
        }

    answer = (state.get("answer") or "").lower()
    routing = state.get("intent", "policy")
    citations = state.get("citations") or []
    citation_ids = {c["source_id"] for c in citations}
    faithfulness = state.get("faithfulness_score")
    should_refuse = case.get("should_refuse", False)

    # ── Routing check ──
    routing_ok = routing == case["expected_routing"]

    # ── Citation check (flexible) ──
    # Rule:
    #   - expected_citations non-empty → check subset match
    #   - expected_citations empty ar should_refuse=True → any citation ok
    #   - expected_citations empty ar should_refuse=False → at least 1 citation
    # ── Citation check (flexible) ──
    # Non-policy routes (escalation, chitchat) → citation lagbe na
    expected_cites = set(case.get("expected_citations", []))
    is_non_policy = routing in ("escalation", "chitchat", "fallback")

    if expected_cites:
        citation_ok = expected_cites.issubset(citation_ids)
    elif should_refuse or is_non_policy:
        citation_ok = True
    else:
        citation_ok = len(citation_ids) >= 1

    # ── Coverage check ──
    expected_keywords = case.get("expected_answer_contains", [])
    if expected_keywords:
        found = sum(1 for kw in expected_keywords if kw.lower() in answer)
        coverage = found / len(expected_keywords)
    else:
        coverage = 1.0
    coverage_ok = coverage >= 0.5

    # ── Refusal check ──
    if should_refuse:
        refusal_markers = ["don't have", "do not have", "can only help", "not about"]
        refusal_ok = any(m in answer for m in refusal_markers)
    else:
        refusal_ok = True

    passed = routing_ok and citation_ok and coverage_ok and refusal_ok

    return {
        "id": case["id"],
        "query": case["query"],
        "routing": routing,
        "expected_routing": case["expected_routing"],
        "routing_ok": routing_ok,
        "citation_ids": sorted(citation_ids),
        "expected_citations": case.get("expected_citations", []),
        "citation_ok": citation_ok,
        "coverage": coverage,
        "coverage_ok": coverage_ok,
        "refusal_ok": refusal_ok,
        "faithfulness": faithfulness,
        "duration": time.time() - start,
        "passed": passed,
    }


async def main() -> int:
    dataset_path = ROOT / "data" / "eval_dataset.json"
    if not dataset_path.exists():
        print(f"[FAIL] Dataset not found: {dataset_path}")
        return 1

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    cases = dataset["test_cases"]
    print("=" * 70)
    print(f"Nexus-AI Evaluation - {len(cases)} cases")
    print("=" * 70)
    print()

    results = []
    for case in cases:
        print(f">> {case['id']}: {case['query'][:60]}...", flush=True)
        result = await evaluate_case(case)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  [{status}] routing={result.get('routing')} "
            f"(expected={result.get('expected_routing')}) "
            f"| cites={result.get('citation_ids')} "
            f"| faith={result.get('faithfulness')} "
            f"| {result['duration']:.1f}s"
        )
        print()

    total = len(results)
    routing_pass = sum(1 for r in results if r.get("routing_ok"))
    citation_pass = sum(1 for r in results if r.get("citation_ok"))
    coverage_pass = sum(1 for r in results if r.get("coverage_ok"))
    overall_pass = sum(1 for r in results if r["passed"])

    faithfulness_scores = [
        r["faithfulness"] for r in results
        if r.get("faithfulness") is not None
    ]
    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores else 0.0
    )

    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    print(f"  Overall pass:     {overall_pass}/{total} ({overall_pass/total*100:.1f}%)")
    print(f"  Routing accuracy: {routing_pass}/{total} ({routing_pass/total*100:.1f}%)")
    print(f"  Citation accuracy: {citation_pass}/{total} ({citation_pass/total*100:.1f}%)")
    print(f"  Coverage accuracy: {coverage_pass}/{total} ({coverage_pass/total*100:.1f}%)")
    print(f"  Avg faithfulness: {avg_faithfulness:.2f}")
    print()

    routing_rate = routing_pass / total
    citation_rate = citation_pass / total
    coverage_rate = coverage_pass / total

    thresholds_ok = (
        routing_rate >= THRESHOLD_ROUTING
        and citation_rate >= THRESHOLD_CITATION
        and coverage_rate >= THRESHOLD_COVERAGE
        and avg_faithfulness >= THRESHOLD_FAITHFULNESS
    )

    if thresholds_ok:
        print("[OK] ALL THRESHOLDS MET - CI pass")
        return 0
    else:
        print("[FAIL] THRESHOLDS FAILED:")
        print(f"   routing: {routing_rate:.2f} (need {THRESHOLD_ROUTING})")
        print(f"   citation: {citation_rate:.2f} (need {THRESHOLD_CITATION})")
        print(f"   coverage: {coverage_rate:.2f} (need {THRESHOLD_COVERAGE})")
        print(f"   faithfulness: {avg_faithfulness:.2f} (need {THRESHOLD_FAITHFULNESS})")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))