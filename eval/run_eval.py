"""
Runs eval/eval_cases.py through the real ManufacturingAssistant pipeline and reports
which structural and content expectations pass or fail per case.

Two check categories, checked separately so a failure report says WHERE the problem
is, not just THAT something failed:

1. STRUCTURAL checks — free, deterministic, don't depend on the LLM's wording:
   - did routing extract the expected machine_id?
   - did routing correctly decide whether a prediction was needed?
   - did the prediction come back found/not-found as expected?
   - did retrieval return chunks limited to the expected machine types?
   - did retrieval return at least the expected number of below-threshold chunks?

2. CONTENT checks — require the actual LLM call, since they inspect the generated
   answer text. Loose by design (substring match, any-of): natural-language answers
   vary in wording even when they're correct, so an exact-match check here would
   mostly measure the model's phrasing choices rather than whether the pipeline
   actually worked.

This directly answers "how do you evaluate whether the RAG system is actually working"
— the answer isn't "I read the outputs and it looked fine," it's this script.

Usage:
    python -m eval.run_eval
"""

import json
import time
from datetime import datetime, timezone

from eval.eval_cases import EVAL_CASES
from orchestration.chain import ManufacturingAssistant, RELEVANCE_DISTANCE_THRESHOLD

# Seconds to wait between cases. Free-tier LLM APIs (e.g. Groq's free tier: 8000
# tokens/minute) rate-limit on tokens per minute, and each question here — with
# retrieved document context and/or an ML prediction folded into the prompt — can
# easily run 1000+ tokens. Twelve cases back-to-back with no pacing reliably trips
# that limit partway through a run. This isn't tuned to any specific provider's
# limit; it's a conservative default that keeps a 12-case run comfortably under
# typical free-tier ceilings. Lower it if you're on a paid tier with more headroom.
SECONDS_BETWEEN_CASES = 3


def check_structural(case: dict, result: dict) -> list[dict]:
    checks = []
    routing = result.get("routing") or {}
    prediction = result.get("prediction")
    chunks = result.get("retrieved_chunks") or []

    if case["expected_machine_id"] is not None or routing.get("machine_id") is not None:
        checks.append({
            "name": "machine_id_extracted",
            "passed": routing.get("machine_id") == case["expected_machine_id"],
            "detail": f"expected={case['expected_machine_id']!r} actual={routing.get('machine_id')!r}",
        })

    checks.append({
        "name": "needs_prediction_routing",
        "passed": (routing.get("needs_prediction") if routing else False) == case["expects_prediction"],
        "detail": f"expected={case['expects_prediction']} actual={routing.get('needs_prediction') if routing else None}",
    })

    if case["expects_prediction_found"] is not None:
        actual_found = prediction.get("found") if prediction else None
        checks.append({
            "name": "prediction_found_status",
            "passed": actual_found == case["expects_prediction_found"],
            "detail": f"expected found={case['expects_prediction_found']} actual found={actual_found}",
        })

    # retrieval type coverage: at least one chunk should come from an expected type.
    # This is an overlap check, not a strict subset — unfiltered retrieval (general
    # questions with no machine context) legitimately surfaces some incidental
    # cross-type matches (e.g. a per-machine troubleshooting SOP that also mentions
    # lockout/tagout), and that's correct behavior, not noise to penalize.
    if case["expected_retrieval_types"] is not None:
        actual_types = {c["machine_type"] for c in chunks}
        overlap = actual_types & case["expected_retrieval_types"]
        checks.append({
            "name": "retrieval_machine_types",
            "passed": len(overlap) > 0,
            "detail": f"expected overlap with {case['expected_retrieval_types']} actual={actual_types}",
        })

    relevant_count = sum(1 for c in chunks if c["distance"] <= RELEVANCE_DISTANCE_THRESHOLD)
    checks.append({
        "name": "min_relevant_chunks",
        "passed": relevant_count >= case["min_relevant_chunks"],
        "detail": f"expected>={case['min_relevant_chunks']} actual={relevant_count}",
    })

    return checks


def check_content(case: dict, answer: str) -> list[dict]:
    if case["answer_contains_any"] is None:
        return []
    answer_lower = answer.lower()
    matched = [kw for kw in case["answer_contains_any"] if kw.lower() in answer_lower]
    return [{
        "name": "answer_contains_any",
        "passed": len(matched) > 0,
        "detail": f"expected any of {case['answer_contains_any']} — matched: {matched or 'none'}",
    }]


def run(assistant=None) -> dict:
    if assistant is None:
        assistant = ManufacturingAssistant()

    report = {"run_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    total_checks = 0
    passed_checks = 0

    for i, case in enumerate(EVAL_CASES):
        try:
            result = assistant.answer(case["question"])
            structural = check_structural(case, result)
            content = check_content(case, result["answer"])
            all_checks = structural + content
            case_report = {
                "id": case["id"],
                "question": case["question"],
                "answer": result["answer"],
                "checks": all_checks,
                "all_passed": all(c["passed"] for c in all_checks),
                "error": None,
            }
        except Exception as e:
            # A single failed case (rate limit, transient network error, etc.)
            # should not lose the results of every other case in the run — record
            # it as a failure with the actual error message and keep going.
            case_report = {
                "id": case["id"],
                "question": case["question"],
                "answer": None,
                "checks": [],
                "all_passed": False,
                "error": f"{type(e).__name__}: {e}",
            }

        report["cases"].append(case_report)
        total_checks += len(case_report["checks"])
        passed_checks += sum(1 for c in case_report["checks"] if c["passed"])

        # Skip the delay after the last case — nothing left to pace against.
        if i < len(EVAL_CASES) - 1:
            time.sleep(SECONDS_BETWEEN_CASES)

    report["summary"] = {
        "total_cases": len(EVAL_CASES),
        "cases_fully_passed": sum(1 for c in report["cases"] if c["all_passed"]),
        "total_checks": total_checks,
        "checks_passed": passed_checks,
    }
    return report


def print_report(report: dict):
    print(f"Eval run at {report['run_at']}\n")
    for case in report["cases"]:
        status = "PASS" if case["all_passed"] else "FAIL"
        print(f"[{status}] {case['id']}: {case['question']}")
        if case.get("error"):
            print(f"    ERROR: {case['error']}")
            print()
            continue
        for c in case["checks"]:
            mark = "  ok " if c["passed"] else " FAIL"
            print(f"   {mark} {c['name']}: {c['detail']}")
        if not case["all_passed"]:
            print(f"   answer: {case['answer'][:200]}")
        print()

    s = report["summary"]
    print("=" * 60)
    print(f"Cases fully passed: {s['cases_fully_passed']}/{s['total_cases']}")
    print(f"Individual checks passed: {s['checks_passed']}/{s['total_checks']}")


if __name__ == "__main__":
    report = run()
    print_report(report)
    with open("eval/results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull results written to eval/results.json")
