"""
Routing logic: given a raw user question, decide what the orchestration chain needs
to gather before calling the LLM — an ML prediction for a specific machine, document
retrieval, or both.

Design decision: this routing is plain Python (a regex + a couple of heuristics), not
a second LLM call asking "what should I do with this question?". At this project's
scope, the routing decision is simple enough (does the question mention a machine ID?)
that spending an extra LLM call on it would add latency and cost for no real gain.
A more open-ended assistant with many possible tools would justify LLM-based routing
(or a proper agent framework); a two-branch decision does not. This is a real
design tradeoff worth stating explicitly rather than defaulting to "more LLM calls
because it's an AI project."
"""

import re

MACHINE_ID_PATTERN = re.compile(r"\bM\d{3}\b", re.IGNORECASE)

# Machine type keywords let retrieval be filtered even when no specific machine ID
# is mentioned (e.g. "how do I maintain a conveyor" -> filter to CONVEYOR + GENERAL docs).
MACHINE_TYPE_KEYWORDS = {
    "CNC_MILL": ["cnc", "mill", "milling", "spindle"],
    "CONVEYOR": ["conveyor", "belt"],
    "PRESS": ["press", "hydraulic"],
    "ROBOT_ARM": ["robot", "robotic", "arm", "actuator", "joint"],
    "PUMP": ["pump", "impeller"],
}


def route_question(question: str) -> dict:
    q_lower = question.lower()

    machine_id_match = MACHINE_ID_PATTERN.search(question)
    machine_id = machine_id_match.group(0).upper() if machine_id_match else None

    machine_type = None
    for mtype, keywords in MACHINE_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            machine_type = mtype
            break

    # Needs a prediction only when a specific machine ID is mentioned — predictions
    # are per-machine, so there's nothing to predict for a general question like
    # "what's our maintenance policy."
    needs_prediction = machine_id is not None

    # Retrieval runs for essentially every question — even a pure prediction request
    # ("is M014 healthy?") benefits from the equipment manual's context on what the
    # prediction actually means for that machine type. The only case where retrieval
    # is skipped is an empty/too-short question, handled as an edge case upstream.
    needs_retrieval = True

    return {
        "machine_id": machine_id,
        "machine_type": machine_type,
        "needs_prediction": needs_prediction,
        "needs_retrieval": needs_retrieval,
    }


if __name__ == "__main__":
    tests = [
        "Is machine M014 at risk of failing soon?",
        "What's the lockout/tagout procedure before maintenance?",
        "How do I troubleshoot rising vibration on a conveyor?",
        "Why does the pump keep failing?",
    ]
    for t in tests:
        print(t, "->", route_question(t))
