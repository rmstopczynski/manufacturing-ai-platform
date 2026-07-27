"""
Evaluation test cases for the manufacturing assistant.

Design note: expected characteristics, not exact strings. Natural-language answers
from an LLM will never match a fixed string exactly, so every check here is either
structural (did routing/prediction/retrieval behave correctly — deterministic, checked
against the actual pipeline internals, not the LLM's wording) or a loose content check
(does the answer contain at least one of a small set of plausible keywords/phrases).
This is deliberately closer to "does the answer have the right shape" than "did the
model say exactly X" — the latter is brittle and would fail on harmless rephrasing.

Each case is a dict:
    id                          - short identifier
    question                    - the raw question sent to the assistant
    expected_machine_id         - str or None: what routing should extract
    expects_prediction          - bool: should routing decide it needs an ML prediction
    expects_prediction_found    - bool or None: for known/unknown machine IDs (None = skip)
    expected_retrieval_types    - set of machine_types acceptable in retrieved chunks,
                                   or None to skip this check entirely
    min_relevant_chunks         - minimum chunks expected below the relevance threshold
    answer_contains_any         - list of case-insensitive substrings; at least one
                                   should appear in the final answer (None = skip)
"""

EVAL_CASES = [
    {
        "id": "known_machine_healthy",
        "question": "Is machine M001 at risk of failing soon?",
        "expected_machine_id": "M001",
        "expects_prediction": True,
        "expects_prediction_found": True,
        "expected_retrieval_types": {"CNC_MILL", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["healthy", "not at risk", "low risk", "0%", "0.0%"],
    },
    {
        "id": "unknown_machine",
        "question": "Is machine M999 at risk of failing soon?",
        "expected_machine_id": "M999",
        "expects_prediction": True,
        "expects_prediction_found": False,
        "expected_retrieval_types": None,
        "min_relevant_chunks": 0,
        "answer_contains_any": ["not found", "no data", "doesn't exist", "no sensor data", "couldn't find"],
    },
    {
        "id": "general_sop_no_machine",
        "question": "What's the lockout/tagout procedure before maintenance?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["lockout", "tagout", "loto", "energy source", "lock and tag"],
    },
    {
        "id": "conveyor_troubleshooting",
        "question": "How do I troubleshoot rising vibration on a conveyor?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"CONVEYOR", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["belt", "conveyor", "misalignment", "vibration"],
    },
    {
        "id": "pump_failure_reason",
        "question": "Why does the pump keep failing?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"PUMP", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["cavitation", "bearing", "impeller", "pump"],
    },
    {
        "id": "maintenance_schedule_general",
        "question": "What is the routine maintenance schedule?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["30", "40", "45", "day", "interval"],
    },
    {
        "id": "off_topic_question",
        "question": "Tell me about quantum computing trends",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": None,
        "min_relevant_chunks": 0,
        "answer_contains_any": [
            "no relevant", "don't have", "not related", "not able to", "no sufficiently relevant",
            "no information", "not documented",
        ],
    },
    {
        "id": "vague_input_shortcircuit",
        "question": "asdf",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": None,
        "min_relevant_chunks": 0,
        "answer_contains_any": ["more detail", "machine id", "clarify", "what topic"],
    },
    {
        "id": "second_known_machine",
        "question": "Is M005 healthy?",
        "expected_machine_id": "M005",
        "expects_prediction": True,
        "expects_prediction_found": True,
        "expected_retrieval_types": {"PUMP", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["healthy", "risk", "%"],
    },
    {
        "id": "robot_arm_maintenance_prep",
        "question": "What should I check before performing maintenance on a robot arm?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"ROBOT_ARM", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["lockout", "tagout", "actuator", "joint", "energy source"],
    },
    {
        "id": "press_operating_range",
        "question": "What's the normal operating temperature range for a hydraulic press?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"PRESS", "GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["58", "72", "65", "degrees", "°c", "temperature"],
    },
    {
        "id": "sop_reference_by_name",
        "question": "What does SOP-001 cover?",
        "expected_machine_id": None,
        "expects_prediction": False,
        "expects_prediction_found": None,
        "expected_retrieval_types": {"GENERAL"},
        "min_relevant_chunks": 1,
        "answer_contains_any": ["lockout", "tagout", "energy", "maintenance"],
    },
]
