"""
The orchestration chain: given a raw user question, this is the full path from
question -> routing -> (ML prediction, doc retrieval) -> prompt construction ->
LLM call -> final answer.

Uses LangChain's LCEL (LangChain Expression Language) for the actual LLM call —
a ChatPromptTemplate piped into ChatOpenAI piped into a string output parser. The
routing, prediction-fetching, and retrieval steps happen in plain Python BEFORE the
chain is invoked, because those steps are deterministic data-gathering, not something
that benefits from being inside a "chain" abstraction. The LangChain piece is doing
exactly the job it's good at — templating a prompt and calling the model — not being
used as a magic wrapper around logic that's really just plain Python.

Why LangChain instead of calling the OpenAI API directly? At this project's scale,
a raw `openai.chat.completions.create()` call would honestly work almost as well
for the LLM step alone. The real justification is consistency and swappability:
the prompt template, output parsing, and model config are decoupled from the call
site, so swapping models (e.g. OpenAI -> a local model via Ollama) or adding
structured output parsing later doesn't require rewriting the calling code — only
the chain definition. For a single-model, single-prompt project this is a modest
win; it becomes a much bigger one the moment there's more than one prompt/model
combination in play, which is the honest answer to give if pushed on this in an
interview rather than overselling LangChain as doing something it isn't here.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from orchestration.router import route_question
from ml.predictor import predict_machine
from rag.retrieve import retrieve

# Retrieved chunks with a distance above this are treated as "not actually relevant"
# rather than being stuffed into the prompt anyway. Chroma's default distance metric
# here is cosine distance (lower = more similar); this threshold was set empirically
# by checking retrieval distances on the sanity-check queries in Phase 2 and picking
# a cutoff above the distances seen for genuinely relevant hits.
RELEVANCE_DISTANCE_THRESHOLD = 1.7

SYSTEM_PROMPT = """You are a manufacturing operations assistant. You help engineers and \
plant managers understand machine health and find relevant procedures/documentation.

Rules:
- Base your answer ONLY on the ML prediction and retrieved documents provided below. \
Do not invent maintenance history, specific dates, or procedures that aren't in the \
provided context.
- If no ML prediction is provided, do not speculate about a specific machine's health.
- If the retrieved documents are marked as not relevant, say so plainly rather than \
answering from general knowledge, and suggest what information would help.
- Be concise and practical — this is for someone on a plant floor, not a report."""

USER_PROMPT_TEMPLATE = """Question: {question}

ML Prediction:
{prediction_context}

Retrieved Documentation:
{retrieval_context}
"""


def _format_prediction_context(prediction: dict | None) -> str:
    if prediction is None:
        return "(No specific machine ID was mentioned, so no prediction was pulled.)"
    if not prediction["found"]:
        return f"(Machine ID was mentioned but not found: {prediction['error']})"
    return (
        f"Machine {prediction['machine_id']}: "
        f"{'AT RISK of failure within 7 days' if prediction['failure_predicted_within_7d'] else 'currently healthy'} "
        f"(failure probability: {prediction['failure_probability']:.1%}). "
        f"Days since last maintenance: {prediction['days_since_maintenance']}. "
        f"Top contributing sensor signals: {prediction['top_contributing_features']}."
    )


def _format_retrieval_context(chunks: list[dict]) -> str:
    relevant = [c for c in chunks if c["distance"] <= RELEVANCE_DISTANCE_THRESHOLD]
    if not relevant:
        return "(No sufficiently relevant documents were found for this question.)"
    parts = []
    for c in relevant:
        parts.append(f"[{c['doc_type']} | {c['title']}]\n{c['text']}")
    return "\n\n".join(parts)


def build_chain(llm=None):
    """Builds the LCEL chain: prompt -> llm -> string output. `llm` is injectable so
    tests can pass a fake/stub model without hitting the real OpenAI API."""
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


class ManufacturingAssistant:
    def __init__(self, llm=None):
        self.chain = build_chain(llm)

    def answer(self, question: str) -> dict:
        question = question.strip()
        # "Too vague" is judged by word count, not character count — a single
        # nonsense word ("asdf") is just as unhelpful as a very short string, and a
        # character-length check alone would let it through.
        if len(question) < 3 or len(question.split()) < 2:
            return {
                "question": question,
                "answer": "Could you provide a bit more detail? For example, mention "
                          "a specific machine ID (e.g. M014) or what topic you need "
                          "documentation on.",
                "prediction": None,
                "retrieved_chunks": [],
            }

        routing = route_question(question)

        prediction = None
        if routing["needs_prediction"]:
            prediction = predict_machine(routing["machine_id"])

        # If the question named a machine ID but no type keyword (e.g. "Is M001 at
        # risk?" mentions nothing about it being a CNC mill), fall back to the
        # machine type resolved from the prediction itself so retrieval still gets
        # filtered to relevant docs instead of searching the whole corpus.
        retrieval_machine_type = routing["machine_type"]
        if retrieval_machine_type is None and prediction and prediction.get("found"):
            retrieval_machine_type = prediction.get("machine_type")

        chunks = []
        if routing["needs_retrieval"]:
            chunks = retrieve(question, k=3, machine_type=retrieval_machine_type)

        answer_text = self.chain.invoke({
            "question": question,
            "prediction_context": _format_prediction_context(prediction),
            "retrieval_context": _format_retrieval_context(chunks),
        })

        return {
            "question": question,
            "answer": answer_text,
            "routing": routing,
            "prediction": prediction,
            "retrieved_chunks": chunks,
        }


if __name__ == "__main__":
    # Requires OPENAI_API_KEY to be set — see .env.example. Not runnable in the
    # sandbox used to build this (see README Phase 3 notes); run locally.
    assistant = ManufacturingAssistant()
    result = assistant.answer("Is machine M014 at risk of failing soon?")
    print(result["answer"])
