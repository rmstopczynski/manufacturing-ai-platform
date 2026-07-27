"""
Streamlit front end: a chat interface backed by /ask, plus a sidebar dashboard of
current machine health pulled from /predict for every known machine.

Talks to the FastAPI backend over HTTP (not by importing the orchestration code
directly), matching how this would actually be deployed — API and UI as separate
services/containers — rather than coupling the UI to internal Python modules.
"""

import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Manufacturing AI Assistant", page_icon="🏭", layout="wide")


@st.cache_data(ttl=60)
def get_known_machine_ids() -> list[str]:
    # Reads the raw sensor log directly for the machine ID list, since that's cheap
    # local data — no need to round-trip through the API just to populate a dropdown.
    try:
        df = pd.read_csv("data/raw/sensor_logs.csv")
        return sorted(df["machine_id"].unique().tolist())
    except FileNotFoundError:
        return []


def check_api_reachable() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def fetch_prediction(machine_id: str) -> dict:
    try:
        # Short timeout deliberately — this is called once per machine (potentially
        # dozens), sequentially. A generous timeout here means a slow/unreachable API
        # blocks the entire page for minutes before anything renders. Fail fast instead;
        # an unreachable backend should surface immediately, not after a long hang.
        resp = requests.post(f"{API_BASE_URL}/predict", params={"machine_id": machine_id}, timeout=3)
        if resp.status_code == 404:
            return {"machine_id": machine_id, "status": "no data"}
        resp.raise_for_status()
        data = resp.json()
        status = "AT RISK" if data["failure_predicted_within_7d"] else "healthy"
        return {
            "machine_id": machine_id,
            "status": status,
            "failure_probability": data["failure_probability"],
            "days_since_maintenance": data["days_since_maintenance"],
        }
    except requests.exceptions.RequestException as e:
        return {"machine_id": machine_id, "status": f"API error: {e}"}


def fetch_all_predictions(machine_ids: list[str]) -> list[dict]:
    # Fetching sequentially here would mean N machines * request latency before the
    # page renders anything — with 60 machines, that adds up even at a few hundred ms
    # each. A thread pool fans the requests out concurrently instead, so the wall-clock
    # cost is roughly one request's worth of latency, not N of them.
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(fetch_prediction, machine_ids))


def render_dashboard():
    st.sidebar.header("Machine Health Dashboard")
    machine_ids = get_known_machine_ids()
    if not machine_ids:
        st.sidebar.warning("No machine data found. Run data/generate_sensor_data.py first.")
        return

    if st.sidebar.button("Refresh health scores"):
        st.cache_data.clear()

    # One cheap health check before looping through every machine — if the API is
    # down, say so once instead of hitting N slow timeouts in a row.
    if not check_api_reachable():
        st.sidebar.error(
            f"Can't reach the API at {API_BASE_URL}. "
            f"Make sure `uvicorn api.main:app --port 8000` is running in another terminal."
        )
        return

    rows = fetch_all_predictions(machine_ids)
    df = pd.DataFrame(rows)

    def highlight_risk(row):
        color = "#ffcccc" if row.get("status") == "AT RISK" else ""
        return [f"background-color: {color}"] * len(row)

    if not df.empty:
        st.sidebar.dataframe(
            df.style.apply(highlight_risk, axis=1),
            width="stretch",
            hide_index=True,
        )


def render_chat():
    st.title("🏭 Manufacturing AI Assistant")
    st.caption(
        "Ask about machine health, maintenance procedures, or troubleshooting. "
        "Mention a machine ID (e.g. M001) to get a live prediction alongside relevant documentation."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("details"):
                with st.expander("What was used to generate this answer"):
                    st.json(msg["details"])

    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=30)
                    if resp.status_code >= 400:
                        # Surface the API's actual `detail` message (e.g. the real LLM
                        # provider error) rather than raise_for_status()'s generic
                        # "502 Server Error" — that generic text hides the one piece of
                        # information actually needed to debug a failure here.
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except ValueError:
                            detail = resp.text
                        answer = f"The assistant API returned an error ({resp.status_code}): {detail}"
                        details = None
                    else:
                        data = resp.json()
                        answer = data["answer"]
                        details = {
                            "routing": data.get("routing"),
                            "prediction": data.get("prediction"),
                            "retrieved_chunks": data.get("retrieved_chunks"),
                        }
                except requests.exceptions.RequestException as e:
                    answer = f"Sorry, I couldn't reach the assistant API: {e}"
                    details = None

            st.markdown(answer)
            if details:
                with st.expander("What was used to generate this answer"):
                    st.json(details)

        st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})


def main():
    render_dashboard()
    render_chat()


if __name__ == "__main__":
    main()
