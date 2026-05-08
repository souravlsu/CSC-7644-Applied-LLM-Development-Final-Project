"""Streamlit UI for Fluent UDF Copilot.

Features:
  - Chat-style prompt/response.
  - Runtime model selection.
  - RAG on/off toggle.
  - Separate report tabs for retrieval, static checks, and evaluation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.assistant import Assistant
from src.config import ARTIFACTS_DIR, CONFIG, INDEX_DIR
from src.evaluate import summarize
from src.llm_client import LLMClient
from src.retriever import Retriever

st.set_page_config(page_title="Fluent UDF Copilot", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def load_retriever(
    index_dir: str, embed_model: str, embed_device: str, embed_batch_size: int
) -> Retriever:
    """Load and cache the retriever to avoid rebuilding per interaction."""
    return Retriever.load(
        Path(index_dir),
        embed_model=embed_model,
        embed_device=embed_device,
        embed_batch_size=embed_batch_size,
    )


def load_eval_artifact(path: Path) -> tuple[list[dict], dict]:
    """Load saved evaluation records and compute summary metrics."""
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize([_RecordAdapter(item) for item in data])  # type: ignore[misc]
    return data, summary


class _RecordAdapter:
    """Adapter to feed saved JSON records into evaluate.summarize."""

    def __init__(self, payload: dict) -> None:
        self.mode = payload.get("mode")
        self.macro_family_correct = payload.get("macro_family_correct")
        self.structural_pass = payload.get("structural_pass", False)
        self.cites_context = payload.get("cites_context")


def assistant_from_sidebar() -> tuple[Assistant | None, str]:
    """Build runtime-configured Assistant from sidebar settings."""
    st.sidebar.header("Runtime Settings")
    index_dir = st.sidebar.text_input("Index directory", value=str(INDEX_DIR))
    base_url = st.sidebar.text_input("LLM base URL", value=CONFIG.llm_base_url)
    api_key = st.sidebar.text_input(
        "LLM API key", value=CONFIG.llm_api_key, type="password"
    )

    model_choices = [
        "llama3.2:3b",
        "llama3.1:8b",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "custom",
    ]
    picked = st.sidebar.selectbox("Model preset", model_choices, index=0)
    custom_model = st.sidebar.text_input("Custom model id", value=CONFIG.llm_model)
    model = custom_model if picked == "custom" else picked

    temp = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(CONFIG.llm_temperature),
        step=0.05,
    )
    max_tokens = st.sidebar.number_input(
        "Max tokens",
        min_value=128,
        max_value=4096,
        value=int(CONFIG.llm_max_tokens),
        step=64,
    )

    embed_model = st.sidebar.text_input("Embedding model", value=CONFIG.embed_model)
    allowed_devices = ["auto", "cpu", "cuda"]
    default_device = (
        CONFIG.embed_device if CONFIG.embed_device in set(allowed_devices) else "auto"
    )
    embed_device = st.sidebar.selectbox(
        "Embedding device",
        allowed_devices,
        index=allowed_devices.index(default_device),
    )
    embed_batch_size = st.sidebar.number_input(
        "Embedding batch size",
        min_value=1,
        max_value=512,
        value=int(CONFIG.embed_batch_size),
        step=1,
    )

    try:
        retriever = load_retriever(
            index_dir, embed_model, embed_device, int(embed_batch_size)
        )
    except Exception as err:
        return None, f"Failed to load index: {err}"

    llm = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(temp),
        max_tokens=int(max_tokens),
    )
    return Assistant(retriever=retriever, llm=llm, config=CONFIG), ""


def render_chat_tab(assistant: Assistant) -> None:
    """Render chat UI and run one-shot generation on user prompts."""
    col_chat, _, col_rag = st.columns([5, 0.35, 2.65], vertical_alignment="top")

    with col_chat:
        st.subheader("Chat")
        st.caption(
            "Ask for UDF generation, explanation, or debugging. "
            "The app keeps a local session chat history."
        )

    with col_rag:
        with st.container(border=True):
            st.markdown(
                """
                <p style="
                    margin: 0;
                    font-size: 0.82rem;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    opacity: 0.95;
                    text-transform: uppercase;
                    text-align: right;
                ">Retrieval (RAG)</p>
                <p style="
                    margin: 6px 0 12px 0;
                    font-size: 0.76rem;
                    opacity: 0.78;
                    line-height: 1.35;
                    text-align: right;
                ">Ground responses in the indexed Fluent manual & example UDFs.</p>
                """,
                unsafe_allow_html=True,
            )
            _toggle_pad, toggle_col = st.columns([0.12, 0.88])
            with toggle_col:
                rag_enabled = st.toggle(
                    "RAG — use manual & examples",
                    value=True,
                    key="rag_enabled_toggle",
                    help="Off: baseline LLM only. On: retrieval-augmented prompts using Top‑K passages below.",
                )
            st.slider(
                "Top‑K passages",
                min_value=1,
                max_value=15,
                value=int(CONFIG.top_k),
                step=1,
                key="ui_top_k",
                help="How many passages are merged into the prompt when RAG is enabled.",
            )

    mode = "rag" if rag_enabled else "baseline"
    top_k = int(st.session_state.get("ui_top_k", CONFIG.top_k))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Example: write a parabolic inlet velocity UDF for a 2D channel"
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                result = assistant.ask(
                    prompt,
                    mode=mode,  # type: ignore[arg-type]
                    top_k=top_k,
                )
            st.markdown(result.answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer}
            )
            st.session_state.last_result = result

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


def render_retrieval_report_tab() -> None:
    """Render retrieved context for the most recent response."""
    st.subheader("Retrieval Report")
    result = st.session_state.get("last_result")
    if not result:
        st.info("Run at least one query in Chat tab to populate retrieval report.")
        return
    if not result.retrieved:
        st.warning(
            "Last run was baseline mode (RAG off), so there are no retrieved chunks."
        )
        return

    st.metric("Retrieved chunks", len(result.retrieved))
    for idx, item in enumerate(result.retrieved, start=1):
        chunk = item.chunk
        with st.expander(
            f"[C{idx}] score={item.score:.3f} | {chunk.section} | "
            f"pp.{chunk.page_start}-{chunk.page_end}"
        ):
            st.write(f"**Source:** `{chunk.source}`")
            st.write(
                f"**Macros:** {', '.join(chunk.macro_families) if chunk.macro_families else '-'}"
            )
            st.code(
                chunk.text[:2000] + ("..." if len(chunk.text) > 2000 else ""),
                language="text",
            )


def render_checker_report_tab() -> None:
    """Render static checker details for the last answer."""
    st.subheader("Static Checker Report")
    result = st.session_state.get("last_result")
    if not result:
        st.info("Run at least one query in Chat tab to populate checker report.")
        return
    check = result.check.to_dict()
    st.json(check)
    if check.get("warnings"):
        st.warning("Warnings found. Review before using code in Fluent.")
    else:
        st.success("No checker warnings on the last response.")


def render_eval_report_tab() -> None:
    """Render summary and raw table from saved evaluation artifacts."""
    st.subheader("Evaluation Report")
    path = st.text_input(
        "Evaluation artifact path", value=str(ARTIFACTS_DIR / "eval_records.json")
    )
    records, summary = load_eval_artifact(Path(path))
    if not records:
        st.info(
            "No evaluation artifact found yet. Run `python -m scripts.evaluate` first."
        )
        return

    st.write("### Summary")
    for mode, info in summary.items():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{mode}: prompts", info.get("prompts", 0))
        with c2:
            value = info.get("macro_accuracy")
            st.metric(
                f"{mode}: macro acc", "n/a" if value is None else f"{100*value:.1f}%"
            )
        with c3:
            value = info.get("structural_pass_rate")
            st.metric(
                f"{mode}: structural", "n/a" if value is None else f"{100*value:.1f}%"
            )
        with c4:
            value = info.get("citation_rate")
            st.metric(
                f"{mode}: citation", "n/a" if value is None else f"{100*value:.1f}%"
            )

    st.write("### Raw Records")
    st.dataframe(records, use_container_width=True)


def main() -> None:
    """Entry point for the Streamlit application."""
    st.title("Fluent UDF Copilot")
    st.caption("Retrieval-grounded UDF generation, explanation, and debugging.")

    assistant, error = assistant_from_sidebar()
    if assistant is None:
        st.error(error)
        st.stop()

    tab_chat, tab_retrieval, tab_checker, tab_eval = st.tabs(
        ["Chat", "Retrieval Report", "Checker Report", "Evaluation Report"]
    )
    with tab_chat:
        render_chat_tab(assistant)
    with tab_retrieval:
        render_retrieval_report_tab()
    with tab_checker:
        render_checker_report_tab()
    with tab_eval:
        render_eval_report_tab()


if __name__ == "__main__":
    main()
