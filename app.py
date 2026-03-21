"""
app.py  –  NPA Management Tool (Streamlit)
==========================================
Single-chat interface powered by the Master Agent.

The master agent automatically routes each message to the correct
specialised tool (Policy Q&A / Account Recommendation / Branch Analysis)
so the user never needs to switch tabs or select a mode.

Manual tool tabs are still available via the sidebar for users who prefer
a structured workflow.
"""

import streamlit as st
from src.master_agent import run_master_agent, reset_agent_memory
from src.rag_setup import qa_chain, retriever, llm, decompose_and_answer
from src.action_recommendation import get_account_recommendation, answer_followup
from src.branch_analysis import get_branch_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPA Management Tool",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 NPA Management Tool")

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.sidebar.radio(
    "Interface Mode",
    ["🤖 Master Agent (Auto-route)", "📋 Manual Tools"],
    index=0,
)

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar.expander("ℹ️ RAG Pipeline", expanded=False):
    st.markdown(
        """
        **Active pipeline:**
        - 📄 Parent-document retrieval
        - 🔀 Hybrid BM25 + Vector (MMR)
        - 🎯 Cohere re-ranking *(if key set)*
        - 🧩 Query decomposition
        - 📊 Table-aware extraction
        """
    )

with st.sidebar.expander("🤖 Master Agent Tools", expanded=False):
    st.markdown(
        """
        The agent selects the right tool automatically:

        | Trigger | Tool |
        |---|---|
        | Account number | Account Recommendation |
        | Branch code / ALL | Branch Analysis |
        | Policy question | Policy Q&A |
        | Follow-up question | Last active tool |
        """
    )

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("agent_chat_history",       []),
    ("manual_chat_history",      []),
    ("qa_chat_history",          []),
    ("last_account_context",     None),
    ("manual_tool",              "Policy Q&A"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def render_messages(history_key: str):
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def add_message(history_key: str, role: str, content: str):
    st.session_state[history_key].append({"role": role, "content": content})


def is_account_number(text: str) -> bool:
    return text.strip().isdigit()


def format_source_chunks(source_documents: list) -> str:
    if not source_documents:
        return ""
    seen, lines = set(), []
    for doc in source_documents:
        meta    = doc.metadata
        page    = meta.get("page", "?")
        ctype   = meta.get("type", "text").upper()
        heading = meta.get("section_heading", "").strip()
        key     = (page, ctype, heading[:50])
        if key in seen:
            continue
        seen.add(key)
        entry = f"- Page {page} [{ctype}]"
        if heading:
            entry += f"  —  *{heading[:80]}*"
        lines.append(entry)
    if not lines:
        return ""
    return "\n\n---\n📄 **Sources used:**\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1:  MASTER AGENT  (Auto-route)
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🤖 Master Agent (Auto-route)":
    st.header("💬 NPA Saarthi — Intelligent Assistant")
    st.caption(
        "Ask anything: enter an **account number**, a **branch code** (or ALL), "
        "or any **policy question**. The agent will route your request automatically."
    )

    render_messages("agent_chat_history")

    user_input = st.chat_input(
        "e.g.  '100234'  |  'Branch B01'  |  'What is the OTS policy for Loss Assets?'"
    )

    if user_input:
        add_message("agent_chat_history", "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("NPA Saarthi is thinking…"):
                response_text = run_master_agent(user_input)
            st.markdown(response_text)

        add_message("agent_chat_history", "assistant", response_text)

    if st.session_state.agent_chat_history:
        if st.button("🗑️ Clear Conversation", key="clear_agent"):
            st.session_state.agent_chat_history = []
            reset_agent_memory()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2:  MANUAL TOOLS  (user picks the tool)
# ═══════════════════════════════════════════════════════════════════════════════

else:
    tool_option = st.sidebar.selectbox(
        "Select Tool",
        ["Policy Q&A", "Account Recommendation", "Branch Analysis"],
    )

    # Clear history on tool switch
    if st.session_state.manual_tool != tool_option:
        st.session_state.manual_tool            = tool_option
        st.session_state.manual_chat_history    = []
        st.session_state.qa_chat_history        = []
        st.session_state.last_account_context   = None

    # ── 2a.  POLICY Q&A ───────────────────────────────────────────────────────
    if tool_option == "Policy Q&A":
        st.header("Ask Questions about the Bank's Recovery Policy")

        show_sources  = st.sidebar.toggle("Show source references", value=True)
        use_decompose = st.sidebar.toggle("Query decomposition (complex Qs)", value=True)

        render_messages("manual_chat_history")

        question = st.chat_input("Ask a question about the NPA recovery policy…")
        if question:
            add_message("manual_chat_history", "user", question)
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analysing policy…"):
                    if use_decompose:
                        response = decompose_and_answer(
                            question=question,
                            base_chain=qa_chain,
                            llm=llm,
                            chat_history=st.session_state.qa_chat_history,
                        )
                    else:
                        response = qa_chain.invoke({
                            "question":     question,
                            "chat_history": st.session_state.qa_chat_history,
                        })

                    answer      = response.get("answer", "")
                    source_docs = response.get("source_documents", [])

                    display_answer = answer
                    if show_sources:
                        display_answer = answer + format_source_chunks(source_docs)

                    st.markdown(display_answer)

            add_message("manual_chat_history", "assistant", display_answer)
            st.session_state.qa_chat_history.append((question, answer))

        if st.session_state.manual_chat_history:
            if st.button("🗑️ Clear Conversation", key="clear_qa"):
                st.session_state.manual_chat_history = []
                st.session_state.qa_chat_history     = []
                st.rerun()

    # ── 2b.  ACCOUNT RECOMMENDATION ──────────────────────────────────────────
    elif tool_option == "Account Recommendation":
        st.header("Get Action Recommendations for NPA Account")
        render_messages("manual_chat_history")

        user_input = st.chat_input(
            "Enter an account number, or ask a follow-up question about the last account…"
        )

        if user_input:
            add_message("manual_chat_history", "user", user_input)
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Processing…"):
                    if is_account_number(user_input):
                        response_text = get_account_recommendation(user_input.strip())
                        st.session_state.last_account_context = response_text
                    else:
                        if st.session_state.last_account_context:
                            response_text = answer_followup(
                                question=user_input,
                                account_context=st.session_state.last_account_context,
                            )
                        else:
                            response_text = (
                                "Please enter an account number first so I have "
                                "context to answer your follow-up question."
                            )
                    st.markdown(response_text)

            add_message("manual_chat_history", "assistant", response_text)

        if st.session_state.manual_chat_history:
            if st.button("🗑️ Clear Conversation", key="clear_acct"):
                st.session_state.manual_chat_history  = []
                st.session_state.last_account_context = None
                st.rerun()

    # ── 2c.  BRANCH ANALYSIS ──────────────────────────────────────────────────
    elif tool_option == "Branch Analysis":
        st.header("Branch-wise NPA Profile")
        render_messages("manual_chat_history")

        user_input = st.chat_input(
            "Enter a branch code (or 'ALL'), or ask a follow-up question…"
        )
        if user_input:
            add_message("manual_chat_history", "user", user_input)
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Generating report…"):
                    try:
                        report = get_branch_analysis(user_input.upper())
                    except Exception as exc:
                        report = f"⚠️ Error generating report: {exc}"
                    st.markdown(report)

            add_message("manual_chat_history", "assistant", report)

        if st.session_state.manual_chat_history:
            if st.button("🗑️ Clear Conversation", key="clear_branch"):
                st.session_state.manual_chat_history = []
                st.rerun()
