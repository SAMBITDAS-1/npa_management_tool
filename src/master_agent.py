"""
master_agent.py  –  NPA Master Agent
=====================================
A single LangChain agent that automatically routes user requests to one of
three specialised tools:

  1. policy_qa            – answers questions about the bank's NPA recovery policy
  2. account_recommendation – fetches an account from accounts.csv and returns
                              corrective-action recommendations + OTS estimate
  3. branch_analysis      – generates a branch-level (or consolidated) NPA report

The agent maintains conversational memory across turns so that follow-up
questions ("what about account 12345?", "now show me branch B02") work
seamlessly without the user having to restart.

Routing heuristics handled by the agent
-----------------------------------------
- Bare account number  → account_recommendation tool
- Branch code / "ALL"  → branch_analysis tool
- Everything else      → policy_qa (default RAG tool)
- Follow-up Qs after an account lookup → account_recommendation (follow-up mode)
"""

from __future__ import annotations

import re
from typing import Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Internal imports ──────────────────────────────────────────────────────────
from .rag_setup import qa_chain, retriever, llm, decompose_and_answer
from .action_recommendation import get_account_recommendation, answer_followup
from .branch_analysis import get_branch_analysis

# ═══════════════════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def policy_qa(question: str) -> str:
    """
    Answer questions about the bank's NPA Recovery Policy document.

    Use this tool when the user asks about:
    - Policy rules, guidelines, or procedures
    - OTS / compromise settlement policy
    - Authority levels and delegation of powers
    - SARFAESI / DRT / legal recovery procedures
    - Provisioning norms or RBI classification rules
    - Any general banking / NPA policy question

    Input: the user's question as a plain string.
    Output: a detailed answer grounded in the policy document.
    """
    try:
        result = decompose_and_answer(
            question=question,
            base_chain=qa_chain,
            llm=llm,
            chat_history=[],          # agent memory handles history externally
        )
        answer      = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        # Append compact source references
        if source_docs:
            seen, lines = set(), []
            for doc in source_docs:
                meta    = doc.metadata
                key     = (meta.get("page", "?"), meta.get("type", "text"),
                           meta.get("section_heading", "")[:50])
                if key in seen:
                    continue
                seen.add(key)
                entry = f"- Page {meta.get('page', '?')} [{meta.get('type','text').upper()}]"
                heading = meta.get("section_heading", "").strip()
                if heading:
                    entry += f"  —  *{heading[:80]}*"
                lines.append(entry)
            if lines:
                answer += "\n\n---\n📄 **Sources:**\n" + "\n".join(lines)

        return answer or "No relevant policy information found."
    except Exception as exc:
        return f"Policy Q&A error: {exc}"


@tool
def account_recommendation(input_text: str) -> str:
    """
    Retrieve NPA account details and generate corrective-action recommendations.

    Use this tool when the user:
    - Provides a numeric account number (digits only)
    - Asks for recommendations, OTS estimate, or recovery actions for a specific account
    - Asks a follow-up question about an account that was already looked up
      (in that case pass the follow-up question as input_text; the tool will
       detect it is not a bare account number and attempt a follow-up answer)

    Input: an account number (digits) OR a follow-up question string.
    Output: structured markdown with classification, recommended actions, OTS range,
            upgrade path, and review timeline.
    """
    text = input_text.strip()

    # Pure account number → fresh lookup
    if re.fullmatch(r"\d+", text):
        return get_account_recommendation(text)

    # Follow-up question – we have no account context here, so ask the user
    return (
        "Please provide the account number first so I can look up the account "
        "details and then answer your follow-up question."
    )


@tool
def branch_analysis(branch_code: str) -> str:
    """
    Generate a branch-level NPA portfolio report.

    Use this tool when the user:
    - Provides a branch code (e.g. "B01", "BR002", "MAIN")
    - Types "ALL" to see a consolidated report across all branches
    - Asks for NPA ratios, top NPA accounts, scheme-wise breakdown, or
      portfolio overview for a branch

    Input: a branch code string or "ALL".
    Output: a structured markdown report with portfolio overview, NPA breakdown,
            top accounts, scheme-wise analysis, and (for ALL) a branch comparison table.
    """
    return get_branch_analysis(branch_code.strip().upper())


# ═══════════════════════════════════════════════════════════════════════════════
# Agent assembly
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are **NPA Saarthi**, an intelligent NPA (Non-Performing Asset) \
management assistant for an Indian public-sector bank.

You have access to three specialised tools:

1. **policy_qa** – for any question about the bank's NPA recovery policy document.
2. **account_recommendation** – for fetching account details and generating \
   corrective-action recommendations / OTS estimates for a specific account number.
3. **branch_analysis** – for generating branch-level or consolidated NPA portfolio reports.

## Routing rules
- If the user message is a **numeric account number** → use `account_recommendation`.
- If the user message is a **branch code** (e.g. "B01") or the word "ALL" → use `branch_analysis`.
- If the user is asking about **policy, rules, guidelines, authority, SARFAESI, DRT, OTS policy, \
  provisioning** → use `policy_qa`.
- If the user asks a **follow-up question** about an account already discussed → use \
  `account_recommendation` with the follow-up text.
- When in doubt, use `policy_qa`.

## Behaviour guidelines
- Always respond in clear, professional English.
- For multi-part questions, call tools sequentially and synthesise the results.
- Never fabricate account data or policy rules; rely entirely on the tools.
- Keep non-tool responses brief and direct; let the tool output carry the detail.
- Greet the user warmly on the first turn and explain the three capabilities.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_TOOLS = [policy_qa, account_recommendation, branch_analysis]

_agent = create_openai_tools_agent(llm=llm, tools=_TOOLS, prompt=_PROMPT)

# Sliding-window memory (last 10 exchanges) to keep context manageable
_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10,
)

agent_executor = AgentExecutor(
    agent=_agent,
    tools=_TOOLS,
    memory=_memory,
    verbose=True,          # set False in production
    handle_parsing_errors=True,
    max_iterations=5,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Public helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_master_agent(user_message: str) -> str:
    """
    Send a message to the master agent and return its response.

    Parameters
    ----------
    user_message : str
        Raw text from the user (account number, branch code, or natural-language question).

    Returns
    -------
    str
        The agent's markdown-formatted response.
    """
    try:
        result = agent_executor.invoke({"input": user_message})
        return result.get("output", "I could not generate a response. Please try again.")
    except Exception as exc:
        return f"⚠️ Agent error: {exc}"


def reset_agent_memory() -> None:
    """Clear the agent's conversation memory (e.g. on session reset)."""
    _memory.clear()
