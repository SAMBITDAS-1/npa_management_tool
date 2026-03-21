"""
action_recommendation.py  –  NPA Account Recommendation Engine
===============================================================
Uses every column in accounts.csv to build a rich account profile,
then retrieves policy guidance via the RAG pipeline in rag_setup.py
before asking the LLM for corrective actions + OTS estimate.

CSV columns used
----------------
account_number  | branch_code   | npa_status  | outstanding_amount
days_past_due   | security_amount | scheme     | customer          | limit

NPA Status codes (RBI classification)
--------------------------------------
4 → Sub-Standard (< 12 months past NPA date)
5 → Doubtful-1   (12–24 months)
6 → Doubtful-2   (24–36 months)
7 → Doubtful-3   (> 36 months)
8 → Loss Asset
"""

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .rag_setup import retriever

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_NPA_LABELS = {
    4: "Sub-Standard",
    5: "Doubtful-1 (12–24 months)",
    6: "Doubtful-2 (24–36 months)",
    7: "Doubtful-3 (> 36 months)",
    8: "Loss Asset",
}


def _npa_label(status_code: int) -> str:
    return _NPA_LABELS.get(status_code, f"Unknown (code {status_code})")


def _coverage_ratio(outstanding: float, security: float) -> str:
    if outstanding <= 0:
        return "N/A"
    return f"{(security / outstanding) * 100:.1f}%"


def _unsecured_amount(outstanding: float, security: float) -> float:
    return max(0.0, outstanding - security)


def _utilisation_ratio(outstanding: float, limit: float) -> str:
    if limit <= 0:
        return "N/A"
    return f"{(outstanding / limit) * 100:.1f}%"


def _build_rag_query(info: pd.Series) -> str:
    npa_label = _npa_label(int(info["npa_status"]))
    sec_note  = "secured" if info["security_amount"] > 0 else "unsecured"
    return (
        f"NPA classification: {npa_label}, "
        f"days past due: {info['days_past_due']}, "
        f"outstanding amount: {info['outstanding_amount']}, "
        f"security/collateral: {info['security_amount']} ({sec_note}), "
        f"loan scheme: {info['scheme'].strip()}, "
        f"sanctioned limit: {info['limit']}, "
        f"corrective actions OTS compromise settlement recovery"
    )


def _build_prompt(info: pd.Series, policy_context: str) -> str:
    npa_label   = _npa_label(int(info["npa_status"]))
    coverage    = _coverage_ratio(float(info["outstanding_amount"]), float(info["security_amount"]))
    unsecured   = _unsecured_amount(float(info["outstanding_amount"]), float(info["security_amount"]))
    utilisation = _utilisation_ratio(float(info["outstanding_amount"]), float(info["limit"]))

    return f"""
You are a senior NPA Recovery Officer at an Indian public-sector bank.

════════════════════════════════════════════════
BANK'S RECOVERY POLICY (retrieved sections)
════════════════════════════════════════════════
{policy_context}

════════════════════════════════════════════════
ACCOUNT PROFILE
════════════════════════════════════════════════
Account Number   : {info['account_number']}
Customer Name    : {info['customer'].strip()}
Branch Code      : {info['branch_code']}

Loan Scheme      : {info['scheme'].strip()}
Sanctioned Limit : ₹{float(info['limit']):,.0f}

NPA Status       : {npa_label}  (code {int(info['npa_status'])})
Days Past Due    : {info['days_past_due']} days

Outstanding Amt  : ₹{float(info['outstanding_amount']):,.0f}
Security / Coll. : ₹{float(info['security_amount']):,.0f}
Unsecured Portion: ₹{unsecured:,.0f}
Security Cover   : {coverage}  (security ÷ outstanding)
Limit Utilisation: {utilisation}  (outstanding ÷ limit)

════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════
Using the policy excerpts above AND your banking expertise, provide:

1. **Current Classification & Risk Assessment**
   - Confirm the NPA category and explain what it means for provisioning.
   - Comment on security cover adequacy and unsecured exposure.
   - Note the scheme-specific risk (e.g. PMRY, SJSRY, Micro Enterprise).

2. **Recommended Corrective Actions** (priority-ordered)
   - List concrete recovery steps aligned with the policy.
   - Specify which authority level should handle this account based on the
     outstanding amount and scheme.
   - Include legal / SARFAESI / DRT actions if warranted by the NPA stage.

3. **OTS / Compromise Settlement Estimate**
   - Suggest a tentative OTS amount range grounded in the policy.
   - Justify the range using: outstanding, security cover, NPA vintage,
     and scheme type.
   - State any conditions the borrower must meet for OTS eligibility.

4. **Upgradation / NPA Exit Path**
   - Describe how this account could be regularised or upgraded,
     and the minimum repayment required.

5. **Next Review / Escalation Timeline**
   - State when the branch should review and escalate if unresolved.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_account_recommendation(account_number: str) -> str:
    """
    Fetch the account row from accounts.csv (all columns), retrieve relevant
    policy sections via RAG, and return a structured recommendation.

    Parameters
    ----------
    account_number : str
        The account number typed by the user (digits only).

    Returns
    -------
    str
        Structured markdown recommendation from the LLM.
    """
    csv_path = os.path.join(DATA_DIR, "accounts.csv")
    df       = pd.read_csv(csv_path)

    try:
        account = df[df["account_number"] == int(account_number)]
    except ValueError:
        return "❌ Invalid account number format. Please enter digits only."

    if account.empty:
        return (
            f"❌ Account **{account_number}** not found in the database. "
            "Please verify the account number and try again."
        )

    info = account.iloc[0]

    # RAG retrieval
    rag_query = _build_rag_query(info)
    try:
        docs           = retriever.get_relevant_documents(rag_query)
        policy_context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:
        policy_context = f"[Policy retrieval unavailable: {exc}]"

    # LLM inference
    prompt   = _build_prompt(info, policy_context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def answer_followup(question: str, account_context: str) -> str:
    """
    Answer a follow-up question about an NPA account using the
    recommendation text already generated for that account.

    Parameters
    ----------
    question : str
        The user's follow-up question.
    account_context : str
        The full recommendation text returned earlier by get_account_recommendation().

    Returns
    -------
    str
        The LLM's answer grounded in the account context.
    """
    system_prompt = (
        "You are an expert NPA (Non-Performing Asset) analyst at an Indian public-sector bank. "
        "A user has already received a detailed account analysis (provided below). "
        "Answer their follow-up question using the information in that analysis "
        "combined with your banking domain knowledge. "
        "Be concise, specific, and refer to the actual account figures where relevant.\n\n"
        f"--- ACCOUNT ANALYSIS ---\n{account_context}\n--- END OF ANALYSIS ---"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])
    return response.content
