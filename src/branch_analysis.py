"""
branch_analysis.py  –  Branch-level NPA Portfolio Reporter
============================================================
Generates a structured markdown report for a single branch or
a consolidated view across all branches.

CSV columns used
----------------
account_number | branch_code | customer | npa_status | outstanding_amount
days_past_due  | security_amount | limit | scheme

NPA Status codes
-----------------
4 → Standard / SMA      (not NPA)
5 → Substandard         (NPA)
6 → Doubtful I          (NPA)
7 → Doubtful II         (NPA)
8 → Loss                (NPA)
"""

import os
import pandas as pd

# ── RBI NPA classification map ────────────────────────────────────────────────
NPA_STATUS_MAP = {
    4: "Standard / SMA",
    5: "Substandard",
    6: "Doubtful I",
    7: "Doubtful II",
    8: "Loss",
}

NPA_CODES = {5, 6, 7, 8}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "accounts.csv")
    df = pd.read_csv(csv_path, dtype={"branch_code": str, "account_number": str})
    df["npa_status"]         = pd.to_numeric(df["npa_status"],         errors="coerce")
    df["outstanding_amount"] = pd.to_numeric(df["outstanding_amount"], errors="coerce").fillna(0)
    df["days_past_due"]      = pd.to_numeric(df["days_past_due"],      errors="coerce").fillna(0)
    df["security_amount"]    = pd.to_numeric(df["security_amount"],    errors="coerce").fillna(0)
    df["limit"]              = pd.to_numeric(df["limit"],              errors="coerce").fillna(0)
    return df


def _fmt_amount(amount: float) -> str:
    """Format large amounts in Indian numbering (lakhs / crores)."""
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} L"
    return f"₹{amount:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_branch_analysis(branch_code: str) -> str:
    """
    Generate a branch-level NPA portfolio report.

    Parameters
    ----------
    branch_code : str
        A specific branch code (e.g. "B01") or "ALL" for a consolidated view.

    Returns
    -------
    str
        A structured markdown report.
    """
    df = _load_data()
    branch_code = branch_code.strip().upper()

    if branch_code == "ALL":
        data  = df.copy()
        title = "All Branches — Consolidated NPA Profile"
    else:
        data  = df[df["branch_code"] == branch_code]
        title = f"Branch {branch_code} — NPA Profile"

    if data.empty:
        return (
            f"⚠️ No data found for branch code **{branch_code}**. "
            "Please check the code and try again."
        )

    # ── Core counts ───────────────────────────────────────────────────────────
    total_accounts = len(data)
    npa_data       = data[data["npa_status"].isin(NPA_CODES)]
    npa_count      = len(npa_data)
    standard_count = total_accounts - npa_count
    npa_ratio      = (npa_count / total_accounts * 100) if total_accounts else 0

    # ── Amount metrics ────────────────────────────────────────────────────────
    total_outstanding      = data["outstanding_amount"].sum()
    npa_outstanding        = npa_data["outstanding_amount"].sum()
    npa_outstanding_ratio  = (npa_outstanding / total_outstanding * 100) if total_outstanding else 0
    total_security         = npa_data["security_amount"].sum()
    net_npa                = max(npa_outstanding - total_security, 0)
    provision_coverage     = (total_security / npa_outstanding * 100) if npa_outstanding else 0
    avg_dpd_all            = data["days_past_due"].mean()
    avg_dpd_npa            = npa_data["days_past_due"].mean() if npa_count else 0

    # ── Status breakdown ──────────────────────────────────────────────────────
    status_lines = []
    for code in sorted(data["npa_status"].dropna().unique()):
        label  = NPA_STATUS_MAP.get(int(code), f"Code {int(code)}")
        subset = data[data["npa_status"] == code]
        flag   = " ⚠️" if code in NPA_CODES else ""
        status_lines.append(
            f"  - {label}{flag}: {len(subset)} accounts | {_fmt_amount(subset['outstanding_amount'].sum())}"
        )
    status_breakdown = "\n".join(status_lines)

    # ── Top 5 NPA accounts ────────────────────────────────────────────────────
    top5 = (
        npa_data.nlargest(5, "outstanding_amount")[
            ["account_number", "customer", "outstanding_amount", "npa_status", "days_past_due", "scheme"]
        ]
        if npa_count else pd.DataFrame()
    )
    top5_lines = []
    for _, row in top5.iterrows():
        status_label = NPA_STATUS_MAP.get(int(row["npa_status"]), str(int(row["npa_status"])))
        top5_lines.append(
            f"  {row['account_number']} | {str(row['customer']).strip()[:28]:<28} | "
            f"{_fmt_amount(row['outstanding_amount']):>12} | {status_label} | DPD: {int(row['days_past_due'])}"
        )

    # ── Scheme-wise NPA breakdown ─────────────────────────────────────────────
    scheme_npa = (
        npa_data.groupby("scheme")
        .agg(accounts=("account_number", "count"), outstanding=("outstanding_amount", "sum"))
        .sort_values("outstanding", ascending=False)
        .head(8)
    )
    scheme_lines = [
        f"  - {str(row.name).strip()[:40]:<40} | {int(row['accounts']):>4} accts | {_fmt_amount(row['outstanding'])}"
        for _, row in scheme_npa.iterrows()
    ] if not scheme_npa.empty else ["  (No NPA accounts)"]

    # ── Branch comparison (ALL only) ──────────────────────────────────────────
    branch_summary_block = ""
    if branch_code == "ALL":
        branch_summary = (
            data.groupby("branch_code")
            .apply(lambda g: pd.Series({
                "total":           len(g),
                "npa":             g["npa_status"].isin(NPA_CODES).sum(),
                "outstanding":     g["outstanding_amount"].sum(),
                "npa_outstanding": g.loc[g["npa_status"].isin(NPA_CODES), "outstanding_amount"].sum(),
            }))
            .reset_index()
        )
        branch_summary["npa_ratio"] = (
            branch_summary["npa"] / branch_summary["total"] * 100
        ).round(1)
        top_branches = branch_summary.sort_values("npa_outstanding", ascending=False).head(10)
        b_lines = [
            f"  Branch {row['branch_code']:>6} | NPA: {int(row['npa']):>4}/{int(row['total']):<4} "
            f"({row['npa_ratio']:>5.1f}%) | NPA O/S: {_fmt_amount(row['npa_outstanding'])}"
            for _, row in top_branches.iterrows()
        ]
        branch_summary_block = (
            "\n\n---\n### 🏢 Top 10 Branches by NPA Outstanding\n" + "\n".join(b_lines)
        )

    # ── Compose report ────────────────────────────────────────────────────────
    top5_header = f"{'Account No.':<14} {'Customer':<30} {'Outstanding':>12}   {'Status':<16} DPD"
    top5_sep    = "-" * 85
    top5_body   = "\n".join(top5_lines) if top5_lines else "  (No NPA accounts)"

    report = f"""
## {title}

---
### 📊 Portfolio Overview
| Metric | Value |
|--------|-------|
| Total Accounts | {total_accounts:,} |
| NPA Accounts | {npa_count:,} |
| Standard / SMA Accounts | {standard_count:,} |
| **NPA Account Ratio** | **{npa_ratio:.2f}%** |
| Total Outstanding | {_fmt_amount(total_outstanding)} |
| NPA Outstanding | {_fmt_amount(npa_outstanding)} |
| **NPA Outstanding Ratio** | **{npa_outstanding_ratio:.2f}%** |
| Security Held (NPA accts) | {_fmt_amount(total_security)} |
| Net NPA (after security) | {_fmt_amount(net_npa)} |
| Security Coverage | {provision_coverage:.1f}% |
| Avg Days Past Due (All) | {avg_dpd_all:.0f} days |
| Avg Days Past Due (NPA) | {avg_dpd_npa:.0f} days |

---
### 🗂️ Breakdown by NPA Classification
{status_breakdown}

---
### 📋 Top 5 NPA Accounts by Outstanding Amount
```
{top5_header}
{top5_sep}
{top5_body}
```

---
### 🏷️ Scheme-wise NPA Breakdown (Top 8)
{chr(10).join(scheme_lines)}{branch_summary_block}
"""
    return report.strip()
