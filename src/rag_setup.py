"""
rag_setup.py  –  Advanced RAG pipeline for the NPA Recovery Policy PDF
=======================================================================

Features
---------
1. **Contextual chunking** – every chunk is prefixed with its section heading
   so isolated chunks don't lose context.
2. **Parent-document retrieval** – small child chunks are retrieved (precision),
   but the *parent* passage is sent to the LLM (recall / coherence).
3. **Multi-row table normalisation** – merges wrapped header rows; cleans
   symbol artifacts (bullets, arrows) so tables are readable downstream.
4. **Cohere re-ranker** – after hybrid retrieval, a cross-encoder re-ranks the
   top-N candidates before passing them to the LLM.
5. **Query decomposition** – complex questions are broken into sub-questions;
   each is answered separately, then the answers are synthesised.
6. **Wider retrieval window** – k raised so dense policy documents are covered.
7. **Metadata enrichment** – section title, clause number, and page carried
   through to every chunk for grounded citations.
"""

import os
import re
import json
from typing import List

import pandas as pd
import pdfplumber
from pydantic import Field
from dotenv import load_dotenv

from langchain.schema import Document, BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

# ── Environment ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../config/.env"))

# ── Constants ─────────────────────────────────────────────────────────────────
CHILD_CHUNK_SIZE    = 400
CHILD_CHUNK_OVERLAP = 60
PARENT_CHUNK_SIZE   = 1200
PARENT_CHUNK_OVERLAP = 150
RETRIEVAL_K         = 12
BM25_K              = 6
RERANK_TOP_N        = 6


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PDF PARSING
# ═══════════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[\.\)]\s+[A-Z]"
    r"|[A-Z][A-Z\s]{4,}$"
    r"|(?:CHAPTER|SECTION|ANNEXURE|CLAUSE)\s+\w+"
    r")",
    re.MULTILINE,
)

_ARTIFACT_RE = re.compile(r"[\uf0b7\uf0dc\uf0de\uf0a7\uf020\u25cf\u2022\u25a0]")


def _clean_text(text: str) -> str:
    text = _ARTIFACT_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_section_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if _HEADING_RE.match(line):
            return line[:120]
    return ""


def _normalise_table_rows(table: list) -> list:
    if not table:
        return table

    cleaned = []
    for row in table:
        cleaned_row = []
        for cell in row:
            val = str(cell).strip() if cell is not None else ""
            val = _ARTIFACT_RE.sub("", val)
            val = re.sub(r"\s+", " ", val).strip()
            cleaned_row.append(val)
        cleaned.append(cleaned_row)

    if len(cleaned) >= 2:
        n = len(cleaned[0])
        row0_empties = sum(1 for c in cleaned[0] if not c)
        row1_empties = sum(1 for c in cleaned[1] if not c)
        if n > 0 and row0_empties / n > 0.4 and row1_empties / n > 0.4:
            merged = [f"{c0} {c1}".strip() for c0, c1 in zip(cleaned[0], cleaned[1])]
            cleaned = [merged] + cleaned[2:]

    return cleaned


def _table_to_markdown(table: list) -> str | None:
    table = _normalise_table_rows(table)
    if not table or len(table) < 2:
        return None

    header = table[0]
    rows   = [r for r in table[1:] if any(c for c in r)]
    if not rows:
        return None

    try:
        df = pd.DataFrame(rows, columns=header)
        return df.to_markdown(index=False)
    except Exception:
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in rows:
            padded = (row + [""] * len(header))[: len(header)]
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)


def load_pdf_with_tables(pdf_path: str) -> list:
    documents: list[Document] = []
    current_heading = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # Tables
            tables       = page.extract_tables()
            table_finder = page.find_tables()
            table_bboxes = [t.bbox for t in table_finder] if table_finder else []

            for t_idx, table in enumerate(tables):
                md = _table_to_markdown(table)
                if md:
                    documents.append(Document(
                        page_content=md,
                        metadata={
                            "source":          pdf_path,
                            "page":            page_num,
                            "type":            "table",
                            "table_index":     t_idx,
                            "section_heading": current_heading,
                        }
                    ))

            # Text (table bboxes cropped out)
            remaining = page
            for bbox in table_bboxes:
                try:
                    remaining = remaining.outside_bbox(bbox)
                except Exception:
                    pass

            raw_text = remaining.extract_text(x_tolerance=2, y_tolerance=3) or ""
            clean    = _clean_text(raw_text)

            if clean:
                heading = _extract_section_heading(clean)
                if heading:
                    current_heading = heading

                documents.append(Document(
                    page_content=clean,
                    metadata={
                        "source":          pdf_path,
                        "page":            page_num,
                        "type":            "text",
                        "section_heading": current_heading,
                    }
                ))

    return documents


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CONTEXTUAL CHUNKING  (parent → child)
# ═══════════════════════════════════════════════════════════════════════════════

def _prefix(doc: Document) -> str:
    parts = []
    if doc.metadata.get("section_heading"):
        parts.append(f"Section: {doc.metadata['section_heading']}")
    parts.append(f"Page {doc.metadata['page']}")
    parts.append(f"[{doc.metadata.get('type', 'text').upper()}]")
    return " | ".join(parts) + "\n\n"


def build_parent_child_docs(raw_docs: list) -> tuple:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_docs: list[Document] = []
    child_docs:  list[Document] = []

    for doc_idx, doc in enumerate(raw_docs):
        prefix = _prefix(doc)

        if doc.metadata.get("type") == "table":
            content     = prefix + doc.page_content
            shared_meta = {**doc.metadata, "parent_id": f"p{doc_idx}"}
            parent_docs.append(Document(page_content=content, metadata=shared_meta))
            child_docs.append(Document(
                page_content=content,
                metadata={**shared_meta, "child_id": f"c{doc_idx}_0"},
            ))
        else:
            parents = parent_splitter.split_documents([doc])
            for p_idx, parent in enumerate(parents):
                parent_id              = f"p{doc_idx}_{p_idx}"
                parent.page_content    = prefix + parent.page_content
                parent.metadata["parent_id"] = parent_id
                parent_docs.append(parent)

                for c_idx, child in enumerate(child_splitter.split_documents([parent])):
                    child.metadata["parent_id"] = parent_id
                    child.metadata["child_id"]  = f"c{doc_idx}_{p_idx}_{c_idx}"
                    child_docs.append(child)

    return parent_docs, child_docs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PARENT-DOCUMENT RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════

class ParentDocumentRetriever(BaseRetriever):
    """Maps child-chunk hits back to their parent documents."""

    child_retriever: object = Field()
    parent_docs_map: dict   = Field()
    top_n: int              = Field(default=RERANK_TOP_N)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        child_hits       = self.child_retriever.get_relevant_documents(query)
        seen_parent_ids  = set()
        ordered_parents: List[Document] = []

        for child in child_hits:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                parent = self.parent_docs_map.get(pid)
                if parent:
                    ordered_parents.append(parent)
            if len(ordered_parents) >= self.top_n:
                break

        return ordered_parents or child_hits[: self.top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  RE-RANKER  (Cohere or passthrough fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_cohere_rerank(query: str, docs: list, top_n: int) -> list:
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        return docs[:top_n]

    try:
        import cohere
        co       = cohere.Client(api_key)
        passages = [d.page_content[:2048] for d in docs]
        results  = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=passages,
            top_n=top_n,
        )
        return [docs[r.index] for r in results.results]
    except Exception as e:
        print(f"[ReRank] Cohere unavailable ({e}); using original order.")
        return docs[:top_n]


class ReRankingRetriever(BaseRetriever):
    """Wraps any retriever and applies Cohere re-ranking post-retrieval."""

    base_retriever: object = Field()
    top_n: int             = Field(default=RERANK_TOP_N)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        return _try_cohere_rerank(query, docs, self.top_n)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document assistant for a Bank's NPA Recovery Policy (2023-24).

CONTEXT PROVIDED:
{context}

STRICT RULES:
- Answer ONLY from the context above. Never use outside knowledge.
- If the answer is not in the context, say exactly:
  "I could not find this information in the provided policy document."
- For tables: read every row and column carefully. Quote column headers when citing figures.
- For authority limits / delegation tables: list all tiers with their limits.
- For procedures: enumerate steps in order.
- Always mention the page number and section when available in the metadata.

QUESTION: {question}

ANSWER (cite page/section where possible):""",
)

DECOMPOSE_PROMPT = """You are a query analysis assistant for a banking policy document.
Break the following complex question into 2-4 simpler sub-questions that together cover the full answer.
Return ONLY a JSON array of strings, e.g. ["sub-q 1", "sub-q 2"].

Question: {question}"""

SYNTHESISE_PROMPT = """You are a document assistant for a Bank's NPA Recovery Policy.
Multiple sub-questions were answered from the policy document. Synthesise them into one
coherent, complete answer.

Sub-questions and answers:
{qa_pairs}

Original question: {question}

Synthesised answer (cite page/section where helpful):"""


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  QUERY DECOMPOSITION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def _is_complex(question: str) -> bool:
    q          = question.lower()
    indicators = ["and", "also", "both", "compare", "difference", "vs",
                  "what are all", "list all", "explain and", "how and"]
    return any(ind in q for ind in indicators) and len(question) > 80


def decompose_and_answer(question: str, base_chain, llm, chat_history: list) -> dict:
    """
    For complex questions: decompose → answer each sub-q → synthesise.
    For simple questions: answer directly.
    """
    if not _is_complex(question):
        return base_chain.invoke({"question": question, "chat_history": chat_history})

    decompose_msg = llm.predict(DECOMPOSE_PROMPT.format(question=question))
    try:
        sub_questions = json.loads(decompose_msg)
        if not isinstance(sub_questions, list):
            raise ValueError
    except Exception:
        return base_chain.invoke({"question": question, "chat_history": chat_history})

    qa_pairs_text   = ""
    all_source_docs = []
    for i, sq in enumerate(sub_questions[:4], 1):
        result = base_chain.invoke({"question": sq, "chat_history": chat_history})
        qa_pairs_text += f"\nSub-question {i}: {sq}\nAnswer: {result.get('answer', '')}\n"
        all_source_docs.extend(result.get("source_documents", []))

    final_answer = llm.predict(
        SYNTHESISE_PROMPT.format(qa_pairs=qa_pairs_text, question=question)
    )
    return {"answer": final_answer, "source_documents": all_source_docs}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_rag():
    npa_path = os.path.join(DATA_DIR, "npa_policy.pdf")
    if not os.path.exists(npa_path):
        raise FileNotFoundError(f"Missing file: {npa_path}")

    # Step 1: Parse PDF
    raw_docs = load_pdf_with_tables(npa_path)
    print(f"[RAG] Loaded {len(raw_docs)} raw docs "
          f"({sum(1 for d in raw_docs if d.metadata['type']=='table')} tables, "
          f"{sum(1 for d in raw_docs if d.metadata['type']=='text')} text blocks)")

    # Step 2: Parent-child split
    parent_docs, child_docs = build_parent_child_docs(raw_docs)
    parent_map = {d.metadata["parent_id"]: d for d in parent_docs}
    print(f"[RAG] {len(parent_docs)} parent chunks, {len(child_docs)} child chunks")

    # Step 3: Embed child chunks
    embeddings      = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore     = FAISS.from_documents(child_docs, embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": 40, "lambda_mult": 0.55},
    )

    # Step 4: BM25 on child chunks
    bm25_retriever   = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = BM25_K

    # Step 5: Hybrid retriever
    hybrid_child_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    # Step 6: Map child hits → parent docs
    parent_retriever = ParentDocumentRetriever(
        child_retriever=hybrid_child_retriever,
        parent_docs_map=parent_map,
        top_n=RERANK_TOP_N + 2,
    )

    # Step 7: Re-ranker
    final_retriever = ReRankingRetriever(
        base_retriever=parent_retriever,
        top_n=RERANK_TOP_N,
    )

    # Step 8: LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Step 9: Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=final_retriever,
        combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain, final_retriever, llm


# Module-level initialisation (imported by app.py and master_agent.py)
qa_chain, retriever, llm = setup_rag()
