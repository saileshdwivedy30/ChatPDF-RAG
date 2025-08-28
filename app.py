# app.py (session-only, append-to-index)
import io
import os
import time
import random
import hashlib
import boto3
import streamlit as st
from botocore.config import Config
from botocore.exceptions import ClientError

from pypdf import PdfReader
from langchain.docstore.document import Document

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from collections import Counter
import re
from difflib import SequenceMatcher


# -------------------------
# Bedrock client (us-east-1 + retries)
# -------------------------
REGION = "us-east-1"
bedrock_cfg = Config(
    retries={"max_attempts": 10, "mode": "standard"},
    read_timeout=60,
    connect_timeout=10,
    max_pool_connections=50,
)
bedrock = boto3.client("bedrock-runtime", region_name=REGION, config=bedrock_cfg)

# -------------------------
# Embeddings (Titan v2)
# -------------------------
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock,
)

# -------------------------
# Claude model (Haiku on demand)
# -------------------------
HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def get_haiku_llm(max_tokens: int = 256, temperature: float = 0.2):
    return ChatBedrock(
        model_id=HAIKU_MODEL_ID,
        client=bedrock,
        streaming=False,
        model_kwargs={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": ["\n\nHuman:", "\n\nSystem:"],
        },
    )

# -------------------------
# Utilities
# -------------------------
def _sha1(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

# -------------------------
# Ingest uploads entirely in memory (no disk writes)
# - Uses getvalue() so it works even after previous reads
# - Adds a stable "doc_id" to metadata to dedupe across appends
# -------------------------
def extract_docs_from_uploads(files) -> list[Document]:
    docs: list[Document] = []
    for f in files:
        data = f.getvalue()  # works regardless of current pointer
        if not data:
            continue
        file_hash = _sha1(data)
        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": f.name,
                        "page": i,
                        "doc_id": f"{f.name}::p{i}::{file_hash}",  # stable id
                    },
                )
            )
    return docs

def split_docs(raw_docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)

# -------------------------
# Vector store management (APPEND)
# - Keeps a set of seen "doc_id" values in session to avoid duplicates
# - If a vectorstore exists: add only new chunks
# - If none exists: create it
# -------------------------
def build_or_update_vectorstore(files) -> FAISS | None:
    raw_docs = extract_docs_from_uploads(files)
    if not raw_docs:
        return st.session_state.get("vs")

    # Track which base pages we have already indexed to avoid duplicates
    seen_ids = st.session_state.setdefault("seen_doc_ids", set())

    # Filter out pages we have already ingested
    new_raw_docs = [d for d in raw_docs if d.metadata.get("doc_id") not in seen_ids]
    if not new_raw_docs:
        return st.session_state.get("vs")

    # Remember these pages
    for d in new_raw_docs:
        seen_ids.add(d.metadata.get("doc_id"))

    chunks = split_docs(new_raw_docs)
    if not chunks:
        return st.session_state.get("vs")

    # Create or append
    if "vs" in st.session_state and st.session_state["vs"] is not None:
        st.session_state["vs"].add_documents(chunks)
        return st.session_state["vs"]
    else:
        return FAISS.from_documents(chunks, bedrock_embeddings)

# -------------------------
# Prompt (short and direct)
# -------------------------
prompt_template = """
Human: Use the context below to answer the question **briefly and directly** (under 120 words).
If the answer isn't in the context, say you don't know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# -------------------------
# Backoff wrapper for Bedrock throttling
# -------------------------
def invoke_with_backoff(qa, query, max_attempts=6):
    base = 0.5
    for attempt in range(max_attempts):
        try:
            return qa.invoke({"query": query})
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in (
                "ThrottlingException",
                "TooManyRequestsException",
                "LimitExceededException",
                "ProvisionedThroughputExceededException",
                "Throttling",
                "ThrottledException",
            ):
                time.sleep(min(8.0, base * (2 ** attempt)) + random.random() * 0.2)
                continue
            raise
    raise RuntimeError("Retry limit reached due to throttling.")

# -------------------------
# Build QA chain, return answer + source chips
# -------------------------

def _answer_tokens(text: str) -> set[str]:
    # 3+ chars, keep numbers/hyphens, lowercase
    return set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower()))

def _support_score(answer: str, text: str) -> float:
    # How well does text support answer? Combines token overlap + fuzzy similarity.
    a = (answer or "").lower()
    t = (text or "").lower()

    a_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", a))
    t_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", t))
    overlap = len(a_tokens & t_tokens) / max(len(a_tokens), 1)

    sim = SequenceMatcher(None, a[:600], t[:2000]).ratio()
    return 0.7 * overlap + 0.3 * sim


def answer_with_sources(llm, vectorstore, query, k=6):
    # Answer and cite only files that *strongly* support the final answer.
    MIN_SUPPORT_ABS = 0.18         # drop weak matches outright
    SECOND_SUPPORT_REL = 0.65      # keep #2 only if close to #1
    MAX_SOURCES = 2

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",                  # mmr keeps diversity for re-ranking
            search_kwargs={"k": k, "fetch_k": 24, "lambda_mult": 0.25},
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    resp = invoke_with_backoff(qa, query)
    answer = (resp.get("result", "") or "").strip()
    docs = resp.get("source_documents", []) or []

    # Per-file best supporting chunk score (fresh each question)
    best_score_by_file, best_doc_by_file = {}, {}
    for d in docs:
        fname = (d.metadata or {}).get("filename", "unknown.pdf")
        s = _support_score(answer, d.page_content or "")
        if s > best_score_by_file.get(fname, -1):
            best_score_by_file[fname] = s
            best_doc_by_file[fname] = d

    ranked = sorted(best_score_by_file.items(), key=lambda x: x[1], reverse=True)

    chosen_files = []
    if ranked:
        top1_name, top1_score = ranked[0]
        if top1_score >= MIN_SUPPORT_ABS:
            chosen_files.append(top1_name)

        if len(ranked) > 1 and len(chosen_files) < MAX_SOURCES:
            top2_name, top2_score = ranked[1]
            if top2_score >= MIN_SUPPORT_ABS and top2_score >= top1_score * SECOND_SUPPORT_REL:
                chosen_files.append(top2_name)

    # Build chips only from chosen files
    chips = []
    for fname in chosen_files:
        d = best_doc_by_file[fname]
        meta = d.metadata or {}
        page = meta.get("page")
        page_disp = (page + 1) if isinstance(page, int) else "?"
        chips.append({
            "name": fname,
            "page": page_disp,
            "snippet": (d.page_content or "").strip()[:400],
        })
        if len(chips) >= MAX_SOURCES:
            break

    return answer, chips

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config("Chat PDF", layout="centered")
    st.header("Chat with your PDFs")
    st.caption("Uploads live only for this session. Closing/refreshing clears everything.")

    uploads = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Append to vectorstore (or create if none)
    if st.button("Add uploads to index"):
        if not uploads:
            st.warning("Please upload at least one PDF.")
        else:
            vs = build_or_update_vectorstore(uploads)
            if vs is None:
                st.error("No readable text found in the uploaded PDFs.")
            else:
                st.session_state["vs"] = vs
                st.success("Indexed new pages (existing index kept).")

    # Ask
    user_q = st.text_input("Ask a question about your PDFs")

    # Optional toggle for seeing snippets
    #show_snippets = st.checkbox("Show source snippets", value=False)

    if st.button("Ask Claude"):
        if not user_q.strip():
            st.warning("Type a question first.")
            return
        vs = st.session_state.get("vs")
        if vs is None:
            st.warning("No index yet. Click 'Add uploads to index' after uploading PDFs.")
            return

        try:
            with st.spinner("Thinking..."):
                llm = get_haiku_llm(max_tokens=256, temperature=0.2)
                answer, chips = answer_with_sources(llm, vs, user_q, k=5)
                st.write(answer)
                if chips:
                    st.markdown("**Sources**")
                    st.write("  ".join([f"`{c['name']} p.{c['page']}`" for c in chips]))
                    # if show_snippets:
                    #     for c in chips:
                    #         st.markdown(f"**{c['name']} — p.{c['page']}**")
                    #         if c["snippet"]:
                    #             st.write(c["snippet"])
                    #         st.markdown("---")
        except ClientError as e:
            msg = str(e)
            if "on-demand throughput isn’t supported" in msg:
                st.error("This model version needs an Inference Profile.")
            else:
                st.error(f"Bedrock error: {e}")
        except Exception as e:
            st.error(str(e))

    # # Reset session (clears index and dedupe set)
    # if st.button("Reset session"):
    #     st.session_state.clear()
    #     st.experimental_rerun()

if __name__ == "__main__":
    main()
