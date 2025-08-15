import os
import re
import json
import tempfile
import hashlib
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# 1) ENVIRONMENT & CONFIG
# -----------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("RECIPE_GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

st.set_page_config(page_title="Scientific Literature RAG", page_icon="🧪", layout="wide")

if not GROQ_API_KEY:
    st.error("Missing GROQ API key. Set RECIPE_GROQ_API_KEY or GROQ_API_KEY in your environment.")
    st.stop()

# -----------------------------------------------------------------------------
# 2) LIGHT STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #0E1117; color: #EAEDF1; }
      h1,h2,h3 { color: #E6E6FA !important; }
      .block-label { color:#a8b3cf; font-size:0.9rem; }
      .uploadedFile { background:#141820; border:1px solid #2c3242; border-radius:10px; padding:10px; }
      pre, code { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3) PROMPT TEMPLATE (kept clear and strict about citations)
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a domain-specific scientific research assistant.
You specialize in {domain} and can interpret equations, technical terms, and research methods.

USER QUESTION:
{user_query}

RETRIEVED LITERATURE (snippets; may include equations and reference list excerpts):
{document_context}

RESPONSE INSTRUCTIONS:
1) Provide a clear, technically accurate answer scoped to the question.
2) If equations appear, briefly define symbols and connect them to the question.
3) Use at least TWO inline citations pulled ONLY from the provided snippets.
4) Format citations as: (Author et al., Year, DOI:xxxx) or (Title, Year, DOI:xxxx) if author unknown.
5) If the context is insufficient, say what’s missing and propose 2–3 concrete follow-ups.
"""

# -----------------------------------------------------------------------------
# 4) CACHED FACTORIES
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    """Load the sentence embedding model (once)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def get_llm():
    """Init the Groq LLM client (once)."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1200,
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore(_embedding):
    """In-memory vector store for prototyping (swap for Chroma/Pinecone in prod)."""
    return InMemoryVectorStore(_embedding)

# -----------------------------------------------------------------------------
# 5) HELPERS
# -----------------------------------------------------------------------------
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def bytes_to_tempfile(file_bytes: bytes, suffix: str) -> str:
    """Write uploaded bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return tmp.name

# --- Equation-aware chunking ---
# Keep $$...$$ blocks intact; only split normal text around them.
EQN_BLOCK_RE = re.compile(r"\$\$[\s\S]*?\$\$", re.MULTILINE)

def preserve_equations_chunks(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """Split text, preserving $$...$$ math blocks as atomic segments."""
    parts: List[str] = []
    pos = 0
    for m in EQN_BLOCK_RE.finditer(text):
        if m.start() > pos:
            parts.append(text[pos:m.start()])       # text before equation
        parts.append(text[m.start():m.end()])       # the equation block
        pos = m.end()
    if pos < len(text):                              # trailing text
        parts.append(text[pos:])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " "],
        add_start_index=True,
    )

    chunks: List[str] = []
    for segment in parts:
        seg = segment.strip()
        if seg.startswith("$$") and seg.endswith("$$"):
            chunks.append(segment)                   # keep eqn blocks whole
        else:
            chunks.extend(splitter.split_text(segment))
    return chunks

def _as_list(obj) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]

def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw record into a consistent shape:
    title, abstract, body_text, citations(list), year, domain.
    """
    citations = rec.get("citations") or rec.get("references") or []
    if isinstance(citations, str):
        try:
            citations = json.loads(citations)
        except Exception:
            citations = _as_list(citations)

    year = rec.get("year")
    if not year:
        y = rec.get("published_date") or rec.get("date") or ""
        if isinstance(y, str) and len(y) >= 4 and y[:4].isdigit():
            year = y[:4]

    return {
        "title": rec.get("title") or "Untitled",
        "abstract": rec.get("abstract") or "",
        "body_text": rec.get("body_text") or rec.get("text") or rec.get("content") or "",
        "citations": citations,
        "year": year,
        "domain": rec.get("domain") or rec.get("field") or "Unknown",
    }

def load_jsonl_as_documents(path: str, sample_rows: Optional[int] = None) -> List[Document]:
    """Load JSONL corpus where each line is a paper-like record."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if sample_rows is not None and len(lines) > sample_rows:
        lines = lines[:sample_rows]

    docs: List[Document] = []
    for i, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except Exception:
            continue
        rec = normalize_record(raw)
        text = f"# {rec['title']}\n\n## Abstract\n{rec['abstract']}\n\n## Body\n{rec['body_text']}"
        docs.append(Document(
            page_content=text,
            metadata={
                "row_index": i,
                "title": rec["title"],
                "year": rec["year"],
                "domain": rec["domain"],
                "citations": rec["citations"],
            },
        ))
    return docs

def load_tabular_as_documents(path: str, is_csv: bool, sample_rows: Optional[int] = None) -> List[Document]:
    """Load CSV/XLSX corpora; best-effort mapping to title/abstract/body/citations."""
    if is_csv:
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1", on_bad_lines="skip")
    else:
        df = pd.read_excel(path)

    if sample_rows is not None and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=13).reset_index(drop=True)

    docs: List[Document] = []
    for idx, row in df.iterrows():
        title = str(row.get("title", f"Doc {idx}"))
        abstract = str(row.get("abstract", ""))
        body = str(row.get("body_text", "")) or " | ".join(map(str, row.values))

        citations = row.get("citations", [])
        if isinstance(citations, str):
            try:
                citations = json.loads(citations)
            except Exception:
                citations = _as_list(citations)

        year = row.get("year")
        year = None if pd.isna(year) else year

        text = f"# {title}\n\n## Abstract\n{abstract}\n\n## Body\n{body}"
        docs.append(Document(
            page_content=text,
            metadata={
                "row_index": int(idx),
                "title": title,
                "year": year,
                "domain": str(row.get("domain", "Unknown")),
                "citations": citations,
            },
        ))
    return docs

def chunk_documents_equation_aware(docs: List[Document], chunk_size=1200, chunk_overlap=200) -> List[Document]:
    """Apply equation-aware splitting to every document."""
    chunks: List[Document] = []
    for d in docs:
        for j, piece in enumerate(preserve_equations_chunks(d.page_content, chunk_size, chunk_overlap)):
            meta = {**d.metadata, "chunk_id": j}
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks

def index_documents(vstore: InMemoryVectorStore, chunks: List[Document]):
    """Embed + add to the vector store in small batches with progress UI."""
    if not chunks:
        return
    batch = 64
    bar = st.progress(0, text="Indexing embeddings…")
    for i in range(0, len(chunks), batch):
        vstore.add_documents(chunks[i:i+batch])
        pct = min(100, int((i + batch) / max(1, len(chunks)) * 100))
        bar.progress(pct)
    bar.progress(100)

def pretty_context(snippets: List[Document], max_chars: int = 1000) -> str:
    """
    Format retrieved passages for the LLM:
    - Include minimal metadata (title/year/domain)
    - Include up to 5 citation stubs per snippet
    """
    blocks = []
    for d in snippets:
        text = d.page_content[:max_chars]
        meta = f"[Title: {d.metadata.get('title','Untitled')} | Year: {d.metadata.get('year','?')} | Domain: {d.metadata.get('domain','?')}]"
        cits = d.metadata.get("citations", [])
        cite_strs = []
        for c in cits[:5]:
            if isinstance(c, dict):
                title_or_text = c.get("text") or c.get("title") or "Unknown"
                doi = c.get("doi") or c.get("DOI") or ""
                cite_strs.append(f"{title_or_text} | DOI:{doi}")
            else:
                cite_strs.append(str(c))
        if cite_strs:
            meta += "\nCitations: " + "; ".join(cite_strs)
        blocks.append(meta + "\n" + text)
    return "\n\n---\n\n".join(blocks)

def retrieve(vstore: InMemoryVectorStore, query: str, k: int = 5) -> List[Document]:
    return vstore.similarity_search(query, k=k)

def answer(llm: ChatGroq, domain: str, user_query: str, context_text: str) -> str:
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm
    out = chain.invoke({"domain": domain, "user_query": user_query, "document_context": context_text})
    return getattr(out, "content", out)

# -----------------------------------------------------------------------------
# 6) SESSION STATE
# -----------------------------------------------------------------------------
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# -----------------------------------------------------------------------------
# 7) APP UI
# -----------------------------------------------------------------------------
st.title("🧪 Scientific Literature RAG")
st.caption("Biology • Chemistry • Physics — equation-aware retrieval with verifiable citations")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.write("Adjust domain and chunking behavior here.")
    domain = st.selectbox("Domain focus", ["Biology", "Chemistry", "Physics", "Unknown"], index=0)
    chunk_size = st.slider("Chunk size", 600, 2200, 1200, step=100, help="Target characters per chunk.")
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, step=50, help="Overlap between consecutive chunks.")
    top_k = st.slider("Top-K retrieved passages", 1, 10, 5)
    st.markdown("---")
    st.markdown("**Embedding model**: `" + EMBEDDING_MODEL + "`")
    st.markdown("**LLM model**: `" + GROQ_MODEL + "`")

st.subheader("📚 Upload your literature dataset")
st.markdown("<span class='block-label'>Recommended: JSONL where each line is a paper-like object.</span>", unsafe_allow_html=True)

with st.form("loader_form"):
    uploaded = st.file_uploader("Upload JSONL / CSV / XLSX", type=["jsonl", "csv", "xlsx"])
    sample_rows = st.number_input("Optional: limit rows (for quick tests)", min_value=0, max_value=50000, value=0, step=1000)
    submit = st.form_submit_button("📤 Process & Index")

# Instantiate models once
embedder = get_embedder()
llm = get_llm()
vstore = get_vectorstore(embedder)

if submit:
    if not uploaded:
        st.error("Please upload a dataset first.")
    else:
        # Save to temp
        raw_bytes = uploaded.getvalue()
        st.session_state.file_hash = _sha256(raw_bytes)
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".jsonl"
        tmp_path = bytes_to_tempfile(raw_bytes, suffix)

        # Parse
        with st.spinner("Parsing dataset…"):
            if suffix == ".jsonl":
                docs = load_jsonl_as_documents(tmp_path, sample_rows or None)
            else:
                docs = load_tabular_as_documents(tmp_path, is_csv=(suffix == ".csv"), sample_rows=sample_rows or None)

        if not docs:
            st.error("No documents parsed. Check file format/columns.")
        else:
            # Chunk + Index
            with st.spinner("Equation-aware chunking…"):
                chunks = chunk_documents_equation_aware(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            with st.spinner("Embedding & indexing (batched)…"):
                index_documents(vstore=vstore, chunks=chunks)

            st.session_state.vector_ready = True
            st.success(f"✅ Indexed successfully — Docs: {len(docs)} | Chunks: {len(chunks)}")

st.divider()
st.subheader("🔎 Ask a scientific question")
query = st.text_input(
    "Example: “Explain how path integrals relate to the quantum harmonic oscillator.”",
    value="Explain how path integrals relate to the quantum harmonic oscillator and include citations.",
)

go = st.button("🤖 Generate Answer")

if go:
    if not st.session_state.vector_ready:
        st.error("Please upload and index a literature dataset first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant passages…"):
            hits = retrieve(vstore, query, k=top_k)
            ctx = pretty_context(hits)

        with st.spinner("Generating answer…"):
            response = answer(llm, domain, query, ctx)

        st.markdown("### 🧠 Answer")
        st.write(response)

        with st.expander("🔍 See retrieved snippets"):
            for i, d in enumerate(hits, start=1):
                meta = f"**[{i}] {d.metadata.get('title','Untitled')}** — Year: {d.metadata.get('year','?')} | Domain: {d.metadata.get('domain','?')}"
                st.markdown(meta)
                st.code(d.page_content[:900])
                st.markdown("---")

# -----------------------------------------------------------------------------
# 8) HELP / NOTES
# -----------------------------------------------------------------------------
with st.expander("⚡ Tips & Setup"):
    st.markdown(
        """
        **Input format (recommended JSONL keys):**
        - `title` *(str)*, `abstract` *(str)*, `body_text` *(str; may contain $$...$$ math)*  
        - `citations` *(list of dicts: `{title|text, doi}`)*, `year` *(int/str)*, `domain` *(str)*

        **Environment variables:**
        - `RECIPE_GROQ_API_KEY` **or** `GROQ_API_KEY`
        - `GROQ_MODEL` (default: `deepseek-r1-distill-llama-70b`)
        - `EMBEDDING_MODEL` (default: `sentence-transformers/all-mpnet-base-v2`)

        **Scaling tips:**
        - Swap `InMemoryVectorStore` for **Chroma** or **Pinecone** for larger corpora.
        - Keep $$...$$ blocks intact (already handled here) to preserve equations.
        - Consider domain-specific embeddings (e.g., SciBERT) if your corpus is very technical.
        """
    )
