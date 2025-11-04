#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Software Bug Fix Helper — RAG with FREE local LLM via Ollama.
Usage:
  # 0) Setup
  #   python -m venv .venv && source .venv/bin/activate
  #   pip install -r requirements.txt
  #   (install ollama, pull a model, ensure it's running)

  # 1) Build index
  #   python bugfix_rag.py index --repo ~/path/to/repo --out rag_index

  # 2) Ask for a fix (uses Ollama by default)
  #   python bugfix_rag.py ask \
  #     --question "CLI crashes when input is empty" \
  #     --out rag_index \
  #     --k 6 \
  #     --model "llama3.1" \
  #     --provider ollama

  # (Optional) narrow retrieval:
  #   ... --file "src/main.py"
"""

import argparse
import os
import re
import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import requests

# -----------------------------
# Config
# -----------------------------
SUPPORTED_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt",
    ".c", ".h", ".cpp", ".cc", ".hpp",
    ".go", ".rs",
}

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast
DEFAULT_TOP_K = 6
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_CHUNK_LINES = 160
DEFAULT_CHUNK_OVERLAP = 40

# -----------------------------
# Utilities
# -----------------------------
def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def iter_source_files(repo: Path) -> List[Path]:
    files = []
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files

def chunk_code(text: str, max_lines: int, overlap: int) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        start = i
        end = min(i + max_lines, n)
        chunk = "\n".join(lines[start:end])
        chunk = re.sub(r"\n+$", "", chunk)
        chunks.append((start + 1, end, chunk))  # 1-indexed
        if end >= n:
            break
        i = end - overlap
        if i <= start:
            i = end
    return chunks

# -----------------------------
# Index schema
# -----------------------------
class ChunkMeta(BaseModel):
    file: str
    start_line: int
    end_line: int
    sha1: str

class IndexBundle(BaseModel):
    metas: List[ChunkMeta]
    tfidf_vocab: List[str]
    tfidf_idf: List[float]
    tfidf_doc_term: Dict[int, Dict[int, float]]
    embed_model_name: str

# -----------------------------
# TF-IDF helpers
# -----------------------------
def build_tfidf(chunks: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )
    X = vec.fit_transform(chunks)
    return vec, X

def sparse_row_to_dict(X, i: int) -> Dict[int, float]:
    row = X.getrow(i)
    return {idx: float(val) for idx, val in zip(row.indices, row.data)}

# -----------------------------
# Index building
# -----------------------------
def build_index(repo: Path, out_dir: Path,
                embed_model_name: str = DEFAULT_EMBED_MODEL,
                max_lines: int = DEFAULT_MAX_CHUNK_LINES,
                overlap: int = DEFAULT_CHUNK_OVERLAP):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_source_files(repo)
    if not files:
        print("No supported source files found.")
        sys.exit(1)

    print(f"Found {len(files)} files. Chunking...")
    metas: List[ChunkMeta] = []
    chunk_texts: List[str] = []

    for fp in tqdm(files, desc="Chunking"):
        text = read_text(fp)
        if not text.strip():
            continue
        for s, e, chunk in chunk_code(text, max_lines, overlap):
            chunk_texts.append(chunk)
            metas.append(ChunkMeta(file=str(fp.relative_to(repo)),
                                   start_line=s, end_line=e, sha1=sha1(chunk)))

    print(f"Total chunks: {len(chunk_texts)}")

    print("Loading embedding model:", embed_model_name)
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    d = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(d)  # cosine via normalized embeddings
    index.add(embeddings.astype("float32"))
    faiss_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    print("Fitting TF-IDF...")
    vec, X = build_tfidf(chunk_texts)
    tfidf_vocab = list(vec.vocabulary_.keys())
    vocab_to_idx = vec.vocabulary_
    tfidf_doc_term = {i: sparse_row_to_dict(X, i) for i in range(X.shape[0])}
    idf_array = vec.idf_.tolist()

    bundle = IndexBundle(
        metas=metas,
        tfidf_vocab=tfidf_vocab,
        tfidf_idf=idf_array,
        tfidf_doc_term=tfidf_doc_term,
        embed_model_name=embed_model_name
    )
    with open(out_dir / "metas.pkl", "wb") as f:
        pickle.dump(bundle.model_dump(), f)

    with open(out_dir / "repo_root.txt", "w") as f:
        f.write(str(repo.resolve()))

    print("Index built:")
    print(" -", faiss_path)
    print(" -", out_dir / "metas.pkl")
    print(" -", out_dir / "repo_root.txt")

# -----------------------------
# Retrieval
# -----------------------------
class Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.repo_root = Path((index_dir / "repo_root.txt").read_text().strip())
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "metas.pkl", "rb") as f:
            data = pickle.load(f)
        self.bundle = IndexBundle(**data)
        self.model = SentenceTransformer(self.bundle.embed_model_name)
        self.vocab_to_idx = {term: i for i, term in enumerate(self.bundle.tfidf_vocab)}
        self.idf = np.array(self.bundle.tfidf_idf, dtype=np.float32)

    def _tfidf_vectorize(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", text.lower())
        grams = tokens[:] + [" ".join(ng) for ng in zip(tokens, tokens[1:])]
        vec = np.zeros(len(self.vocab_to_idx), dtype=np.float32)
        for g in grams:
            idx = self.vocab_to_idx.get(g)
            if idx is not None:
                vec[idx] += 1.0
        vec = vec * self.idf
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def search(self, query: str, k: int = DEFAULT_TOP_K, file_hint: Optional[str] = None):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.faiss_index.search(q_emb, min(k*5, len(self.bundle.metas)))
        I = I[0].tolist()
        D = D[0].tolist()

        q_tfidf = self._tfidf_vectorize(query)
        scored = []
        for idx, emb_score in zip(I, D):
            meta = self.bundle.metas[idx]
            if file_hint and file_hint not in meta["file"]:
                continue
            # build doc tfidf vec
            doc_map = self.bundle.tfidf_doc_term[idx]
            doc_vec = np.zeros_like(q_tfidf)
            for t_idx, val in doc_map.items():
                if int(t_idx) < len(doc_vec):
                    doc_vec[int(t_idx)] = val
            doc_vec = normalize(doc_vec.reshape(1, -1))[0]
            tfidf_score = float(np.dot(q_tfidf, doc_vec))
            score = 0.65 * emb_score + 0.35 * tfidf_score
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        results = []
        for score, idx in top:
            m = self.bundle.metas[idx]
            fpath = self.repo_root / m.file
            text = read_text(fpath)
            lines = text.splitlines()
            chunk = "\n".join(lines[m.start_line-1: m.end_line])
            results.append({
                "score": float(score),
                "file": m.file,
                "start_line": m.start_line,
                "end_line": m.end_line,
                "chunk": chunk
            })
        return results

# -----------------------------
# LLM via Ollama (free local)
# -----------------------------
SYSTEM_PROMPT = """You are a senior software engineer.
Given a natural-language bug report and a set of code chunks (with file paths and line ranges),
you must:
1) Diagnose the likely root cause(s).
2) Propose a minimal, safe patch as a unified diff (git-style).
3) Explain the fix briefly.

Rules:
- Output one unified diff block per file you modify.
- Use correct file paths as shown in context, with `a/` and `b/` prefixes.
- Keep changes minimal; do not refactor unless necessary.
- After the diff(s), add a short explanation under a heading "Rationale".
"""

def call_ollama(model: str, question: str, contexts: List[Dict], host: str = "http://localhost:11434") -> str:
    context_strs = []
    for c in contexts:
        header = f"FILE: {c['file']}  LINES: {c['start_line']}-{c['end_line']}\n"
        context_strs.append(header + c["chunk"])
    context_blob = "\n\n" + ("\n\n" + ("-"*80) + "\n\n").join(context_strs)

    user_prompt = f"""Bug report / request:
{question}

Top {len(contexts)} retrieved code chunks:
{context_blob}

Please provide:
- Unified diff(s)
- Then "Rationale" section.
"""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {'message': {'content': ...}, ...}
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    # some variants return a 'response' field
    if "response" in data:
        return data["response"]
    return json.dumps(data, indent=2)

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Software Bug Fix Helper (RAG, free via Ollama)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build index over a repo")
    p_index.add_argument("--repo", required=True, type=str, help="Path to source repo")
    p_index.add_argument("--out", required=True, type=str, help="Output dir for index")
    p_index.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    p_index.add_argument("--max-lines", type=int, default=DEFAULT_MAX_CHUNK_LINES)
    p_index.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    p_ask = sub.add_parser("ask", help="Ask for a bug fix suggestion")
    p_ask.add_argument("--out", required=True, type=str, help="Index directory")
    p_ask.add_argument("--question", required=True, type=str, help="Bug report / question")
    p_ask.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-k chunks")
    p_ask.add_argument("--file", type=str, default=None, help="Optional file hint to narrow search")
    p_ask.add_argument("--provider", type=str, default="ollama", choices=["ollama"], help="LLM provider")
    p_ask.add_argument("--model", type=str, default="llama3.1", help="Model name for provider (e.g., llama3.1, deepseek-coder:6.7b)")
    p_ask.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama host")

    args = parser.parse_args()

    if args.cmd == "index":
        build_index(Path(args.repo).expanduser().resolve(),
                    Path(args.out).expanduser().resolve(),
                    embed_model_name=args.embed_model,
                    max_lines=args.max_lines,
                    overlap=args.overlap)
    elif args.cmd == "ask":
        idx = Retriever(Path(args.out).expanduser().resolve())
        results = idx.search(args.question, k=args.k, file_hint=args.file)
        if not results:
            print("No results found. Try different query or rebuild index.")
            sys.exit(2)
        print(f"\n[Retrieved {len(results)} chunks]")
        for i, r in enumerate(results, 1):
            print(f"#{i} {r['file']}:{r['start_line']}-{r['end_line']}  (score={r['score']:.3f})")

        print("\n[Generating patch with local model via Ollama...]")
        answer = call_ollama(args.model, args.question, results, host=args.ollama_host)
        print("\n" + "="*80 + "\nLLM PROPOSED PATCH\n" + "="*80)
        print(answer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
