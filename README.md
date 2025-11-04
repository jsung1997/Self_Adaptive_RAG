
Overview:
Traditional static analysis tools detect syntax-level issues but often fail to reason about semantic bugs (e.g., missing condition checks, null pointer issues, incorrect logic).
This project addresses that gap by combining vector-based retrieval and natural language reasoning to locate likely bug sources and propose contextually relevant patches.




Key Features:
Semantic Retrieval – Uses transformer-based embeddings (SentenceTransformers) to understand natural-language queries and map them to code semantics.
Hybrid Ranking – Combines FAISS vector similarity with TF-IDF token matching for better recall on identifiers and keywords.
Local LLM Reasoning – Employs Ollama to run models such as llama3.1, deepseek-coder:6.7b, or qwen2.5-coder:7b without internet or API costs.
Automated Diff Generation – Outputs human-readable git diff patches ready for manual validation.
Explainable Output – Provides a short “Rationale” explaining why the change is needed.
Offline & Reproducible – Fully local; no cloud dependencies or privacy risks.
Modular Design – Easily replaceable embeddings, retrieval engine, or LLM backend.





Technical Components:
| Layer               | Description                                                                                   | Library              |
| ------------------- | --------------------------------------------------------------------------------------------- | -------------------- |
| **Chunking**        | Line-based overlapping segmentation for contextual integrity. Default: 160 lines, 40 overlap. | Native Python        |
| **Embeddings**      | Semantic representation of code using transformer models (`all-MiniLM-L6-v2`).                | SentenceTransformers |
| **Vector Index**    | Approximate nearest neighbor search using cosine similarity.                                  | FAISS                |
| **Lexical Ranking** | Sparse token weighting for identifier-level matching.                                         | scikit-learn TF-IDF  |
| **Hybrid Scoring**  | Weighted ensemble: `score = 0.65*embed + 0.35*tfidf`.                                         | Custom               |
| **Retriever**       | Returns top-k chunks with file path, line range, and similarity scores.                       | Custom               |
| **LLM Interface**   | REST API to local Ollama instance.                                                            | requests             |
| **LLM Models**      | `llama3.1`, `deepseek-coder:6.7b`, `qwen2.5-coder:7b`                                         | Ollama               |



Performance:
Metric	Small Repo (1k chunks)	Medium Repo (10k chunks)
Indexing Time	~5s	~45s
Retrieval Latency	< 0.1s	< 0.3s
LLM Generation	3–10s	Depends on model
Index Size	~2 MB	~15 MB



Research Relevance:
This project lies at the intersection of:
RAG-based code understanding
LLM-assisted software maintenance
Self-adapting machine learning (SHML)
AI-driven software defect prediction
Potential applications include:
Automatic bug triage in large-scale repositories
Context-aware patch suggestion systems
Foundation for “self-healing software” pipelines where the model adapts to recurring failure patterns.
