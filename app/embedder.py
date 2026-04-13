"""
embedder.py
-----------
Generates dense vector embeddings for text chunks and performs
similarity-based retrieval using FAISS.

Model: all-MiniLM-L6-v2 (22M params, runs fast on CPU)
Index: FAISS FlatL2 — exact nearest-neighbor search, no approximation.
"""

import numpy as np
import faiss
from typing import Optional
from sentence_transformers import SentenceTransformer

# Load embedding model once at module level (cached after first load)
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def build_faiss_index(chunks: list[str]) -> tuple[faiss.IndexFlatL2, np.ndarray]:
    """
    Embed all text chunks and build a FAISS index.

    Args:
        chunks: List of text chunks from the document.

    Returns:
        Tuple of (faiss_index, embeddings_array).
    """
    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


def retrieve_top_chunks(
    query: str,
    chunks: list[str],
    index: faiss.IndexFlatL2,
    top_k: int = 5,
) -> list[str]:
    """
    Find the top-K most semantically similar chunks to the query.

    Args:
        query:  The user's question.
        chunks: Original list of text chunks (maps to index positions).
        index:  The FAISS index built from those chunks.
        top_k:  How many chunks to return.

    Returns:
        List of the most relevant text chunks, ordered by similarity.
    """
    model = _get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    k = min(top_k, len(chunks))
    distances, indices = index.search(query_vec, k)

    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results
