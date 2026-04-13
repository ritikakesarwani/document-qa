"""
pdf_processor.py
----------------
Handles PDF text extraction and chunking.

Uses PyMuPDF (fitz) for fast, reliable text extraction across all PDF types.
Text is split into overlapping chunks to preserve context across boundaries.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Full extracted text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)
    doc.close()
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Overlapping ensures that context spanning chunk boundaries is preserved,
    reducing the chance of missing relevant information during retrieval.

    Args:
        text:       The full document text.
        chunk_size: Number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap

    return chunks
