"""
routes.py
---------
Flask route handlers for PDF upload and Q&A endpoints.

Session state (chunks + FAISS index) is stored in a module-level dict
keyed by a session token, keeping the system stateless between requests
while retaining document context within a browser session.
"""

import os
import uuid
from flask import Blueprint, render_template, request, jsonify, session

from app.pdf_processor import extract_text_from_pdf, chunk_text
from app.embedder import build_faiss_index, retrieve_top_chunks
from app.qa_engine import answer_question

bp = Blueprint("main", __name__)

# In-memory session store: { session_id -> { "chunks": [...], "index": faiss_index } }
_session_store: dict = {}

ALLOWED_EXTENSIONS = {"pdf"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Handle PDF upload.

    Workflow:
    1. Validate the uploaded file is a PDF.
    2. Save it temporarily to the uploads/ folder.
    3. Extract text and split into chunks.
    4. Build a FAISS index over the chunks.
    5. Store chunks + index in the session store.
    6. Return metadata to the frontend.
    """
    from flask import current_app

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Generate a unique session ID for this document
    session_id = str(uuid.uuid4())
    safe_filename = f"{session_id}.pdf"
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    pdf_path = os.path.join(upload_folder, safe_filename)

    try:
        file.save(pdf_path)

        # Process PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return jsonify({"error": "Could not extract text from this PDF. It may be scanned/image-based."}), 422

        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "Document appears to be empty after processing."}), 422

        # Build vector index
        faiss_index, _ = build_faiss_index(chunks)

        # Store in session store
        _session_store[session_id] = {
            "chunks": chunks,
            "index": faiss_index,
            "filename": file.filename,
        }

        return jsonify({
            "session_id": session_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "message": "Document processed successfully. You can now ask questions.",
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

    finally:
        # Clean up the uploaded file — we only need the index
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@bp.route("/ask", methods=["POST"])
def ask():
    """
    Handle a Q&A request.

    Workflow:
    1. Retrieve the FAISS index and chunks for this session.
    2. Find the top-K most relevant chunks for the question.
    3. Pass context + question to FLAN-T5 to generate an answer.
    4. Return the answer and the source chunks used.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request body."}), 400

    session_id = data.get("session_id", "").strip()
    question = data.get("question", "").strip()

    if not session_id:
        return jsonify({"error": "Missing session_id. Please upload a document first."}), 400

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if session_id not in _session_store:
        return jsonify({"error": "Session not found. Please upload a document again."}), 404

    try:
        session_data = _session_store[session_id]
        chunks = session_data["chunks"]
        faiss_index = session_data["index"]

        # Retrieve top relevant chunks
        top_chunks = retrieve_top_chunks(question, chunks, faiss_index, top_k=5)

        # Generate answer using LLM
        answer = answer_question(question, top_chunks)

        return jsonify({
            "answer": answer,
            "sources": top_chunks[:3],  # Return top 3 source excerpts to the UI
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500
