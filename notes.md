# 📘 DocuQuery — Complete Technical Notes

> A deep-dive explanation of every step, every technology choice, and the full architecture of the Document Q&A system.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The Big Picture — How It Works End to End](#2-the-big-picture--how-it-works-end-to-end)
3. [Core Concept — RAG (Retrieval-Augmented Generation)](#3-core-concept--rag-retrieval-augmented-generation)
4. [Full Architecture Diagram](#4-full-architecture-diagram)
5. [Technology Stack — What & Why](#5-technology-stack--what--why)
6. [Step-by-Step Pipeline Explained](#6-step-by-step-pipeline-explained)
   - [Step 1 — PDF Upload](#step-1--pdf-upload)
   - [Step 2 — Text Extraction](#step-2--text-extraction)
   - [Step 3 — Chunking](#step-3--chunking)
   - [Step 4 — Embedding](#step-4--embedding)
   - [Step 5 — Vector Index (FAISS)](#step-5--vector-index-faiss)
   - [Step 6 — Question Embedding & Retrieval](#step-6--question-embedding--retrieval)
   - [Step 7 — Answer Generation (LLM)](#step-7--answer-generation-llm)
7. [Project Structure Explained](#7-project-structure-explained)
8. [File-by-File Code Explanation](#8-file-by-file-code-explanation)
9. [API Design](#9-api-design)
10. [Frontend Design Decisions](#10-frontend-design-decisions)
11. [Why No External APIs?](#11-why-no-external-apis)
12. [Limitations & Future Improvements](#12-limitations--future-improvements)
13. [Glossary](#13-glossary)

---

## 1. What This Project Does

DocuQuery is a **Document Question Answering (QA) system** that:

- Accepts a PDF file from the user
- Reads and understands the content of the document
- Lets the user ask any natural language question
- Returns an accurate answer **based only on what the document says**

The entire system runs **locally on your machine** — no internet connection needed at runtime, no data sent to any server, no API keys required.

---

## 2. The Big Picture — How It Works End to End

When you upload a PDF and ask a question, the following happens:

```
YOU upload a PDF
        │
        ▼
Text is extracted from the PDF (PyMuPDF)
        │
        ▼
Text is split into small, overlapping chunks
        │
        ▼
Each chunk is converted into a vector (a list of numbers)
using a local AI embedding model (MiniLM)
        │
        ▼
All vectors are stored in a fast search index (FAISS)
        │
        ▼
YOU ask a question
        │
        ▼
The question is also converted into a vector (same model)
        │
        ▼
FAISS finds the top 5 chunks most similar to your question
        │
        ▼
Those chunks are given to a local LLM (FLAN-T5)
as context, along with your question
        │
        ▼
FLAN-T5 reads the context and generates an answer
        │
        ▼
YOU see the answer + source excerpts in the UI
```

---

## 3. Core Concept — RAG (Retrieval-Augmented Generation)

This system is built using a technique called **RAG**.

### What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It combines two ideas:

| Component | What it does |
|---|---|
| **Retrieval** | Find the most relevant parts of the document for the question |
| **Generation** | Use an AI model to write an answer *based on those parts* |

### Why not just feed the entire PDF to the AI?

AI language models have a **token limit** — they can only process a certain amount of text at once. For example, FLAN-T5-base can handle roughly 512–1024 tokens (≈ 700–800 words) of input.

A typical PDF might have **10,000–100,000 words**. You cannot feed the whole thing to the model.

RAG solves this by:
1. Splitting the document into small pieces (chunks)
2. Finding only the **relevant** pieces for each question
3. Sending only those pieces to the LLM

This makes the system:
- **Scalable** — works on any sized document
- **Accurate** — the AI focuses only on relevant content
- **Grounded** — the answer must come from the document, not the model's memory

### RAG vs Fine-tuning

| Approach | Training needed? | Works on new docs? | Needs GPU? |
|---|---|---|---|
| **RAG (our approach)** | ❌ No | ✅ Yes, instantly | ❌ No |
| Fine-tuning | ✅ Yes (expensive) | ❌ No (fixed to trained data) | ✅ Usually |

RAG is the right choice here because we want to support **any PDF at runtime**.

---

## 4. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER'S BROWSER                                  │
│                                                                          │
│   ┌──────────────────────────┐    ┌──────────────────────────────────┐  │
│   │     UPLOAD PANEL         │    │         CHAT PANEL               │  │
│   │                          │    │                                  │  │
│   │  [Drag & Drop PDF Zone]  │    │  [Empty state / Chat messages]  │  │
│   │  [Document info card]    │    │  [Q&A bubbles with sources]     │  │
│   │  [How it works steps]    │    │  [Question input + Send button] │  │
│   └──────────┬───────────────┘    └────────────────┬─────────────────┘  │
│              │ POST /upload                        │ POST /ask           │
└──────────────│─────────────────────────────────────│────────────────────┘
               │                                     │
               ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FLASK BACKEND  (run.py)                         │
│                                                                          │
│  ┌────────────────┐   ┌──────────────────┐   ┌────────────────────────┐ │
│  │ pdf_processor  │   │    embedder       │   │      qa_engine         │ │
│  │                │   │                  │   │                        │ │
│  │ 1. PyMuPDF     │   │ 3. SentenceT.    │   │ 6. AutoTokenizer       │ │
│  │    reads PDF   │──►│    encodes       │   │    AutoModelForSeq2Seq │ │
│  │                │   │    chunks        │   │                        │ │
│  │ 2. chunk_text  │   │                  │   │ 7. Prompt engineering  │ │
│  │    splits text │   │ 4. FAISS index   │   │    → generate answer   │ │
│  │    (500 words, │   │    built from    │──►│                        │ │
│  │     50 overlap)│   │    embeddings    │   │ Returns: answer string │ │
│  └────────────────┘   │                  │   └────────────────────────┘ │
│                        │ 5. similarity   │                               │
│                        │    search for   │                               │
│                        │    top-K chunks │                               │
│                        └──────────────────┘                              │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        routes.py                                   │  │
│  │  /upload → calls pdf_processor + embedder → stores in session     │  │
│  │  /ask    → calls embedder (search) + qa_engine → returns answer   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
               │                         │
               ▼                         ▼
┌─────────────────────┐    ┌──────────────────────────┐
│   LOCAL FILE SYSTEM │    │   IN-MEMORY SESSION STORE │
│                     │    │                          │
│ uploads/ (temp PDF) │    │ { session_id: {          │
│ models/  (AI cache) │    │     chunks: [...],       │
└─────────────────────┘    │     faiss_index: ...     │
                           │   }}                     │
                           └──────────────────────────┘
```

---

## 5. Technology Stack — What & Why

### 🐍 Python
**Why:** Python is the standard language for AI/ML. All major AI libraries (PyTorch, HuggingFace Transformers, FAISS, sentence-transformers) are Python-first. It also has Flask for web development.

---

### 🌐 Flask
**What:** A lightweight Python web framework.

**Why chosen over alternatives:**

| Framework | Why NOT chosen |
|---|---|
| Django | Too heavy, too much config for a simple app |
| FastAPI | Great choice too, but adds async complexity |
| **Flask** ✅ | Minimal, easy to understand, perfect for this scope |

Flask handles:
- Serving the HTML frontend
- Exposing `/upload` and `/ask` REST endpoints
- Managing file uploads (via Werkzeug)

---

### 📄 PyMuPDF (fitz)
**What:** A Python library for reading PDF files.

**Why chosen over alternatives:**

| Library | Why NOT chosen |
|---|---|
| pdfplumber | Slower on large PDFs |
| PyPDF2 / pypdf | Less reliable text extraction |
| pdfminer | Complex API, slower |
| **PyMuPDF** ✅ | Fastest, most reliable, handles complex layouts |

PyMuPDF extracts raw text from each page of the PDF. It works on text-based PDFs (not scanned/image PDFs).

---

### 🧠 sentence-transformers — `all-MiniLM-L6-v2`
**What:** A pre-trained model that converts text into a 384-dimensional vector (embedding).

**Why embeddings?**
Computers can't directly compare sentences like "What is the capital of France?" with "Paris is the capital of France." But if we convert both to vectors, we can measure how **mathematically similar** they are. Similar meanings → similar vectors → close in vector space.

**Why `all-MiniLM-L6-v2` specifically?**

| Model | Size | Speed | Quality |
|---|---|---|---|
| all-MiniLM-L6-v2 ✅ | 22 MB | Very fast | Excellent |
| all-mpnet-base-v2 | 420 MB | Slow | Slightly better |
| text-embedding-ada-002 | External API | Fast | Best |

`all-MiniLM-L6-v2` is the sweet spot — tiny, fast, and high quality for semantic search tasks.

---

### 🔍 FAISS (Facebook AI Similarity Search)
**What:** A library for fast similarity search over large collections of vectors.

**Why FAISS?**
Once we have embeddings for all chunks, we need to find the ones most similar to the user's question embedding. FAISS does this extremely fast — even with thousands of vectors, it finds the top matches in milliseconds.

**Why `IndexFlatL2`?**

| Index Type | Speed | Accuracy | When to use |
|---|---|---|---|
| FlatL2 ✅ | Fast (small data) | 100% exact | < 100k vectors |
| IVFFlat | Faster (large data) | ~99% | > 100k vectors |
| HNSW | Fastest | ~95% | Millions of vectors |

For a single document (typically 50–500 chunks), `FlatL2` gives perfect accuracy with no overhead.

**What is L2?**
L2 = Euclidean distance. Two vectors are "similar" if their L2 distance is small (they are close together in 384-dimensional space).

---

### 🤖 FLAN-T5-base (`google/flan-t5-base`)
**What:** A 250-million parameter language model by Google, fine-tuned on hundreds of NLP tasks including question answering.

**Why FLAN-T5?**

| Model | Size | CPU speed | Quality | Needs GPU? |
|---|---|---|---|---|
| FLAN-T5-base ✅ | 900 MB | 2-5 sec/query | Good | ❌ No |
| FLAN-T5-large | 3 GB | 15-20 sec | Better | Optional |
| LLaMA 3.2 3B | 6+ GB | Very slow | Best | ✅ Recommended |
| GPT-4 | External API | Fast | Best | N/A |

FLAN-T5-base is the only model that:
1. Runs acceptably fast on CPU (no GPU required)
2. Is small enough to download and cache locally
3. Is instruction-tuned (understands commands like "Answer based only on the context")

**How it generates answers:**
FLAN-T5 is an **encoder-decoder** model (T5 architecture). We give it:
```
Answer the question based only on the given context.
If the answer is not in the context, say 'I could not find an answer.'

Context:
[chunk 1 text]
[chunk 2 text]
...

Question: What is X?

Answer:
```
The model reads this entire prompt and generates the answer text.

---

### 🎨 Vanilla HTML/CSS/JavaScript
**Why not React/Vue/Next.js?**

| Framework | Why NOT chosen |
|---|---|
| React | Needs Node.js, npm, build pipeline |
| Vue | Same overhead |
| **Vanilla** ✅ | Zero dependencies, instant load, Flask serves it directly |

The UI is a single HTML file with separate CSS and JS files. Flask's `render_template` serves it directly — no separate frontend server needed.

---

## 6. Step-by-Step Pipeline Explained

### Step 1 — PDF Upload

**File:** `app/routes.py` → `/upload` endpoint

**What happens:**
1. User drags a PDF onto the browser drop zone
2. JavaScript sends the file to `POST /upload` as `multipart/form-data`
3. Flask validates: is it a PDF? Is it under 50 MB?
4. Flask saves it temporarily to `uploads/<uuid>.pdf`
5. Processing begins...

**Why save temporarily?**
We need the file path to pass to PyMuPDF. After extraction, the file is immediately deleted — we don't store PDFs.

**Why UUID for filename?**
If two users upload `report.pdf` at the same time, UUIDs prevent filename collisions.

```python
safe_filename = f"{session_id}.pdf"  # e.g. "a3f92b1c-....pdf"
```

---

### Step 2 — Text Extraction

**File:** `app/pdf_processor.py` → `extract_text_from_pdf()`

**What happens:**
```python
doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text("text")  # Extract raw text from page
```

PyMuPDF reads each page and extracts all selectable text. Pages are joined with double newlines to preserve paragraph structure.

**Output:** One large string containing all the document's text.

**Limitation:** This only works on PDFs with selectable text. Scanned PDFs (photos of pages) return empty strings and would need OCR (Optical Character Recognition) like Tesseract.

---

### Step 3 — Chunking

**File:** `app/pdf_processor.py` → `chunk_text()`

**The problem:** The full document text might be 50,000 words. We can't embed it as one piece (too large for the model) and we can't search it efficiently as one blob.

**The solution:** Split it into overlapping windows.

```
Document words: [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 ...]

Chunk 1: w1 → w500
Chunk 2: w451 → w950   ← 50-word overlap with Chunk 1
Chunk 3: w901 → w1400  ← 50-word overlap with Chunk 2
...
```

**Why overlap?**
Without overlap, if an important sentence falls at the boundary between two chunks, it gets cut in half and neither chunk has the full context. Overlap ensures boundary content appears in at least one complete chunk.

**Why 500 words per chunk?**
- Large enough to contain a full thought/paragraph
- Small enough to fit in the LLM's context window alongside other chunks
- Small enough that FAISS can find very targeted matches

---

### Step 4 — Embedding

**File:** `app/embedder.py` → `build_faiss_index()`

**What happens:**
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)  # shape: (num_chunks, 384)
```

Each chunk becomes a **384-dimensional vector** — a list of 384 decimal numbers.

**What do these numbers mean?**
They represent the semantic meaning of the text in a high-dimensional mathematical space. Texts with similar meanings have similar vectors (small distance between them). Texts with different meanings have different vectors (large distance).

**Example (simplified to 3D for illustration):**

```
"The company revenue grew by 20%" → [0.82, 0.31, 0.67, ...]
"Annual sales increased significantly" → [0.79, 0.33, 0.71, ...]  ← similar!
"The weather is nice today" → [0.12, 0.89, 0.04, ...]  ← very different
```

---

### Step 5 — Vector Index (FAISS)

**File:** `app/embedder.py` → `build_faiss_index()`

```python
dim = embeddings.shape[1]   # 384
index = faiss.IndexFlatL2(dim)
index.add(embeddings)        # Store all chunk vectors
```

FAISS builds an index over all the chunk vectors. This index allows us to search for the **nearest neighbors** (most similar vectors) extremely fast.

**Think of it like:**
Instead of comparing your question to every chunk one by one (slow), FAISS organizes the vectors in memory so it can jump directly to the similar ones (fast).

---

### Step 6 — Question Embedding & Retrieval

**File:** `app/embedder.py` → `retrieve_top_chunks()`

When the user asks a question:

```python
# 1. Embed the question using the SAME model
query_vec = model.encode(["What is the revenue growth?"])
# → shape: (1, 384) — one 384-dim vector

# 2. Search FAISS for the 5 closest chunk vectors
distances, indices = index.search(query_vec, k=5)
# → indices: [42, 17, 8, 63, 29]  (positions of top-5 chunks)

# 3. Return those chunks
results = [chunks[i] for i in indices[0]]
```

**Why the same model for both chunks and questions?**
Both must live in the same 384-dimensional vector space for the distance comparison to be meaningful. If you embedded chunks with model A and questions with model B, the coordinate systems wouldn't match.

---

### Step 7 — Answer Generation (LLM)

**File:** `app/qa_engine.py` → `answer_question()`

The top 5 retrieved chunks are combined into a context string and given to FLAN-T5:

```python
prompt = f"""Answer the question based only on the given context.
If the answer is not in the context, say 'I could not find an answer in the document.'

Context:
{context}

Question: {question}

Answer:"""

inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Key parameters:**
| Parameter | Value | Why |
|---|---|---|
| `max_length=1024` | Input limit | FLAN-T5-base's max context |
| `max_new_tokens=256` | Output limit | Enough for a paragraph answer |
| `num_beams=4` | Beam search | Better quality than greedy decoding |
| `no_repeat_ngram_size=3` | 3 | Prevents repetition in answers |

**What is beam search?**
Instead of always picking the single most likely next word (greedy), beam search keeps track of the top 4 candidate sequences at each step and picks the overall best sequence. It produces more coherent, higher quality answers.

---

## 7. Project Structure Explained

```
Document QA/
│
├── app/                        ← Python package (the backend)
│   ├── __init__.py             ← Creates Flask app (app factory pattern)
│   ├── routes.py               ← HTTP endpoints: /upload, /ask
│   ├── pdf_processor.py        ← PDF text extraction + chunking
│   ├── embedder.py             ← Embeddings + FAISS vector search
│   └── qa_engine.py            ← LLM answer generation
│
├── static/                     ← Frontend static files (served by Flask)
│   ├── css/style.css           ← All UI styles (dark glassmorphism theme)
│   └── js/main.js              ← All frontend logic (upload, chat, etc.)
│
├── templates/                  ← HTML templates (Jinja2)
│   └── index.html              ← The single-page app shell
│
├── uploads/                    ← Temporary PDF storage (auto-deleted)
│   └── .gitkeep                ← Keeps folder in git even though it's empty
│
├── models/                     ← HuggingFace model cache
│   └── .gitkeep                ← Keeps folder in git even though it's empty
│
├── docs/                       ← Screenshots, demo video for README
│
├── run.py                      ← Entry point: python run.py
├── requirements.txt            ← All Python dependencies
├── .gitignore                  ← Files to exclude from git
├── LICENSE                     ← MIT License
├── README.md                   ← Setup guide and documentation
└── notes.md                    ← This file — detailed technical notes
```

**Why the `app/` package structure?**
Separating each concern into its own file makes the code:
- Easy to read and understand
- Easy to test each part independently
- Easy to swap one component (e.g., replace FLAN-T5 with a different model without touching other files)

**Why use Flask app factory (`create_app`)?**
The factory pattern (`app/__init__.py` returning a Flask app) makes it easy to create multiple app instances with different configs (e.g., testing vs production).

---

## 8. File-by-File Code Explanation

### `run.py`
The entry point. Imports `create_app`, creates the Flask app, and starts the development server on port 5000.

### `app/__init__.py`
The **app factory**:
- Creates the Flask app
- Sets `UPLOAD_FOLDER` config
- Sets a random `secret_key` for session security
- Creates the `uploads/` directory if it doesn't exist
- Registers the routes blueprint

### `app/routes.py`
Two endpoints:

**`POST /upload`:**
- Validates file is a PDF and under 50 MB
- Saves to `uploads/<uuid>.pdf`
- Calls `extract_text_from_pdf()` → `chunk_text()` → `build_faiss_index()`
- Stores `{ chunks, faiss_index }` in `_session_store[session_id]`
- Deletes the PDF file
- Returns `{ session_id, chunk_count, filename }`

**`POST /ask`:**
- Looks up session data by `session_id`
- Calls `retrieve_top_chunks(question, chunks, index, top_k=5)`
- Calls `answer_question(question, top_chunks)`
- Returns `{ answer, sources }`

**Why in-memory session store instead of a database?**
For simplicity. A real production system would use Redis or a database. For local/single-user use, a Python dict is perfectly fine.

### `app/pdf_processor.py`
- `extract_text_from_pdf()` — opens PDF, iterates pages, extracts text, joins with `\n\n`
- `chunk_text()` — splits on word boundaries, creates overlapping windows

### `app/embedder.py`
- `_get_embed_model()` — lazy-loads the SentenceTransformer model (only downloads/loads once)
- `build_faiss_index()` — encodes all chunks → creates FAISS FlatL2 index
- `retrieve_top_chunks()` — encodes query → searches index → returns top-K chunks

### `app/qa_engine.py`
- `_get_qa_model()` — lazy-loads FLAN-T5 tokenizer + model
- `answer_question()` — builds prompt → tokenizes → generates → decodes → returns string

### `static/js/main.js`
- Handles drag-and-drop and click-to-browse upload
- Sends `FormData` to `/upload`
- Shows animated progress bar and document info card
- Chat interface: appends user/AI message bubbles
- Shows "thinking" animation while waiting for answer
- Sources accordion toggle
- Auto-growing textarea
- Enter-to-send keyboard shortcut

### `static/css/style.css`
- CSS custom properties (variables) for the entire design system
- Dark glassmorphism panels with `backdrop-filter: blur()`
- Animated background gradient orbs
- Responsive grid layout (side-by-side on desktop, stacked on mobile)
- Smooth micro-animations for all interactions

---

## 9. API Design

### `POST /upload`

**Request:**
```
Content-Type: multipart/form-data
Body: file = <PDF binary>
```

**Success Response (200):**
```json
{
  "session_id": "a3f92b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "filename": "annual_report.pdf",
  "chunk_count": 142,
  "message": "Document processed successfully. You can now ask questions."
}
```

**Error Response (400/422/500):**
```json
{
  "error": "Only PDF files are supported."
}
```

---

### `POST /ask`

**Request:**
```json
{
  "session_id": "a3f92b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "question": "What was the total revenue in 2023?"
}
```

**Success Response (200):**
```json
{
  "answer": "The total revenue in 2023 was $4.2 billion, representing a 12% increase from the previous year.",
  "sources": [
    "...In fiscal year 2023, the company reported total revenues of $4.2 billion...",
    "...Revenue grew by 12% year-over-year, driven by strong performance in..."
  ]
}
```

---

## 10. Frontend Design Decisions

### Dark Glassmorphism Theme
- **Background:** Deep space dark (`#07090f`)
- **Panels:** Semi-transparent with `backdrop-filter: blur(20px)` → frosted glass effect
- **Accent color:** Indigo (`#6366f1`) — professional, modern, not harsh on eyes
- **Gradient orbs:** Three large blurred orbs that slowly drift — creates depth

### Two-panel Layout
- **Left:** Upload panel (fixed width 360px) — document management
- **Right:** Chat panel (flexible) — Q&A conversation
- Switches to stacked layout on mobile (< 900px)

### Chat Interface
- Messages are color-coded: user (indigo gradient bubble) vs AI (dark elevated bubble)
- Animated "thinking dots" while waiting for an answer
- Sources accordion — expandable section showing which document excerpts were used
- Example question chips — clickable prompts to guide users

---

## 11. Why No External APIs?

The task requirement was **no external APIs (OpenAI, etc.)**. Here's why this matters and how we solved it:

| Concern | External API | Our Approach |
|---|---|---|
| **Privacy** | Data leaves your machine | Everything runs locally |
| **Cost** | Pay per token (~$0.002/1K tokens) | Free after model download |
| **Internet** | Required always | Only for first model download |
| **Reliability** | API downtime affects you | Always works offline |
| **Speed** | Network latency adds delay | Only limited by local CPU |

The trade-off: local models (FLAN-T5-base) are less powerful than GPT-4. But for document QA on focused content, they perform very well.

---

## 12. Limitations & Future Improvements

### Current Limitations

| Limitation | Reason | Potential Fix |
|---|---|---|
| Scanned PDFs not supported | PyMuPDF can't read image-based PDFs | Add `pytesseract` OCR step |
| Single document per session | Simple in-memory store | Add persistent storage (SQLite/Redis) |
| No conversation memory | Each question is independent | Add chat history to LLM prompt |
| Answer quality limited by FLAN-T5-base | Small model | Upgrade to flan-t5-large or TinyLlama |
| No multi-user support | Shared in-memory store | Add proper session management |
| Tables/figures poorly handled | Extracted as raw text | Add specialized table extraction |

### Possible Upgrades

1. **Better LLM:** Replace FLAN-T5-base with `Mistral-7B-Instruct` (quantized) for much better answers — requires more RAM (~4 GB)
2. **OCR support:** Add `pytesseract` + `pdf2image` for scanned PDFs
3. **Multi-document:** Allow uploading multiple PDFs and querying across all
4. **Persistent sessions:** Store indexes in SQLite so they survive server restarts
5. **Streaming answers:** Use `model.generate()` with streaming to show the answer word-by-word as it's generated
6. **Better chunking:** Use semantic/sentence-aware chunking instead of word-count windows

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — combining search with language model generation |
| **Embedding** | Converting text into a fixed-size vector of numbers that captures its meaning |
| **Vector** | A list of numbers representing a point in high-dimensional space |
| **FAISS** | Facebook AI Similarity Search — a library for fast vector lookup |
| **Cosine Similarity** | A measure of how similar two vectors are (angle between them) |
| **L2 Distance** | Euclidean distance between two vectors — used by FAISS FlatL2 |
| **Token** | A word or word-fragment that AI models process as a unit |
| **Chunking** | Splitting a large text into smaller, manageable pieces |
| **Overlap** | Shared words between consecutive chunks to preserve boundary context |
| **Beam Search** | A decoding strategy that tracks multiple candidate outputs for better quality |
| **LLM** | Large Language Model — an AI trained on vast text to generate language |
| **FLAN-T5** | Google's T5 model fine-tuned with instructions (Fine-tuned Language Net) |
| **Encoder-Decoder** | Architecture where one part reads input (encoder) and another writes output (decoder) |
| **Blueprint** | Flask's way of organizing routes into separate files/modules |
| **App Factory** | A function that creates and returns a Flask app — allows flexible configuration |
| **Session Store** | In-memory dictionary mapping session IDs to user document data |
| **Glassmorphism** | UI design style using frosted glass effect with blur and transparency |
| **Semantic Search** | Finding results by meaning/intent rather than exact keyword matching |
