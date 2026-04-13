/**
 * main.js — DocuQuery Frontend Logic
 *
 * Handles:
 *  - Drag & drop / click-to-browse PDF upload
 *  - POST /upload → display document info card
 *  - Chat-style Q&A via POST /ask
 *  - Auto-growing textarea
 *  - Sources accordion toggle
 */

(() => {
  "use strict";

  /* ── DOM References ── */
  const dropZone       = document.getElementById("drop-zone");
  const fileInput      = document.getElementById("file-input");
  const browseBtn      = document.getElementById("browse-btn");
  const uploadProgress = document.getElementById("upload-progress");
  const progressBar    = document.getElementById("progress-bar");
  const progressFile   = document.getElementById("progress-filename");
  const progressStatus = document.getElementById("progress-status");
  const docCard        = document.getElementById("doc-card");
  const docCardName    = document.getElementById("doc-card-name");
  const docCardMeta    = document.getElementById("doc-card-meta");
  const removeDocBtn   = document.getElementById("remove-doc-btn");
  const emptyState     = document.getElementById("empty-state");
  const chatMessages   = document.getElementById("chat-messages");
  const questionInput  = document.getElementById("question-input");
  const sendBtn        = document.getElementById("send-btn");

  /* ── State ── */
  let sessionId    = null;
  let isProcessing = false;

  /* ══════════════════════════════════════════
     UPLOAD LOGIC
  ══════════════════════════════════════════ */

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  /* Drag & Drop */
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type === "application/pdf") {
      handleFile(file);
    } else {
      showError("Only PDF files are supported.");
    }
  });

  /** Upload a PDF file to /upload */
  async function handleFile(file) {
    if (file.size > 50 * 1024 * 1024) {
      showError("File size exceeds 50 MB limit.");
      return;
    }

    // Reset state
    resetSession();

    // Show progress
    progressFile.textContent = file.name;
    progressStatus.textContent = "Uploading…";
    progressBar.style.width = "30%";
    uploadProgress.classList.remove("hidden");
    progressBar.classList.add("indeterminate");

    const formData = new FormData();
    formData.append("file", file);

    try {
      progressStatus.textContent = "Extracting text…";

      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Upload failed.");
      }

      // Success
      progressBar.classList.remove("indeterminate");
      progressBar.style.width = "100%";
      progressStatus.textContent = "Ready ✓";

      setTimeout(() => {
        uploadProgress.classList.add("hidden");
        progressBar.style.width = "0%";
        showDocCard(file.name, data.chunk_count);
      }, 800);

      sessionId = data.session_id;
      enableChat();

    } catch (err) {
      progressBar.classList.remove("indeterminate");
      uploadProgress.classList.add("hidden");
      showError(err.message);
    }

    // Reset file input so same file can be re-uploaded
    fileInput.value = "";
  }

  function showDocCard(filename, chunkCount) {
    docCardName.textContent = filename;
    docCardMeta.textContent = `${chunkCount} text chunks indexed · Ready to answer questions`;
    docCard.classList.remove("hidden");
  }

  removeDocBtn.addEventListener("click", () => {
    resetSession();
  });

  function resetSession() {
    sessionId = null;
    docCard.classList.add("hidden");
    uploadProgress.classList.add("hidden");
    disableChat();
    clearChat();
  }

  /* ══════════════════════════════════════════
     CHAT LOGIC
  ══════════════════════════════════════════ */

  function enableChat() {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
    emptyState.classList.remove("hidden");
    chatMessages.classList.add("hidden");
  }

  function disableChat() {
    questionInput.disabled = true;
    sendBtn.disabled = true;
    emptyState.classList.remove("hidden");
    chatMessages.classList.add("hidden");
  }

  function clearChat() {
    chatMessages.innerHTML = "";
  }

  /* Auto-growing textarea */
  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
    sendBtn.disabled = questionInput.value.trim() === "" || !sessionId;
  });

  /* Send on Enter (Shift+Enter = new line) */
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled && !isProcessing) submitQuestion();
    }
  });

  sendBtn.addEventListener("click", () => {
    if (!isProcessing) submitQuestion();
  });

  async function submitQuestion() {
    const question = questionInput.value.trim();
    if (!question || !sessionId || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    questionInput.disabled = true;
    questionInput.value = "";
    questionInput.style.height = "auto";

    // Show chat messages area
    emptyState.classList.add("hidden");
    chatMessages.classList.remove("hidden");

    // Add user message
    appendUserMessage(question);

    // Add thinking bubble
    const thinkingId = appendThinkingBubble();

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question }),
      });

      const data = await response.json();

      removeThinkingBubble(thinkingId);

      if (!response.ok) {
        appendAIMessage(data.error || "An error occurred.", [], true);
      } else {
        appendAIMessage(data.answer, data.sources || []);
      }

    } catch (err) {
      removeThinkingBubble(thinkingId);
      appendAIMessage("Network error. Please check that the server is running.", [], true);
    }

    isProcessing = false;
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }

  /** Append a user message bubble */
  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message-group";
    div.innerHTML = `
      <div class="message user-message">
        <div class="avatar" aria-hidden="true">You</div>
        <div class="bubble">${escapeHtml(text)}</div>
      </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  /** Append a pulsing "thinking" bubble, return its ID */
  function appendThinkingBubble() {
    const id = "thinking-" + Date.now();
    const div = document.createElement("div");
    div.className = "message-group";
    div.id = id;
    div.innerHTML = `
      <div class="message ai-message">
        <div class="avatar" aria-hidden="true">AI</div>
        <div class="bubble thinking-bubble" aria-label="Generating answer">
          <span class="thinking-dot" aria-hidden="true"></span>
          <span class="thinking-dot" aria-hidden="true"></span>
          <span class="thinking-dot" aria-hidden="true"></span>
        </div>
      </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
    return id;
  }

  function removeThinkingBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  /** Append an AI answer bubble with optional sources */
  function appendAIMessage(text, sources, isError = false) {
    const div = document.createElement("div");
    div.className = "message-group";

    let sourcesHtml = "";
    if (sources && sources.length > 0 && !isError) {
      const sourceItems = sources.map((s, i) => `
        <div class="source-item">
          <div class="source-label">Excerpt ${i + 1}</div>
          <div>${escapeHtml(s.substring(0, 300))}${s.length > 300 ? "…" : ""}</div>
        </div>`).join("");

      sourcesHtml = `
        <button class="sources-toggle" onclick="toggleSources(this)" aria-expanded="false">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
          View ${sources.length} source excerpt${sources.length > 1 ? "s" : ""}
        </button>
        <div class="sources-list">${sourceItems}</div>`;
    }

    div.innerHTML = `
      <div class="message ai-message">
        <div class="avatar" aria-hidden="true">AI</div>
        <div class="bubble ${isError ? "error-bubble" : ""}">
          ${escapeHtml(text)}
          ${sourcesHtml}
        </div>
      </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  /* ══════════════════════════════════════════
     HELPERS
  ══════════════════════════════════════════ */

  function showError(msg) {
    const div = document.createElement("div");
    div.style.cssText = `
      position:fixed; bottom:24px; right:24px; z-index:999;
      background:#1a0a0a; border:1px solid rgba(248,81,73,0.4);
      color:#fca5a5; border-radius:10px; padding:14px 18px;
      font-size:0.85rem; max-width:340px; box-shadow:0 8px 32px rgba(0,0,0,0.6);
      animation: msg-in 0.3s ease;
    `;
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 5000);
  }

  function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(str).replace(/[&<>"']/g, (m) => map[m]);
  }

  /* ══════════════════════════════════════════
     GLOBAL HELPERS (called via onclick)
  ══════════════════════════════════════════ */

  window.toggleSources = function (btn) {
    btn.classList.toggle("open");
    const list = btn.nextElementSibling;
    list.classList.toggle("open");
    btn.setAttribute("aria-expanded", list.classList.contains("open"));
  };

  window.setExampleQuestion = function (btn) {
    if (!sessionId) {
      showError("Please upload a PDF document first.");
      return;
    }
    questionInput.value = btn.textContent;
    questionInput.dispatchEvent(new Event("input"));
    questionInput.focus();
  };

})();
