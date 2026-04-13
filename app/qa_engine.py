"""
qa_engine.py
------------
Generates answers to user questions using retrieved document context.

Model: google/flan-t5-base (250M params, CPU-friendly, instruction-tuned)

FLAN-T5 is trained on a mixture of NLP tasks including question answering,
making it well-suited for extractive and abstractive answers from context.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_QA_MODEL_NAME = "google/flan-t5-base"
_qa_model = None
_qa_tokenizer = None


def _get_qa_model():
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        _qa_tokenizer = AutoTokenizer.from_pretrained(_QA_MODEL_NAME)
        _qa_model = AutoModelForSeq2SeqLM.from_pretrained(_QA_MODEL_NAME)
        _qa_model.eval()
    return _qa_model, _qa_tokenizer


def answer_question(question: str, context_chunks: list[str]) -> str:
    """
    Generate an answer to `question` using the provided context chunks.

    The prompt is formatted as an instruction for FLAN-T5:
        "Answer the question based only on the given context. ..."

    Args:
        question:       The user's question string.
        context_chunks: Relevant chunks retrieved from the document.

    Returns:
        A string answer grounded in the document context.
    """
    model, tokenizer = _get_qa_model()

    # Combine retrieved chunks into a single context block
    context = "\n\n".join(context_chunks)

    # Truncate context to avoid exceeding model token limit
    max_context_chars = 3000
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    prompt = (
        "Answer the question based only on the given context. "
        "If the answer is not in the context, say 'I could not find an answer in the document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()
