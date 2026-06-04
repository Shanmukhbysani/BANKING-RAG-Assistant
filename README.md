# RAG-Based Banking Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions over banking policy documents — with **grounded answers and source attribution**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shanmukhbysani/BANKING-RAG-Assistant/blob/main/banking_rag_assistant.ipynb)

> Click the badge to run the interactive chatbot in your browser.

---

## What it does

Bank staff and customers shouldn't have to read 50-page policy PDFs to find one answer. This assistant lets them ask in plain English — *"What's the overdraft interest rate?"* — and returns a precise answer grounded in the bank's documents, citing which document it came from.

## How RAG works here

```
Documents → chunk → embed → store in vector DB
                                     │
User question → embed → semantic search ──► top relevant chunks
                                     │
                          LLM answers using ONLY those chunks → answer + sources
```

1. **Chunking (LangChain):** policy documents are split into overlapping chunks so retrieval is precise.
2. **Embeddings (Sentence Transformers):** each chunk is converted to a vector with `all-MiniLM-L6-v2` — runs locally, no API needed.
3. **Vector store (ChromaDB):** chunks are indexed for fast semantic search.
4. **Retrieval:** the question is embedded and the most similar chunks are pulled back.
5. **Generation (LLM):** the model answers using only the retrieved context, and cites its sources.

## Why RAG instead of just asking an LLM?

- **Grounding:** the model answers from the bank's actual documents, not its training data — so it can't make up a fee or rate.
- **Source attribution:** every answer cites which document it came from, which is essential for trust and compliance in banking.
- **Updatable:** change a policy document and the assistant updates instantly — no expensive model retraining.

> **Interview line:** "RAG beats fine-tuning here because banking policies change constantly. With RAG I just update the document and the answer updates — and every answer is traceable to a source, which matters for compliance."

## Tech stack

`LangChain` · `ChromaDB` · `Sentence Transformers` · `Groq (LLM)` · `Gradio` · `Python`

## How to run

**In Colab (easiest):** click the badge above → run the single cell → paste your free Groq key (console.groq.com/keys) → the chat UI appears in the output.

**Permanent live demo (Hugging Face Spaces):** create a free Space (Gradio SDK), upload the app code + `requirements.txt`, add `GROQ_API_KEY` in Settings → Secrets. You get a permanent public URL.

## Limitations

- The knowledge base here is synthetic; production would ingest the bank's real policy PDFs via a document loader.
- Retrieval quality depends on chunk size and embedding model — both would be tuned on real documents.

## Author

**Shanmukh Bysani** — B.E. AI & Data Science, CBIT Hyderabad
