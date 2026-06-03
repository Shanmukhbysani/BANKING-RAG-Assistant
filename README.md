# RAG Banking Assistant

A lightweight Streamlit app for uploading banking policy documents, chunking them into semantically meaningful passages, indexing them in ChromaDB, and answering questions with retrieved source context.

This project is designed as a prototype for secure, grounded question answering over internal banking documents. It emphasizes retrieval-first answers, source attribution, and minimal infrastructure so it can be demoed or iterated on quickly.

## What It Does

- Upload `.txt` policy or procedure documents from the sidebar.
- Split documents into smaller overlapping chunks with `RecursiveCharacterTextSplitter`.
- Embed chunks with a local Hugging Face sentence-transformer model.
- Store and search embeddings in ChromaDB.
- Retrieve the most relevant chunks for each question.
- Generate a grounded response that stays close to the retrieved context.
- Show source chunks so users can verify where each answer came from.

## Why This Design

Banking and policy workflows usually need two things at the same time:

1. Answers that are useful and concise.
2. Traceability back to the original policy text.

This project follows a retrieval-augmented generation pattern so the assistant does not rely on memory alone. Instead, it searches the document store first and then answers from the most relevant text it found.

## Project Structure

```text
banking-rag-assistant/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Tech Stack

- Streamlit for the UI.
- ChromaDB for local vector storage and similarity search.
- sentence-transformers via Hugging Face embeddings for local text vectorization.
- LangChain community loaders and text splitters for ingestion and chunking.

## How It Works

### 1. Document ingestion

Users upload one or more `.txt` files in the sidebar. Each file is written to a temporary path, loaded with `TextLoader`, split into chunks, and added to the vector store.

### 2. Chunking

The app uses `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and 50 characters of overlap. That balance keeps chunks small enough for retrieval while preserving surrounding context.

### 3. Embedding and indexing

The app uses `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`, a compact model that works well for semantic search and is practical for local prototyping.

### 4. Retrieval

When a user asks a question, the app searches the vector store for the top 3 most relevant chunks using similarity search.

### 5. Grounded answer generation

The current implementation uses a mock grounded response function that demonstrates the control flow for a secure production LLM. The intended pattern is to pass retrieved context into an LLM and constrain it to answer only from that context.

### 6. Source attribution

The UI shows each retrieved chunk in an expandable section so the user can inspect the exact supporting text.

## Setup

### Prerequisites

- Python 3.10 or newer.
- A virtual environment is recommended.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal.

## Usage

1. Start the app.
2. Upload one or more `.txt` banking policy documents in the sidebar.
3. Wait for indexing to complete.
4. Ask a policy question in the query field.
5. Review the answer and inspect the matching source chunks.

### Example questions

- What is the verification window for a high-value wire transfer?
- Which documents are required before approving the loan exception?
- What is the retention period for customer onboarding records?
- When must a manual review be escalated to compliance?

## Configuration Notes

The app is currently optimized for local development and demos.

- Vector storage is in-memory by default.
- Document processing is limited to `.txt` files.
- The generation function is a placeholder for a secure LLM integration.

If you want to turn this into a more durable internal tool, consider adding persistence, authentication, and a real model backend.

## Recommended Production Improvements

- Persist ChromaDB to disk with a `persist_directory`.
- Replace the mock generator with a secure LLM endpoint or on-prem model.
- Add support for PDF, DOCX, and HTML ingestion.
- Add metadata filters for document type, department, or policy version.
- Track upload history and indexing status per user session.
- Add citation formatting with page numbers or section labels.
- Add evaluation tests for retrieval quality and answer faithfulness.

## Security Considerations

Because this is a banking-oriented assistant, the production version should be designed carefully.

- Do not send sensitive policy data to external services unless that is explicitly approved.
- Validate and restrict uploaded file types.
- Add authentication and role-based access control before deployment.
- Log access to documents and generated answers if auditability is required.
- Prefer local or private model inference for confidential content.

## Troubleshooting

### No answers are returned

- Make sure at least one document has been uploaded.
- Confirm the text contains policy language that matches the question.
- Try rephrasing the question using terms that appear in the source document.

### Upload succeeds but nothing seems indexed

- Check that the file is plain text and not a renamed binary file.
- Verify the app has permission to create temporary files.
- Restart the app and re-upload the documents.

### Embedding model download is slow

- The first run may download the Hugging Face model.
- Use a stable internet connection for the initial setup.
- After the model is cached, startup should be faster.

## Limitations

- The current example only supports `.txt` files.
- The vector store is in-memory, so data is lost when the app restarts.
- The response function is demonstrative rather than a real LLM integration.
- The app assumes the retrieved chunks are sufficient for the answer.

## Suggested Next Step

If you want, the next useful upgrade is to add a real `app.py` and `requirements.txt` that match this README, including persistent Chroma storage and a proper grounded LLM call.
