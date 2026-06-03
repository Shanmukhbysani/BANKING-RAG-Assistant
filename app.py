import os
import re
import tempfile

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="RAG Banking Assistant", layout="wide")
st.title("🏦 RAG-Based Banking Document Assistant")
st.caption("Secure, Grounded Question-Answering with Explicit Source Attribution")


@st.cache_resource
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(embedding_function=embeddings)
    return vector_db


vector_store = get_vector_db()


def generate_grounded_response(query, retrieved_chunks):
    if not retrieved_chunks:
        return (
            "I am sorry, but I cannot find any relevant banking policies in the provided documents to answer your question.",
            [],
        )

    query_terms = {
        token.lower()
        for token in re.findall(r"\b\w+\b", query)
        if len(token) > 2
    }

    scored_sentences = []
    for doc in retrieved_chunks:
        source_name = doc.metadata.get("source", "Unknown")
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
        for sentence in sentences:
            sentence_terms = {token.lower() for token in re.findall(r"\b\w+\b", sentence)}
            overlap = len(query_terms & sentence_terms)
            if overlap:
                scored_sentences.append((overlap, source_name, sentence.strip()))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: (-item[0], len(item[2])))
        best_match = scored_sentences[0]
        response = (
            f"The most relevant policy language I found says: {best_match[2]} "
            f"(source: {best_match[1]})."
        )
    else:
        top_chunk = retrieved_chunks[0]
        response = (
            "I found related document content, but no sentence matched the question terms closely enough to answer with confidence. "
            f"The closest source is {top_chunk.metadata.get('source', 'Unknown')}: "
            f"{top_chunk.page_content[:200].strip()}..."
        )

    return response, retrieved_chunks


sidebar = st.sidebar
sidebar.header("📁 Document Ingestion Dashboard")

uploaded_files = sidebar.file_uploader(
    "Upload Policy Documents (.txt)",
    type=["txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_docs_loaded = False
    for uploaded_file in uploaded_files:
        if f"processed_{uploaded_file.name}" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            try:
                loader = TextLoader(temp_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)

                vector_store.add_documents(chunks)
                st.session_state[f"processed_{uploaded_file.name}"] = True
                new_docs_loaded = True
            finally:
                os.remove(temp_path)

    if new_docs_loaded:
        sidebar.success("Successfully chunked and indexed documents into ChromaDB!")


st.subheader("Query Interface")
user_query = st.text_input(
    "Enter your policy question:",
    placeholder="e.g., What is the verification window for a high-value mortgage wire transfer?",
)

if user_query:
    with st.spinner("Retrieving relevant semantic contexts..."):
        matched_context = vector_store.similarity_search(user_query, k=3)
        answer, sources = generate_grounded_response(user_query, matched_context)

    st.markdown("### 🤖 Guarded Assistant Response")
    st.write(answer)

    st.markdown("---")
    st.markdown("### 🔍 Mathematical Grounding & Source Attribution")

    if sources:
        for idx, doc in enumerate(sources):
            with st.expander(f"Reference Match {idx + 1} | Source: {doc.metadata.get('source')}"):
                st.info(f"**Verbatim Chunk Content:**\n\n{doc.page_content}")
    else:
        st.warning("No text components matched this query vector deeply enough.")
else:
    st.info(
        "💡 To start testing, upload a text document containing structural rules or banking policies in the sidebar."
    )
