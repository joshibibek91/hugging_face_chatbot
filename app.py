import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA

from transformers import pipeline
import torch
import tempfile
import os

st.set_page_config(page_title="📄 Document Q&A", layout="centered")
st.title("📄 Document Q&A with Hugging Face + LangChain")

# Upload file
uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

# Input query
query = st.text_input("Ask a question based on the document")

if uploaded_file and query:
    with st.spinner("Processing the document..."):
        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Load document
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        # Embedding model
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Vector DB
        db = FAISS.from_documents(split_docs, embedding)
        retriever = db.as_retriever()

        # Hugging Face LLM (Falcon-7B or GPT2)
        model_name = "google/flan-t5-base"
        # try:
        hf_pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=256,
            device= -1
        )
        # except Exception as e:
        #     st.warning(f"Falcon-7B model too large or failed: {e}. Falling back to GPT2.")
        #     hf_pipeline = pipeline(
        #         "text-generation",
        #         model="gpt2",
        #         max_new_tokens=256,
        #         device=-1
        #     )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Retrieval Q&A
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        result = qa({"query": query})

        # Output
        st.subheader("🧠 Answer")
        st.success(result["result"])

        # Cleanup
        os.remove(file_path)