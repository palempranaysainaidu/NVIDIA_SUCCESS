import streamlit as st
import os
import time
from dotenv import load_dotenv

# NVIDIA LangChain imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# LangChain community imports (latest structure)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings  # fixed import

# Core LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# ✅ Set NVIDIA API key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# ✅ Optional: Set USER_AGENT to avoid warnings
os.environ["USER_AGENT"] = "MyNvidiaNimApp/1.0"

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# -------------------------------
# Function: Create Vector Embeddings
# -------------------------------
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading and embedding PDF documents..."):
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("us_census")  # directory path
            st.session_state.docs = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=50
            )

            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:30]
            )

            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )

        st.success("✅ FAISS Vector Store DB is ready!")


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 NVIDIA NIM RAG Demo")

prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate and concise response.

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

user_question = st.text_input("💬 Enter your question about the documents:")

if st.button("📚 Create Document Embeddings"):
    vector_embedding()

# Ensure LLM is available
if "llm" not in st.session_state:
    st.session_state.llm = llm

if user_question:
    if "vectors" not in st.session_state:
        st.warning("Please create embeddings first by clicking the button above.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            documents_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, documents_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_question})
            elapsed = time.process_time() - start

        st.subheader("🧾 Answer:")
        st.write(response['answer'])
        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        with st.expander("📄 Document Similarity Search Context"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("-------------------------------------")
