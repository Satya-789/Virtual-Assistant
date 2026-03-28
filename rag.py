from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os
import shutil

# Fix USER_AGENT warning
os.environ["USER_AGENT"] = "streamlit-app"

# Load env
load_dotenv()

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Constants
CHUNK_SIZE = 1000
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "assistant"


# ✅ LLM
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.5,
        max_tokens=500
    )


# ✅ Vector Store with embeddings (IMPORTANT FIX)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )


# ✅ Process URLs
def process_urls(urls):
    yield "Initializing... ✅"

    # Reset DB
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)

    vector_store = get_vector_store()

    yield "Loading URLs... ✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Splitting text... ✅"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    yield "Storing in DB... ✅"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Done! ✅"


# ✅ Generate Answer
def generate_answer(query):
    vector_store = get_vector_store()
    llm = get_llm()

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain.invoke(
        {"question": query},
        return_only_outputs=True
    )

    return result.get("answer", ""), result.get("sources", "")
