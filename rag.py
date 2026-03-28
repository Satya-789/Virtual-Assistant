from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
load_dotenv()

# Constants
CHUNK_SIZE = 1000
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "assistant"


# ✅ LLM factory
def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.5,
        max_tokens=500
    )


# ✅ Vector DB
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
    yield "Initializing...✅"

    vector_store = get_vector_store()

    yield "Resetting DB...✅"
    vector_store.reset_collection()

    yield "Loading URLs...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Splitting text...✅"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    yield "Storing in DB...✅"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Done...✅"


# ✅ Generate Answer
def generate_answer(query):
    vector_store = get_vector_store()
    llm = get_llm()

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain.invoke({"question": query})

    return result.get("answer", ""), result.get("sources", "")
