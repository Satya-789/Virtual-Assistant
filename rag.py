from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

# ✅ Fix USER_AGENT warning
os.environ["USER_AGENT"] = "streamlit-app"

# Load env variables
load_dotenv()

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma   # ✅ FIXED IMPORT
from langchain_groq import ChatGroq

# Constants
CHUNK_SIZE = 1000
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "assistant"


# ✅ LLM factory
def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.5,
        max_tokens=500
    )


# ✅ Vector DB (lightweight, no embeddings needed)
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR)
    )


# ✅ Process URLs → store in vector DB
def process_urls(urls):
    yield "Initializing... ✅"

    vector_store = get_vector_store()

    yield "Resetting database... ✅"
    vector_store.reset_collection()

    yield "Loading URLs... ✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Splitting text... ✅"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    yield "Storing in vector database... ✅"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Done! ✅"


# ✅ Generate Answer
def generate_answer(query):
    vector_store = get_vector_store()
    llm = get_llm()

    if not vector_store:
        raise RuntimeError("Vector DB not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain.invoke(
        {"question": query},
        return_only_outputs=True
    )

    answer = result.get("answer", "")
    sources = result.get("sources", "")

    return answer, sources


# ✅ Run standalone (for testing)
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for step in process_urls(urls):
        print(step)

    answer, sources = generate_answer(
        "What is the 30-year fixed mortgage rate?"
    )

    print("\nAnswer:", answer)
    print("\nSources:", sources)
