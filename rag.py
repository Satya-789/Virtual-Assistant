from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ✅ Load env variables FIRST
load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"


# ✅ LLM factory (IMPORTANT)
def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.7,
        max_tokens=500
    )


# ✅ Vector DB factory
def get_vector_store():
    ef = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True}
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR)
    )


# ✅ Process URLs → store in vector DB
def process_urls(urls):
    yield "Initializing components...✅"

    vector_store = get_vector_store()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text...✅"
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = splitter.split_documents(data)

    yield "Adding to vector DB...✅"
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done...✅"


# ✅ Generate answer
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

    return result.get("answer", ""), result.get("sources", "")


# ✅ Run example
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for step in process_urls(urls):
        print(step)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with the date?"
    )

    print("\nAnswer:", answer)
    print("\nSources:", sources)
