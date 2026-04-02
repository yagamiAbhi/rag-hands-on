import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()


def get_embedding_model():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Set GOOGLE_API_KEY or OPENAI_API_KEY in environment")

    os.environ.setdefault("OPENAI_API_KEY", api_key)

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
    except Exception:
        pass

    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    except Exception as e:
        raise ImportError("No supported embedding implementation found.") from e


def load_vector_store(persist_directory="db/chroma_db"):
    embedding_model = get_embedding_model()
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db


def main():
    print("=== RAG Retrieval Pipeline ===")

    db = load_vector_store()

    query = "How much did Microsoft pay to acquire GitHub?"

    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}: {doc.page_content}\n")

    return relevant_docs


if __name__ == "__main__":
    main()
