import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


def get_embedding_model():
    """Return embeddings model; prefer Google Palm (Studio) and fallback to OpenAI."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Set GOOGLE_API_KEY or OPENAI_API_KEY in your environment (or .env)")

    # If using Google AI Studio key, mirror to OPENAI_API_KEY to support OpenAI-style wrappers.
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    # Try Google Generative AI from langchain-google-genai
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        print("Using GoogleGenerativeAIEmbeddings for embedding")
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
    except Exception:
        pass

    # Fallback to OpenAI embeddings for compatibility with existing sample
    from langchain_openai import OpenAIEmbeddings

    print("Using OpenAIEmbeddings fallback")
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))


def load_documents(docs_path="docs"):
    """Load all text documents from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add documents.")

    loader_kwargs = {"encoding": "utf-8", "autodetect_encoding": True}
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs=loader_kwargs,
    )

    try:
        documents = loader.load()
    except Exception as e:
        print("Warning: failed to load with utf-8/autodetect - retrying with cp1252 fallback.")
        loader_kwargs = {"encoding": "cp1252", "autodetect_encoding": True}
        loader = DirectoryLoader(
            path=docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs=loader_kwargs,
        )
        documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)}")
        print(chunk.page_content)

    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create the vector store and persist to disk."""
    print("Creating vector store...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store created at {persist_directory}")
    return vectorstore


def main():
    print("=== RAG Ingestion Pipeline ===")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    if os.path.exists(persist_directory):
        print("Existing vector store found, loading...")
        embedding_model = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"Loaded existing store")
        return vectorstore

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persist_directory)

    print("Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    main()
