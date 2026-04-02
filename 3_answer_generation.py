import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

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

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))


def get_chat_model():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Set GOOGLE_API_KEY or OPENAI_API_KEY in environment")

    os.environ.setdefault("OPENAI_API_KEY", api_key)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash"), temperature=0)
    except Exception:
        pass

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"), temperature=0)


def load_vector_store(persist_directory="db/chroma_db"):
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    print("=== RAG Answer Generation ===")

    db = load_vector_store()

    query = "How much did Microsoft pay to acquire GitHub?"
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}: {doc.page_content}\n")

    context_text = "\n".join(f"- {doc.page_content}" for doc in relevant_docs)
    combined_input = (
        f"Based on the following documents, please answer this question: {query}\n\n"
        f"Documents:\n{context_text}\n\n"
        "Please provide a clear, helpful answer using only the information from these documents. "
        "If you can't find the answer in the documents, say 'I don't have enough information to answer that question based on the provided documents.'"
    )

    model = get_chat_model()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)

    print("--- Generated Response ---")
    print(result.content)


if __name__ == "__main__":
    main()
