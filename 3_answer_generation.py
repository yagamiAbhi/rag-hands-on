import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings


class GoogleAIStudioEmbeddings(Embeddings):
    def __init__(self, model="gemini-embedding-001", output_dimensionality=768):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY in your environment (or .env)")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality

    def _normalize(self, vec):
        norm = sum(x * x for x in vec) ** 0.5
        return vec if norm == 0 else [x / norm for x in vec]

    def embed_documents(self, texts):
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [self._normalize(e.values) for e in result.embeddings]

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return self._normalize(result.embeddings[0].values)


def get_embedding_model():
    return GoogleAIStudioEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
        output_dimensionality=int(os.getenv("GOOGLE_EMBEDDING_DIM", "768")),
    )


def get_chat_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in environment")

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GOOGLE_CHAT_MODEL", "models/gemini-flash-latest")

    class GoogleChat:
        def __init__(self, client, model):
            self.client = client
            self.model = model

        def _extract_text(self, response):
            if not response.candidates:
                return ""
            candidate = response.candidates[0]
            if candidate.content is None:
                return ""
            if candidate.content.parts:
                return "".join((part.text or "") for part in candidate.content.parts if getattr(part, "text", None))
            return ""
        def invoke(self, messages):
            prompt = "\n".join([m.content for m in messages if hasattr(m, "content")])
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return type("x", (), {"content": self._extract_text(response)})()

    return GoogleChat(client, model_name)


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
