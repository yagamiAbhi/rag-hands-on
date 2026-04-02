import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

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
