# rag-hands-on
A repo to learn and practice RAG

## Quickstart (Google AI Studio)

1. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`:
   - `GOOGLE_API_KEY=your_google_ai_studio_key_here`
2. Install dependencies (same as rag-for-beginners):
   - `pip install -r ../rag-for-beginners/requirements.txt`
3. Place your source docs in `docs/` (plain text `.txt`).
4. Run ingestion:
   - `python 1_ingestion_pipeline.py`
5. Test retrieval:
   - `python 2_retrieval_pipeline.py`
6. Generate answer:
   - `python 3_answer_generation.py`

## Notes
- The code keeps same architecture as `rag-for-beginners` modules 1-3.
- It attempts Google Palm classes first, then falls back to OpenAI wrappers for compatibility.

