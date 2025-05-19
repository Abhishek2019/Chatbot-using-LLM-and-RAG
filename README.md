
# Q-A Chatbot using LLM and RAG

This is a Retrieval-Augmented Generation (RAG) based chatbot built entirely with open-source tools. It allows users to query documents such as resumes, reports, and internal documents using natural language. The system retrieves relevant document chunks from a vector store and generates contextual answers using a local language model.

## Features

- **Document-based Q&A**: Ask questions and get answers grounded in your private documents.
- **Vector Search**: Uses ChromaDB to store and retrieve document embeddings.
- **LLM Integration**: Uses Hugging Face open-source models (e.g., Phi-1.5, Flan-T5) for generating answers.
- **No Paid Services**: 100% open-source, no proprietary APIs or cloud costs.
- **Developed from scratch**: Fully built, trained, and deployed locally.

## Tech Stack

- **LangChain** - Retrieval + chaining logic
- **ChromaDB** - Local vector database for embedding storage and search
- **Hugging Face Transformers** - Tokenizer and language model inference
- **SentenceTransformers** - Document embedding via `all-MiniLM-L6-v2`

## Folder Structure

```
Q-A-Chatbot-using-LLM-and-RAG/
├── chatbot.py            # User interaction script
├── ingest.py             # Parses and stores document embeddings in ChromaDB
├── token_secrets.py      # HuggingFace token (optional for gated models)
├── paths.py              # Contains CHROMA_DIR, HF_HOME paths
├── data/                 # Folder with documents to parse (PDF, DOCX, TXT)
├── chroma_db/            # Persisted vector store
└── README.md             # Project documentation
```

## How It Works

1. **Ingest Documents**: Uses `ingest.py` to load, split, and embed documents into ChromaDB.
2. **Start Chatbot**: Launch `chatbot.py` which loads the vector store and LLM pipeline.
3. **Ask Questions**: The chatbot retrieves top relevant chunks and uses the LLM to answer.
4. **Fallbacks**: If no relevant docs found, a small LLM like Phi-1.5 handles open-ended questions.

## Models Used

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLMs (select one):
  - `microsoft/phi-1_5`

All models are run locally — **no external API or cloud service used**.

## Setup Instructions

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run document ingestion:

```bash
python build_chroma_db.py
```

3. Start the chatbot:

```bash
python chatbot.py
```

## Notes

- Optimized for CPU or small GPU setups.
- Add PDF, TXT, DOCX files inside the `data/` directory before running `ingest.py`.
- No internet or Hugging Face login required unless you use gated models.

---

Built with using open-source LLMs and vector databases. Perfect for private, local document-based Q&A.

