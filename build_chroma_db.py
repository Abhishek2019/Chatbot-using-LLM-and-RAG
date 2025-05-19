import os
import json
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from document_loader import load_documents

from paths import root, DOCS_PATH, CHROMA_DIR, META_PATH, HF_HOME

# Set paths
# root = "/N/slate/awalaval"
# DOCS_PATH = f"{root}/LLM_code/LLM_test/Resume_Chatbot/Chatbot-using-LLM-and-RAG/documents"
# CHROMA_DIR = "./vector_store/chroma"
# META_PATH = "./vector_store/metadata.json"

# Set HF cache
# os.environ['HF_HOME'] = f"{root}/model_cache"
os.environ['HF_HOME'] = HF_HOME

# Load SentenceTransformer and wrap for LangChain
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = SentenceTransformerEmbeddings(model_name=model_name)

# Helper: Split text into chunks
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Load and chunk documents
chunks = []
metadata = []

docs = load_documents(DOCS_PATH)
for fname, text in docs:
    for i, chunk in enumerate(chunk_text(text)):
        chunks.append(chunk)
        metadata.append({"source": fname, "chunk_id": i})

print(f"Total chunks: {len(chunks)}")

# Wrap text and metadata into LangChain Document format
documents = [
    Document(page_content=chunk, metadata=meta)
    for chunk, meta in zip(chunks, metadata)
]

# Create or load Chroma vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_fn,
    persist_directory=CHROMA_DIR
)

# Save metadata separately if needed
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Chroma vector index and metadata saved.")
