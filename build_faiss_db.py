import os
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from document_loader import load_documents 
from secrets import secret
import numpy as np
from transformers import AutoConfig
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

root = "/N/slate/awalaval"
os.environ['HF_HOME'] = f"{root}/model_cache"

# Authentication
from huggingface_hub import login
from huggingface_hub import hf_hub_download

login(token=secret["huggingface_token"])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

DOCS_PATH = "/N/slate/awalaval/LLM_code/LLM_test/Resume_Chatbot/Chatbot-using-LLM-and-RAG/documents"
INDEX_PATH = "./vector_store/faiss_index.index"
META_PATH = "./vector_store/metadata.json"

def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        return embeddings.cpu().numpy()


chunks = []
metadata = []

docs = load_documents(DOCS_PATH)
for fname, text in docs:
    for i, chunk in enumerate(chunk_text(text)):
        chunks.append(chunk)
        metadata.append({"source": fname, "chunk_id": i})

print(f"Total chunks: {len(chunks)}")


embeddings = []
BATCH_SIZE = 32

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    batch_embeddings = encode_texts(batch)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ FAISS index and metadata saved.")
