import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from token_secrets import secret
from paths import CHROMA_DIR, HF_HOME

os.environ['HF_HOME'] = HF_HOME

# Authentication
from huggingface_hub import login
login(token=secret["huggingface_token"])

# Load embeddings and DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# Load LLM
llm_model = "microsoft/phi-1_5"  # or any HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(llm_model, trust_remote_code=True, device_map=None)

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# # Chat loop
# while True:
#     query = input("Ask a question: ")
#     if query.lower() in ["exit", "quit"]:
#         break
#     answer = qa.invoke({"query": query})
#     print("Answer:", answer)


def is_retrieval_useful(query, retriever, threshold=0.7):
    """Check if retrieval returns anything semantically meaningful"""
    results = retriever.get_relevant_documents(query)
    return len(results) > 0

# Chat loop
while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break

    # Check if there's anything to retrieve
    if is_retrieval_useful(query, vectordb.as_retriever()):
        answer = qa.invoke({"query": query})
    else:
        # Use raw LLM for open-ended prompts like "Hi"
        answer = llm_pipeline(query)[0]["generated_text"]

    print("Answer:", answer)