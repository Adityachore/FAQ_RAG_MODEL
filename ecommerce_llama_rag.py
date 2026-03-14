import json
import os
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# 🔐 LOAD ENV VARIABLES
# =========================

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL")
# Use a specific meta model compatible with the NVIDIA API.
MODEL_NAME = "meta/llama-3.1-8b-instruct"

# =========================
# 📂 STEP 1 — Load JSON Dataset
# =========================

with open("Ecommerce_FAQ_Chatbot_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data["questions"]:
    text = f"Q: {item['question']}\nA: {item['answer']}"
    documents.append(Document(page_content=text))

# =========================
# ✂️ STEP 2 — Chunk Text
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# =========================
# 🔢 STEP 3 — Create Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# 🗄️ STEP 4 — Store in FAISS
# =========================

vector_db = FAISS.from_documents(docs, embeddings)

retriever = vector_db.as_retriever()

# =========================
# 🤖 FUNCTION: CALL LLaMA API
# =========================

def ask_llama(prompt):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    response = requests.post(
        LLAMA_API_URL,
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# =========================
# 💬 STEP 5 — CHAT LOOP
# =========================

print("\n🛒 Ecommerce FAQ Chatbot Ready! Type 'exit' to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    # 🔎 Retrieve relevant FAQ entries
    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are an ecommerce customer support assistant.
Answer the question using ONLY the information below.

Context:
{context}

Question: {query}

Answer:
"""

    answer = ask_llama(prompt)

    print("\nBot:", answer, "\n")