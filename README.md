# 🛒 E-Commerce FAQ Chatbot — RAG Model with LLaMA

A **Retrieval-Augmented Generation (RAG)** powered FAQ chatbot for e-commerce customer support. It uses a local FAISS vector database for semantic search over a curated FAQ dataset and queries the **Meta LLaMA 3.1** model via the NVIDIA API to generate accurate, context-aware answers.

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  FAISS Vector DB (Retriever)│  ← Semantic search over FAQ embeddings
└─────────────────────────────┘
    │  Top-K relevant FAQ chunks
    ▼
┌─────────────────────────────┐
│   Prompt Builder            │  ← Combines context + user query
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  LLaMA 3.1 8B (NVIDIA API) │  ← Generates final answer
└─────────────────────────────┘
    │
    ▼
  Answer displayed to user
```

---

## 🗂️ Project Structure

```
FAQ_RAG_MODEL/
│
├── ecommerce_llama_rag.py           # Main chatbot script
├── Ecommerce_FAQ_Chatbot_dataset.json  # FAQ dataset (Q&A pairs)
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (not pushed to GitHub)
└── .gitignore
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Meta LLaMA 3.1 8B Instruct (via NVIDIA API) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Framework** | LangChain |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Adityachore/FAQ_RAG_MODEL.git
cd FAQ_RAG_MODEL
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
LLAMA_API_KEY=your_nvidia_api_key_here
LLAMA_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
```

> 🔑 Get your free API key at [build.nvidia.com](https://build.nvidia.com)

### 5. Run the Chatbot

```bash
python ecommerce_llama_rag.py
```

---

## 💬 Example Usage

```
🛒 Ecommerce FAQ Chatbot Ready! Type 'exit' to quit.

You: What is your return policy?

Bot: You can return any item within 30 days of purchase for a full refund,
     provided it is in its original condition and packaging...

You: How do I track my order?

Bot: You can track your order by visiting the 'My Orders' section in your
     account and clicking on the tracking link next to your order...

You: exit
```

---

## 🔧 Configuration

You can tune the following parameters in `ecommerce_llama_rag.py`:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Size of each text chunk for splitting |
| `chunk_overlap` | `50` | Overlap between consecutive chunks |
| `temperature` | `0.2` | LLM creativity (lower = more factual) |
| `max_tokens` | `1024` | Maximum response length |
| `top_p` | `0.7` | Nucleus sampling parameter |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Aditya Chore** — [GitHub](https://github.com/Adityachore)
