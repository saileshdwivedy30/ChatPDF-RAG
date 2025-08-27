# 📄 Chat with Your PDFs (RAG with LangChain on AWS Bedrock)


**Chat with your PDFs using a Retrieval-Augmented Generation (RAG) pipeline built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Amazon Bedrock](https://aws.amazon.com/bedrock/).**  
This project demonstrates how to combine embeddings, vector search, and LLM inference into a clean end-to-end application.  

Upload PDFs → index them on the fly with FAISS → query with Claude 3 Haiku → get short, grounded answers with citations.

---

## 🚀 Features
- **PDF ingestion (in-memory):** Upload multiple PDFs, extract text per page, no disk writes.  
- **Deduplication:** Stable `doc_id` prevents duplicate indexing across sessions.  
- **Chunking:** Smart splitting using LangChain’s RecursiveCharacterTextSplitter.  
- **Embeddings:** [Amazon Titan Embeddings v2](https://aws.amazon.com/bedrock/) via Bedrock.  
- **Vector Store:** [FAISS](https://faiss.ai/) for fast similarity search (kept in session).  
- **LLM:** [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-family) via Bedrock (fast + low-latency).  
- **Custom QA chain:**  
  - Short, citation-first prompt.  
  - Exponential backoff for throttling.  
  - Per-file support scoring (token overlap + fuzzy match).  
  - Shows only **strongest supporting sources** (≤2 per answer).  
- **UI:** Streamlit interface — upload, index, and query in one page.  

---

## 🛠️ Tech Stack
- **Framework:** [LangChain](https://www.langchain.com/) – orchestration of ingestion, chunking, retriever + LLM chain.  
- **UI:** [Streamlit](https://streamlit.io/) – lightweight frontend for PDF upload + Q&A.  
- **Embeddings:** [Amazon Titan Embeddings v2](https://aws.amazon.com/bedrock/) via `langchain-aws`.  
- **LLM:** [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-family) via Bedrock.  
- **Vector Store:** [FAISS](https://faiss.ai/) – similarity search over chunk embeddings.  
- **Infra / Cloud:** [Amazon Bedrock](https://aws.amazon.com/bedrock/) – unified API for embeddings + LLMs.  
- **Parsing:** [pypdf](https://pypi.org/project/pypdf/) – PDF text extraction.  

---

## 🏗️ Architecture
```mermaid
flowchart TD
    A[Upload PDFs] --> B[Extract text per page]
    B --> C[Chunking (2000/200)]
    C --> D[Embeddings (Titan v2 on Bedrock)]
    D --> E[FAISS Vector Store in session_state]
    E -->|Similarity Search (k=5)| F[Retriever]
    F --> G[Claude 3 Haiku (via Bedrock)]
    G --> H[Answer + Sources]
```

---

## 📂 Project Structure
```
app.py        # Streamlit app with full RAG pipeline
```

---

## Quickstart

### Prerequisites
- Python 3.9+
- AWS account with **Bedrock access enabled** (region: us-east-1)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

`requirements.txt` (minimal):
```txt
streamlit
boto3
pypdf
langchain
langchain-community
langchain-aws
```

### Run locally
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## 🎯 Usage
1. Upload one or more PDFs.  
2. Click **“Add uploads to index”** to build/append to the FAISS index.  
3. Type your question.  
4. Click **“Ask Claude”** → get a short answer + cited sources.  

---

## 🧩 Key Design Choices
- **Session-only:** No data persists after browser refresh.  
- **Deduplication:** Uses SHA-1 hash per page -> avoids duplicate embeddings if file re-uploaded.  
- **Retry logic:** Exponential backoff for Bedrock API throttling.  
- **Source filtering:** Keeps answers grounded by citing only documents that strongly support the response.  
