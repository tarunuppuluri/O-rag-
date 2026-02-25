# ⚡ O(RAG): Multi-Document Hybrid Search AI Tutor

> A production-grade Retrieval-Augmented Generation (RAG) system engineered to process dense Data Structures, Algorithms, and Mathematics textbooks. 



## 🚀 The Engineering Problem & Solution
Standard "Chat with PDF" wrappers rely solely on Semantic Vector Search (Cosine Similarity). While excellent for general concepts, vector search frequently fails at retrieving **exact keyword matches**, highly specific variable names (e.g., `sys.argv`), or mathematical notations (e.g., $O(n \log n)$).

**O(RAG) solves this via a Hybrid Search Pipeline.** By running a mathematical fusion of **Dense Vector Embeddings** and **Sparse Keyword Indexing (BM25)**, the system achieves near-perfect retrieval accuracy for complex computer science terminology, dramatically reducing LLM hallucination.

---

## 🛠️ Technology Stack

| Component | Technology Used | Engineering Purpose |
| :--- | :--- | :--- |
| **LLM Backend** | Google Gemini 1.5 Flash | High-speed inference with streaming support |
| **Vector Engine** | `sentence-transformers` / NumPy | Dense semantic retrieval (understanding "vibes") |
| **Keyword Engine** | `rank_bm25` (Okapi) | Sparse retrieval for exact code/variable matches |
| **Data Ingestion** | PyMuPDF (`fitz`) | Layout-aware text extraction across multiple PDFs |
| **Frontend UI** | Streamlit | Real-time chat and document management |

---

## 🧠 System Architecture & Core Features

### 1. Hybrid Score Normalization (The Math)
To fuse the scores from both search engines, O(RAG) normalizes the unbounded BM25 scores against the $0.0 - 1.0$ constraints of cosine similarity using **Min-Max Scaling**. The system then calculates a weighted hybrid score:

$Final\_Score = (\alpha \times Vector\_Score) + ((1 - \alpha) \times BM25\_Score)$

*(System defaults to $\alpha = 0.5$ for an even 50/50 fusion of meaning and keyword matching).*

### 2. Multi-Document Knowledge Base
* Ingests multiple PDFs simultaneously into a unified vector space.
* Attaches strict `source_name` and `page_number` metadata to every text chunk.
* Enables cross-document reasoning (e.g., "Compare the definition of QuickSort in Textbook A vs. Lecture Notes B").

### 3. Asynchronous Token Streaming
Reduces perceived latency and Time-To-First-Byte (TTFB) to near-zero. Instead of blocking the main thread while the LLM generates a response, O(RAG) utilizes Python `yield` generators to stream tokens to the Streamlit UI in real-time.

### 4. Zero-Hallucination Citation Engine
Forces the LLM to ground its answers strictly in the provided text. The UI enforces transparency by displaying exact **Source Documents** and **Page Numbers**, complete with expandable text excerpts to verify the AI's logic.

### 5. Automated Study Guide Generator ("Professor Mode")
Bypasses the strict anti-hallucination prompt to act as an expert tutor. It analyzes the entire user chat history in session memory, filters out unanswered questions, and synthesizes successfully explained concepts into a downloadable `.md` or `.txt` study guide.

---

## 📂 Project Structure

```text
O-rag-/
├── app.py                 # Streamlit UI, session state, and core app loop
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API Keys - Git Ignored)
├── .gitignore             # Security and cache rules
└── src/
    ├── ingestion.py       # PyMuPDF processing, text cleaning, and chunking
    ├── retrieval.py       # Hybrid Engine (FAISS-style Vector + BM25 Fusion)
    └── generation.py      # Gemini API integration, streaming, and prompting
```

---

## ⚙️ Local Installation & Setup

### Prerequisites
* Python 3.9 or higher
* A valid Google AI Studio API Key

### Step-by-Step Guide

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/O-rag-.git](https://github.com/YOUR_GITHUB_USERNAME/O-rag-.git)
cd O-rag-
```

**2. Create a Virtual Environment (Recommended)**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and securely add your API key:
```bash
GOOGLE_API_KEY="your_gemini_api_key_here"
```

**5. Run the Application**
```bash
streamlit run app.py
```

---

## 🔮 Future Roadmap
- [ ] **GraphRAG Integration:** Map structural relationships between linked algorithmic concepts (e.g., Trees -> Graphs -> BFS).
- [ ] **Persistent Vector Storage:** Migrate from in-memory NumPy arrays to ChromaDB or Pinecone to save knowledge bases between sessions.
- [ ] **OCR Capabilities:** Integrate `pytesseract` to extract mathematical formulas from scanned, image-heavy lecture slides.

---
*Architected and engineered by Tarun.*