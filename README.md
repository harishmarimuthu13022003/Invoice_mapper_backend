# Invoice Mapper Backend

The backend of the Invoice Mapper is a robust **FastAPI-powered** system that handles document processing, database interactions (MongoDB), and our core **Retrieval-Augmented Generation (RAG)** pipeline.

## 🚀 Key Technologies
*   **FastAPI** - High-performance web framework for Python.
*   **MongoDB + Motor** - Asynchronous NoSQL database for flexible data schemas and requests.
*   **Groq API (Llama 3)** - Primary ultra-fast LLM interface.
*   **Google Gemini API** - Fallback intelligent inference layer.
*   **Pinecone** - Serverless vector database.
*   **PyPDF** - Document processing and chunking.

---

## 🛠️ The Intelligent RAG Workflow

Our core feature is an autonomous pipeline resolving chaotic PDF invoices into highly-structured mapping data, without manual intervention:

### 1. Invoice Upload Phase
When a Supplier uploads an invoice (`/invoices/upload`), the backend immediately buffers the file, passes it to the `RAG Engine`, and creates initial structural representations in MongoDB.

### 2. Multi-Stage Extraction & Parsing
*   **Text Extraction:** `pypdf` strips multi-page elements and retains the exact spatial relationships of service tables.
*   **Line Filter Engine:** We clean out heavy boilerplate ("Bill To", "VAT", "BSB", "Dates") and enforce a condition that only rows ending in valid floating dollar amounts (`$XX.XX`) are evaluated.

### 3. Classification & Retrieval (RAG)
1.  **Keyword Detection Fast-Lane:** Basic heuristics map obvious services (e.g. "wound dressing" -> Nursing) to lighten API calls.
2.  **Vector Retrieval:** If unclear, we use **TF-IDF mapping** (backed by Pinecone) against known Service Codes (e.g. `01_111_021_01`) returning the top 5 `similar_codes`.
3.  **LLM Reasoning:** The primary AI receives our dynamic list of active `CATEGORY_SERVICE_CODES` & the `similar_codes`, returning a fully-typed `RAGResponse` (confidence score, suggested category, and specific billing code). 

### 4. Dynamic Discovery (The "Learn" Flow)
*If the LLM determines a service description falls outside all known categories, it:*
1.  Generates `is_new_category = true`.
2.  Triggers a secondary prompt to design a *brand new* NDIS Category Name and a structured Service Code template.
3.  Automatically generates a `CategoryRequest` ticket sent to the Admin portal while flagging the invoice items as `PENDING_APPROVAL`.
4.  Once the Admin clicks **Approve**, the backend:
    *   Adds it to the static Vector Store.
    *   Saves it natively into the `service_codes` MongoDB schema.
    *   *Hot-reloads it immediately into memory* so any sequential uploads use the newly formed category without requiring server restarts.

---

## 🏃 Running the Backend Locally

1. Ensure **Python 3.10+** is installed.
2. Clone, setup an environment, and install `requirements.txt`.
3. Create your `.env` defining:
    * `MONGODB_URI`, `PINECONE_API_KEY`
    * `GROQ_API_KEY`, `GEMINI_API_KEY`
4. Start the server (runs via Uvicorn):
   ```bash
   python main.py
   ```
   *Available on `http://localhost:8003`*
