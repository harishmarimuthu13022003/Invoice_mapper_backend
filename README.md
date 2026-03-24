# Invoice Mapper - Backend

A FastAPI-based backend for AI-powered invoice processing with RAG (Retrieval Augmented Generation) integration.

## Overview

The backend provides RESTful APIs for:
- Invoice upload and processing
- AI-powered service code mapping using RAG
- Multi-role authentication
- Audit logging and monitoring

## Features

- FastAPI REST API
- MongoDB database integration
- ChromaDB/Pinecone vector store for RAG
- LLM integration (OpenAI compatible)
- PDF text extraction
- Role-based access control
- Comprehensive audit logging

## Project Structure

```
backend/
├── main.py                    # FastAPI application entry point
├── models.py                  # Pydantic data models
├── rag_engine.py             # RAG engine for service code mapping
├── database/
│   ├── mongodb_schema.json   # MongoDB schema definitions
│   └── service_codes_seed.json # Initial service codes
├── uploads/                   # Uploaded invoice PDFs
├── run.sh                    # Linux/Mac startup script
└── run.bat                   # Windows startup script
```

## Getting Started

### Prerequisites

- Python 3.9+
- MongoDB (local or Atlas)
- Pinecone API key (for vector store)
- OpenAI API key (for LLM)

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and MongoDB connection
   ```

5. Start the server:
   ```bash
   # Linux/Mac
   ./run.sh
   
   # Windows
   run.bat
   ```

The server will start at `http://localhost:8001`

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | User login |

### Invoices

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/invoices` | GET | Get all invoices (role-based) |
| `/invoices/upload` | POST | Upload and process invoice |
| `/invoices/{id}` | GET | Get specific invoice |
| `/invoices/{id}/confirm` | PUT | Supplier confirmation |
| `/invoices/{id}/finance-approve` | PUT | Finance approval |
| `/invoices/{id}/finance-reject` | PUT | Finance rejection |

### Service Codes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/service-codes` | GET | Get all service codes |
| `/service-codes/seed` | POST | Seed service codes |
| `/rag/seed` | POST | Seed RAG vector store |

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audit-logs` | GET | Get audit logs |
| `/audit-logs/stats` | GET | Get audit statistics |
| `/rag-stats` | GET | Get RAG statistics |

### Category Requests

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/category-requests` | GET | Get category requests |
| `/category-requests/{id}/approve` | PUT | Approve request |
| `/category-requests/{id}/reject` | PUT | Reject request |

## RAG Workflow

The RAG (Retrieval Augmented Generation) system maps invoice line items to service codes using:

### 1. PDF Text Extraction
```
Invoice PDF → PyMuPDF → Raw Text
```

### 2. Line Item Parsing
```
Raw Text → Regex/NLP → Line Items (description, quantity, amount)
```

### 3. Semantic Search (Retrieval)
```
Line Item Description → Embedding Model → Vector
Vector → ChromaDB/Pinecone → Top-K Similar Service Codes
```

### 4. Code Suggestion (Augmented Generation)
```
Top-K Codes + Line Item → LLM (OpenAI/GPT) → Suggested Service Code + Confidence
```

### 5. Feedback Loop
```
User Override → Update Vector Store → Improved Future Suggestions
```

## RAG Architecture

```
┌─────────────────┐
│   Invoice PDF   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extract   │  (PyMuPDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Line Items     │  (Parsing)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │  (text-embedding-ada-002)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  (ChromaDB/Pinecone)
└────────┬────────┘
         │
    ┌────┴────┐
    │ Search  │
    │ Top-K   │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│     LLM         │  (GPT-4/GPT-3.5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Service Code    │
│ + Confidence   │
└─────────────────┘
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=invoice_mapper

# Pinecone (Vector Store)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=service-codes

# OpenAI (LLM)
OPENAI_API_KEY=your_openai_api_key

# App Settings
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## User Roles

| Role | Permissions |
|------|-------------|
| Supplier | Upload invoices, view own invoices |
| Finance | Approve/reject invoices |
| Administrator | Manage service codes, sync RAG |
| Technical Monitor | View logs and metrics |

## Database Collections

### users
```json
{
  "username": "string",
  "password": "hashed_string",
  "role": "Administrator|Supplier|Finance|Technical Monitor",
  "email": "string"
}
```

### invoices
```json
{
  "invoice_id": "string",
  "supplier_id": "string",
  "pdf_path": "string",
  "status": "Pending|Approved",
  "gender": "Male|Female|Unknown",
  "line_items": [
    {
      "description": "string",
      "suggested_code": "string",
      "confidence_score": 0.95,
      "reasoning": "string",
      "final_code": "string",
      "flagged": false
    }
  ],
  "uploaded_at": "datetime"
}
```

### service_codes
```json
{
  "code": "string",
  "description": "string",
  "category": "string",
  "status": "Active|Inactive"
}
```

### audit_logs
```json
{
  "invoice_id": "string",
  "action": "string",
  "user_id": "string",
  "timestamp": "datetime",
  "details": "string",
  "level": "INFO|WARN|ERROR",
  "status": "string"
}
```

### category_requests
```json
{
  "description": "string",
  "category": "string",
  "suggested_code": "string",
  "status": "Pending|Approved|Rejected",
  "requested_at": "datetime"
}
```

## Technology Stack

- **Framework**: FastAPI
- **Database**: MongoDB
- **Vector Store**: ChromaDB / Pinecone
- **LLM**: OpenAI GPT-4
- **PDF Processing**: PyMuPDF
- **Authentication**: JWT

## License

MIT License
