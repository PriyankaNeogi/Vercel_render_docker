# Copilot Instructions for Multimodal RAG Chatbot

## Architecture Overview

This is a **full-stack multimodal RAG (Retrieval-Augmented Generation) application** with:
- **Backend (Python/FastAPI)**: Processes PDFs (text + images), indexes with FAISS vector store, answers queries using Groq LLM
- **Frontend (Next.js/TypeScript)**: User interface for PDF upload and chat interface
- **Data Flow**: PDF → extract text/images → embed with CLIP → store in FAISS → retrieve top-5 matches → generate response via Groq LLM

## Development Workflow

### Two-Terminal Setup (Critical)
The backend and frontend **must run in separate terminals simultaneously**:
```bash
# Terminal 1: Backend (port 8000)
cd backend
source venv/bin/activate  # macOS/Linux
python -m uvicorn app:app --reload

# Terminal 2: Frontend (port 3000)
cd frontend
npm run dev
```

### Environment Setup
- Backend requires `.env` file in `backend/` folder with `GROQ_API_KEY=your_key_here`
- Frontend API routes proxy requests to hardcoded backend URL: `https://multimodal-rag-backend-g1i2.onrender.com` (see [app/api/](app/api/))
- Local development requires updating backend URL in API routes if running locally

## Key Architecture Patterns

### Backend: Multi-Stage PDF Processing

1. **Upload Endpoint** (`/upload-pdf` in [app.py](backend/app.py#L115)):
   - Accepts PDF files (5MB limit for Render free tier)
   - Extracts text chunks using `RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)`
   - Extracts images as base64, stores in `image_store` dict keyed by `page_{num}_img_{idx}`
   - Embeds all chunks using CLIP (text and image embeddings)
   - Stores in global `vector_store` (FAISS instance)

2. **Embedding Pipeline** ([app.py](backend/app.py#L65-L95)):
   - Text: `embed_text()` uses CLIPModel for semantic representation
   - Images: `embed_image()` uses same CLIP model for visual embeddings
   - Memory optimization: Models loaded per-call, deleted after use, garbage collected
   - Embeddings L2-normalized for cosine similarity

3. **Query Endpoint** (`/query` in [app.py](backend/app.py#L193)):
   - Embeds user question with CLIP
   - Retrieves top-5 semantically similar chunks (text + image metadata)
   - Constructs context prompt with page references and "[IMAGE on page X]" placeholders
   - Invokes Groq LLM (`llama-3.1-8b-instant`) for generation

### Frontend: API Proxy Pattern

Next.js API routes ([app/api/](app/api/)) act as **middleware**:
- `POST /api/upload` → proxies to backend `/upload-pdf`
- `POST /api/chat` → proxies to backend `/query`
- Extracts form data or JSON from client, passes through to backend
- Returns backend response to client (see [chat/route.ts](frontend/app/api/chat/route.ts))

### Global State Management

- `vector_store` (global in [app.py](backend/app.py#L43)): Single FAISS instance shared across requests
- `image_store` (dict): In-memory base64 storage for extracted images
- **Note**: Ephemeral storage; resets on backend restart

## Critical Dependencies

**Backend** (see [requirements.txt](backend/requirements.txt)):
- `langchain` + `langchain-community` + `langchain-groq`: LLM chain orchestration
- `transformers` + `torch`: CLIP model for multimodal embeddings
- `faiss-cpu`: Vector similarity search
- `pymupdf` (fitz): PDF text/image extraction
- `fastapi` + `uvicorn`: REST API server

**Frontend** (see [package.json](frontend/package.json)):
- `next@14.2.5`: Framework
- Minimal dependencies (no UI library); inline React state management

## Common Patterns & Conventions

### Error Handling
- Backend validates file size upfront (5MB limit for Render free tier)
- Returns `{"error": "message"}` on failures; clients must check response
- No explicit error boundaries; uncaught exceptions crash backend

### API Contract
All endpoints expect/return JSON:
- Upload: Multipart form data with `file` field
- Query: `{"question": "string"}` → `{"answer": "string"}` or `{"error": "string"}`

### Resource Management
- CLIP models loaded per-request (not cached) to minimize memory footprint
- `torch.no_grad()` blocks disable gradients for inference
- Garbage collection called after model cleanup

## Modification Patterns

When working on this codebase:

1. **Adding new LLM endpoints**: Follow `/query` pattern—embed query, search FAISS, construct prompt, invoke LLM
2. **Changing CLIP embeddings**: Update both `embed_text()` and `embed_image()` consistently
3. **Modifying chunking strategy**: Edit `RecursiveCharacterTextSplitter` params in `/upload-pdf` (currently 500/100)
4. **Frontend changes**: Update Next.js page in [app/page.tsx](frontend/app/page.tsx); ensure API route proxies match backend endpoints
5. **Backend URL**: Hardcoded in [app/api/chat/route.ts](frontend/app/api/chat/route.ts) and [app/api/upload/route.ts](frontend/app/api/upload/route.ts)—update both for local development

## Testing Approach

- **Manual workflow**: Upload PDF → verify chunks count matches output → query with known information → validate response accuracy
- No automated tests present; focus on integration testing via UI
- Memory constraints on Render free tier require careful PDF size limits

## Deployment Context

- Backend deployed on Render (free tier: 512MB RAM, ephemeral storage)
- Frontend on Vercel
- Hardcoded backend URL in frontend implies production is pre-configured; local dev requires manual URL override
