

---

# Multimodal RAG Chatbot (Text + Image + PDF)

This repository contains a **Multimodal Retrieval-Augmented Generation (RAG) Chatbot** that can process **PDF documents containing both text and images**, index them, and answer user queries using **Groq LLM**, **CLIP-based embeddings**, and **FAISS vector search**.

---

## Important Tip (Read This First)

**The backend and frontend must be running in two separate terminals.**

* The **backend (FastAPI)** runs on port `8000`
* The **frontend (Vercel / Next.js)** runs on port `3000`

Both servers must be running **at the same time** for the application to work correctly.

---

## Tech Stack

### Backend

* Python 3.10+
* FastAPI
* LangChain
* Groq API
* CLIP (Transformers)
* FAISS
* PyMuPDF

### Frontend

* Next.js
* Vercel AI SDK

---

## Project Structure

```
multimodal-rag-chatbot/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env.example
│
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   ├── package.json
│   └── next.config.js
│
└── README.md
```

---

## Environment Variables

Create a `.env` file inside the `backend` folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Backend Setup (Terminal 1)



## Important Tip: Activate Virtual Environment (Local Development) - 


Before running the backend on your local system, always activate the virtual environment.
If the virtual environment is not activated, Python may not find the required packages and the backend will fail to start.
 

```
Max Os / Linux -
cd backend
source venv/bin/activate

Windows (PowerShell) -
cd backend
venv\Scripts\activate
```

After activation, your terminal will show something like:
(venv)
This confirms that the correct Python environment is active.


---
### Step 1: Open Terminal 1 and go to backend

```bash
cd backend
```

### Step 2: Create and activate virtual environment

```bash
python3 -m venv 
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start backend server

```bash
uvicorn app:app --reload --port 8000
```

Backend URL:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```
<img width="1438" height="815" alt="Screenshot 2025-12-22 at 2 09 50 PM" src="https://github.com/user-attachments/assets/fac6868f-1bca-4729-8958-b564d23416d2" />

<img width="1438" height="815" alt="Screenshot 2025-12-22 at 2 10 02 PM" src="https://github.com/user-attachments/assets/9e016313-72cf-42a7-90d4-67f5258dded5" />

<img width="1438" height="815" alt="Screenshot 2025-12-22 at 2 10 08 PM" src="https://github.com/user-attachments/assets/637eb55f-c6f0-45dd-866c-a144492e5c03" />

<img width="1438" height="815" alt="Screenshot 2025-12-22 at 2 10 14 PM" src="https://github.com/user-attachments/assets/57528591-ba42-4a08-a407-c59f7510f694" />


---

## Frontend Setup (Terminal 2)

### Step 1: Open a **new terminal window**

Do **not** stop the backend server.

### Step 2: Navigate to frontend

```bash
cd frontend
```

### Step 3: Install dependencies

```bash
npm install
```

### Step 4: Start frontend server

```bash
npm run dev
```

Frontend URL:

```
http://localhost:3000
```

<img width="1438" height="815" alt="Screenshot 2025-12-22 at 2 10 19 PM" src="https://github.com/user-attachments/assets/cb3dd2e2-ad48-4674-b55c-648bef00c645" />

---

## How the Two Terminals Work Together

* **Terminal 1**
  Runs FastAPI backend
  Handles PDF upload, embedding, retrieval, and LLM responses

* **Terminal 2**
  Runs Next.js frontend
  Sends requests to backend APIs and displays responses

If **either terminal is stopped**, the application will not function.

---

## API Example

### Query Endpoint

```json
POST /query
{
  "question": "What do images indicate?"
}
```

Response:

```json
{
  "answer": "Based on the provided text and image context..."
}
```


---

## Development Workflow Summary - 

* Backend (FastAPI + Vercel AI SDK patterns)
* Frontend (Next.js + Vercel AI SDK)
* Both run locally in separate terminals during development

---

## Author

Developed by **Priyanka Neogi**

---

