# =========================
# ENVIRONMENT SETUP
# =========================
import os
import io
import gc
import base64
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# =========================
# CORE IMPORTS
# =========================
import fitz  # PyMuPDF
import torch
from PIL import Image

# =========================
# FASTAPI
# =========================
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# LANGCHAIN
# =========================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# =========================
# TRANSFORMERS (CLIP)
# =========================
from transformers import CLIPProcessor, CLIPModel

# =========================
# APP INIT
# =========================
app = FastAPI(title="Multimodal RAG (Render Free Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL STATE
# =========================
vector_store = None
image_store = {}

clip_model = None
clip_processor = None

is_indexing = False
lock = threading.Lock()

# =========================
# LOAD CLIP ONCE (CPU SAFE)
# =========================
def load_clip():
    global clip_model, clip_processor

    if clip_model is None:
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to("cpu").eval()

        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_fast=False
        )

        torch.set_grad_enabled(False)

# =========================
# EMBEDDINGS
# =========================
def embed_text(text: str):
    load_clip()
    with torch.no_grad():
        inputs = clip_processor(
            text=text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77
        )
        features = clip_model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()


def embed_image(image: Image.Image):
    load_clip()
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt")
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

# =========================
# LLM
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# =========================
# REQUEST MODEL
# =========================
class QueryRequest(BaseModel):
    question: str

# =========================
# PDF PROCESSOR (SERIALIZED)
# =========================
def process_pdf_sync(file: UploadFile):
    global vector_store, image_store, is_indexing

    with lock:
        is_indexing = True

        pdf = fitz.open(stream=file.file.read(), filetype="pdf")

        docs = []
        embeddings = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80
        )

        for page_num, page in enumerate(pdf):

            text = page.get_text()
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"page": page_num, "type": "text"}
                )
                for chunk in splitter.split_documents([doc]):
                    docs.append(chunk)
                    embeddings.append(embed_text(chunk.page_content))

            for img_index, img in enumerate(page.get_images(full=True)):
                base = pdf.extract_image(img[0])
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")

                image_id = f"page_{page_num}_img_{img_index}"
                image_store[image_id] = base64.b64encode(
                    base["image"]
                ).decode()

                docs.append(
                    Document(
                        page_content=f"[IMAGE on page {page_num}]",
                        metadata={
                            "page": page_num,
                            "type": "image",
                            "image_id": image_id
                        }
                    )
                )

                embeddings.append(embed_image(pil))

        pdf.close()

        vector_store = FAISS.from_embeddings(
            text_embeddings=list(
                zip([d.page_content for d in docs], embeddings)
            ),
            embedding=None,
            metadatas=[d.metadata for d in docs]
        )

        gc.collect()
        torch.cuda.empty_cache()

        is_indexing = False

# =========================
# UPLOAD PDF (SYNC, SAFE)
# =========================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if is_indexing:
        return {"error": "Another PDF is processing. Please wait."}

    process_pdf_sync(file)
    return {"status": "PDF indexed successfully"}

# =========================
# QUERY
# =========================
@app.post("/query")
def query_rag(request: QueryRequest):
    if is_indexing:
        return {"error": "Indexing in progress. Try again shortly."}

    if vector_store is None:
        return {"error": "No document indexed yet"}

    query_embedding = embed_text(request.question)

    results = vector_store.similarity_search_by_vector(
        query_embedding,
        k=4
    )

    context = []
    for doc in results:
        context.append(
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
        )

    prompt = f"""
You are a multimodal assistant.

CONTEXT:
{chr(10).join(context)}

QUESTION:
{request.question}

Answer clearly and accurately.
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# =========================
# STATUS ENDPOINT
# =========================
@app.get("/status")
def status():
    return {
        "is_indexing": is_indexing,
        "indexed": vector_store is not None
    }

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok"}
