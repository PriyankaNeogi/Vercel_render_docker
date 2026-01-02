# =========================
# ENVIRONMENT SETUP
# =========================
import os
import io
import gc
import base64
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL STATE
# =========================
vector_store = None
image_store = {}
is_indexing = False

clip_model = None
clip_processor = None
clip_lock = threading.Lock()

# =========================
# LAZY LOAD CLIP (SAFE)
# =========================
def load_clip():
    global clip_model, clip_processor

    with clip_lock:
        if clip_model is None:
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            clip_model.eval()

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
            padding=True,
            truncation=True,
            max_length=77
        )
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().cpu().numpy()

def embed_image(image: Image.Image):
    load_clip()
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt")
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().cpu().numpy()

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
# PDF PROCESSOR (THREAD)
# =========================
def process_pdf_sync(file_bytes: bytes):
    global vector_store, image_store, is_indexing

    is_indexing = True
    image_store.clear()

    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    docs = []
    embeddings = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    #  HARD LIMIT FOR FREE TIER
    MAX_PAGES = 3

    for page_num, page in enumerate(pdf):
        if page_num >= MAX_PAGES:
            break

        text = page.get_text()
        if text.strip():
            base_doc = Document(
                page_content=text,
                metadata={"page": page_num, "type": "text"}
            )
            for chunk in splitter.split_documents([base_doc]):
                docs.append(chunk)
                embeddings.append(embed_text(chunk.page_content))

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                base = pdf.extract_image(img[0])
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")

                img_id = f"p{page_num}_i{img_index}"
                buf = io.BytesIO()
                pil.save(buf, format="PNG")

                image_store[img_id] = base64.b64encode(buf.getvalue()).decode()

                docs.append(
                    Document(
                        page_content="[IMAGE]",
                        metadata={
                            "page": page_num,
                            "type": "image",
                            "image_id": img_id
                        }
                    )
                )

                embeddings.append(embed_image(pil))
            except:
                pass

    pdf.close()

    vector_store = FAISS.from_embeddings(
        list(zip([d.page_content for d in docs], embeddings)),
        embedding=None,
        metadatas=[d.metadata for d in docs]
    )

    gc.collect()
    is_indexing = False

# =========================
# UPLOAD ENDPOINT
# =========================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global is_indexing

    if is_indexing:
        return {"error": "Indexing already in progress"}

    data = file.file.read()

    thread = threading.Thread(
        target=process_pdf_sync,
        args=(data,)
    )
    thread.start()

    return {"status": "PDF upload started. Processing in background."}

# =========================
# QUERY ENDPOINT
# =========================
@app.post("/query")
def query_rag(req: QueryRequest):
    if is_indexing:
        return {"error": "PDF is still processing. Please wait 30 seconds."}

    if vector_store is None:
        return {"error": "No document indexed yet"}

    query_vec = embed_text(req.question)

    results = vector_store.similarity_search_by_vector(query_vec, k=3)

    text_ctx = []
    img_ctx = []

    for d in results:
        if d.metadata["type"] == "text":
            text_ctx.append(f"[Page {d.metadata['page']}] {d.page_content}")
        else:
            img_ctx.append(f"Image on page {d.metadata['page']}")

    prompt = f"""
You are a multimodal assistant.

TEXT CONTEXT:
{chr(10).join(text_ctx)}

IMAGE CONTEXT:
{chr(10).join(img_ctx)}

QUESTION:
{req.question}

Answer clearly.
"""

    return {"answer": llm.invoke(prompt).content}

# =========================
# STATUS (IMPORTANT)
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
