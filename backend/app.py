# =========================
# ENVIRONMENT SETUP
# =========================
import os
import io
import gc
import base64

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
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
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
app = FastAPI(title="Multimodal RAG (Groq + CLIP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL STATE (IMPORTANT)
# =========================
vector_store = None
image_store = {}

clip_model = None
clip_processor = None

# Render-safe flags
is_indexing = False
is_ready = False

# =========================
# LAZY LOAD CLIP (SAFE)
# =========================
def load_clip():
    global clip_model, clip_processor

    if clip_model is None or clip_processor is None:
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
# EMBEDDING FUNCTIONS
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
# BACKGROUND PDF PROCESSOR
# =========================
def process_pdf(file: UploadFile):
    global vector_store, image_store, is_indexing, is_ready

    is_indexing = True
    is_ready = False

    try:
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")

        docs = []
        embeddings = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        for page_num, page in enumerate(pdf):

            text = page.get_text()
            if text.strip():
                temp_doc = Document(
                    page_content=text,
                    metadata={"page": page_num, "type": "text"}
                )

                for chunk in splitter.split_documents([temp_doc]):
                    docs.append(chunk)
                    embeddings.append(embed_text(chunk.page_content))

            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    base = pdf.extract_image(img[0])
                    image_bytes = base["image"]

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    image_id = f"page_{page_num}_img_{img_index}"

                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")

                    image_store[image_id] = base64.b64encode(
                        buf.getvalue()
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

                    embeddings.append(embed_image(pil_image))

                except Exception as e:
                    print("Image error:", e)

        pdf.close()

        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip([d.page_content for d in docs], embeddings)),
            embedding=None,
            metadatas=[d.metadata for d in docs]
        )

        is_ready = True

    finally:
        is_indexing = False
        gc.collect()
        torch.cuda.empty_cache()

# =========================
# PDF UPLOAD ENDPOINT
# =========================
@app.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    background_tasks.add_task(process_pdf, file)
    return {"status": "PDF upload started. Processing in background."}

# =========================
# STATUS ENDPOINT (CRITICAL)
# =========================
@app.get("/status")
def status():
    return {
        "is_indexing": is_indexing,
        "indexed": is_ready
    }

# =========================
# QUERY ENDPOINT (RENDER SAFE)
# =========================
@app.post("/query")
def query_rag(request: QueryRequest):

    if is_indexing:
        return {
            "error": "PDF is still processing. Please wait and try again."
        }

    if not is_ready or vector_store is None:
        return {
            "error": "No document indexed yet."
        }

    query_embedding = embed_text(request.question)

    results = vector_store.similarity_search_by_vector(
        query_embedding,
        k=5
    )

    text_context = []
    image_context = []

    for doc in results:
        if doc.metadata["type"] == "text":
            text_context.append(
                f"[Page {doc.metadata['page']}]: {doc.page_content}"
            )
        else:
            image_context.append(
                f"Image detected on page {doc.metadata['page']}"
            )

    prompt = f"""
You are a multimodal assistant.

TEXT CONTEXT:
{chr(10).join(text_context)}

IMAGE CONTEXT:
{chr(10).join(image_context)}

QUESTION:
{request.question}

Answer clearly and accurately.
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok"}
