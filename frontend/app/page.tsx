"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [loadingAsk, setLoadingAsk] = useState(false);
  const [status, setStatus] = useState("");

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

  // -------------------------
  // PDF UPLOAD
  // -------------------------
  const uploadPDF = async () => {
    if (!file) {
      alert("Please select a PDF first");
      return;
    }

    setLoadingUpload(true);
    setStatus("Uploading PDF and indexing…");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BACKEND_URL}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setStatus(data.status || "PDF uploaded. Indexing in background.");
    } catch (err) {
      console.error(err);
      setStatus("Upload failed");
    } finally {
      setLoadingUpload(false);
    }
  };

  // -------------------------
  // ASK QUESTION
  // -------------------------
  const askQuestion = async () => {
    if (!question.trim()) {
      alert("Please enter a question");
      return;
    }

    setLoadingAsk(true);
    setAnswer("");

    try {
      const res = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setAnswer(data.answer || "No answer received");
    } catch (err) {
      console.error(err);
      setAnswer("Error while asking question");
    } finally {
      setLoadingAsk(false);
    }
  };

  return (
    <main style={{ padding: "40px", fontFamily: "Arial, sans-serif" }}>
      <h1>Multimodal RAG Chat</h1>

      {/* PDF Upload */}
      <div style={{ marginBottom: "20px" }}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <br />
        <button
          onClick={uploadPDF}
          disabled={loadingUpload}
          style={{ marginTop: "10px" }}
        >
          {loadingUpload ? "Uploading…" : "Upload PDF"}
        </button>
        <p>{status}</p>
      </div>

      <hr />

      {/* Ask Question */}
      <div style={{ marginTop: "20px" }}>
        <textarea
          rows={4}
          cols={80}
          placeholder="Ask a question about the document…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <br />
        <button
          onClick={askQuestion}
          disabled={loadingAsk}
          style={{ marginTop: "10px" }}
        >
          {loadingAsk ? "Thinking…" : "Ask"}
        </button>
      </div>

      {/* Answer */}
      {answer && (
        <div style={{ marginTop: "30px" }}>
          <h3>Answer</h3>
          <p>{answer}</p>
        </div>
      )}
    </main>
  );
}
