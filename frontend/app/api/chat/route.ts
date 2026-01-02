import { NextResponse } from "next/server";

const BACKEND_URL =
  "https://multimodal-rag-backend-g1i2.onrender.com";

export async function POST(req: Request) {
  const body = await req.json();

  const res = await fetch(`${BACKEND_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: body.question
    })
  });

  const data = await res.json();
  return NextResponse.json(data);
}
