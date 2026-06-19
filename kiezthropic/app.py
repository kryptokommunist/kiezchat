"""Kiezthropic — RAG chat app over Kiez Burn wiki, backed by SAP AI Core."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import requests as req_lib
from flask import Flask, Response, render_template, request, stream_with_context
from openai import OpenAI

import rag

app = Flask(__name__)

# ---------------------------------------------------------------------------
# AI Core credentials — from VCAP_SERVICES (CF) or env vars (local)
# ---------------------------------------------------------------------------

def _load_aicore_credentials() -> dict:
    vcap = os.environ.get("VCAP_SERVICES")
    if vcap:
        services = json.loads(vcap)
        creds = services["aicore"][0]["credentials"]
        return {
            "client_id": creds["clientid"],
            "client_secret": creds["clientsecret"],
            "auth_url": creds["url"],
            "api_url": creds["serviceurls"]["AI_API_URL"],
        }
    return {
        "client_id": os.environ["AICORE_CLIENT_ID"],
        "client_secret": os.environ["AICORE_CLIENT_SECRET"],
        "auth_url": os.environ["AICORE_AUTH_URL"],
        "api_url": os.environ["AICORE_API_URL"],
    }


AICORE_CREDS = _load_aicore_credentials()
DEPLOYMENT_ID = os.environ.get("AICORE_DEPLOYMENT_ID", "d34c832f51430c83")
RESOURCE_GROUP = os.environ.get("AICORE_RESOURCE_GROUP", "default")
MODEL = os.environ.get("AICORE_MODEL", "gpt-4o")

_token_cache: dict = {"token": "", "expires_at": 0.0}
_token_lock = threading.Lock()


def get_token() -> str:
    with _token_lock:
        if time.time() < _token_cache["expires_at"] - 60:
            return _token_cache["token"]
        resp = req_lib.post(
            f"{AICORE_CREDS['auth_url']}/oauth/token",
            data={"grant_type": "client_credentials"},
            auth=(AICORE_CREDS["client_id"], AICORE_CREDS["client_secret"]),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
        return _token_cache["token"]


def get_openai_client() -> OpenAI:
    base_url = f"{AICORE_CREDS['api_url']}/v2/inference/deployments/{DEPLOYMENT_ID}/v1/"
    return OpenAI(
        base_url=base_url,
        api_key=get_token(),
        default_headers={"AI-Resource-Group": RESOURCE_GROUP},
    )


# ---------------------------------------------------------------------------
# RAG index — loaded once at startup
# ---------------------------------------------------------------------------

WIKI_DIR = str(Path(__file__).parent)

print("Loading pre-built FAISS index…")
rag.load_prebuilt(WIKI_DIR)
print("Index ready.")

# ---------------------------------------------------------------------------
# Query routing — detect listing/aggregation queries that need more context
# ---------------------------------------------------------------------------

_LIST_PATTERNS = re.compile(
    r"\ball\b.*\b(camp|kiez|install|dream|art|stage|space|bar|pub|spa|workshop|temple)\b"
    r"|\b(list|show|give me|what are|enumerate|which|tell me).{0,30}\b(camp|kiez|install|dream|art|space)\b"
    r"|\b(camp|kiez|install|dream|art|space)s\b.{0,30}\b(list|all|every|complete|there|exist|at kiez)\b"
    r"|\bwhat\b.{0,20}\b(camps|installations|kieze)\b.{0,30}\bat\b",
    re.IGNORECASE,
)

LIST_TOP_K = 30
DEFAULT_TOP_K = 6


def is_listing_query(query: str) -> bool:
    return bool(_LIST_PATTERNS.search(query))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Kiezthropic, a helpful assistant for Kiez Burn — a Burning Man-inspired community event near Berlin.
Answer questions using the wiki context provided. Be friendly and concise.
When asked to list all camps or installations, provide a complete list from the context — do not truncate or summarize.
If the context doesn't cover the question, say so honestly."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return {"error": "empty message"}, 400

    if is_listing_query(user_message):
        camps_chunks = rag.retrieve_by_source("camps_list")
        faiss_chunks = rag.retrieve(user_message, top_k=LIST_TOP_K)
        seen_texts = {c["text"] for c in camps_chunks}
        extra = [c for c in faiss_chunks if c["text"] not in seen_texts]
        chunks = camps_chunks + extra
    else:
        chunks = rag.retrieve(user_message, top_k=DEFAULT_TOP_K)

    context_text = "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in chunks
    )
    max_tokens = 4096 if is_listing_query(user_message) else 2048

    def generate():
        client = get_openai_client()
        stream = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Wiki context:\n{context_text}\n\nQuestion: {user_message}",
                },
            ],
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield f"data: {json.dumps({'text': delta.content})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
