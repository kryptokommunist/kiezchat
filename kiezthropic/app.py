"""Kiezthropic — RAG chat app over Kiez Burn wiki, backed by SAP AI Core."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests as req_lib
from flask import Flask, Response, render_template, request, stream_with_context
from openai import BadRequestError
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI

import rag

app = Flask(__name__)
# Trust one proxy hop (Cloudflare or CF router)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

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
# Storage — SQLite
# On CF: corrections persist between restarts (ephemeral disk, reset on redeploy).
# Approved corrections committed to corrections_seed.json are re-seeded on deploy.
# Request stats are best-effort (survive restarts, reset on redeploy).
# ---------------------------------------------------------------------------

WIKI_DIR = Path(__file__).parent
DB_PATH = WIKI_DIR / "kiezthropic.db"
SEED_PATH = WIKI_DIR / "corrections_seed.json"
ADMIN_PASSWORD_FILE = WIKI_DIR / "admin_password.txt"

_COST_PER_1M_INPUT = 2.50   # GPT-4o input $/1M tokens
_COST_PER_1M_OUTPUT = 10.00  # GPT-4o output $/1M tokens

_db_lock = threading.Lock()


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db_lock:
        conn = _db_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                bad_answer TEXT,
                correction TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT,
                question TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS seed_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        conn.commit()

        if SEED_PATH.exists():
            seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
            seed_version = str(seed.get("version", "1"))
            row = conn.execute("SELECT value FROM seed_meta WHERE key='version'").fetchone()
            if not row or row["value"] != seed_version:
                conn.execute("DELETE FROM corrections WHERE status='seeded'")
                for entry in seed.get("corrections", []):
                    if not entry.get("correction"):
                        continue
                    conn.execute(
                        "INSERT INTO corrections (question,bad_answer,correction,status,created_at) VALUES (?,?,?,'seeded',?)",
                        (entry.get("question"), entry.get("bad_answer"), entry["correction"],
                         entry.get("created_at", datetime.now(timezone.utc).isoformat())),
                    )
                conn.execute("INSERT OR REPLACE INTO seed_meta (key,value) VALUES ('version',?)", (seed_version,))
                conn.commit()
        conn.close()


def _load_admin_password() -> str:
    if ADMIN_PASSWORD_FILE.exists():
        return ADMIN_PASSWORD_FILE.read_text(encoding="utf-8").strip()
    return os.environ.get("ADMIN_PASSWORD", "")


ADMIN_PASSWORD = _load_admin_password()

# ---------------------------------------------------------------------------
# RAG index — loaded once at startup
# ---------------------------------------------------------------------------

print("Loading pre-built FAISS index…")
rag.load_prebuilt(str(WIKI_DIR))
print("Index ready.")

print("Initialising database…")
_init_db()
print("Database ready.")


def _log_request(ip: str, question: str, prompt_tokens: int, completion_tokens: int):
    with _db_lock:
        conn = _db_conn()
        conn.execute(
            "INSERT INTO requests (ip,question,prompt_tokens,completion_tokens) VALUES (?,?,?,?)",
            (ip, (question or "")[:500], prompt_tokens, completion_tokens),
        )
        conn.commit()
        conn.close()


def _load_corrections() -> str:
    with _db_lock:
        conn = _db_conn()
        rows = conn.execute(
            "SELECT question, correction FROM corrections WHERE status IN ('approved','seeded') ORDER BY created_at"
        ).fetchall()
        conn.close()
    lines = []
    for row in rows:
        q, c = row["question"], row["correction"]
        if not c:
            continue
        lines.append(f"Q: {q}\nA: {c}" if q else c)
    return "\n\n".join(lines)

# ---------------------------------------------------------------------------
# Agentic RAG tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the Kiez Burn wiki using both semantic (vector) and keyword (BM25) search in parallel. "
                "Results are tagged with their match type: 'both' (matched both), 'vector' (semantic), or 'keyword' (exact terms). "
                "Returns titles, short excerpts, match types, and IDs. Use multiple searches with different terms to cover the topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — use specific terms relevant to the question."},
                    "top_k": {"type": "integer", "description": "Number of results per method (1–15, default 6).", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_context",
            "description": (
                "Mark specific chunk IDs (from search results) to include as full-text context when generating the answer. "
                "Use this for chunks that contain detailed information needed for the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "integer"}, "description": "List of chunk IDs to fetch full text for."}
                },
                "required": ["ids"],
            },
        },
    },
]

AGENTIC_SYSTEM = """You are Kiezthropic, a helpful assistant for Kiez Burn — a Burning Man-inspired community event near Berlin.

You have access to a wiki search tool that runs semantic + keyword search in parallel. Follow this process:

1. Call search() with specific terms related to the question.
2. Call add_to_context() with the IDs of the 2-4 most relevant chunks. Always do this — snippets are truncated and the full text contains more detail.
3. Optionally search again (up to 3 times total) with different terms if you need more information.
4. Once you have added relevant chunks, stop calling tools and give your final answer.

Important rules:
- ALWAYS call add_to_context() after each search with the most relevant IDs — never skip this step.
- Results tagged "both" matched semantic + keyword search and are usually most relevant.
- For listing questions (camps, installations, etc.): search "camps list 2026", add those IDs.
- For factual questions: add the IDs of chunks most likely to contain the answer, even if the snippet seems incomplete.
- After adding context, answer directly without calling more tools unless a second search is clearly needed.
- If the question is vague or a general greeting, treat it as "tell me about Kiez Burn" and search for an overview.
- NEVER ask the user to clarify — always make your best guess at what they want and answer it."""

ANSWER_SYSTEM_BASE = """You are Kiezthropic, a helpful assistant for Kiez Burn — a Burning Man-inspired community event near Berlin.
Answer questions using the provided wiki context. Be friendly, direct, and concise. Do not use emojis.
When listing camps or installations, provide a complete list — do not truncate or summarize.
If the question is vague, give a useful overview of Kiez Burn rather than asking for clarification.
If the context doesn't cover the question, say so briefly and suggest what to ask instead."""


def _build_answer_system() -> str:
    corrections = _load_corrections()
    if not corrections:
        return ANSWER_SYSTEM_BASE
    return (
        "IMPORTANT CORRECTIONS — these override any conflicting information in the wiki:\n\n"
        + corrections + "\n\n---\n\n" + ANSWER_SYSTEM_BASE
    )


def _format_search_results(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        snippet = c["text"][:600].replace("\n", " ")
        lines.append(f'ID:{c["idx"]} [{c.get("match","vector")}] | {c["title"]} | {snippet}…')
    return "\n".join(lines)


def _msg_to_dict(m) -> dict:
    if isinstance(m, dict):
        return m
    d: dict = {"role": m.role, "content": m.content or ""}
    if hasattr(m, "tool_calls") and m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in m.tool_calls
        ]
    return d


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return {"error": "empty message"}, 400
    # history: list of {role, content} from the frontend (preceding turns)
    history = data.get("history", [])
    history = [
        {"role": h["role"], "content": str(h.get("content", ""))[:2000]}
        for h in history
        if isinstance(h, dict) and h.get("role") in ("user", "assistant")
    ][-10:]

    # Cloudflare sets CF-Connecting-IP; fall back to X-Forwarded-For then remote addr
    client_ip = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0]
    ).strip()

    def generate():
        client = get_openai_client()
        messages: list = [
            {"role": "system", "content": AGENTIC_SYSTEM},
            {"role": "user", "content": user_message},
        ]
        collected_ids: set[int] = set()
        search_count = 0
        MAX_SEARCHES = 3
        total_prompt_tokens = 0
        total_completion_tokens = 0

        try:
            # --- agentic loop ---
            while search_count < MAX_SEARCHES:
                response = client.chat.completions.create(
                    model=MODEL, max_tokens=1024, tools=TOOLS, tool_choice="auto", messages=messages,
                )
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

                msg = response.choices[0].message
                messages.append(_msg_to_dict(msg))

                if not msg.tool_calls:
                    break

                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if fn == "search":
                        query = args.get("query", user_message)
                        top_k = max(1, min(int(args.get("top_k", 6)), 15))
                        yield f"data: {json.dumps({'status': f'Searching: {query}'})}\n\n"
                        results = rag.retrieve_combined(query, top_k=top_k)
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": _format_search_results(results) or "No results found."})
                        search_count += 1

                    elif fn == "add_to_context":
                        ids = [int(i) for i in args.get("ids", [])]
                        collected_ids.update(ids)
                        yield f"data: {json.dumps({'status': f'Loading {len(ids)} chunk(s) in full'})}\n\n"
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": f"Added chunk IDs {ids} to context."})

            # --- build final context ---
            context_parts = []
            if collected_ids:
                full_chunks = rag.get_chunks_by_ids(list(collected_ids))
                if any("camps_list" in c.get("source", "") for c in full_chunks):
                    camps_all = rag.retrieve_by_source("camps_list")
                    camps_ids = {c["idx"] for c in camps_all}
                    extra = rag.get_chunks_by_ids([i for i in camps_ids if i not in collected_ids])
                    full_chunks = full_chunks + extra
                for c in full_chunks:
                    context_parts.append(f"[{c['title']}]\n{c['text']}")

            full_context_block = (
                "Full text for selected chunks:\n\n" + "\n\n---\n\n".join(context_parts) + "\n\n"
                if context_parts else ""
            )
            search_history = [m for m in messages[1:] if isinstance(m, dict) and m.get("role") == "tool"]
            snippet_block = ""
            if search_history:
                snippet_block = "Search result snippets from earlier:\n\n" + "\n\n".join(
                    m["content"] for m in search_history if "ID:" in m.get("content", "")
                ) + "\n\n"

            final_messages = [
                {"role": "system", "content": _build_answer_system()},
                *history,
                {"role": "user", "content": f"{full_context_block}{snippet_block}Question: {user_message}"},
            ]

            stream = client.chat.completions.create(
                model=MODEL, max_tokens=4096, stream=True,
                stream_options={"include_usage": True},
                messages=final_messages,
            )
            for chunk in stream:
                if chunk.usage:
                    total_prompt_tokens += chunk.usage.prompt_tokens
                    total_completion_tokens += chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
            yield "data: [DONE]\n\n"

        except BadRequestError as e:
            err_body = getattr(e, 'body', {}) or {}
            if isinstance(err_body, dict) and err_body.get('error', {}).get('code') == 'content_filter':
                yield f"data: {json.dumps({'text': 'The content filter blocked this response. Try rephrasing your question.'})}\n\n"
            else:
                yield f"data: {json.dumps({'text': f'Request error: {e.message}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        threading.Thread(
            target=_log_request,
            args=(client_ip, user_message, total_prompt_tokens, total_completion_tokens),
            daemon=True,
        ).start()

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/correct", methods=["POST"])
def correct():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    bad_answer = data.get("bad_answer", "").strip()
    correction = data.get("correction", "").strip()
    if not correction:
        return {"error": "correction required"}, 400
    with _db_lock:
        conn = _db_conn()
        conn.execute(
            "INSERT INTO corrections (question,bad_answer,correction,status) VALUES (?,?,?,'pending')",
            (question or None, bad_answer or None, correction),
        )
        conn.commit()
        conn.close()
    return {"ok": True}


def _auth_required() -> bool:
    return request.cookies.get("admin_auth") == ADMIN_PASSWORD


@app.route("/admin", methods=["GET", "POST"])
def admin():
    error = None
    if request.method == "POST":
        if request.form.get("password", "") == ADMIN_PASSWORD:
            resp = Response("", status=302, headers={"Location": "/admin"})
            resp.set_cookie("admin_auth", ADMIN_PASSWORD, httponly=True, samesite="Lax")
            return resp
        error = "Wrong password"

    if not _auth_required():
        return render_template("admin_login.html", error=error)

    with _db_lock:
        conn = _db_conn()
        pending = [dict(r) for r in conn.execute(
            "SELECT id, question, bad_answer, correction, created_at FROM corrections WHERE status='pending' ORDER BY created_at"
        ).fetchall()]
        approved = [dict(r) for r in conn.execute(
            "SELECT id, question, bad_answer, correction, created_at FROM corrections WHERE status IN ('approved','seeded') ORDER BY created_at"
        ).fetchall()]
        conn.close()

    return render_template("admin.html", pending=pending, approved=approved, tab="corrections")


@app.route("/admin/stats")
def admin_stats():
    if not _auth_required():
        return Response("", status=302, headers={"Location": "/admin"})

    with _db_lock:
        conn = _db_conn()
        totals = dict(conn.execute("""
            SELECT COUNT(*) total_requests,
                   COUNT(DISTINCT ip) unique_ips,
                   COALESCE(SUM(prompt_tokens),0) total_prompt_tokens,
                   COALESCE(SUM(completion_tokens),0) total_completion_tokens
            FROM requests
        """).fetchone())
        daily = [dict(r) for r in conn.execute("""
            SELECT DATE(created_at) day, COUNT(*) reqs,
                   COALESCE(SUM(prompt_tokens),0) pt,
                   COALESCE(SUM(completion_tokens),0) ct
            FROM requests
            WHERE created_at >= DATE('now','-30 days')
            GROUP BY day ORDER BY day
        """).fetchall()]
        recent = [dict(r) for r in conn.execute("""
            SELECT ip, question, prompt_tokens, completion_tokens, created_at
            FROM requests ORDER BY created_at DESC LIMIT 50
        """).fetchall()]
        conn.close()

    total_cost = (
        totals["total_prompt_tokens"] / 1_000_000 * _COST_PER_1M_INPUT
        + totals["total_completion_tokens"] / 1_000_000 * _COST_PER_1M_OUTPUT
    )
    return render_template("admin.html", totals=totals, daily=daily, recent=recent,
                           total_cost=total_cost,
                           cost_per_1m_in=_COST_PER_1M_INPUT,
                           cost_per_1m_out=_COST_PER_1M_OUTPUT,
                           tab="stats")


@app.route("/admin/approve", methods=["POST"])
def admin_approve():
    if not _auth_required():
        return {"error": "unauthorized"}, 403
    row_id = int(request.get_json(force=True).get("id", 0))
    with _db_lock:
        conn = _db_conn()
        conn.execute("UPDATE corrections SET status='approved' WHERE id=?", (row_id,))
        conn.commit()
        conn.close()
    return {"ok": True}


@app.route("/admin/delete", methods=["POST"])
def admin_delete():
    if not _auth_required():
        return {"error": "unauthorized"}, 403
    row_id = int(request.get_json(force=True).get("id", 0))
    with _db_lock:
        conn = _db_conn()
        conn.execute("DELETE FROM corrections WHERE id=?", (row_id,))
        conn.commit()
        conn.close()
    return {"ok": True}


@app.route("/admin/export")
def admin_export():
    if not _auth_required():
        return {"error": "unauthorized"}, 403
    with _db_lock:
        conn = _db_conn()
        rows = [dict(r) for r in conn.execute(
            "SELECT question, bad_answer, correction, created_at FROM corrections WHERE status IN ('approved','seeded') ORDER BY created_at"
        ).fetchall()]
        conn.close()
    seed = {"version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"), "corrections": rows}
    return app.response_class(
        json.dumps(seed, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=corrections_seed.json"},
    )


@app.route("/stats.json")
def stats_json():
    """Public aggregate stats — no auth, no PII (IPs not exposed)."""
    with _db_lock:
        conn = _db_conn()
        totals = dict(conn.execute("""
            SELECT COUNT(*) total_requests,
                   COUNT(DISTINCT ip) unique_ips,
                   COALESCE(SUM(prompt_tokens),0) total_prompt_tokens,
                   COALESCE(SUM(completion_tokens),0) total_completion_tokens
            FROM requests
        """).fetchone())
        daily = [dict(r) for r in conn.execute("""
            SELECT DATE(created_at) day, COUNT(*) reqs
            FROM requests
            WHERE created_at >= DATE('now','-30 days')
            GROUP BY day ORDER BY day
        """).fetchall()]
        conn.close()
    total_cost = (
        totals["total_prompt_tokens"] / 1_000_000 * _COST_PER_1M_INPUT
        + totals["total_completion_tokens"] / 1_000_000 * _COST_PER_1M_OUTPUT
    )
    return app.response_class(
        json.dumps({
            "total_requests": totals["total_requests"],
            "unique_ips": totals["unique_ips"],
            "total_tokens": totals["total_prompt_tokens"] + totals["total_completion_tokens"],
            "total_cost": round(total_cost, 4),
            "daily": daily,
        }),
        mimetype="application/json",
    )


@app.route("/admin/logout")
def admin_logout():
    resp = Response("", status=302, headers={"Location": "/admin"})
    resp.delete_cookie("admin_auth")
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
