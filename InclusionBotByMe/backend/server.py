"""SPED InclusionBot V2 - web server (API + serves the front-end folder).

Run:   python backend/server.py        (from the project folder)
Open:  http://127.0.0.1:8000

Needs the same setup as app.py: Ollama running and the custom model built with
       ollama create inclusionbot -f backend/Modelfile

The server reuses app.py and rag.py (rules, RAG pipeline, <user_data> wrapping), so the
web chat follows exactly the same rules as the terminal chat. It uses only the
Python standard library plus the `ollama` package that app.py already needs.

Privacy (Section 8.7): the server keeps no chat history. The browser holds the
conversation in memory and sends the last few turns with each question; nothing
is written to disk, and closing or reloading the page clears it.
"""
import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama

import rag
from app import MAX_TURNS, MODEL, plan, wrap

BACKEND_DIR = Path(__file__).resolve().parent

HOST = "127.0.0.1"  # local only: do not expose this on a network without review
PORT = 8000
FRONTEND_DIR = BACKEND_DIR.parent / "front-end"
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 2000


def clean_history(raw):
    """Keep only well-formed user/assistant turns, newest MAX_TURNS exchanges."""
    history = []
    if not isinstance(raw, list):
        return history
    for item in raw[-MAX_TURNS * 2:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content[:MAX_MESSAGE_CHARS * 4]
        if role == "assistant":
            content = rag.strip_footer(content)  # sources list is for the reader, not the model
        # User turns are always re-wrapped here, never trusted from the browser.
        history.append({"role": role, "content": wrap(content) if role == "user" else content})
    return history


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, fmt, *args):
        # Log only the request line, never question text.
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            try:
                ollama.show(MODEL)
                model_ready, note = True, ""
            except ollama.ResponseError as e:
                model_ready = False
                note = (f"Model '{MODEL}' not found. Build it first: ollama create {MODEL} -f backend/Modelfile"
                        if e.status_code == 404 else f"Ollama error: {e.error}")
            except Exception:
                model_ready, note = False, "Could not reach Ollama. Start the Ollama app and try again."
            try:
                info = rag.index_info()
            except Exception as e:
                info = {"documents": [], "chunks": 0, "retrieval": ""}
                model_ready, note = False, f"Reference document index could not be loaded: {e}"
            self.send_json(HTTPStatus.OK, {"model": MODEL, "ready": model_ready, "note": note, **info})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Message is too long."})
            return
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request."})
            return

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, str) or not message.strip():
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Please type a question."})
            return
        if len(message) > MAX_MESSAGE_CHARS:
            self.send_json(HTTPStatus.BAD_REQUEST,
                           {"error": f"Please keep questions under {MAX_MESSAGE_CHARS} characters."})
            return

        # RAG: guards -> router -> retrieve from the reference document -> gate -> generate.
        # Rules are re-read each time so Modelfile changes apply at once.
        try:
            reply_plan = plan(message.strip(), clean_history(data.get("history")))
            stream = rag.generate(reply_plan, MODEL)
            first = next(stream)  # surfaces connection / model errors before we reply 200
        except ollama.ResponseError as e:
            msg = (f"A model is missing. Run: ollama create {MODEL} -f backend/Modelfile "
                   f"and ollama pull {rag.EMBED_MODEL}"
                   if e.status_code == 404 else f"Ollama error: {e.error}")
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": msg})
            return
        except StopIteration:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": "The model returned no answer."})
            return
        except Exception:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Could not reach Ollama. Start the Ollama app and try again."})
            return

        # Stream plain text; the connection closes when the answer is complete.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(first.encode("utf-8"))
            self.wfile.flush()
            for piece in stream:
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # user closed the page or pressed Stop
        except Exception as e:  # model failed mid-answer (e.g. Ollama crash): tell the reader
            detail = e.error if isinstance(e, ollama.ResponseError) else "could not reach Ollama"
            try:
                self.wfile.write(f"\n\n(The answer was cut off - Ollama error: {detail}. "
                                 f"Please try again.)".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    try:
        rag.get_retriever(verbose=True)  # builds the document index on first run
        info = rag.index_info()
    except Exception as e:
        sys.exit(f"Could not load the reference document index: {e}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SPED InclusionBot web chat: http://{HOST}:{PORT}")
    print(f"(reference: {'; '.join(info['documents'])} | {info['chunks']} passages, "
          f"{info['retrieval']} retrieval)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
