"""SPED InclusionBot V2 - terminal chat app (document-grounded RAG).

Run:   python backend/app.py        (from the project folder)
Needs: Ollama running, the custom model built with
       ollama create inclusionbot -f backend/Modelfile
and the embedding model:
       ollama pull nomic-embed-text

Answers come only from the reference document(s) in docs/source_documents/.
See rag.py for the pipeline.
"""
import re
import sys
from pathlib import Path

import ollama

import rag

MODEL = "inclusionbot"
BACKEND_DIR = Path(__file__).resolve().parent
MODELFILE = BACKEND_DIR / "Modelfile"
MAX_TURNS = 8     # remember only the last 8 exchanges, in memory only (Section 8.7)
PROMPT_TURNS = 4  # exchanges sent to the model, so the retrieved passages always fit


def load_system_rules():
    """Read the SYSTEM prompt from the Modelfile so the rules live in one place.

    The app sends its own system message (rules). Ollama drops the Modelfile's
    SYSTEM when a request already starts with a system message, so the rules
    must be included here too.
    """
    text = MODELFILE.read_text(encoding="utf-8")
    match = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    if not match:
        sys.exit("Could not find a SYSTEM \"\"\" ... \"\"\" block in the Modelfile.")
    return match.group(1).strip()


def build_system_prompt():
    """Rules for the model (from the Modelfile)."""
    return load_system_rules()


def wrap(user_text):
    """Mark user input as data, not instructions (Lab 4.2 defence)."""
    clean = user_text.replace("<user_data>", "").replace("</user_data>", "")
    return f"<user_data>{clean}</user_data>"


def prompt_history(history):
    """Recent turns for the model, without the sources footers the app added."""
    out = []
    for m in history[-PROMPT_TURNS * 2:]:
        content = rag.strip_footer(m["content"]) if m["role"] == "assistant" else m["content"]
        out.append({"role": m["role"], "content": content})
    return out


def plan(user_text, history, system_prompt=None):
    """Run the RAG steps (guards, router, retrieval, relevance gate). Returns a rag.Plan."""
    if system_prompt is None:
        system_prompt = build_system_prompt()
    return rag.plan_reply(user_text, prompt_history(history), MODEL, system_prompt)


def ask(system_prompt, history, user_text, stream=False):
    """Answer one question through the RAG pipeline. Returns the reply text
    (prints it live when stream=True)."""
    p = plan(user_text, history, system_prompt)
    reply = ""
    for piece in rag.generate(p, MODEL):
        reply += piece
        if stream:
            print(piece, end="", flush=True)
    if stream:
        print()
    return reply


def main():
    history = []
    try:
        rag.get_retriever(verbose=True)  # builds the document index on first run
        info = rag.index_info()
    except Exception as e:
        sys.exit(f"Could not load the reference document index: {e}")
    print("SPED InclusionBot V2  |  type /bye to quit, /reset to clear the chat")
    print(f"(reference: {'; '.join(info['documents'])} | {info['chunks']} passages, "
          f"{info['retrieval']} retrieval)")
    print("Reminder: please don't type learners' names or personal records.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("/bye", "/exit", "/quit"):
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("(chat cleared)\n")
            continue

        print("InclusionBot: ", end="", flush=True)
        try:
            system_prompt = build_system_prompt()
            reply = ask(system_prompt, history, user_text, stream=True)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(f"\nA model is missing. Build/pull them first:\n"
                      f"  ollama create {MODEL} -f backend/Modelfile\n"
                      f"  ollama pull {rag.EMBED_MODEL}")
                break
            print(f"\nOllama error: {e.error}")
            continue
        except Exception as e:  
            print(f"\nCould not reach Ollama ({e}). Start the Ollama app and try again.")
            break

        history.append({"role": "user", "content": wrap(user_text)})
        history.append({"role": "assistant", "content": reply})
        del history[:-MAX_TURNS * 2]
        print()


if __name__ == "__main__":
    main()
