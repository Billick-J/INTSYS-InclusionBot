# SPED InclusionBot

A question-and-answer assistant for Special Education (SPED) and Inclusive Education,
built for teachers and parents in Baguio City, Philippines. It runs locally with Ollama
(`llama3:8b`) and uses **Retrieval-Augmented Generation (RAG)**: every answer is taken
from the **InclusionBot Q&A Knowledge Base** only, never from the model's general knowledge or
any outside source. If the answer is not in the knowledge base, the bot says that the information
is not covered in the knowledge base.

Knowledge base: `docs/source_documents/InclusionBot-QA-Knowledge-Base.pdf` (the only file listed in
`sources.json`). It has 64 question-and-answer entries in 12 categories plus a "Gaps" section listing
what it does not cover (65 passages in the index). The `.md` copy and the original paper
`373662eng.pdf` in the same folder are not indexed (a `[skipped]` line is printed for each).

Every answer follows the matching entry: the answer, then **Real-life example**, **What to do** and
**Important note**. Answers never say they come from a document: phrases such as "the document
explains that ..." are rewritten as plain statements, and no titles or page numbers are shown.

## How InclusionBot answers

```
question
  │
  ├─ 1. Guards ─────── greeting / thanks ─────────────► fixed welcome
  │                     child-safety report ──────────► fixed safety referral
  │                     learner name / records ───────► name removed + privacy notice
  │
  ├─ 2. Router (LLM) ─ is it about SPED / inclusive education?
  │                     + rewrite as a standalone English search query
  │                       (handles follow-ups such as "give an example" and Taglish)
  │
  ├─ 3. Retrieve ───── hybrid search over 65 knowledge-base passages
  │                     (BM25 keywords + nomic-embed-text vectors, Reciprocal Rank Fusion)
  │
  ├─ 4. Gate ───────── off-topic ─────────────────────► polite refusal (no answer, not even partly)
  │                     on-topic but nothing relevant ─► "not covered in my knowledge base"
  │
  └─ 5. Generate ───── model answers ONLY from the top 5 passages, in plain words
                        (no passage numbers, sections or page numbers)
```

The model is never asked to answer without retrieved passages, and if the passages it
receives don't actually answer the question it must say so instead of guessing.

## Folder structure

```
InclusionBot/
├── backend/                   Python code, model and data
│   ├── app.py                 Rules, terminal chat
│   ├── rag.py                 RAG pipeline: PDF ingestion, index, retrieval, scope gate, generation
│   ├── server.py              Web server: API for the browser chat, serves front-end/
│   ├── Modelfile              Ollama model definition and system prompt (rules)
│   ├── rag_index/             Generated search index (rebuilt automatically, not edited by hand)
│   └── tests/
│       └── run_tests.py       Test set; reports are saved here as results_<date>.md
├── front-end/                 Browser chat (blue and red theme)
├── docs/
│   └── source_documents/      Reference document(s) used by RAG + sources.json (titles)
├── change_log.md              Record of changes and who approved them
├── requirements.txt
└── README.md
```

## Setup (once)

Run these from the `InclusionBot` folder in PowerShell:

```powershell
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama create inclusionbot -f backend/Modelfile
```

Run `ollama create` again whenever you change `backend/Modelfile`.
`nomic-embed-text` is the small embedding model used for semantic search. Without it the
bot still works, using keyword search only (less accurate).

## Run

Make sure the Ollama app is running first, then from the `InclusionBot` folder:

| What | Command |
|------|---------|
| Web chat | `python backend/server.py`, then open <http://127.0.0.1:8000> |
| Terminal chat | `python backend/app.py` |
| Test set | `python backend/tests/run_tests.py` (or e.g. `... run_tests.py D1 X1` for some tests) |
| Rebuild the document index | `python backend/rag.py --build` |
| See what retrieval finds | `python backend/rag.py --search "what is reasonable accommodation"` |

The first start builds the document index (about a minute). Press **Ctrl + C** to stop the web server.

## Changing or adding reference documents

Only documents about **Special Education (SPED) or Inclusive Education** may be added.

1. Put the PDF (or `.txt` / `.md`) file in `docs/source_documents/`.
2. Add its title, author, year, country and `category` to `docs/source_documents/sources.json`.
   The category must be one of `SPED_CATEGORIES` in `backend/rag.py` (Special Education, Inclusive
   Education, Philippine SPED Policies, Philippine SPED Services, Teaching Strategies, Early Childhood
   Development). A file that is not listed, or has another category, is **not indexed** and a
   `[skipped]` warning is printed.
3. Restart the app. The index is rebuilt automatically when a document changes.

Chapters, sections and pages are still recorded in the index (shown by `rag.py --search`),
but answers no longer show them.

**Q&A knowledge-base files** (`.pdf` or `.md`) are split differently from other PDFs. A PDF with
`Question:` entries is first turned back into lines (running page headers removed, wrapped lines
joined, category headings taken from its "Categories" list). Then every `Question:` entry up to the next `---` is one passage,
the `# N. Category` heading above it is its chapter, the `Related Questions` list is left out, and
the pages in its `Source/Reference: ... Location in document:` line are recorded but not shown to
the model. A section with no `Question:` entries (the "Gaps" section) is one passage. Older files
that use `### ` headings are still supported. Long entries are split at paragraph breaks.

## School-specific information

The knowledge base comes from an international paper: it does not describe this school's own
enrollment steps, fees, services or contacts, or where SPED centers are. Questions about those
get a reply saying the information is not covered in the knowledge base and telling the user to
contact the school's SPED department (no outside links or addresses).

## Voice features (web chat)

- **Listen** under every answer reads it aloud, with **Pause**, **Resume** and **Stop**.
- The **microphone** turns a spoken question into text in the box; the user checks it and presses Send.
- **Voice** (next to the box) turns on voice mode: every answer is read aloud automatically.
- Settings › Voice: language (Automatic / English / Filipino), reading voice, reading speed. **Esc** stops both.

These use the browser's built-in Web Speech API; the RAG backend is unchanged. Reading aloud
uses the voices installed on the device (works offline). The microphone works in Chrome and
Edge only, needs internet (the browser sends the audio to Google or Microsoft), and only on
`127.0.0.1`/`localhost` or HTTPS pages.

## Tuning

In `backend/rag.py`: `TOP_K` (passages per answer), `CHUNK_CHARS` (passage size) and
`DENSE_MIN` (how similar a passage must be to count as relevant; raise it for stricter
"not in the document" replies, lower it if good questions are refused). The test report
lists the retrieved passages and scores for every question, to help with tuning.

## Privacy

- The server keeps no chat history: it only receives the last 8 exchanges with each question
  and never writes them to disk.
- The web chat saves past chats (**Recent chats** in the sidebar, up to 30) in the user's own
  browser only (localStorage). Users can delete one chat, delete all, or turn saving off in
  Settings › Chat history. Turn it off on shared computers.
- Learner names typed as "my student Juan Dela Cruz" are replaced with "[learner]" before
  anything is sent to the model, and the user is reminded not to share personal records.
- The web server does not log question text and only listens on `127.0.0.1`.
