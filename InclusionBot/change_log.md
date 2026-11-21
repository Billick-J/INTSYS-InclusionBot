# InclusionBot change log

| Date | Entry / file | What changed | Approved by |
|------|--------------|--------------|-------------|
| 2026-11-15 | Modelfile | V1 system prompt with Section 4.4 scope rule, decision rule, safety and privacy rules | (pending SPED review) |
| 2026-11-15 | front-end/ | Added web chat front end (blue and red theme) reusing app.py rules and knowledge base | (pending SPED review) |
| 2026-11-15 | (folders) | Reorganised into backend/, front-end/ and docs/; added README.md and requirements.txt | (pending SPED review) |
| 2026-11-20 | backend/rag.py (new) | V2: Retrieval-Augmented Generation. Reference document *Inclusive education: Children with disabilities* (Hunt, 2020, UNESCO GEM background paper) indexed into 206 section-labelled passages; hybrid BM25 + nomic-embed-text retrieval; LLM scope router; relevance gate ("not in the document" reply); answers cite passages with section and page | (pending SPED review) |
| 2026-11-20 | Modelfile | Added RAG grounding rules (answer only from retrieved passages, cite [n], say when the document lacks the answer); removed "general information" allowances; temperature 0.2 → 0.1 | (pending SPED review) |
| 2026-11-20 | app.py, server.py | Both chats now answer through the RAG pipeline; approved KB entries are retrieved like document passages; safety, privacy (learner names removed) and diagnosis referral guards; Taglish fixed replies | (pending SPED review) |
| 2026-11-20 | front-end/ | Reference-document card, citation badges, Sources block, headings in answers, document-based suggestion chips | (pending SPED review) |
| 2026-11-20 | tests/run_tests.py | Added document-grounding (D1–D8) and stricter scope (X5–X6) tests; report shows route, search query and retrieved passages | (pending SPED review) |
| 2026-11-20 | docs/source_documents/ | Added 373662eng.pdf and sources.json | (pending SPED review) |
| 2026-11-20 | app.py, rag.py, server.py, Modelfile, tests | Removed the school knowledge base (no SPED entries); answers come only from the UNESCO reference document | (pending SPED review) |
| 2026-11-20 | rag.py, Modelfile, tests | Answers no longer show passage numbers [n], section names, page numbers or a Sources list | (pending SPED review) |
| 2026-11-20 | rag.py, tests | Answers end with one Source line naming the document (title, author, UNESCO, year), without sections or pages | (pending SPED review) |
| 2026-11-20 | rag.py, Modelfile | Answers name the document by its exact title ("According to *Inclusive education: Children with disabilities*...") instead of "according to the document" | (pending SPED review) |
| 2026-11-20 | front-end/ | Voice accessibility: Listen / Pause / Resume / Stop on every answer (text-to-speech), microphone to speak a question into the box (speech-to-text), Voice mode to read answers automatically, voice language / voice / speed settings, Esc to stop, keyboard focus outlines. RAG backend unchanged | (pending SPED review) |
| 2026-11-20 | rag.py, tests | Removed the Source line at the bottom of answers (answers still name the document by its title in the text) | (pending SPED review) |
| 2026-11-20 | front-end/js/script.js | Reading aloud prefers natural / neural voices (e.g. Edge "Online (Natural)", Google voices) and falls back to a device voice when offline | (pending SPED review) |
| 2026-11-20 | front-end/ | Reading aloud: default speed 0.85×, short natural pauses between sentences and points, short forms read in full (e.g. -> for example) | (pending SPED review) |
| 2026-11-20 | front-end/ | Chat history: Recent chats list in the sidebar (saved in the browser only, up to 30), open a past chat and continue it, delete one / delete all, option to turn saving off. Server still stores nothing | (pending SPED review) |
| 2026-11-21 | backend/rag.py, Modelfile, sources.json, tests | Each answer now names the document it actually used; Philippine sources are preferred (country field); source cautions (date_context, age_group) are passed to the model | (pending SPED review) |
| 2026-11-21 | backend/rag.py, front-end/js/script.js | Questions asking where SPED centers or ILRCs are now get the NCDA list link (https://ncda.gov.ph/list-of-sped-centers/); .gov.ph links are clickable | (pending SPED review) |
| 2026-11-21 | backend/rag.py | A condition named in the question (dyslexia, ADHD, autism...) must appear in the passages used for the answer; early-childhood (0-4) sources rank lower for school-age questions; greeting no longer lists the documents; added python backend/rag.py --calibrate | (pending SPED review) |
