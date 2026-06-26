# AI Helpdesk Assistant

An AI-powered chatbot that answers IT support questions by searching internal documentation and generating grounded, accurate answers — built using Retrieval-Augmented Generation (RAG).

## What it does

Instead of relying on the AI's general knowledge (which can be wrong or made up), this assistant:

1. **Stores** a knowledge base of IT support documents (WiFi troubleshooting, password resets, printer issues, etc.) as embeddings (numerical representations of meaning)
2. **Retrieves** the most relevant document for any user question using semantic similarity search
3. **Generates** an answer strictly grounded in that document's content
4. **Flags uncertainty** — if no document is a good enough match, it says so honestly instead of guessing, and suggests escalating to a human agent

## Why I built this

Most IT helpdesks have internal knowledge bases that are underused because search is poor or staff don't have time to dig through them. This project demonstrates how AI can make that knowledge instantly accessible and reduce repetitive first-line support work — while staying grounded in real documentation rather than hallucinating answers.

## Tech stack

- Python 3
- Google Gemini API (`google-genai`) — for both embeddings and answer generation
- NumPy — for similarity calculations
- `python-dotenv` for secure API key handling

## How it works (RAG pipeline)

1. `build_knowledge_base.py` reads all `.txt` files in `knowledge_base/`, converts each to an embedding using Gemini's embedding model, and saves them to `knowledge_base_embeddings.json`
2. `ask.py` embeds the user's question, compares it to every document using cosine similarity, and picks the best match
3. The matched document is passed to Gemini as context, with instructions to answer only from that content
4. If similarity confidence is below a threshold, the assistant declines rather than guessing

## Setup

1. Clone this repo
2. Install dependencies:
pip3 install google-genai python-dotenv numpy
3. Create a `.env` file with your own Gemini API key:
GEMINI_API_KEY=your_key_here
4. Build the knowledge base (only needed once, or after editing documents):
python3 build_knowledge_base.py
5. Start chatting:
python3 ask.py

## Example interaction
You: my password is not working, what should I do now?

(Matched: password_reset.txt, confidence: 0.647)

Assistant: To help you reset your password, I first need to verify your identity...

## Possible future improvements

- Support PDF/Word documents in the knowledge base, not just plain text
- Web-based chat interface instead of terminal
- Multi-turn conversation memory
- Slack/Teams integration for real deployment