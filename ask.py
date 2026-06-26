import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

EMBEDDINGS_FILE = "knowledge_base_embeddings.json"

def load_knowledge_base():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def cosine_similarity(vec_a, vec_b):
    """Measures how similar two embeddings are. 1 = identical meaning, 0 = unrelated."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(question, knowledge_base):
    """Compares the question's embedding to every document, returns the closest match."""
    question_embedding = get_embedding(question)

    best_doc = None
    best_score = -1

    for doc in knowledge_base:
        score = cosine_similarity(question_embedding, doc["embedding"])
        print(f"  Similarity to {doc['filename']}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc, best_score

def generate_answer(question, context_doc):
    """Asks Gemini to answer the question, using the matched document as the source of truth."""
    prompt = f"""
You are an IT helpdesk assistant. Answer the user's question using ONLY the information in the document below. 
If the document doesn't contain enough information to answer, say so honestly rather than guessing.

Document:
{context_doc['content']}

User's question: {question}

Give a clear, helpful answer in 2-4 sentences, written as if speaking directly to the user.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def main():
    knowledge_base = load_knowledge_base()
    print("AI Helpdesk Assistant — ask a question, or type 'quit' to exit.\n")

    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        best_doc, score = find_best_match(question, knowledge_base)

        # If confidence is too low, don't pretend we found a good answer
        if score < 0.5:
            print("\nAssistant: I couldn't find anything relevant in the knowledge base for that. This might need a human agent.\n")
            continue

        answer = generate_answer(question, best_doc)
        print(f"\n(Matched: {best_doc['filename']}, confidence: {score:.3f})")
        print(f"Assistant: {answer}\n")

if __name__ == "__main__":
    main()