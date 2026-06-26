import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

KB_FOLDER = "knowledge_base"
OUTPUT_FILE = "knowledge_base_embeddings.json"

def load_documents(folder):
    """Reads every .txt file in the folder and returns a list of (filename, content) pairs."""
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"filename": filename, "content": content})
    return documents

def get_embedding(text):
    """Sends text to Gemini's embedding model, gets back a list of numbers representing its meaning."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def main():
    documents = load_documents(KB_FOLDER)
    print(f"Loaded {len(documents)} documents from {KB_FOLDER}\n")

    for doc in documents:
        print(f"Embedding: {doc['filename']}...")
        doc["embedding"] = get_embedding(doc["content"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f)

    print(f"\nDone! Saved embeddings for {len(documents)} documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()