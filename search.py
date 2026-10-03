import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Load textbook chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks")


# Load embeddings
embeddings = np.load("embeddings.npy")

print("Embeddings shape:", embeddings.shape)


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors")


# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Ask a question
question = input("\nAsk a textbook question: ")


# Convert question into embedding
question_embedding = model.encode(
    [question]
)

question_embedding = np.array(
    question_embedding
).astype("float32")


# Search for the 3 most relevant chunks
distances, indices = index.search(
    question_embedding,
    3
)


# Display results
print("\n========== RESULTS ==========")

for rank, (distance, index_number) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):

    chunk = chunks[index_number]

    print(f"\n--- Result {rank} ---")

    print("Subject:", chunk["subject"])
    print("Book:", chunk["book"])
    print("Page:", chunk["page"])
    print("Distance:", distance)

    print("\nText:")
    print(chunk["text"])