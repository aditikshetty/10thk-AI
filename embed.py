import json
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the chunks we created earlier
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks")


# 2. Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# 3. Take only the text from each chunk
texts = [chunk["text"] for chunk in chunks]


# 4. Convert every chunk into an embedding
print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# 5. Convert to float32
embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# 6. Save the embeddings
np.save("embeddings.npy", embeddings)

print("Saved embeddings.npy")