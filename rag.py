import json
import numpy as np
import faiss
import re
import ollama

from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD TEXTBOOK CHUNKS
# ==========================================

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks")


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

embeddings = np.load("embeddings.npy")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} chunks")


# ==========================================
# EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# NORMALIZE TEXT
# ==========================================

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# KEYWORD SEARCH
# ==========================================

def keyword_search(question):

    question = normalize(question)

    query_words = set(question.split())

    results = []

    for i, chunk in enumerate(chunks):

        text = normalize(chunk["text"])

        text_words = set(text.split())

        matches = query_words.intersection(text_words)

        score = len(matches)

        if score > 0:
            results.append((score, i))

    results.sort(reverse=True)

    return results


# ==========================================
# SEMANTIC SEARCH
# ==========================================

def semantic_search(question, top_k=5):

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        results.append(
            (
                float(distance),
                int(index_number)
            )
        )

    return results


# ==========================================
# HYBRID SEARCH
# ==========================================

def hybrid_search(question):

    keyword_results = keyword_search(question)

    semantic_results = semantic_search(
        question,
        top_k=5
    )

    scores = {}

    # --------------------------------------
    # Keyword results
    # --------------------------------------

    for keyword_score, index_number in keyword_results[:20]:

        scores[index_number] = (
            scores.get(index_number, 0)
            + keyword_score * 10
        )

    # --------------------------------------
    # Semantic results
    # --------------------------------------

    for rank, (distance, index_number) in enumerate(
        semantic_results
    ):

        semantic_score = 5 - rank

        scores[index_number] = (
            scores.get(index_number, 0)
            + semantic_score
        )

    # --------------------------------------
    # Sort final results
    # --------------------------------------

    final_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return final_results


# ==========================================
# BUILD TEXTBOOK CONTEXT
# ==========================================

def build_context(results, max_results=3):

    context_parts = []

    for rank, (index_number, score) in enumerate(
        results[:max_results],
        start=1
    ):

        chunk = chunks[index_number]

        context_parts.append(
            f"""
SOURCE {rank}

Subject: {chunk["subject"]}
Book: {chunk["book"]}
Page: {chunk["page"]}

Textbook text:
{chunk["text"]}
"""
        )

    return "\n".join(context_parts)


# ==========================================
# ASK LLAMA
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are 10th K AI, a Karnataka SSLC Class 10
textbook tutor.

Your job is to answer using ONLY the supplied
textbook evidence.

STRICT RULES:

1. Use ONLY the supplied textbook evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Do NOT add information that is not present
   in the textbook evidence.

5. Do NOT translate textbook terms unless the
   translation is explicitly present in the
   textbook evidence.

6. Do NOT explain the meaning of a term using
   your own knowledge.

7. When quoting textbook evidence, copy the
   wording exactly.

8. If the evidence does not contain enough
   information, say:
   "The textbook passage does not provide that
   information."

9. The answer must directly answer the student's
   question.

10. Keep the explanation short and suitable for
    a Class 10 student.

TEXTBOOK EVIDENCE:
------------------
{context}
------------------

STUDENT QUESTION:
{question}

Return EXACTLY this structure:

🎯 Answer:
Give the direct answer in one sentence.
Use only information supported by the evidence.

📚 Textbook Evidence:
Copy only the exact relevant sentence(s)
from the supplied textbook evidence.

💡 Simple Explanation:
Explain the answer simply using ONLY the
information present in the evidence.

Do NOT add definitions, translations,
historical facts, or other knowledge.

🔎 Question Pattern:
Describe what information the student needs
to identify to answer this question.

For example:
"Person → Title"
"Cause → Effect"
"Formula → Numerical calculation"
"Event → Person"
"Definition → Term"

Do NOT give a generic explanation about
words such as "who", "what", or "which".

🧠 Memory Trick:
Give ONE short memory association using the
names or terms in the evidence.

📝 Similar Practice Question:

Create ONE NEW question using ONLY the supplied
textbook evidence.

The new question MUST:

- test the SAME pattern as the student's question
- ask about a DIFFERENT fact from the evidence
- NOT repeat the student's question
- NOT use outside knowledge
- have an answer explicitly supported by the evidence

For a "Person → Title" question, choose a
DIFFERENT title from the evidence.

For example, if the evidence says:

"He had titles like Karnataka 'Kavichakravarthi',
'Aprathima veera', 'Thenkanaraja' and
'Navakoti Narayana'."

and the student asks:

"Who had the title Karnataka Kavichakravarthi?"

then a valid practice question would be:

"Which title was held by Chikkadevaraja Wodeyar
besides Karnataka Kavichakravarthi?"

Answer:
"Aprathima veera"

Do NOT use this exact example every time.
Generate the question from the supplied evidence.

Format:

Question:
<new question>

Answer:
<short answer supported by the evidence>

"""


    response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0
    }
)

    return response["message"]["content"]


# ==========================================
# MAIN PROGRAM
# ==========================================

question = input("\nAsk a question: ")

print("\nSearching textbook...")

results = hybrid_search(question)

# ------------------------------------------
# Show retrieved chunks
# ------------------------------------------

print("\nTop textbook matches:")

for rank, (index_number, score) in enumerate(
    results[:3],
    start=1
):

    chunk = chunks[index_number]

    print(
        f"\n{rank}. "
        f"{chunk['subject']} | "
        f"Page {chunk['page']} | "
        f"Score {score}"
    )


# ------------------------------------------
# Build context
# ------------------------------------------

context = build_context(
    results,
    max_results=3
)


# ------------------------------------------
# Generate answer
# ------------------------------------------

print("\nGenerating answer with Llama...\n")

answer = generate_answer(
    question,
    context
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)

print("10th K AI")

print("=" * 60)

print(answer)

print("\n" + "=" * 60)