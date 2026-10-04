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
# STOP WORDS
# ==========================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "does",
    "do",
    "did",
    "how",
    "why",
    "which",
    "who",
    "called",
    "known",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its"
}


# ==========================================
# KEYWORD SEARCH
# ==========================================

def keyword_search(question, subject=None):

    question_normalized = normalize(question)

    query_words = [
        word
        for word in question_normalized.split()
        if word not in STOP_WORDS
    ]

    results = []

    for i, chunk in enumerate(chunks):

        # --------------------------------------
        # HARD SUBJECT FILTER
        # --------------------------------------

        if subject:

            if chunk["subject"].lower() != subject.lower():
                continue

        text = normalize(chunk["text"])

        text_words = set(text.split())

        # --------------------------------------
        # MATCH IMPORTANT WORDS
        # --------------------------------------

        matches = set(query_words).intersection(
            text_words
        )

        if not matches:
            continue

        # Number of matching important words
        score = len(matches)

        # --------------------------------------
        # WORD COVERAGE BONUS
        # --------------------------------------

        if len(query_words) > 0:

            coverage = (
                len(matches)
                / len(set(query_words))
            )

            score += coverage * 5

        # --------------------------------------
        # EXACT PHRASE BONUS
        # --------------------------------------

        important_phrase = " ".join(query_words)

        if (
            important_phrase
            and important_phrase in text
        ):
            score += 10

        results.append(
            (
                score,
                i
            )
        )

    # Highest score first
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results


# ==========================================
# SEMANTIC SEARCH
# ==========================================

def semantic_search(
    question,
    top_k=50,
    subject=None
):

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # --------------------------------------
    # FAISS SEARCH
    # --------------------------------------

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        # --------------------------------------
        # HARD SUBJECT FILTER
        # --------------------------------------

        if subject:

            if (
                chunks[index_number]["subject"].lower()
                != subject.lower()
            ):
                continue

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

def hybrid_search(
    question,
    subject=None
):

    keyword_results = keyword_search(
        question,
        subject
    )

    semantic_results = semantic_search(
        question,
        top_k=50,
        subject=subject
    )

    scores = {}


    # ======================================
    # KEYWORD SCORE
    # ======================================

    for keyword_score, index_number in (
        keyword_results[:50]
    ):

        scores[index_number] = (
            scores.get(index_number, 0)
            + keyword_score * 20
        )


    # ======================================
    # SEMANTIC SCORE
    # ======================================

    for rank, (
        distance,
        index_number
    ) in enumerate(
        semantic_results
    ):

        # Semantic ranking gives a smaller
        # contribution than keyword matching
        semantic_score = max(
            1,
            20 - rank
        )

        scores[index_number] = (
            scores.get(index_number, 0)
            + semantic_score
        )


    # ======================================
    # FINAL SORT
    # ======================================

    final_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return final_results


# ==========================================
# BUILD TEXTBOOK CONTEXT
# ==========================================

def build_context(
    results,
    max_results=1
):

    context_parts = []

    for rank, (
        index_number,
        score
    ) in enumerate(
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

def generate_answer(
    question,
    context
):

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

8. If the supplied textbook evidence does not contain enough information
to answer the student's question, DO NOT guess.

Say exactly:

"The textbook text I found does not provide enough information to answer this question. It may refer to a textbook diagram or figure."

Do not use outside knowledge to fill the missing information.

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


. Answer:

Give the direct answer in one sentence.

Use only information supported by the evidence.


Do NOT include a "📚 Textbook Evidence" section.
The application will display the textbook evidence separately.


 Simple Explanation:

Explain the answer simply using ONLY the
information present in the evidence.

Do NOT add definitions, translations,
historical facts, or other knowledge.


 Question Pattern:

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


 Memory Trick:

Give ONE short memory association using the
names or terms in the evidence.


 Similar Practice Question:

Create ONE NEW question using ONLY the supplied textbook evidence.

STRICT RULES:

- The answer MUST be directly stated in the supplied evidence.
- Do NOT make the answer from inference.
- Do NOT use outside knowledge.
- Do NOT ask about something that is only mentioned as a question in the textbook.
- The new question must test the SAME question pattern.
- The new question must ask about a DIFFERENT fact from the student's question.
- The answer must be clearly found in the evidence.

If there is not enough information to create a valid
similar practice question, write:

"Not enough textbook evidence to create a similar practice question."

For example, if the pattern is "Phenomenon → Name",
ask about another phenomenon whose name is explicitly present
in the supplied evidence.

"""

    # ======================================
    # OLLAMA
    # ======================================

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

