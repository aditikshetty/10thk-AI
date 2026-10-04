import json
import os
import re

import numpy as np
import faiss
import streamlit as st
import ollama

from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD TEXTBOOK DATA
# ============================================================

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks")


# ============================================================
# LOAD EMBEDDINGS + FAISS
# ============================================================

embeddings = np.load("embeddings.npy")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} chunks")


# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

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


# ============================================================
# STOP WORDS
# ============================================================

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


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(question, subject=None):

    question_normalized = normalize(question)

    query_words = [
        word
        for word in question_normalized.split()
        if word not in STOP_WORDS
    ]

    results = []

    for i, chunk in enumerate(chunks):

        if subject:

            if chunk["subject"].lower() != subject.lower():
                continue

        text = normalize(chunk["text"])

        text_words = set(text.split())

        matches = set(query_words).intersection(
            text_words
        )

        if not matches:
            continue

        score = len(matches)

        if len(query_words) > 0:

            coverage = (
                len(matches)
                / len(set(query_words))
            )

            score += coverage * 5

        important_phrase = " ".join(query_words)

        if (
            important_phrase
            and important_phrase in text
        ):

            score += 10

        results.append(
            (score, i)
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results


# ============================================================
# SEMANTIC SEARCH
# ============================================================

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

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

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


# ============================================================
# HYBRID SEARCH
# ============================================================

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

    # Keyword scores

    for keyword_score, index_number in (
        keyword_results[:50]
    ):

        scores[index_number] = (
            scores.get(index_number, 0)
            + keyword_score * 20
        )

    # Semantic scores

    for rank, (
        distance,
        index_number
    ) in enumerate(
        semantic_results
    ):

        semantic_score = max(
            1,
            20 - rank
        )

        scores[index_number] = (
            scores.get(index_number, 0)
            + semantic_score
        )

    final_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return final_results


# ============================================================
# BUILD CONTEXT
# ============================================================

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


# ============================================================
# OLLAMA CLIENT
# ============================================================

def get_ollama_client():

    """
    Hybrid Ollama setup.

    LOCAL:
        Uses Ollama running on the user's computer.

    CLOUD:
        Uses Ollama Cloud through ollama.com.
    """

    # --------------------------------------------------------
    # Try Streamlit Cloud secret first
    # --------------------------------------------------------

    api_key = None

    try:

        api_key = st.secrets.get(
            "OLLAMA_API_KEY"
        )

    except Exception:

        api_key = None

    # --------------------------------------------------------
    # Also check environment variable
    # --------------------------------------------------------

    if not api_key:

        api_key = os.getenv(
            "OLLAMA_API_KEY"
        )

    # --------------------------------------------------------
    # CLOUD MODE
    # --------------------------------------------------------

    if api_key:

        print(
            "Using Ollama Cloud"
        )

        return ollama.Client(
            host="https://ollama.com",
            headers={
                "Authorization":
                    f"Bearer {api_key}"
            }
        )

    # --------------------------------------------------------
    # LOCAL MODE
    # --------------------------------------------------------

    print(
        "Using local Ollama"
    )

    return ollama.Client(
        host="http://localhost:11434"
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

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

8. If the supplied textbook evidence does not
   contain enough information to answer the
   student's question, say:

"The textbook text I found does not provide enough
information to answer this question. It may refer
to a textbook diagram or figure."

Do not use outside knowledge to fill the
missing information.

9. The answer must directly answer the
   student's question.

10. Keep the explanation short and suitable
    for a Class 10 student.


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

Examples:

"Person → Title"
"Cause → Effect"
"Formula → Numerical calculation"
"Event → Person"
"Definition → Term"

Do NOT give a generic explanation about
words such as "who", "what", or "which".


🧠 Memory Trick:

Give ONE short memory association using
the names or terms in the evidence.


📝 Similar Practice Question:

Create ONE NEW question using ONLY the
supplied textbook evidence.

The new question MUST:

- test the SAME pattern as the student's question
- ask about a DIFFERENT fact from the evidence
- NOT repeat the student's question
- NOT use outside knowledge
- have an answer explicitly supported by the evidence

For a "Person → Title" question, choose a
DIFFERENT title from the evidence.


Format:

Question:
<new question>

Answer:
<short answer supported by the evidence>
"""

    # ========================================================
    # GET OLLAMA CLIENT
    # ========================================================

    client = get_ollama_client()


    # ========================================================
    # SELECT MODEL
    # ========================================================

    # Local model
    local_model = "llama3.2"

    # Ollama Cloud model
    cloud_model = "gpt-oss:120b"


    # Detect whether API key exists

    api_key = None

    try:

        api_key = st.secrets.get(
            "OLLAMA_API_KEY"
        )

    except Exception:

        api_key = None

    if not api_key:

        api_key = os.getenv(
            "OLLAMA_API_KEY"
        )


    if api_key:

        selected_model = cloud_model

    else:

        selected_model = local_model


    # ========================================================
    # CALL OLLAMA
    # ========================================================

    try:

        response = client.chat(

            model=selected_model,

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

        return response[
            "message"
        ]["content"]


    except Exception as e:

        return f"""
⚠️ AI generation failed.

Ollama connection error:

{str(e)}

If you are running locally, make sure Ollama
is running and that llama3.2 is installed.

If you are running on Streamlit Cloud,
make sure OLLAMA_API_KEY is configured in
Streamlit Secrets.
"""