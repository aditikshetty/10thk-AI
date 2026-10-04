import streamlit as st
import re

import rag
from rag import hybrid_search, build_context, generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="10th K AI",
    page_icon="🟡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   MAIN PAGE
   ============================================================ */

.stApp {
    background-color: #FFF8E1;
}

.main {
    background-color: #FFF8E1;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HEADER
   ============================================================ */

.brand {
    font-size: 42px;
    font-weight: 800;
    color: #3E2723;
    margin-bottom: 0;
}

.brand span {
    color: #FFB300;
}

.tagline {
    color: #6D5D4B;
    font-size: 17px;
    margin-bottom: 35px;
}


/* ============================================================
   QUESTION CARD
   ============================================================ */

.question-card {
    background-color: #FFFFFF;
    border: 1px solid #FFE082;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
}


/* ============================================================
   ANSWER CARD
   ============================================================ */

.answer-card {
    background-color: #FFFFFF;
    border: 1px solid #FFE082;
    border-radius: 20px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
}


/* ============================================================
   EVIDENCE CARD
   ============================================================ */

.evidence-card {
    background-color: #FFFFFF;
    border: 1px solid #FFE082;
    border-radius: 20px;
    padding: 25px;
    margin-top: 18px;
}


/* ============================================================
   PATTERN CARD
   ============================================================ */

.pattern-card {
    background-color: #FFF3BF;
    border: 1px solid #F4D35E;
    border-radius: 20px;
    padding: 25px;
    margin-top: 18px;
}


/* ============================================================
   MEMORY CARD
   ============================================================ */

.memory-card {
    background-color: #FFF9DF;
    border-left: 6px solid #FFB300;
    border-radius: 16px;
    padding: 22px;
    margin-top: 18px;
}


/* ============================================================
   PRACTICE CARD
   ============================================================ */

.practice-card {
    background-color: #FFFFFF;
    border: 2px solid #FFB300;
    border-radius: 20px;
    padding: 25px;
    margin-top: 18px;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #3E2723;
    margin-bottom: 12px;
}

.pattern-text {
    font-size: 27px;
    font-weight: 800;
    color: #3E2723;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background-color: #FFD54F;
    border-right: 1px solid #FFCA28;
}

[data-testid="stSidebar"] * {
    color: #3E2723;
}


/* ============================================================
   TEXT AREA
   ============================================================ */

textarea {
    background-color: #FFFFFF !important;
    color: #3E2723 !important;
    border: 1px solid #FFB300 !important;
    border-radius: 12px !important;
}


/* ============================================================
   SELECT BOX
   ============================================================ */

[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #FFB300;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    background-color: #FFB300;
    color: #3E2723;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    height: 45px;
}

.stButton > button:hover {
    background-color: #FFA000;
    color: #3E2723;
}


/* ============================================================
   GENERAL TEXT
   ============================================================ */

p,
label {
    color: #3E2723;
}

h1,
h2,
h3 {
    color: #3E2723;
}


/* ============================================================
   DIVIDER
   ============================================================ */

hr {
    border-color: #FFE082;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:800;
            color:#3E2723;
        ">
            🟡 10th K AI
        </div>

        <div style="
            color:#5D4037;
            margin-top:5px;
            margin-bottom:25px;
        ">
            Your Class 10 AI Study Tutor
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📚 Subject")

    subject = st.selectbox(
        "Choose your subject",
        [
            "All Subjects",
            "Mathematics",
            "Science",
            "Social Science"
        ]
    )

    st.markdown("---")

    st.markdown("### ✨ What I can do")

    st.markdown("""
    📖 Find your textbook answer

    🔎 Recognize the question pattern

    💡 Explain it simply

    🧠 Create a memory trick

    📝 Give you a similar question
    """)

    st.markdown("---")

    st.caption(
        "RAG • FAISS • Sentence Transformers • Ollama"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="brand">
        <span>10th K</span> AI
    </div>

    <div class="tagline">
        Understand the concept. Recognize the pattern.
        Solve with confidence.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    """
    <div class="question-card">
        <div class="section-title">
            👋 What are you studying today?
        </div>
    """,
    unsafe_allow_html=True
)

question = st.text_area(
    "Ask a question from your textbook",
    placeholder=(
        'Example: Who had the title '
        '"Karnataka Kavichakravarthi?"'
    ),
    height=100,
    label_visibility="collapsed"
)

ask = st.button(
    "✨ Ask 10th K AI",
    use_container_width=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_section(text, title, next_titles):

    if next_titles:

        next_pattern = "|".join(
            re.escape(t) for t in next_titles
        )

        pattern = (
            rf"{re.escape(title)}\s*:\s*"
            rf"(.*?)"
            rf"(?={next_pattern}\s*:|$)"
        )

    else:

        pattern = (
            rf"{re.escape(title)}\s*:\s*"
            rf"(.*)$"
        )

    match = re.search(
        pattern,
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


# ============================================================
# INITIALIZE VARIABLES
# ============================================================

results = []
answer = ""
context = ""


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask:

    if not question.strip():

        st.warning("Please enter a question first.")

    else:

        # ----------------------------------------------------
        # CONVERT UI SUBJECT TO CHUNK SUBJECT
        # ----------------------------------------------------

        if subject == "All Subjects":

            selected_subject = None

        elif subject == "Social Science":

            selected_subject = "social_science"

        else:

            selected_subject = subject.lower()


        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "📚 Searching your textbook..."
        ):

            results = hybrid_search(
                question,
                subject=selected_subject
            )

            context = build_context(
                results,
                max_results=1
            )


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        if results:

            with st.spinner(
                "🤖 10th K AI is thinking..."
            ):

                answer = generate_answer(
                    question,
                    context
                )

        else:

            st.warning(
                "No matching textbook content was found."
            )


        # ====================================================
        # EXTRACT AI SECTIONS
        # ====================================================

        if answer:

            answer_text = get_section(
                answer,
                "🎯 Answer",
                [
                    "💡 Simple Explanation",
                    "🔎 Question Pattern",
                    "🧠 Memory Trick",
                    "📝 Similar Practice Question"
                ]
            )

            explanation_text = get_section(
                answer,
                "💡 Simple Explanation",
                [
                    "🔎 Question Pattern",
                    "🧠 Memory Trick",
                    "📝 Similar Practice Question"
                ]
            )

            pattern_text = get_section(
                answer,
                "🔎 Question Pattern",
                [
                    "🧠 Memory Trick",
                    "📝 Similar Practice Question"
                ]
            )

            memory_text = get_section(
                answer,
                "🧠 Memory Trick",
                [
                    "📝 Similar Practice Question"
                ]
            )

            practice_text = get_section(
                answer,
                "📝 Similar Practice Question",
                []
            )


            # ====================================================
            # ANSWER CARD
            # ====================================================

            if answer_text:

                st.markdown(
                    f"""
                    <div class="answer-card">

                        <div class="section-title">
                            🎯 Answer
                        </div>

                        <div style="
                            font-size:19px;
                            line-height:1.7;
                            color:#3E2723;
                        ">
                            {answer_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # SIMPLE EXPLANATION CARD
            # ====================================================

            if explanation_text:

                st.markdown(
                    f"""
                    <div class="answer-card">

                        <div class="section-title">
                            💡 Simple Explanation
                        </div>

                        <div style="
                            font-size:17px;
                            line-height:1.7;
                            color:#3E2723;
                        ">
                            {explanation_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # QUESTION PATTERN CARD
            # ====================================================

            if pattern_text:

                st.markdown(
                    f"""
                    <div class="pattern-card">

                        <div class="section-title">
                            🔎 Question Pattern
                        </div>

                        <div class="pattern-text">
                            {pattern_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # MEMORY CARD
            # ====================================================

            if memory_text:

                st.markdown(
                    f"""
                    <div class="memory-card">

                        <div class="section-title">
                            🧠 Memory Trick
                        </div>

                        <div style="
                            font-size:18px;
                            line-height:1.6;
                            color:#3E2723;
                        ">
                            {memory_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # PRACTICE CARD
            # ====================================================

            if practice_text:

                st.markdown(
                    f"""
                    <div class="practice-card">

                        <div class="section-title">
                            📝 Similar Practice Question
                        </div>

                        <div style="
                            font-size:17px;
                            line-height:1.7;
                            color:#3E2723;
                        ">
                            {practice_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # TEXTBOOK EVIDENCE
            # ====================================================

           # ====================================================
# TEXTBOOK EVIDENCE
# ====================================================

if results:

    index_number, score = results[0]

    chunk = rag.chunks[index_number]

    st.markdown("### 📚 Textbook Evidence")

    with st.container(border=True):

        st.markdown("**Source 1**")

        st.markdown(
            f"**Subject:** {chunk['subject']}"
        )

        st.markdown(
            f"**Book:** {chunk['book']}"
        )

        st.markdown(
            f"**Page:** {chunk['page']}"
        )

        st.markdown("---")

        st.markdown(
            f"> {chunk['text']}"
        )

else:

    if ask and question.strip():

        st.info(
            "No matching textbook passage was found."
        )