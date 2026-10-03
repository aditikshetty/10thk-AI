import pymupdf
from pathlib import Path
import json
import re

pdf_folder = Path("textbooks")

MAX_WORDS = 80
OVERLAP_SENTENCES = 1

chunks = []


def clean_text(text):
    text = re.sub(r"\s+", " ", text)

    remove_patterns = [
        r"Government of Karnataka",
        r"Karnataka Textbook Society.*?",
        r"NOT TO BE REPUBLISHED",
        r"@KTBS",
    ]

    for pattern in remove_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):
    # Split after ., ! or ?
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


for pdf_path in pdf_folder.rglob("*.pdf"):

    print(f"\nReading: {pdf_path}")

    subject = pdf_path.parent.name

    with pymupdf.open(pdf_path) as document:

        print(f"Pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):

            text = page.get_text()

            if not text.strip():
                continue

            text = clean_text(text)

            if not text:
                continue

            sentences = split_sentences(text)

            current_sentences = []
            current_words = 0

            for sentence in sentences:

                sentence_words = len(sentence.split())

                # If adding this sentence makes the chunk too large,
                # save the current chunk first.
                if (
                    current_sentences
                    and current_words + sentence_words > MAX_WORDS
                ):

                    chunk_text = " ".join(current_sentences)

                    chunks.append({
                        "subject": subject,
                        "book": pdf_path.name,
                        "page": page_number,
                        "text": chunk_text
                    })

                    # Keep last sentence for overlap
                    current_sentences = current_sentences[
                        -OVERLAP_SENTENCES:
                    ]

                    current_words = sum(
                        len(s.split())
                        for s in current_sentences
                    )

                current_sentences.append(sentence)
                current_words += sentence_words

            # Save remaining sentences
            if current_sentences:

                chunk_text = " ".join(current_sentences)

                chunks.append({
                    "subject": subject,
                    "book": pdf_path.name,
                    "page": page_number,
                    "text": chunk_text
                })


print("\n--------------------------------")
print(f"Total chunks created: {len(chunks)}")
print("--------------------------------")

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )

print("Saved chunks to chunks.json")