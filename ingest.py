import fitz
from pathlib import Path

pdf_folder = Path("textbooks")

pdf_files = list(pdf_folder.rglob("*.pdf"))

if not pdf_files:
    print("No PDF found. Add a textbook to the textbooks folder.")
else:
    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path}")

        with fitz.open(pdf_path) as document:
            print(f"Pages: {len(document)}")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text()

                if text.strip():
                    print(f"Page {page_number}:")
                    print(text[:300])
                    break