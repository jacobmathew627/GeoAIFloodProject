import fitz
import re

print("Extracting LSTM context from original PDF...")
try:
    doc = fitz.open("Phase1Report_Fixed.pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if re.search(r'(?i)lstm', text):
            # Split by common sentence delimiters or newlines
            sentences = re.split(r'(?<=[.!?]) +|\n', text)
            for sentence in sentences:
                if re.search(r'(?i)lstm', sentence):
                    # Print sentence with some surrounding context if words are few
                    print(f"--- Page {page_num + 1} ---")
                    print(sentence.strip().replace('\n', ' '))
                    print("=" * 60)
except Exception as e:
    print("Error:", e)
