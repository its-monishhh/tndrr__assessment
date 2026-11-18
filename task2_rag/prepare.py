"""python prepare.py path/to/doc.pdf  ->  data/doc_text.txt (the exact text your questions must be copied from)."""
import sys
from rag import load_doc
full, starts = load_doc(sys.argv[1])
open("data/doc_text.txt", "w", encoding="utf-8").write(full)
print(f"pages: {len(starts)} | words: {len(full.split())} | chars: {len(full)}")
if len(starts) < 30 and sys.argv[1].lower().endswith(".pdf"):
    print("WARNING: the assignment needs at least 30 pages.")
if len(full.split()) / max(1, len(starts)) < 100:
    print("WARNING: very little text per page - the PDF may be scanned images (no extractable text).")
