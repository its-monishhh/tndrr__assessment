"""python find.py "keyword" [doc]  -> every occurrence with page number and context, for copying gold passages."""
import re, sys
from rag import load_doc, page_of
kw = sys.argv[1]
full, starts = load_doc(sys.argv[2] if len(sys.argv) > 2 else "data/doc.pdf")
for m in list(re.finditer(re.escape(kw), full, re.IGNORECASE))[:15]:
    print(f"--- page {page_of(m.start(), starts)} ---")
    print(full[max(0, m.start() - 250): m.end() + 250], "\n")
