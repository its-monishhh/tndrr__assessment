# Task 2 - RAG from scratch (no LangChain / LlamaIndex)

Only: pypdf (text extraction), sentence-transformers (embedding model), FAISS, NumPy. BM25 and rank fusion are hand-written in rag.py.

    pip install -q pypdf sentence-transformers faiss-cpu numpy
    python prepare.py data/doc.pdf          # writes data/doc_text.txt, prints page count
    python find.py "some keyword"           # locate passages, copy them exactly for questions.json
    python eval_retrieval.py                # recall@1/3/5 for chunk sizes 100 and 250 words
    python eval_retrieval.py --hybrid       # optional: adds BM25 and BM25+dense (RRF)
    python ask.py "your question" --backend prompt

Hit definition: a retrieved chunk is a hit if it covers >= 50% of the gold passage's characters (--min_overlap).
