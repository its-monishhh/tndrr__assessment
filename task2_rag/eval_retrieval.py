import argparse, json, sys
from rag import *

ap = argparse.ArgumentParser()
ap.add_argument("--doc", default="data/doc.pdf")
ap.add_argument("--questions", default="questions.json")
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250], help="chunk sizes in words")
ap.add_argument("--overlap_frac", type=float, default=0.2)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
ap.add_argument("--min_overlap", type=float, default=0.5)
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--hybrid", action="store_true", help="also evaluate BM25 and BM25+dense (RRF)")
a = ap.parse_args()

full, starts = load_doc(a.doc)
qs = [q for q in json.load(open(a.questions, encoding="utf-8")) if "FILL" not in q["gold_passage"]]
for q in qs:
    q["span"] = find_span(full, q["gold_passage"])
    if q["span"] is None:
        sys.exit(f"Gold passage not found in the document text:\n  {q['question']}\n"
                 "Copy it exactly from data/doc_text.txt (find.py helps).")
print(f"{len(qs)} questions | model {a.model}")

emb = Embedder(a.model)
qvecs = emb.queries([q["question"] for q in qs])
rows, details = [], []
for size in a.sizes:
    overlap = int(size * a.overlap_frac)
    chunks = chunk(full, size, overlap)
    K = min(max(10, max(a.ks)), len(chunks))
    _, ids = Dense(emb.docs([c["text"] for c in chunks])).search(qvecs, K)
    methods = {"dense": [[int(i) for i in r] for r in ids]}
    if a.hybrid:
        bm = BM25([c["text"] for c in chunks])
        methods["bm25"] = [[int(i) for i in bm.search(q["question"], K)] for q in qs]
        methods["hybrid_rrf"] = [rrf([methods["dense"][j], methods["bm25"][j]], top=K) for j in range(len(qs))]
    for name, ranked in methods.items():
        ranks = []
        for q, r in zip(qs, ranked):
            rank = next((j + 1 for j, ci in enumerate(r) if overlaps(chunks[ci], q["span"], a.min_overlap)), None)
            ranks.append(rank)
            details.append({"chunk_words": size, "method": name, "question": q["question"],
                            "gold_page": page_of(q["span"][0], starts), "first_hit_rank": rank,
                            "top": [{"page": page_of(chunks[ci]["start"], starts), "text": chunks[ci]["text"][:300]}
                                    for ci in r[:5]]})
        row = {"chunk_words": size, "overlap_words": overlap, "n_chunks": len(chunks), "method": name}
        for k in a.ks:
            row[f"recall@{k}"] = round(sum(1 for x in ranks if x and x <= k) / len(qs), 3)
        rows.append(row)
        print(row)

json.dump(rows, open("results/retrieval_summary.json", "w"), indent=2)
json.dump(details, open("results/retrieval_details.json", "w"), indent=2)
kmax = max(a.ks)
print(f"\nDense misses at k={kmax}:")
for d in details:
    if d["method"] == "dense" and not (d["first_hit_rank"] and d["first_hit_rank"] <= kmax):
        print(f"\n[{d['chunk_words']}w] Q: {d['question']}  (gold on page {d['gold_page']}, first hit rank: {d['first_hit_rank']})")
        for t in d["top"][:3]:
            print(f"   p{t['page']}: {t['text'][:160]}...")
