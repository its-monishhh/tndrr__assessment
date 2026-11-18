"""RAG building blocks, no frameworks: text extraction, chunking, embeddings, FAISS, BM25, rank fusion."""
import math, re, unicodedata, zlib
from bisect import bisect_right
from collections import Counter
import numpy as np


def normalise(t):
    t = unicodedata.normalize("NFKC", t)  # also turns ligatures like 'ﬁ' into 'fi'
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"-\s*\n\s*", "", t)       # re-join words hyphenated across line breaks
    return re.sub(r"\s+", " ", t).strip()


def load_doc(path):
    """Returns (full_text, page_start_offsets). All whitespace collapsed to single spaces."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    else:
        pages = [open(path, encoding="utf-8").read()]
    full, starts = "", []
    for p in pages:
        starts.append(len(full))
        full += normalise(p) + " "
    return full, starts


def page_of(offset, starts):
    return bisect_right(starts, offset)  # 1-based page number


def chunk(full, size, overlap):
    """Sliding window of `size` words, moving forward by size-overlap words. Keeps character offsets."""
    words = list(re.finditer(r"\S+", full))
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        w = words[i:i + size]
        out.append({"text": full[w[0].start():w[-1].end()], "start": w[0].start(), "end": w[-1].end()})
        if i + size >= len(words):
            break
    return out


class Embedder:
    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        self.name = name
        if name == "hash":  # offline stub for testing the pipeline only
            return
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)
        self.m.max_seq_length = 512  # inputs longer than this are silently truncated
        self.qprefix = "Represent this sentence for searching relevant passages: " if "bge" in name else ""

    def docs(self, texts):
        if self.name == "hash":
            v = np.zeros((len(texts), 512), dtype="float32")
            for i, t in enumerate(texts):
                for w in re.findall(r"[a-z0-9]+", t.lower()):
                    v[i, zlib.crc32(w.encode()) % 512] += 1
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return self.m.encode(texts, normalize_embeddings=True, batch_size=32,
                             show_progress_bar=False).astype("float32")

    def queries(self, texts):
        if self.name == "hash":
            return self.docs(texts)
        return self.docs([self.qprefix + t for t in texts])


class Dense:
    """Exact nearest-neighbour search. Vectors are unit length, so inner product = cosine similarity."""
    def __init__(self, vecs):
        import faiss
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, qvecs, k):
        return self.index.search(qvecs, k)


def tokenize(t):
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg, self.k1, self.b = self.len.mean(), k1, b
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                f = d.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

    def search(self, query, k):
        return np.argsort(-self.scores(query))[:k]


def rrf(rankings, k=60, top=10):
    """Reciprocal rank fusion: merge several ranked lists using only the ranks."""
    sc = Counter()
    for r in rankings:
        for rank, i in enumerate(r):
            sc[i] += 1.0 / (k + rank + 1)
    return [i for i, _ in sc.most_common(top)]


def _fold(ch):
    low = ch.lower()
    return low if len(low) == 1 else ch


def find_span(full, passage):
    """Locate the gold passage in the document text, ignoring whitespace and letter case, so that
    PDF-extraction spacing quirks ('ISO /IEC', 'GOVERN ,') cannot break the match."""
    p = "".join(_fold(c) for c in normalise(passage) if not c.isspace())
    idx = [i for i, c in enumerate(full) if not c.isspace()]
    stripped = "".join(_fold(full[i]) for i in idx)
    j = stripped.find(p)
    if j == -1 or not p:
        return None
    return idx[j], idx[j + len(p) - 1] + 1


def overlaps(chunk, span, min_overlap=0.5):
    """A chunk is a hit if it covers at least `min_overlap` of the gold passage."""
    ov = max(0, min(chunk["end"], span[1]) - max(chunk["start"], span[0]))
    return ov >= min_overlap * (span[1] - span[0])
