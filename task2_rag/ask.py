"""python ask.py "your question" [--backend prompt|hf]  -> retrieve top chunks, then answer with an LLM."""
import argparse
from rag import *

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--doc", default="data/doc.pdf")
ap.add_argument("--size", type=int, default=250)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--backend", default="prompt", choices=["prompt", "hf"])
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
a = ap.parse_args()

full, starts = load_doc(a.doc)
chunks = chunk(full, a.size, int(a.size * 0.2))
emb = Embedder(a.model)
_, ids = Dense(emb.docs([c["text"] for c in chunks])).search(emb.queries([a.question]), a.k)
ctx = "\n\n".join(f"[page {page_of(chunks[i]['start'], starts)}] {chunks[i]['text']}" for i in ids[0])
prompt = ("Answer the question using only the context. If the answer is not in the context, say you don't know. "
          f"Mention the page numbers you used.\n\nContext:\n{ctx}\n\nQuestion: {a.question}\nAnswer:")
if a.backend == "prompt":
    print(prompt)  # paste into any LLM chat
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda()
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = lm.generate(**enc, max_new_tokens=250, do_sample=False)
    print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
