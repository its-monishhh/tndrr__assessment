import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import build_prompt, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", default=None, help="omit to evaluate the base model")
ap.add_argument("--data", default="data/test.jsonl")
ap.add_argument("--tag", required=True, help="name for the results files, e.g. base or lora")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()


def strict(s):
    try:
        return json.loads(s.strip())
    except Exception:
        return None


def lenient(s):  # pull out the first {...}; forgives ```json fences and chatter
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


rows = load_jsonl(a.data)
preds = []
for i in range(0, len(rows), 16):
    batch = rows[i:i + 16]
    enc = tok([build_prompt(tok, r["input"]) for r in batch], return_tensors="pt",
              padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

KEYS = ["intent", "title", "date", "time", "priority"]
n = len(rows)
strict_ok = lenient_ok = exact = 0
field = {k: 0 for k in KEYS}
records = []
for r, p in zip(rows, preds):
    gold = json.loads(r["output"])
    ps, pl = strict(p), lenient(p)
    strict_ok += ps is not None
    lenient_ok += pl is not None
    ok = isinstance(pl, dict) and pl == gold
    exact += ok
    for k in KEYS:
        field[k] += isinstance(pl, dict) and pl.get(k) == gold[k]
    records.append({"input": r["input"], "gold": gold, "raw_output": p, "exact": bool(ok)})

summary = {"tag": a.tag, "n": n,
           "strict_valid_json_pct": 100 * strict_ok / n,
           "lenient_valid_json_pct": 100 * lenient_ok / n,
           "exact_match_pct": 100 * exact / n,
           "field_accuracy_pct": {k: 100 * v / n for k, v in field.items()}}
os.makedirs("results", exist_ok=True)
json.dump(summary, open(f"results/{a.tag}_summary.json", "w"), indent=2)
json.dump(records, open(f"results/{a.tag}_predictions.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\nFirst failures:")
for rec in [x for x in records if not x["exact"]][:8]:
    print("IN  :", rec["input"]); print("GOLD:", json.dumps(rec["gold"])); print("PRED:", rec["raw_output"], "\n")
