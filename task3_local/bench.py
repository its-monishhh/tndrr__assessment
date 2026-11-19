"""Benchmark one Ollama model: tokens/second, memory, and saved outputs for the 10 prompts.
    python bench.py qwen2.5:1.5b-instruct-q4_K_M
    python bench.py qwen2.5:1.5b-instruct-fp16
Needs the Ollama app running (it serves http://localhost:11434)."""
import json, statistics, sys, urllib.request

HOST = "http://localhost:11434"


def call(path, payload=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def generate(model, prompt):
    return call("/api/generate", {"model": model, "prompt": prompt, "stream": False, "keep_alive": "10m",
                                  "options": {"temperature": 0, "seed": 0, "num_ctx": 2048, "num_predict": 200}})


model = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
prompts = json.load(open("prompts.json", encoding="utf-8"))

generate(model, "Say hi.")  # warm-up: loads the model so load time is not counted
ps = [m for m in call("/api/ps")["models"] if m["name"].startswith(model.split(":")[0])]
mem = {"size_bytes": ps[0]["size"], "size_vram_bytes": ps[0].get("size_vram", 0)} if ps else {}
print("loaded model memory (from /api/ps):", mem)

rows = []
for i, p in enumerate(prompts, 1):
    for r in range(runs):
        o = generate(model, p)
        tps = o["eval_count"] / (o["eval_duration"] / 1e9)
        rows.append({"prompt_id": i, "run": r, "prompt": p, "output": o["response"], "tokens": o["eval_count"],
                     "gen_tokens_per_s": tps,
                     "prompt_tokens_per_s": o["prompt_eval_count"] / (o["prompt_eval_duration"] / 1e9)
                     if o.get("prompt_eval_duration") else None})
        print(f"prompt {i} run {r}: {o['eval_count']} tokens, {tps:.1f} tok/s")

summary = {"model": model, "runs_per_prompt": runs, "memory": mem,
           "mean_gen_tokens_per_s": statistics.mean(x["gen_tokens_per_s"] for x in rows),
           "median_gen_tokens_per_s": statistics.median(x["gen_tokens_per_s"] for x in rows),
           "stdev_gen_tokens_per_s": statistics.pstdev(x["gen_tokens_per_s"] for x in rows)}
print(json.dumps(summary, indent=2))
tag = model.replace(":", "_").replace("/", "_")
json.dump({"summary": summary, "rows": rows}, open(f"results/{tag}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
