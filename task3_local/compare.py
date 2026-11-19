"""Writes results/side_by_side.md with the 4-bit and fp16 answers for each prompt, for your manual comparison."""
import glob, json

files = {p: json.load(open(p, encoding="utf-8"))["rows"] for p in glob.glob("results/*.json")}
q4 = next(k for k in files if "q4_K_M" in k)
fp = next(k for k in files if "fp16" in k)
with open("results/side_by_side.md", "w", encoding="utf-8") as o:
    for a, b in zip(files[q4], files[fp]):
        if a["run"] != 0:  # first run of each prompt only
            continue
        o.write(f"## Prompt {a['prompt_id']}: {a['prompt']}\n\n**4-bit:**\n{a['output']}\n\n**fp16:**\n{b['output']}\n\n---\n\n")
print("wrote results/side_by_side.md")
