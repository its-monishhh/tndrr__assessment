"""Back-of-envelope GPU memory to fine-tune a 7B model. Edit the assumptions, rerun, show the working."""
P = 7e9                       # parameters
d, ff, L, r = 4096, 11008, 32, 16   # Llama-7B-like shape; LoRA rank
ACT_GB = 3.0                  # ASSUMPTION: activations + CUDA overhead (batch ~4, seq ~512, grad checkpointing)
GB = 1e9

lora = L * (4 * r * (d + d) + 3 * r * (d + ff))   # q,k,v,o + gate,up,down, each adds r*(in+out)
lora_gb = lora * (4 + 4 + 8) / GB                 # fp32 weights + fp32 grads + Adam m,v (fp32)

full = {"weights bf16 (2 B/param)": 2 * P / GB, "grads bf16 (2 B/param)": 2 * P / GB,
        "Adam m+v fp32 (8 B/param)": 8 * P / GB, "fp32 master weights (4 B/param)": 4 * P / GB}
lora_run = {"frozen base bf16 (2 B/param)": 2 * P / GB, f"LoRA params+grads+Adam ({lora/1e6:.0f}M)": lora_gb}
qlora = {"frozen base NF4 (0.5 B/param)": 0.5 * P / GB,
         "quant constants, double-quant (~0.127 bit/param)": P * 0.127 / 8 / GB,
         f"LoRA params+grads+Adam ({lora/1e6:.0f}M)": lora_gb}

for name, parts in [("(a) Full fine-tune", full), ("(b) LoRA", lora_run), ("(c) QLoRA", qlora)]:
    print(f"\n{name}")
    for k, v in parts.items():
        print(f"  {k:55s}{v:8.2f} GB")
    print(f"  {'activations/overhead (assumed)':55s}{ACT_GB:8.2f} GB")
    print(f"  {'TOTAL':55s}{sum(parts.values()) + ACT_GB:8.2f} GB")
