import argparse, json, os, time
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from common import build_prompt, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--out", default="adapter")
a = ap.parse_args()

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                         target_modules=TARGETS, task_type="CAUSAL_LM"))
model.print_trainable_parameters()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())


def encode(ex):
    p = tok(build_prompt(tok, ex["input"]), add_special_tokens=False)["input_ids"]
    t = tok(ex["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    # loss only on the answer tokens: prompt positions are masked with -100
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


ds = Dataset.from_list(load_jsonl("data/train.jsonl")).map(encode, remove_columns=["input", "output"])


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
args = TrainingArguments(
    output_dir="runs", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=10, save_strategy="no", bf16=bf16, fp16=not bf16, report_to="none",
    remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
stats = {"trainable_params": trainable, "total_params": total,
         "trainable_pct": 100 * trainable / total, "train_seconds": time.time() - t0,
         "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
         "final_logged_loss": next((l["loss"] for l in reversed(trainer.state.log_history) if "loss" in l), None),
         "config": vars(a)}
os.makedirs("results", exist_ok=True)
json.dump(stats, open("results/train_stats.json", "w"), indent=2)
print(stats)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
