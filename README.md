# LLM Engineering Assignment

**Author:** Monish U  
**Repository Structure:** Three standalone task folders (`task1_lora/`, `task2_rag/`, `task3_local/`) with reproducible scripts, datasets, and saved evaluation artifacts.

---

## Repository Layout

```text
tndrr__assessment-main/
├── README.md
├── task1_lora/                  # Task 1: LoRA fine-tuning on Qwen2.5-0.5B-Instruct
│   ├── common.py                # Shared prompt & JSON schema helpers
│   ├── gen_data.py              # Generates 1,000 train / 100 val / 100 test examples
│   ├── train.py                 # LoRA training script (answer-only loss masking)
│   ├── evaluate.py              # Strict/lenient JSON & per-field evaluation
│   ├── memory_calc.py           # 7B GPU memory estimator (Full / LoRA / QLoRA)
│   ├── data/                    # train.jsonl, val.jsonl, test.jsonl
│   ├── adapter/                 # Saved LoRA adapter weights & tokenizer files
│   └── results/                 # base_summary.json, lora_summary.json, train_stats.json, predictions
├── task2_rag/                   # Task 2: Retrieval-Augmented Generation (RAG) from scratch
│   ├── prepare.py               # Extracts text from the 47-page NIST AI RMF 1.0 PDF
│   ├── rag.py                   # From-scratch chunking, dense embeddings, BM25 & hybrid RRF
│   ├── eval_retrieval.py        # Evaluates Recall@1, Recall@3, Recall@5 across chunk sizes
│   ├── find.py                  # Interactive chunk retrieval inspection
│   ├── ask.py                   # End-to-end question answering over retrieved passages
│   ├── questions.json           # 15 author-written questions with ground-truth source passages
│   ├── data/                    # doc.pdf (NIST AI 100-1) & extracted doc_text.txt
│   └── results/                 # retrieval_summary.json & retrieval_details.json
└── task3_local/                 # Task 3: Local inference benchmark & quantisation analysis
    ├── bench.py                 # Benchmarks generation speed & memory via Ollama
    ├── compare.py               # Generates side-by-side output comparison across 10 prompts
    ├── prompts.json             # 10 evaluation prompts (reasoning, code, JSON, translation, QA)
    └── results/                 # fp16 & q4_K_M JSON logs + side_by_side.md
```

---

## Summary of Results

### Task 1 — Fine-Tuning with LoRA (`task1_lora/`)
* **Task:** Convert plain-English requests into a fixed JSON schema (`intent`, `title`, `date`, `time`, `priority`) with `YYYY-MM-DD` dates, 24-hour `HH:MM` times, and `null` for missing fields.
* **Model & LoRA Setup:** `Qwen/Qwen2.5-0.5B-Instruct` with `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, all 7 linear projections, `lr=2e-4`, `3` epochs (`8,798,208` / `502,830,976` trainable params = **1.75%**; peak GPU memory **9.04 GB**; runtime **649.1 s**; final loss **4.41e-05**).

| Metric (n = 100 test) | Base Model | LoRA Fine-Tuned |
|---|---|---|
| Strict Valid JSON | 97.0% | **100.0%** |
| Exact Match | 0.0% | **78.0%** |
| Intent / Title Accuracy | 70.0% / 15.0% | **84.0% / 89.0%** |
| Date / Time / Priority Accuracy | 49.0% / 49.0% / 35.0% | **100.0% / 100.0% / 100.0%** |

**How to run:**
```bash
cd task1_lora
pip install -q transformers peft datasets accelerate
python evaluate.py --tag base
python train.py
python evaluate.py --tag lora --adapter adapter
python memory_calc.py
```

---

### Task 2 — Retrieval (RAG) from Scratch (`task2_rag/`)
* **Setup:** Built from scratch using dense vector embeddings, custom BM25, Reciprocal Rank Fusion (`hybrid_rrf`), and NumPy (without LangChain or LlamaIndex) over the 47-page **NIST AI Risk Management Framework (AI RMF 1.0)** (`data/doc.pdf`), evaluated on **15 hand-written questions** (`questions.json`).

| Chunking Config | Chunks | Dense R@1 / R@3 / R@5 | BM25 R@1 / R@3 / R@5 | Hybrid RRF R@1 / R@3 / R@5 |
|---|---|---|---|---|
| 100 words (20 overlap) | 196 | 0.600 / 0.733 / **0.933** | 0.400 / 0.467 / 0.533 | 0.600 / 0.733 / 0.733 |
| 250 words (50 overlap) | 79 | **0.667** / 0.733 / **0.933** | **0.533 / 0.667 / 0.733** | 0.600 / **0.867 / 0.933** |

**How to run:**
```bash
cd task2_rag
python prepare.py
python eval_retrieval.py
python ask.py "What are the four functions of the AI RMF Core?"
```

---

### Task 3 — Running a Model Locally & Optimisation (`task3_local/`)
* **Setup:** Benchmarked `Qwen2.5-1.5B-Instruct` locally via Ollama in 16-bit (`fp16`) vs. 4-bit (`q4_K_M`) across 10 prompts (`prompts.json`).

| Model Setting | Footprint (`size_bytes`) | Mean Gen Speed | Median Gen Speed | Std. Dev. |
|---|---|---|---|---|
| `qwen2.5:1.5b-instruct-fp16` | 3.22 GB (`3,219,946,208` B) | 9.05 tok/s | 9.78 tok/s | 2.43 tok/s |
| `qwen2.5:1.5b-instruct-q4_K_M` | **1.11 GB** (`1,112,318,934` B) | **25.55 tok/s (2.82×)** | **26.14 tok/s** | 2.46 tok/s |

**How to run:**
```bash
cd task3_local
python bench.py
python compare.py
```
