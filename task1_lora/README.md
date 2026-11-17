# Task 1 – LoRA fine-tuning (text → JSON)

Run in Colab/Kaggle (GPU runtime):

    pip install -q transformers peft datasets accelerate
    python gen_data.py
    python evaluate.py --tag base                      # baseline, no adapter
    python train.py                                    # writes adapter/ and results/train_stats.json
    python evaluate.py --tag lora --adapter adapter    # fine-tuned model on the held-out test set
    python memory_calc.py                              # 7B memory estimate

Hyperparameter choices are compared on `data/val.jsonl` (`--data data/val.jsonl`), never on test.
