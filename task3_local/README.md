# Task 3 - local model, two quantisation levels (Ollama, CPU)

    ollama pull qwen2.5:1.5b-instruct-q4_K_M
    ollama pull qwen2.5:1.5b-instruct-fp16
    python bench.py qwen2.5:1.5b-instruct-q4_K_M
    python bench.py qwen2.5:1.5b-instruct-fp16

Same prompts, temperature 0, seed 0, context 2048, 200 max new tokens for both. Close other heavy apps while benchmarking.
Outputs and numbers are saved in results/.
