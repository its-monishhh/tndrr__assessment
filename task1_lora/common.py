import json

SYSTEM = (
    "Convert the user's request into JSON with exactly these keys, in this order: "
    "intent (reminder | meeting | task), title (string), date (YYYY-MM-DD or null), "
    "time (HH:MM, 24-hour, or null), priority (low | medium | high | null). "
    "If no year is given, assume 2026. Dates written with slashes are DD/MM/YYYY. "
    "Output only the JSON object."
)


def build_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
