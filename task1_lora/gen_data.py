"""Synthetic dataset: plain-English request -> fixed JSON schema.
Test set deliberately uses UNSEEN titles, UNSEEN phrasings, and a few UNSEEN
date/time styles, so the held-out score measures generalisation, not recall."""
import argparse, datetime as dt, json, os, random

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

VERB_TRAIN = ["call the dentist", "submit the lab report", "pay the electricity bill",
              "water the plants", "buy groceries", "renew my passport", "book train tickets",
              "email the professor", "back up my laptop", "pick up the parcel",
              "return the library books", "charge the power bank"]
VERB_TEST = ["collect my transcripts", "refill my prescription", "update the project readme",
             "clean the fish tank", "fix the bike tyre", "send the invoice to the client"]
TOPIC_TRAIN = ["the Q3 budget", "sprint planning", "the thesis outline", "hackathon prep",
               "the vendor contract", "onboarding of new interns", "the website redesign",
               "lab safety review", "the placement drive", "the cloud migration",
               "the mentor feedback", "the demo rehearsal"]
TOPIC_TEST = ["the capstone review", "the data privacy audit", "the customer churn analysis",
              "the alumni meetup", "server cost optimisation", "the roadmap workshop"]

TEMPLATES = {
    "reminder": ["remind me to {t}", "set a reminder to {t}", "don't let me forget to {t}"],
    "meeting": ["schedule a meeting about {t}", "book a call to discuss {t}",
                "set up a meeting for {t}"],
    "task": ["add a task to {t}", "new task: {t}", "put '{t}' on my to-do list"],
}
TEMPLATES_TEST_EXTRA = {
    "reminder": ["can you ping me to {t}"],
    "meeting": ["I need a sync on {t}"],
    "task": ["log '{t}' as something I have to do"],
}
PRIO = {"high": ["it's urgent", "high priority", "top priority"],
        "medium": ["medium priority", "normal priority"],
        "low": ["low priority", "no rush"]}


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"


def fmt_date(d, rng, extra):
    styles = ["md", "dm", "dm_ord", "mdy", "slash", "iso"]
    if extra:
        styles += ["short", "the_of"]
    s = rng.choice(styles)
    M = MONTHS[d.month - 1]
    return {
        "md": f"{M} {d.day}", "dm": f"{d.day} {M}", "dm_ord": f"{ordinal(d.day)} {M}",
        "mdy": f"{M} {ordinal(d.day)}, {d.year}", "slash": f"{d.day:02d}/{d.month:02d}/{d.year}",
        "iso": d.isoformat(), "short": f"{M[:3]} {d.day}",
        "the_of": f"the {ordinal(d.day)} of {M}",
    }[s]


def fmt_time(h, m, rng, extra):
    h12, ap = (h % 12 or 12), ("am" if h < 12 else "pm")
    styles = ["12h", "24h", "words"] + (["dot", "military"] if extra else [])
    s = rng.choice(styles)
    if s == "12h":
        return f"{h12}{':%02d' % m if m else ''}{rng.choice(['', ' '])}{ap}"
    if s == "24h":
        return f"{h:02d}:{m:02d}"
    if s == "words":
        if h == 12 and m == 0:
            return "noon"
        if m == 0:
            part = "morning" if h < 12 else "afternoon" if h < 17 else "evening"
            return f"{h12} in the {part}"
        return f"{h12}:{m:02d} {ap}"
    if s == "dot":
        return f"{h12}.{m:02d} {ap}"
    return f"{h:02d}{m:02d} hours"


def make(rng, split):
    test = split == "test"
    intent = rng.choice(["reminder", "meeting", "task"])
    pool = (TOPIC_TEST if test else TOPIC_TRAIN) if intent == "meeting" else (VERB_TEST if test else VERB_TRAIN)
    title = rng.choice(pool)
    tpls = TEMPLATES[intent] + (TEMPLATES_TEST_EXTRA[intent] if test else [])
    has_date, has_time = rng.random() < 0.8, rng.random() < 0.7
    prio = rng.choice([None, None, "low", "medium", "high"])
    d = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(365))
    h, m = rng.randrange(6, 22), rng.choice([0, 15, 30, 45])
    dtxt = fmt_date(d, rng, test) if has_date else None
    ttxt = fmt_time(h, m, rng, test) if has_time else None
    if dtxt and ttxt:
        when = rng.choice([f" on {dtxt} at {ttxt}", f" at {ttxt} on {dtxt}"])
    elif dtxt:
        when = f" on {dtxt}"
    elif ttxt:
        when = f" at {ttxt}"
    else:
        when = ""
    ptxt = f", {rng.choice(PRIO[prio])}" if prio else ""
    text = rng.choice(tpls).format(t=title) + when + ptxt
    if rng.random() < 0.2:
        text = "please " + text
    out = {"intent": intent, "title": title, "date": d.isoformat() if has_date else None,
           "time": f"{h:02d}:{m:02d}" if has_time else None, "priority": prio}
    return {"input": text, "output": json.dumps(out)}


def build(n, split, rng, seen):
    rows = []
    while len(rows) < n:
        ex = make(rng, split)
        if ex["input"] in seen:
            continue
        seen.add(ex["input"])
        rows.append(ex)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng, seen = random.Random(a.seed), set()
    os.makedirs("data", exist_ok=True)
    for name, n, split in [("train", a.n_train, "train"), ("val", a.n_val, "train"), ("test", a.n_test, "test")]:
        rows = build(n, split, rng, seen)
        with open(f"data/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(name, len(rows), "| example:", rows[0])
