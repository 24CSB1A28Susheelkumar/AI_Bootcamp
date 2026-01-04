import json
import os
import re
from dotenv import load_dotenv
from generate import GenerateEmail
from evaluate import LLMEvaluator
import matplotlib.pyplot as plt

# ---------------- ENV ----------------
load_dotenv()

MODEL_GEN = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
MODEL_JUDGE = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

generator = GenerateEmail(model=MODEL_GEN)
evaluator = LLMEvaluator(model=MODEL_JUDGE)

DATASETS = {
    "SHORTEN": ("C:\\Users\\User\\Desktop\\ServicenowAI\\ai_bootcamp_starter\\datasets\\shorten.jsonl", "shorten"),
    "LENGTHEN": ("C:\\Users\\User\\Desktop\\ServicenowAI\\ai_bootcamp_starter\\datasets\\lengthen.jsonl", "lengthen"),
    "TONE": ("C:\\Users\\User\\Desktop\\ServicenowAI\\ai_bootcamp_starter\\datasets\\tone.jsonl", "tone"),
}

# ---------------- HELPERS ----------------
def load_jsonl(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_score(text):
    if not text:
        return None
    match = re.search(r"([0-5])\s*(/5)?", text)
    return int(match.group(1)) if match else None


def plot_averages(name, faith, comp, rob):
    plt.figure()
    plt.bar(
        ["Faithfulness", "Completeness", "Robustness"],
        [sum(faith) / len(faith), sum(comp) / len(comp), sum(rob) / len(rob)]
    )
    plt.title(f"{name} – Average Evaluation Scores")
    plt.ylim(0, 5)
    plt.ylabel("Score (0–5)")
    plt.show()


def plot_trends(name, faith, comp, rob):
    plt.figure()
    plt.plot(faith, label="Faithfulness")
    plt.plot(comp, label="Completeness")
    plt.plot(rob, label="Robustness")
    plt.title(f"{name} – Score Trend Across Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Score (0–5)")
    plt.ylim(0, 5)
    plt.legend()
    plt.show()


def evaluate_dataset(records, name, action, max_samples=10):
    faith, comp, rob = [], [], []

    print(f"\n{name} RESULTS")
    print("-" * 40)

    used = 0

    for rec in records:
        if used >= max_samples:
            break

        original = rec.get("content", "").strip()
        if not original:
            continue

        try:
            if action == "tone":
                generated = generator.generate(
                    "tone", original, tone_type="Professional"
                )
            else:
                generated = generator.generate(action, original)
        except Exception as e:
            print("⚠️ Generation failed:", e)
            continue

        try:
            f = extract_score(
                evaluator.judge_faithfulness(original, generated)
            )
            c = extract_score(
                evaluator.judge_completeness(original, generated)
            )
            r = extract_score(
                evaluator.judge_robustness(original, generated)
            )
        except Exception as e:
            print("⚠️ Evaluation failed:", e)
            continue

        if None not in (f, c, r):
            faith.append(f)
            comp.append(c)
            rob.append(r)
            used += 1
            print(f"✓ Evaluated sample {used}")

    if used == 0:
        print("⚠️ No valid samples evaluated")
        return

    print(f"Faithfulness Avg : {sum(faith) / used:.2f}")
    print(f"Completeness Avg : {sum(comp) / used:.2f}")
    print(f"Robustness Avg   : {sum(rob) / used:.2f}")
    print(f"Samples Used    : {used}")

    # 📊 VISUALIZATION
    plot_averages(name, faith, comp, rob)
    plot_trends(name, faith, comp, rob)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    for name, (path, action) in DATASETS.items():
        records = load_jsonl(path)
        evaluate_dataset(records, name, action)


