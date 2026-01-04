import os
import json
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- ENV SETUP ----------------
load_dotenv()

# ---------------- SYNTHETIC EMAIL GENERATOR ----------------
class SyntheticEmailGenerator:
    def __init__(self, model: str):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
        self.model = model

    def generate_email(self, email_id: int, topic: str, tone: str, length: str) -> dict:
        system_prompt = (
            "You are a professional assistant generating synthetic email data.\n"
            "Rules:\n"
            "- Output ONLY valid JSON\n"
            "- No markdown, no explanations\n"
            "- JSON keys: id, subject, content\n"
            "- No greetings or signatures\n"
        )

        user_prompt = f"""
Write a {tone.lower()} email about {topic}.
Length: {length}.

Return ONLY JSON:
{{
  "id": {email_id},
  "subject": "...",
  "content": "..."
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8
        )

        return json.loads(response.choices[0].message.content.strip())


# ---------------- DATA CONFIG ----------------
TOPICS = [
    "a project deadline delay",
    "a team meeting reminder",
    "customer issue resolution",
    "interview follow-up"
]

TONES = ["Professional", "Friendly", "Sympathetic"]
LENGTHS = ["Short", "Medium", "Long"]

TASKS = []
eid = 1
for t in TOPICS:
    for tone in TONES:
        for l in LENGTHS:
            TASKS.append((eid, t, tone, l))
            eid += 1


# ---------------- SEQUENTIAL GENERATION ----------------
def generate_sequential(generator, output_path):
    start = time.time()
    results = []

    for task in TASKS:
        results.append(generator.generate_email(*task))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return time.time() - start


# ---------------- PARALLEL GENERATION ----------------
def generate_parallel(generator, output_path, max_workers=5):
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generator.generate_email, *task)
            for task in TASKS
        ]

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x["id"])

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return time.time() - start


# ---------------- MAIN BENCHMARK ----------------
def benchmark():
    os.makedirs("synthetic_datasets", exist_ok=True)

    generator = SyntheticEmailGenerator(model="gpt-4.1")

    seq_file = "synthetic_datasets/synthetic_sequential.jsonl"
    par_file = "synthetic_datasets/synthetic_parallel.jsonl"

    print(" Running SEQUENTIAL generation...")
    seq_time = generate_sequential(generator, seq_file)

    print(" Running PARALLEL generation...")
    par_time = generate_parallel(generator, par_file)

    print("\n PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"Sequential Time : {seq_time:.2f} seconds")
    print(f"Parallel Time   : {par_time:.2f} seconds")
    print(f"Speedup         : {seq_time / par_time:.2f}x faster")
    print("-" * 40)
    print(" Files generated:")
    print(f" - {seq_file}")
    print(f" - {par_file}")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    benchmark()


