import os
import json
import time
import random
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- ENV SETUP ----------------
load_dotenv()

# ---------------- EXPERIMENTAL SYNTHETIC EMAIL GENERATOR ----------------
class ExperimentalSyntheticEmailGenerator:
    """
    Experimental generator for robustness & diversity testing.
    """

    def __init__(self, model: str):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.model = model

    def generate_email(
        self,
        email_id: int,
        topic: str,
        tone: str,
        length: str,
        structure_type: str,
        ambiguity_level: str,
        noise_level: str
    ) -> dict:

        system_prompt = (
            "You are generating synthetic emails for AI evaluation experiments.\n"
            "Rules:\n"
            "- Output ONLY valid JSON\n"
            "- No markdown or explanations\n"
            "- No greetings or signatures\n"
            "- Include at least one URL and one image reference\n"
            "- Define selected_excerpt copied verbatim from content\n"
        )

        user_prompt = f"""
Write a {tone.lower()} email about {topic}.
Length: {length}.

Structural format: {structure_type}
Ambiguity level: {ambiguity_level}
Noise level: {noise_level}

Return ONLY JSON:
{{
  "id": {email_id},
  "subject": "...",
  "content": "...",
  "selected_excerpt": "...",
  "technical_assets": ["https://portal.company.com/ticket/123", "<img src='https://cdn.company.com/image.png'>"],
  "structure_type": "{structure_type}",
  "ambiguity_level": "{ambiguity_level}",
  "noise_level": "{noise_level}"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(raw)
            parsed["id"] = email_id
            return parsed

        except Exception as e:
            # FALLBACK — NEVER DROP RECORDS
            return {
                "id": email_id,
                "subject": f"Fallback subject for {topic}",
                "content": f"This is a fallback synthetic email about {topic}.",
                "selected_excerpt": f"This is a fallback synthetic email about {topic}.",
                "technical_assets": [],
                "structure_type": "unknown",
                "ambiguity_level": "unknown",
                "noise_level": "unknown",
                "error": str(e)
            }


# ---------------- EXPERIMENT CONFIG ----------------
TOPICS = [
    "a project deadline delay",
    "a team meeting reminder",
    "customer issue resolution",
    "interview follow-up"
]

TONES = ["Professional", "Friendly", "Sympathetic"]
LENGTHS = ["Short", "Medium", "Long"]

STRUCTURES = ["paragraph", "bullets", "numbered", "mixed"]
AMBIGUITY = ["low", "medium", "high"]
NOISE = ["low", "medium", "high"]

TASKS = []
eid = 1
for t in TOPICS:
    for tone in TONES:
        for l in LENGTHS:
            TASKS.append((
                eid,
                t,
                tone,
                l,
                random.choice(STRUCTURES),
                random.choice(AMBIGUITY),
                random.choice(NOISE)
            ))
            eid += 1


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
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print("Worker crashed:", e)

    print(f"Collected {len(results)} records")

    results.sort(key=lambda x: x.get("id", float("inf")))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return time.time() - start


# ---------------- ENTRY POINT ----------------
def run_experiment():
    os.makedirs("synthetic_datasets", exist_ok=True)

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    generator = ExperimentalSyntheticEmailGenerator(model=model_name)

    out_file = "synthetic_datasets/synthetic_experimental.jsonl"

    print("Running EXPERIMENTAL synthetic generation...")
    duration = generate_parallel(generator, out_file)

    print("\nEXPERIMENT COMPLETE")
    print("-" * 40)
    print(f"Records generated : {len(TASKS)}")
    print(f"Output file       : {out_file}")
    print(f"Time taken        : {duration:.2f} seconds")
    print("-" * 40)


if __name__ == "__main__":
    run_experiment()


