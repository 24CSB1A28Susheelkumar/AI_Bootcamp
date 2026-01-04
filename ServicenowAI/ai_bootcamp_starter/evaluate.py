import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMEvaluator:
    def __init__(self, model: str):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
        self.model = model

    def _call_judge(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends prompts to the judge LLM.
        Temperature = 0 ensures deterministic, strict judging.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    # ---------------- FAITHFULNESS ----------------
    def judge_faithfulness(self, original: str, generated: str) -> str:
        system_prompt = """
You are an expert evaluator judging FAITHFULNESS of a rewritten text.

Definition:
- Meaning must be preserved.
- No hallucinated facts.
- No intent distortion.

Scoring:
5 = Perfectly faithful
4 = Minor wording changes
3 = Slight meaning drift
2 = Important meaning changes
1 = Major distortions
0 = Completely unfaithful

You MUST explain the score clearly.
"""

        user_prompt = f"""
ORIGINAL TEXT:
{original}

GENERATED TEXT:
{generated}

Respond in EXACT format:

Score: <0-5> / 5
Verdict: <Short label>

Reasoning:
- Bullet-point justification
- Mention added, altered, or removed facts
"""

        return self._call_judge(system_prompt, user_prompt)

    # ---------------- COMPLETENESS ----------------
    def judge_completeness(self, original: str, generated: str) -> str:
        system_prompt = """
You are an expert evaluator judging COMPLETENESS.

Definition:
- All key ideas must be retained.
- Minor shortening allowed.
- Missing important info lowers score.

Scoring:
5 = Fully complete
4 = One minor detail missing
3 = Some details missing
2 = Many details missing
1 = Barely complete
0 = Almost nothing preserved
"""

        user_prompt = f"""
ORIGINAL TEXT:
{original}

GENERATED TEXT:
{generated}

Respond in EXACT format:

Score: <0-5> / 5
Verdict: <Short label>

Reasoning:
- Bullet-point explanation
- Mention retained and missing points

Missing Elements:
- List missing ideas or "None"
"""

        return self._call_judge(system_prompt, user_prompt)

    # ---------------- ROBUSTNESS (NEW) ----------------
    def judge_robustness(self, original: str, generated: str) -> str:
        """
        Robustness evaluates:
        - Stability under ambiguity
        - Resistance to hallucination
        - Consistency and clarity
        """

        system_prompt = """
You are an expert evaluator judging ROBUSTNESS of a rewritten text.

Definition of Robustness:
- Output should remain stable and sensible.
- Should not hallucinate under ambiguity.
- Should avoid overconfidence or unsafe assumptions.

Scoring:
5 = Very robust and reliable
4 = Mostly robust, minor weaknesses
3 = Some instability or vague assumptions
2 = Fragile response
1 = Very unstable or misleading
0 = Unsafe or nonsensical

Explain your score clearly.
"""

        user_prompt = f"""
ORIGINAL TEXT:
{original}

GENERATED TEXT:
{generated}

Evaluate ROBUSTNESS and respond in EXACT format:

Score: <0-5> / 5
Verdict: <Short label>

Reasoning:
- Bullet-point explanation
- Mention ambiguity handling
- Mention hallucination risks
"""

        return self._call_judge(system_prompt, user_prompt)




