import os
import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)


class GenerateEmail:
    def __init__(self, model: str):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
        self.model = model

    def _call_api(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_prompt(self, action, role, **kwargs):
        if action not in prompts:
            raise ValueError(f"Prompt action '{action}' not found in prompts.yaml")

        if role not in prompts[action]:
            raise ValueError(f"Role '{role}' missing under '{action}' in prompts.yaml")

        template = prompts[action][role]
        return template.format(**kwargs)

    @staticmethod
    def _clean_body(text: str) -> str:
        banned_starts = ("subject:", "dear", "regards:", "sincerely,", "best,")
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            low = line.lower().strip()
            if len(low.split()) < 4 and low.startswith(banned_starts):
                continue
            cleaned.append(line)

        return "\n".join(cleaned).strip() or text

    def generate(self, action: str, selected_text: str, tone_type: str = "Professional") -> str:
        args = {
            "selected_text": selected_text,
            "tone_type": tone_type,
        }

        system_prompt = (
            "You are a professional writing assistant.\n"
            "Rules:\n"
            "- Output ONLY the rewritten paragraph content.\n"
            "- Do NOT include subject lines, greetings, or signatures.\n"
            "- Do NOT provide explanations.\n"
            "- Return plain text only."
        )

        user_prompt = self.get_prompt(action, "user", **args)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = self._call_api(messages)
        return self._clean_body(raw_output)


