import streamlit as st
import json
import os
from dotenv import load_dotenv
from generate import GenerateEmail
from evaluate import LLMEvaluator

load_dotenv()

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Email Studio",
    page_icon="✉️",
    layout="wide"
)

MODEL_NAME1 = os.getenv("AZURE_GPT_41_DEPLOYMENT", "gpt-4.1")
MODEL_NAME2 = os.getenv("AZURE_GPT_4O_MINI_DEPLOYMENT", "gpt-4o-mini")

generator = GenerateEmail(model=MODEL_NAME1)
evaluator = LLMEvaluator(model=MODEL_NAME2)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_emails_from_jsonl(file_path):
    path = file_path if os.path.exists(file_path) else f"datasets/{file_path}"
    if not os.path.exists(path):
        return {"emails": {}, "ids": []}

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    emails = {item["id"]: item for item in data}
    return {"emails": emails, "ids": list(emails.keys())}

data_files = {
    "shorten": load_emails_from_jsonl("shorten.jsonl"),
    "lengthen": load_emails_from_jsonl("lengthen.jsonl"),
    "tone": load_emails_from_jsonl("tone.jsonl"),
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## Controls")

    action = st.radio(
        "Choose Action",
        ["shorten", "lengthen", "tone"],
        format_func=lambda x: x.capitalize()
    )

    email_ids = data_files[action]["ids"]
    emails = data_files[action]["emails"]

    if not email_ids:
        st.error(f"{action}.jsonl missing or empty.")
        st.stop()

    email_id = st.selectbox("Select Email", email_ids)

    if action == "tone":
        tone_choice = st.selectbox(
            "Tone Style",
            ["Friendly", "Sympathetic", "Professional"]
        )
    else:
        tone_choice = None

# ---------------- SESSION STATE ----------------
email = emails[email_id]
original_body = email.get("content", "")

orig_key = f"original_{action}_{email_id}"
gen_key = f"generated_{action}_{email_id}"

if orig_key not in st.session_state:
    st.session_state[orig_key] = original_body

if gen_key not in st.session_state:
    st.session_state[gen_key] = ""

# ---------------- HEADER ----------------
st.markdown("## AI Email Studio")
st.divider()

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**From**")
        st.write(email.get("sender", "-"))
    with col2:
        st.markdown("**Subject**")
        st.write(email.get("subject", "-"))

# ---------------- EMAIL PANELS ----------------
st.markdown("### Email Comparison")

col1, col2 = st.columns(2)

with col1:
    st.text_area(
    label="",
    value=st.session_state[orig_key],
    height=320,
    key=f"readonly_{orig_key}",
    help="Original email (read-only reference)"
)


with col2:
    st.markdown("#### Generated Email")
    st.text_area(
        label="",
        key=gen_key,
        height=320,
        placeholder="Click 'Apply AI' to generate output..."
    )

# ---------------- ACTIONS ----------------
def run_ai():
    content = st.session_state[orig_key].strip()
    if not content:
        return

    if action == "tone":
        generated = generator.generate(
            "tone",
            content,
            tone_type=tone_choice
        )
    else:
        generated = generator.generate(action, content)

    if generated.startswith("Error"):
        st.error(generated)
        return

    st.session_state[gen_key] = generated

def reset_generated():
    st.session_state[gen_key] = ""

st.divider()

col1, col2, col3 = st.columns([2, 2, 6])

with col1:
    st.button(
        "✨ Apply AI",
        use_container_width=True,
        on_click=run_ai
    )

with col2:
    st.button(
        "↩ Reset Output",
        use_container_width=True,
        on_click=reset_generated
    )

with col3:
    st.markdown(
        f"""
        **Current Mode:** `{action.capitalize()}`  
        **Email ID:** `{email_id}`
        """
    )

# ---------------- EVALUATION ----------------
st.divider()
st.markdown("### Model Evaluation (LLM-as-a-Judge)")

if st.button("Evaluate Response"):
    original_text = st.session_state[orig_key]
    generated_text = st.session_state[gen_key]

    if not generated_text.strip():
        st.warning("Generate an email before evaluation.")
    else:
        with st.spinner("Evaluating with LLM-as-a-Judge..."):
            faithfulness_report = evaluator.judge_faithfulness(
                original_text,
                generated_text
            )

            completeness_report = evaluator.judge_completeness(
                original_text,
                generated_text
            )

            robustness_report = evaluator.judge_robustness(
                original_text,
                generated_text
            )

        st.success("Evaluation Complete")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Faithfulness")
            st.text_area(
                "Faithfulness Report",
                value=faithfulness_report,
                height=260
            )

        with col2:
            st.markdown("#### Completeness")
            st.text_area(
                "Completeness Report",
                value=completeness_report,
                height=260
            )
        
        with col3:
            st.markdown("#### Robustness")
            st.text_area(
                "Robustness Report",
                value=robustness_report,
                height=260
            )








