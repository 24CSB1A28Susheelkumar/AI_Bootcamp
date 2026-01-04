import streamlit as st
import json
import os
from dotenv import load_dotenv
from evaluate import LLMEvaluator
from generate import GenerateEmail

# ---------------- ENV ----------------
load_dotenv()

st.set_page_config(
    page_title="Experimental Email Evaluation Studio",
    page_icon="🧪",
    layout="wide"
)

MODEL_GEN = os.getenv("AZURE_GPT_41_DEPLOYMENT", "gpt-4.1")
MODEL_JUDGE = os.getenv("AZURE_GPT_4O_MINI_DEPLOYMENT", "gpt-4o-mini")

generator = GenerateEmail(model=MODEL_GEN)
evaluator = LLMEvaluator(model=MODEL_JUDGE)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_experimental_data():
    path = "C:\\Users\\User\\Desktop\\ServicenowAI\\synthetic_datasets\\synthetic_experimental.jsonl"
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


data = load_experimental_data()

if not data:
    st.error("synthetic_experimental.jsonl not found or empty.")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧪 Experiment Controls")

    structure_filter = st.selectbox(
        "Structure Type",
        ["All"] + sorted({d["structure_type"] for d in data})
    )

    ambiguity_filter = st.selectbox(
        "Ambiguity Level",
        ["All"] + sorted({d["ambiguity_level"] for d in data})
    )

    noise_filter = st.selectbox(
        "Noise Level",
        ["All"] + sorted({d["noise_level"] for d in data})
    )

    action = st.radio(
        "AI Action",
        ["shorten", "lengthen", "tone"],
        format_func=lambda x: x.capitalize()
    )

    tone_choice = None
    if action == "tone":
        tone_choice = st.selectbox(
            "Target Tone",
            ["Professional", "Friendly", "Sympathetic"]
        )

    filtered = []
    for d in data:
        if structure_filter != "All" and d["structure_type"] != structure_filter:
            continue
        if ambiguity_filter != "All" and d["ambiguity_level"] != ambiguity_filter:
            continue
        if noise_filter != "All" and d["noise_level"] != noise_filter:
            continue
        filtered.append(d)

    if not filtered:
        st.warning("No records match selected filters.")
        st.stop()

    record = st.selectbox(
        "Select Email",
        filtered,
        format_func=lambda x: f'ID {x["id"]}'
    )

# ---------------- MAIN VIEW ----------------
st.title("🧪 Experimental Email Evaluation Studio")
st.caption("Controlled stress testing for structure, ambiguity & noise")

# Metadata
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Structure", record["structure_type"])
with m2:
    st.metric("Ambiguity", record["ambiguity_level"])
with m3:
    st.metric("Noise", record["noise_level"])

st.divider()

# ---------------- EMAIL PANELS ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Synthetic Input Email")
    st.markdown(f"**Subject:** {record['subject']}")
    st.text_area(
        "Email Body",
        record["content"],
        height=340,
        disabled=True
    )

    st.markdown("#### 🎯 Selected Excerpt (Reference)")
    st.info(record["selected_excerpt"])

with col2:
    st.markdown("### 📤 Model Output")
    gen_key = f"gen_{record['id']}"

    if gen_key not in st.session_state:
        st.session_state[gen_key] = ""

    st.text_area(
        "Generated / Model Response",
        key=gen_key,
        height=420
    )

# ---------------- GENERATION ----------------
def run_generation():
    source_text = record["content"]

    if action == "tone":
        result = generator.generate(
            "tone",
            source_text,
            tone_type=tone_choice
        )
    else:
        result = generator.generate(action, source_text)

    st.session_state[gen_key] = result


st.divider()

colA, colB, colC = st.columns([2, 2, 6])

with colA:
    st.button(
        "✨ Apply AI",
        use_container_width=True,
        on_click=run_generation
    )

with colB:
    st.button(
        "↩ Reset Output",
        use_container_width=True,
        on_click=lambda: st.session_state.update({gen_key: ""})
    )

with colC:
    st.markdown(
        f"""
        **Mode:** `{action.capitalize()}`  
        **Record ID:** `{record["id"]}`
        """
    )

# ---------------- EVALUATION ----------------
st.divider()
st.header("⚖️ LLM-as-a-Judge Evaluation")

if st.button("Run Evaluation", type="primary"):
    if not st.session_state[gen_key].strip():
        st.warning("Generate or paste model output first.")
    else:
        with st.spinner("Evaluating..."):
            faith = evaluator.judge_faithfulness(
                record["selected_excerpt"],
                st.session_state[gen_key]
            )

            comp = evaluator.judge_completeness(
                record["selected_excerpt"],
                st.session_state[gen_key]
            )

            rob = evaluator.judge_robustness(
                record["content"],
                st.session_state[gen_key]
            )

        st.success("Evaluation Complete")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Faithfulness")
            st.text_area("", faith, height=260)
        with c2:
            st.markdown("#### Completeness")
            st.text_area("", comp, height=260)
        with c3:
            st.markdown("#### Robustness")
            st.text_area("", rob, height=260)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Experimental Evaluation Studio | Structural × Ambiguity × Noise Stress Testing"
)

