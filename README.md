AI-Powered Email Helper App

This repository contains an experimental AI-powered email assistance application built using Streamlit and Large Language Models (LLMs). The system focuses on email rewriting, tone and length modification, and automated quality evaluation.

The project starts with an original email dataset and extends to synthetic dataset generation to study model behavior and robustness.

Overview:

Key functionalities include:

.Email rewriting and enhancement

.Tone modification and length control

.LLM-based evaluation (completeness, faithfulness, robustness)

.Synthetic email generation for experimentation

App Details:

app.py:
Core Streamlit application operating on the original dataset. Acts as the baseline implementation for email editing and evaluation.

optional_app.py:
Alternate interface focused on synthetic dataset generation and comparative evaluation across different LLM roles.

Tech Stack:

.Python

.Streamlit

.OpenAI APIs
