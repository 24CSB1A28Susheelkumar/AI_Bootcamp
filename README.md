AI-Powered Email Helper App

This repository contains an experimental AI-powered email assistance system built using Streamlit and Large Language Models (LLMs). The application focuses on generating, rewriting, and evaluating emails across multiple dimensions such as completeness, faithfulness, tone, and robustness.

The project also explores synthetic dataset generation and LLM-based evaluation pipelines to analyze model behavior on real-world email editing tasks.

Project Overview

The application supports:

Email generation and rewriting

Tone modification (formal, polite, concise, etc.)

Length control (shorten or expand)

Automated evaluation using LLM-based judges

Synthetic dataset generation for robustness testing

The system was developed in phases, starting with a real/original dataset and later extended to include synthetically generated datasets for experimentation.

File Descriptions
app.py

Main Streamlit application

Works with the original (real) email dataset

Provides the baseline interface for:

Email rewriting

Tone and length transformations

LLM-based evaluation (completeness, faithfulness, robustness)

This file represents the core implementation of the system.

optional_app.py

Alternative Streamlit interface

Designed for synthetic dataset experimentation

Includes:

Synthetic email generation

Evaluation on model-generated data

Comparative analysis across different LLM roles (generator vs evaluator)

This interface is optional and intended for experimental and analytical purposes.

data/

datasets/: Contains real or manually curated email samples

synthetic_datasets/: Contains LLM-generated emails used for testing and evaluation

evaluators/

Implements LLM-based evaluation logic

Metrics include:

Completeness

Faithfulness

Tone consistency

Robustness

utils/

Shared utility functions

Prompt templates

Common helper logic used across applications

Experimental Focus

This project investigates:

Behavioral differences between real and synthetic datasets

Trade-offs between completeness, faithfulness, and robustness

Impact of prompt design and model selection on email quality

Technology Stack

Python

Streamlit

Azure OpenAI / OpenAI APIs

dotenv

Concurrent futures for parallel evaluation

How to Run

Clone the repository

git clone <repository-url>
cd <repository-name>


Install dependencies

pip install -r requirements.txt


Configure environment variables (.env)

AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_api_key


Run the application

streamlit run app.py


For synthetic dataset experiments:

streamlit run optional_app.py

Notes

AI assistance was used primarily for boilerplate generation, rewriting, and evaluation

All experiments were conducted for learning and analytical purposes.
