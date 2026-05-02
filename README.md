# Lexical Semantic Change Detection (LSCD) via Graded Jensen-Shannon Divergence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the implementation of a novel, end-to-end pipeline for **Lexical Semantic Change Detection (LSCD)**. The project extends the binary Semantic Distance Metric Learning approach proposed by Aida & Bollegala (ACL 2024) by introducing continuous, graded change scoring and generative interpretability.

To demonstrate the efficacy of this pipeline, we apply it to modern social media corpora (Reddit), successfully quantifying and explaining the semantic drift of vocabulary triggered by the COVID-19 pandemic (e.g., *mask*, *quarantine*, *variant*).

---

## 🎯 Core Academic Contributions
1. **Continuous Graded Scoring (Flaw Resolution):** Addressed the limitations of binary change classification by implementing a continuous Jensen-Shannon Divergence (JSD) metric. This allows for the nuanced detection of *partial* semantic drift rather than forcing a 1/0 label.
2. **Generative Interpretability:** Integrated `google/flan-t5-base` to automatically generate natural language definitions for clustered usage embeddings, transitioning the pipeline from a "black-box" scorer to an interpretable linguistic tool.
3. **Live API Integration:** Developed a robust, dynamic scraping module utilizing the **Arctic Shift API** to perform real-time chronological inference on arbitrary user-provided vocabulary.

---

## 🏗️ System Architecture

The pipeline consists of four primary stages:
1. **Data Acquisition:** Dynamic retrieval of temporally separated social media text (Pre-2020 vs. Post-2020) via the Arctic Shift API.
2. **Contextual Encoding:** A custom `SenseEncoder` utilizing fine-tuned `bert-base-uncased` to extract contextualized word embeddings.
3. **Distributional Scoring:** K-Means clustering of embeddings followed by Jensen-Shannon Divergence calculation to output a graded drift score (0.0 to 1.0).
4. **Interactive UI / Interpretability:** A Flask-based frontend that visualizes the sense clusters and prompts a T5 LLM to define the emerging usages.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+
- At least 8GB RAM (for local LLM inference)

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/badaravikiranreddy23/lscd.git
cd lscd
pip install -r requirements.txt
```

### Running the Live Interactive Dashboard
The primary way to interact with the pipeline is through the local web dashboard.
```bash
cd lscd_frontend
python app.py
```
*Navigate to `http://127.0.0.1:5000` in your web browser. You can type any arbitrary word to initiate a live scraping and semantic drift analysis.*

---

## 📊 Evaluation & Datasets

This methodology was evaluated and benchmarked against standard NLP corpora:
- **Sense Encoder Fine-Tuning:** Trained on the **Word-in-Context (WiC)** dataset to enforce sense-boundary awareness.
- **Methodology Validation:** Evaluated against the **SemEval-2020 Task 1 Subtask 2 (English)** benchmark using Spearman's rank correlation.
- **Social Media Application:** Applied chronologically to real-world Reddit comments (2018-2019 vs 2021-2022).

**Expected Results:** Pandemic-affected vocabulary (*mask*, *variant*, *lockdown*) consistently yields **High Change** scores (JSD > 0.61) with the T5 model correctly isolating the new medical/societal contexts. Control vocabulary (*remote*, *spread*) yields appropriately lower JSD scores, validating the pipeline's precision.

---

## 📖 Key References
- Aida, T., & Bollegala, D. (2024). *A Semantic Distance Metric Learning approach for Lexical Semantic Change Detection.* Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024).
- Fedorova, V., et al. (2024). *Definition Generation for Lexical Semantic Change Detection.* ACL 2024.
- Schlechtweg, D., et al. (2020). *SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection.*
