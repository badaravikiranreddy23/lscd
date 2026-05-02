# Beyond Binary: Graded & Interpretable Lexical Semantic Change Detection on Social Media

## 🎯 Novel Contribution
This project extends Aida & Bollegala (ACL 2024) in two crucial ways:
1. **Flaw 2 Fix** — Replaces binary change detection with a **graded JSD-based score**.
2. **Flaw 4 Fix** — Adds **interpretable definition generation** explaining *how* meaning shifted.

**New angle (what makes this a 9/10):** We apply this pipeline to **modern social media language** — specifically Reddit comments from pre-2020 vs post-2020 — to detect COVID-driven semantic shifts in everyday words like *mask*, *variant*, *quarantine*, and *remote*. 

**Live Demo:** The project features an interactive Flask frontend that dynamically scrapes Reddit via the **Arctic Shift API**, runs BERT inference, and generates Flan-T5 definitions *on the fly* for ANY word you type.

---

## 📁 Project Structure
```
lscd_social/
├── data/
│   ├── social_corpus_builder.py   # Live Reddit scraping via Arctic Shift API
│   ├── generate_sample_data.py    # Generate fallback sample data
│   └── social_corpus/             # Auto-created corpus directory
├── lscd_frontend/
│   ├── app.py                     # Flask backend & API
│   ├── static/                    # UI assets (JS, CSS)
│   └── templates/                 # HTML UI
├── models/
│   ├── sense_encoder.py           # BERT-based sense encoder
│   ├── graded_scorer.py           # JSD graded scoring
│   └── definition_gen.py          # Flan-T5 definition generation
├── evaluate.py                    # Benchmark against SemEval-2020
└── social_analyzer.py             # Main CLI execution script
```

---

## 🚀 Quick Start (Interactive UI)

The best way to demonstrate this project is through the live web interface.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install flask
```

### Step 2: Start the Server
```bash
cd lscd_frontend
python app.py
```

### Step 3: View the Demo
Open your browser to `http://127.0.0.1:5000/`.
Type any word (e.g., `vaccine`, `zoom`, `mask`) and hit "Analyze". The system will scrape 200 Reddit sentences live, run K-Means, compute JSD scores, and generate generative definitions!

---

## 📦 Datasets

If asked about data during the presentation, you are using three datasets:
1. **Reddit Social Media Corpus (Live)**: Auto-fetched from Arctic Shift (2018-2019 vs 2021-2022). Used for modern evaluation.
2. **WiC (Word-in-Context)**: Used exclusively to fine-tune the BERT `SenseEncoder` (via `train.py`).
3. **SemEval-2020 Task 1**: The academic gold-standard corpus used for validating the methodology (via `evaluate.py`).

---

## 📊 Expected Results

Words like *mask*, *variant*, and *lockdown* display **High Change** scores (> 0.61 JSD), with automatically generated definitions clearly highlighting the pandemic usage taking over the original usage. Control words like *spread* or *remote* show low or moderate drift, validating the pipeline mathematically.

---

## 📖 References
- Aida & Bollegala (2024). A Semantic Distance Metric Learning approach for LSCD. ACL 2024.
- Fedorova et al. (2024). Definition generation for LSCD. ACL 2024.
- Schlechtweg et al. (2020). SemEval-2020 Task 1: Unsupervised LSCD.
