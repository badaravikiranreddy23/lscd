# Beyond Binary: Graded & Interpretable Lexical Semantic Change Detection on Social Media

## 🎯 Novel Contribution
This project extends Aida & Bollegala (ACL 2024) in two ways:
1. **Flaw 2 Fix** — Replaces binary change detection with a **graded JSD-based score**
2. **Flaw 4 Fix** — Adds **interpretable definition generation** explaining *how* meaning shifted

**New angle (what makes this a 9/10):** We apply this pipeline to **modern social media language** — specifically Reddit comments from pre-2020 vs post-2020 — to detect COVID-driven semantic shifts in everyday words like *mask*, *variant*, *remote*, *viral*.

---

## 📁 Project Structure
```
lscd_social/
├── data/
│   ├── social_corpus_builder.py   # Fetch from Reddit via Pushshift API
│   ├── generate_sample_data.py    # Generate sample data (no API needed)
│   └── social_corpus/             # Auto-created corpus directory
├── models/
│   ├── sense_encoder.py           # BERT-based sense encoder (Stage 1)
│   ├── graded_scorer.py           # JSD graded scoring (Flaw 2 fix)
│   └── definition_gen.py          # Flan-T5 definition generation (Flaw 4 fix)
├── utils/
│   ├── data_loader.py             # SemEval + general data loading
│   └── evaluation.py              # Spearman rho + F1 metrics
├── social_analyzer.py             # MAIN script — social media pipeline
├── visualize_results.py           # Plot change scores bar chart
├── train.py                       # Fine-tune sense encoder on WiC
├── evaluate.py                    # Benchmark on SemEval-2020
└── requirements.txt
```

---

## 📦 Datasets Required

### For the Social Media Experiment (main contribution)
| Dataset | What it is | How to get it |
|---|---|---|
| Reddit via Pushshift | Pre/post 2020 comments | Auto-fetched by social_corpus_builder.py |
| Sample data (testing) | Hand-crafted sentences per word | Run generate_sample_data.py (no API) |

### For Training the Sense Encoder
| Dataset | What it is | Link |
|---|---|---|
| WiC (Word-in-Context) | Sentence pairs, same/diff sense | https://pilehvar.github.io/wic/ |

### For Benchmarking (SemEval comparison)
| Dataset | What it is | Link |
|---|---|---|
| SemEval-2020 Task 1 | Historical LSCD benchmark | https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/ |

---

## 🚀 Quick Start (No API needed)

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Generate sample data
```bash
python data/generate_sample_data.py
```

### Step 3: Analyze a single word
```bash
python social_analyzer.py --word mask --use_sample_data
```

### Step 4: Analyze ALL words + generate report
```bash
python social_analyzer.py --all --use_sample_data
```

### Step 5: Visualize results
```bash
python visualize_results.py
```

---

## 🌐 Full Reddit Corpus (Better Results)
```bash
# Fetch real Reddit data (takes ~20 min, free, no auth needed)
python data/social_corpus_builder.py

# Then run analysis
python social_analyzer.py --all
```

---

## 🏋️ Train the Sense Encoder (Optional but recommended)
```bash
# Download WiC first from https://pilehvar.github.io/wic/
python train.py --wic_dir data/wic
```

---

## 📊 Expected Results

Words like mask, variant, lockdown should show HIGH change scores (new COVID meanings dominate post-2020). Words like remote show MODERATE change (work-from-home sense gains ground). This validates that our graded score captures real-world semantic drift.

---

## 📖 References
- Aida & Bollegala (2024). A Semantic Distance Metric Learning approach for LSCD. ACL 2024.
- Fedorova et al. (2024). Definition generation for LSCD. ACL 2024.
- Schlechtweg et al. (2020). SemEval-2020 Task 1: Unsupervised LSCD.
