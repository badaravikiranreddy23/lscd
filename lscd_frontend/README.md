# LSCD Interactive Frontend
## Beyond Binary: Graded & Interpretable Lexical Semantic Change Detection

---

## What this is

A full web frontend for your LSCD NLP project.  
- Enter any word → get graded JSD score + before/after Flan-T5 definitions  
- Connects to your real `lscd_social` pipeline if available  
- Falls back to demo data if pipeline isn't set up yet  

---

## Folder structure

```
lscd_frontend/
├── app.py                  ← Flask backend (run this)
├── requirements.txt
├── templates/
│   └── index.html          ← Main page
└── static/
    ├── css/style.css       ← All styles
    └── js/main.js          ← All JS / API calls
```

---

## Setup

### Step 1 — Install Flask
```bash
cd lscd_frontend
pip install -r requirements.txt
```

### Step 2 — Point to your LSCD project
Open `app.py` and update this line (around line 18):
```python
LSCD_PROJECT_PATH = os.path.join(os.path.dirname(__file__), '..', 'lscd_social')
```
Change `'../lscd_social'` to wherever your `lscd_social` folder lives.

### Step 3 — Run
```bash
python app.py
```
Open browser: **http://localhost:5000**

---

## Two modes

| Mode | When | What happens |
|------|------|-------------|
| **Live pipeline** | `lscd_social` is reachable + corpus files exist | Runs real BERT + JSD + Flan-T5 |
| **Demo mode** | Pipeline not found (default on first run) | Uses pre-computed sample data for 8 words |

The mode is shown in the result UI (top-right of score banner).

---

## Adding new words (demo mode)

Open `app.py` and add an entry to the `DEMO_DATA` dict following the existing pattern:

```python
"yourword": {
    "score": 0.55,
    "verdict": "Moderate change",
    "clusters": [
        {"name": "Old sense name", "pre": 70, "post": 30},
        {"name": "New sense name", "pre": 30, "post": 70},
    ],
    "old_definition": {
        "era": "Pre-2020",
        "title": "Short title of old meaning",
        "text": "Longer description of the old sense.",
        "examples": ["Example sentence 1.", "Example sentence 2."]
    },
    "new_definition": {
        "era": "Post-2020",
        "title": "Short title of new meaning",
        "text": "Longer description of the new sense.",
        "examples": ["Example sentence 1.", "Example sentence 2."]
    }
}
```

---

## For your Thursday presentation

1. Run `python app.py`
2. Open browser, click **mask** → show score 0.83 + High change
3. Click **spread** → show score 0.24 + Low change
4. The contrast is your key result — pandemic words shift, control words don't
5. Show the "What this fixes in Aida & Bollegala" card at the bottom — it explains your contribution live

---

## API endpoints

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/` | GET | — | HTML page |
| `/api/analyze` | POST | `{"word": "mask"}` | Score, clusters, definitions |
| `/api/words` | GET | — | List of available demo words |
