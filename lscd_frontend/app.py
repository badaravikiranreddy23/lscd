"""
LSCD Frontend - Flask Backend
Connects the web UI to the actual LSCD pipeline.
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import sys
import traceback

app = Flask(__name__)

# Add parent directory to path so we can import the LSCD modules
# Adjust this path to point to your lscd_social project folder
LSCD_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LSCD_PROJECT_PATH)

# --- Try to import real pipeline ---
PIPELINE_AVAILABLE = False
try:
    from models.sense_encoder import SenseEncoder
    from models.graded_scorer import compute_graded_change_score
    from models.definition_gen import DefinitionGenerator, generate_change_explanation
    import torch
    PIPELINE_AVAILABLE = True
    print("[INFO] Real LSCD pipeline loaded successfully.")
except ImportError as e:
    print(f"[WARN] Could not load real pipeline: {e}")
    print("[INFO] Running in DEMO mode with sample data.")


# --- Sample data for demo/fallback mode ---
DEMO_DATA = {
    "mask": {
        "score": 0.83,
        "verdict": "High change",
        "clusters": [
            {"name": "Costume / disguise", "pre": 72, "post": 8},
            {"name": "Medical / PPE",       "pre": 18, "post": 81},
            {"name": "Metaphor / hide",     "pre": 10, "post": 11}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Mask as disguise or costume",
            "text": "A covering worn over the face for concealment, disguise, or theatrical performance. Associated with Halloween, masquerades, or performance arts.",
            "examples": [
                "She wore a venetian mask at the ball.",
                "The robber's mask hid his identity."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Mask as protective medical device",
            "text": "A face covering (N95, surgical, cloth) worn to prevent transmission of airborne pathogens. Became central to public health mandates during COVID-19.",
            "examples": [
                "Masks are required indoors per the mandate.",
                "She double-masked on the subway."
            ]
        }
    },
    "lockdown": {
        "score": 0.79,
        "verdict": "High change",
        "clusters": [
            {"name": "Prison / security",          "pre": 85, "post": 9},
            {"name": "Public health restriction",  "pre": 5,  "post": 87},
            {"name": "Device / software lock",     "pre": 10, "post": 4}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Lockdown as a security protocol",
            "text": "An emergency security measure in prisons or public spaces restricting movement to contain a threat or dangerous situation.",
            "examples": [
                "The prison went into lockdown after the escape.",
                "School lockdown drill was practiced today."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Lockdown as a public health measure",
            "text": "Government-imposed restrictions on movement and business activity to reduce viral spread. Became the defining word of pandemic-era governance worldwide.",
            "examples": [
                "The third national lockdown began in January.",
                "Lockdown fatigue is affecting mental health."
            ]
        }
    },
    "variant": {
        "score": 0.76,
        "verdict": "High change",
        "clusters": [
            {"name": "Generic variation",       "pre": 78, "post": 12},
            {"name": "Viral / genomic strain",  "pre": 9,  "post": 83},
            {"name": "Product / design version","pre": 13, "post": 5}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Variant as a general alternative form",
            "text": "A different or modified version of something — a spelling variant, product variant, or design variant. Primarily used in linguistics and manufacturing.",
            "examples": [
                "Color variants are available in the catalog.",
                "'Grey' is a British spelling variant of 'gray'."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Variant as a viral mutation strain",
            "text": "A genetically distinct version of a virus (e.g. Delta, Omicron) with different transmissibility or immune escape properties. Dominated public health discourse.",
            "examples": [
                "The new variant evades prior immunity.",
                "Scientists are monitoring the XBB variant."
            ]
        }
    },
    "essential": {
        "score": 0.68,
        "verdict": "High change",
        "clusters": [
            {"name": "Important / necessary",  "pre": 80, "post": 42},
            {"name": "Essential worker",       "pre": 5,  "post": 52},
            {"name": "Essential oils",         "pre": 15, "post": 6}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Essential as fundamentally necessary",
            "text": "Something absolutely necessary or indispensable. Often used in phrases like 'essential ingredient' or 'essential quality.'",
            "examples": [
                "Water is essential for survival.",
                "Patience is an essential quality for teachers."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Essential as a worker classification",
            "text": "Designating workers in critical sectors (healthcare, grocery, transit) who continued working during lockdowns. Carried strong social connotations of sacrifice and inequality.",
            "examples": [
                "Essential workers deserve hazard pay.",
                "She was classified as essential and couldn't stay home."
            ]
        }
    },
    "bubble": {
        "score": 0.71,
        "verdict": "High change",
        "clusters": [
            {"name": "Soap / physical bubble", "pre": 55, "post": 14},
            {"name": "Social / support bubble","pre": 4,  "post": 62},
            {"name": "Financial bubble",       "pre": 23, "post": 16},
            {"name": "Sports bubble",          "pre": 3,  "post": 8}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Bubble as a physical or financial sphere",
            "text": "A thin-walled sphere of liquid enclosing air, or metaphorically an unsustainable market expansion. Also used in children's play contexts.",
            "examples": [
                "The housing bubble burst in 2008.",
                "Kids played with soap bubbles in the garden."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Bubble as a social containment unit",
            "text": "A defined group of people (a 'support bubble') permitted to interact during lockdowns. Also used for isolated sports competitions like the NBA bubble.",
            "examples": [
                "You can only meet with people in your bubble.",
                "The NBA finished its season inside a bubble."
            ]
        }
    },
    "remote": {
        "score": 0.38,
        "verdict": "Moderate change",
        "clusters": [
            {"name": "Geographically distant", "pre": 48, "post": 22},
            {"name": "Remote work / learning", "pre": 22, "post": 58},
            {"name": "Remote control",         "pre": 30, "post": 20}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Remote as distant or detached",
            "text": "Far away in place or time; also a TV remote control. The work sense existed but was niche — fewer than 1 in 4 usages referred to work contexts.",
            "examples": [
                "A remote village in the mountains.",
                "Have you seen the TV remote?"
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Remote as a work/learning modality",
            "text": "The dominant sense shifted to describing work or education conducted outside a physical office. 'Remote' became a default qualifier for professional life.",
            "examples": [
                "We're hiring for a fully remote position.",
                "Remote learning has mixed outcomes for students."
            ]
        }
    },
    "viral": {
        "score": 0.29,
        "verdict": "Low change",
        "clusters": [
            {"name": "Internet fame",           "pre": 53, "post": 49},
            {"name": "Biological / pathogenic", "pre": 37, "post": 41},
            {"name": "Marketing / spread",      "pre": 10, "post": 10}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Viral — already dual meaning",
            "text": "Both 'rapidly spreading on the internet' and 'relating to a virus' were well-established pre-COVID. No dominant sense shift expected — used as a control word.",
            "examples": [
                "That cat video went viral.",
                "A viral infection can cause fatigue."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Viral — slight biological weight increase",
            "text": "The biological sense gained slightly more salience during COVID, but the internet sense remained strong. Overall distributions are quite similar.",
            "examples": [
                "Viral load determines transmission risk.",
                "Her dance went viral overnight."
            ]
        }
    },
    "spread": {
        "score": 0.24,
        "verdict": "Low change",
        "clusters": [
            {"name": "Physical expansion",    "pre": 40, "post": 38},
            {"name": "Disease transmission",  "pre": 28, "post": 35},
            {"name": "Food spread / butter",  "pre": 20, "post": 16},
            {"name": "Financial spread",      "pre": 12, "post": 11}
        ],
        "old_definition": {
            "era": "Pre-2020",
            "title": "Spread as general expansion",
            "text": "To distribute or extend across a surface or area; also a food spread or a financial term. The disease sense existed but was not dominant.",
            "examples": [
                "Spread the butter evenly.",
                "The fire spread quickly through the dry grass."
            ]
        },
        "new_definition": {
            "era": "Post-2020",
            "title": "Spread with mild contagion emphasis",
            "text": "The disease transmission sense increased slightly but did not displace other senses. Confirms this as a low-change control word — our pipeline correctly identifies it.",
            "examples": [
                "Wear a mask to limit the spread.",
                "She spread jam on her toast."
            ]
        }
    }
}


def run_real_pipeline(word):
    """Run the actual LSCD pipeline on a word. Scrape live if not found."""
    import torch
    from data.social_corpus_builder import fetch_reddit_sentences_pushshift, PRE_COVID, POST_COVID

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(LSCD_PROJECT_PATH, 'models', 'sense_encoder.pt')
    encoder = SenseEncoder()
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()

    corpus_dir = os.path.join(LSCD_PROJECT_PATH, 'data', 'social_corpus', word)
    pre_file  = os.path.join(corpus_dir, 'pre2020.txt')
    post_file = os.path.join(corpus_dir, 'post2020.txt')

    pre_sents = []
    post_sents = []

    # Check if files exist locally
    if os.path.exists(pre_file) and os.path.exists(post_file):
        with open(pre_file, 'r', encoding='utf-8')  as f: pre_sents  = [l.strip() for l in f if l.strip()]
        with open(post_file, 'r', encoding='utf-8') as f: post_sents = [l.strip() for l in f if l.strip()]

    # If missing or empty, scrape live via Arctic Shift
    if len(pre_sents) < 20 or len(post_sents) < 20:
        print(f"[INFO] '{word}' not found locally or insufficient data. Scraping live via Arctic Shift...")
        os.makedirs(corpus_dir, exist_ok=True)
        
        pre_sents = fetch_reddit_sentences_pushshift(
            word=word, after=PRE_COVID["start"], before=PRE_COVID["end"], max_results=100
        )
        post_sents = fetch_reddit_sentences_pushshift(
            word=word, after=POST_COVID["start"], before=POST_COVID["end"], max_results=100
        )
        
        # Save scraped data
        with open(pre_file, "w", encoding="utf-8") as f:
            for s in pre_sents: f.write(s + "\n")
        with open(post_file, "w", encoding="utf-8") as f:
            for s in post_sents: f.write(s + "\n")

    if len(pre_sents) < 20 or len(post_sents) < 20:
        raise ValueError(f"Word '{word}' is too rare. Found {len(pre_sents)} pre-2020 and {len(post_sents)} post-2020 sentences. Minimum 20 required.")

    score, embs_pre, embs_post = compute_graded_change_score(encoder, word, pre_sents, post_sents, device)

    def_gen = DefinitionGenerator()
    explanation = generate_change_explanation(
        def_gen=def_gen,
        word=word,
        sentences_c1=pre_sents,
        embs_c1=embs_pre,
        sentences_c2=post_sents,
        embs_c2=embs_post,
        graded_score=score
    )

    if score >= 0.61:   verdict = "High change"
    elif score >= 0.31: verdict = "Moderate change"
    else:               verdict = "Low change"

    # Format for frontend UI
    old_defs = explanation.get("old_period_senses", {})
    new_defs = explanation.get("new_period_senses", {})
    
    old_main = old_defs.get(0, {})
    new_main = new_defs.get(0, {})
    
    # We don't have exact pre/post distribution percentages from the scorer
    # so we'll just format the senses for the UI
    clusters_ui = []
    for i in range(max(len(old_defs), len(new_defs))):
        pre_val = 100 // len(old_defs) if i < len(old_defs) else 0
        post_val = 100 // len(new_defs) if i < len(new_defs) else 0
        clusters_ui.append({
            "name": f"Extracted Sense {i+1}",
            "pre": pre_val,
            "post": post_val
        })

    return {
        "score": round(float(score), 2),
        "verdict": verdict,
        "clusters": clusters_ui,
        "old_definition": {
            "era": "Pre-2020",
            "title": f"Sense 1: {old_main.get('definition', 'Unknown')[:30]}...",
            "text": old_main.get('definition', 'No definition generated.'),
            "examples": old_main.get('example_sentences', [])
        },
        "new_definition": {
            "era": "Post-2020",
            "title": f"Sense 1: {new_main.get('definition', 'Unknown')[:30]}...",
            "text": new_main.get('definition', 'No definition generated.'),
            "examples": new_main.get('example_sentences', [])
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    word = data.get('word', '').strip().lower()

    if not word:
        return jsonify({"error": "No word provided"}), 400

    # Try real pipeline first, fall back to demo data
    if PIPELINE_AVAILABLE:
        try:
            result = run_real_pipeline(word)
            result['mode'] = 'live'
            return jsonify(result)
        except Exception as e:
            print(f"[WARN] Real pipeline failed for '{word}': {e}")
            traceback.print_exc()
            if word not in DEMO_DATA:
                return jsonify({"error": f"Live pipeline failed: {str(e)}"}), 500

    # Demo fallback
    if word in DEMO_DATA:
        result = DEMO_DATA[word].copy()
        result['mode'] = 'demo'
        return jsonify(result)

    return jsonify({
        "error": f"Word '{word}' not found in dataset. Try: " + ", ".join(DEMO_DATA.keys())
    }), 404


@app.route('/api/words', methods=['GET'])
def get_words():
    """Return list of available demo words."""
    pandemic = ['mask', 'lockdown', 'variant', 'essential', 'bubble']
    control  = ['remote', 'viral', 'spread']
    return jsonify({"pandemic": pandemic, "control": control})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  LSCD Interactive Demo")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
