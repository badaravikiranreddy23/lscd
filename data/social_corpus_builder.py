"""
social_corpus_builder.py
Builds two time-separated corpora from Reddit data using the Arctic Shift API.
(Arctic Shift replaces the now-defunct Pushshift API)

PRE-2020 corpus  (C1): Jan 2018 – Dec 2019  → "normal" usage
POST-2020 corpus (C2): Jan 2021 – Dec 2022  → "pandemic/shifted" usage

Target words: mask, variant, remote, viral, bubble, essential, lockdown, spread
"""

import os
import time
import json
import requests
from typing import List, Dict
from datetime import datetime


# Target words with subreddits to search (Arctic Shift requires a subreddit filter)
TARGET_WORDS = {
    "mask":      ["AskReddit", "news", "coronavirus", "worldnews", "pics"],
    "variant":   ["AskReddit", "coronavirus", "science", "worldnews", "news"],
    "remote":    ["AskReddit", "cscareerquestions", "jobs", "technology", "news"],
    "viral":     ["AskReddit", "videos", "funny", "science", "entertainment"],
    "bubble":    ["AskReddit", "nba", "economics", "news", "worldnews"],
    "essential": ["AskReddit", "coronavirus", "jobs", "news", "antiwork"],
    "lockdown":  ["AskReddit", "coronavirus", "unitedkingdom", "news", "worldnews"],
    "spread":    ["AskReddit", "coronavirus", "science", "news", "worldnews"],
}

# Time windows as Unix timestamps
PRE_COVID = {
    "start": int(datetime(2018, 1, 1).timestamp()),
    "end":   int(datetime(2019, 12, 31).timestamp()),
    "label": "pre2020"
}
POST_COVID = {
    "start": int(datetime(2021, 1, 1).timestamp()),
    "end":   int(datetime(2022, 12, 31).timestamp()),
    "label": "post2020"
}

# Arctic Shift API base URL
ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"


def fetch_reddit_sentences_pushshift(
    word: str,
    after: int,
    before: int,
    max_results: int = 500
) -> List[str]:
    """
    Fetch sentences containing a word from Reddit via Arctic Shift API.
    (Drop-in replacement for the defunct Pushshift/PullPush endpoint)

    Arctic Shift requires a subreddit filter, so we iterate over the
    subreddits listed for this word in TARGET_WORDS.

    Free, no auth required.
    """
    sentences = []
    seen = set()  # Deduplication

    # Get subreddits for this word
    subreddits = TARGET_WORDS.get(word, ["AskReddit", "news", "worldnews"])

    for subreddit in subreddits:
        if len(sentences) >= max_results:
            break

        remaining = max_results - len(sentences)
        # Fetch up to 100 per request (API limit)
        limit = min(100, remaining)

        params = {
            "body": word,
            "subreddit": subreddit,
            "after": after,
            "before": before,
            "limit": limit,
        }

        retries = 0
        while retries < 3:
            try:
                resp = requests.get(ARCTIC_SHIFT_URL, params=params, timeout=30)
                if resp.status_code != 200:
                    print(f"    API error {resp.status_code} for r/{subreddit}, retrying...")
                    time.sleep(2)
                    retries += 1
                    continue

                data = resp.json()
                comments = data.get("data", [])
                if not comments:
                    break

                for item in comments:
                    body = item.get("body", "").strip()
                    # Skip deleted/removed/bot content
                    if not body or body in ("[deleted]", "[removed]"):
                        continue
                    if word.lower() not in body.lower():
                        continue
                    if len(body) < 20:
                        continue

                    # Split into sentences and filter
                    for sent in body.replace("!", ".").replace("?", ".").split("."):
                        sent = sent.strip()
                        if (word.lower() in sent.lower()
                                and len(sent.split()) >= 5
                                and len(sent) < 500
                                and sent not in seen):
                            seen.add(sent)
                            sentences.append(sent)

                print(f"    r/{subreddit}: got {len(comments)} comments → {len(sentences)} sentences so far")
                break  # Success, move to next subreddit

            except requests.exceptions.Timeout:
                print(f"    Timeout for r/{subreddit}, retrying...")
                retries += 1
                time.sleep(3)
            except Exception as e:
                print(f"    Error for r/{subreddit}: {e}")
                retries += 1
                time.sleep(2)

        time.sleep(0.5)  # Rate limiting

    return sentences[:max_results]


def build_social_corpus(
    output_dir: str = "data/social_corpus",
    max_per_word: int = 300
) -> None:
    """
    Build pre/post COVID corpora for all target words.
    Saves as JSON and plain text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for word in TARGET_WORDS:
        print(f"\n{'='*50}")
        print(f"Collecting: '{word}'")
        word_dir = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)

        for period in [PRE_COVID, POST_COVID]:
            label = period["label"]
            out_txt = os.path.join(word_dir, f"{label}.txt")
            out_json = os.path.join(word_dir, f"{label}.json")

            if os.path.exists(out_txt):
                print(f"  [{label}] Already exists, skipping.")
                continue

            print(f"  Fetching [{label}] {datetime.fromtimestamp(period['start']).year}–{datetime.fromtimestamp(period['end']).year}...")
            sents = fetch_reddit_sentences_pushshift(
                word=word,
                after=period["start"],
                before=period["end"],
                max_results=max_per_word
            )
            print(f"  Collected {len(sents)} sentences")

            # Save as plain text
            with open(out_txt, "w", encoding="utf-8") as f:
                for s in sents:
                    f.write(s + "\n")

            # Save as JSON with metadata
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "word": word,
                    "period": label,
                    "start_year": datetime.fromtimestamp(period["start"]).year,
                    "end_year": datetime.fromtimestamp(period["end"]).year,
                    "sentences": sents
                }, f, indent=2)

            summary[f"{word}_{label}"] = len(sents)
            time.sleep(1)

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Corpus built. Summary saved to {summary_path}")


def load_social_corpus(
    word: str,
    data_dir: str = "data/social_corpus"
) -> tuple:
    """Load pre/post sentences for a word from saved corpus."""
    sents_pre, sents_post = [], []

    pre_path = os.path.join(data_dir, word, "pre2020.txt")
    post_path = os.path.join(data_dir, word, "post2020.txt")

    if os.path.exists(pre_path):
        with open(pre_path, encoding="utf-8") as f:
            sents_pre = [l.strip() for l in f if l.strip()]

    if os.path.exists(post_path):
        with open(post_path, encoding="utf-8") as f:
            sents_post = [l.strip() for l in f if l.strip()]

    return sents_pre, sents_post


if __name__ == "__main__":
    print("Building social media corpus...")
    print("Using Arctic Shift API (replacement for defunct Pushshift)")
    print(f"Target words: {list(TARGET_WORDS.keys())}\n")
    build_social_corpus(output_dir="data/social_corpus", max_per_word=300)
