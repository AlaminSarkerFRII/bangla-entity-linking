import os
import sys
import json
from tqdm import tqdm

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print(">>> Script started")

from ner.silver_ner import tokenize_bn, bio_tag

INPUT_PATH = "data/processed/wiki_entity_links.json"
OUTPUT_PATH = "data/processed/silver_ner_dataset.json"

os.makedirs("data/processed", exist_ok=True)

def build_ner_dataset():
    print(">>> Loading input JSON")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(">>> Total input samples:", len(data))

    ner_samples = []

    for item in tqdm(data, desc="Building NER dataset"):
        sentence = item.get("sentence")
        mention = item.get("mention")

        if not sentence or not mention:
            continue

        tokens = tokenize_bn(sentence)
        labels = bio_tag(tokens, mention)

        if labels is None:
            continue

        ner_samples.append({
            "tokens": tokens,
            "labels": labels
        })

    print(">>> Final NER samples:", len(ner_samples))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ner_samples, f, ensure_ascii=False, indent=2)

    print(">>> Dataset saved at", OUTPUT_PATH)

if __name__ == "__main__":
    build_ner_dataset()
