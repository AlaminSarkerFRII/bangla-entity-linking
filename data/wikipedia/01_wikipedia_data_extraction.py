"""
Bangla Wikipedia Entity Link Extractor (Fixed for Colab)
"""

import os
import re
import wikipediaapi
import json
from tqdm import tqdm
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

wiki = wikipediaapi.Wikipedia(
    user_agent='WikiExtractor/1.0',
    language='bn',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def extract_entity_mentions(page_title, max_sentences=30):
    page = wiki.page(page_title)
    if not page.exists():
        return None

    samples = []
    sentences = nltk.sent_tokenize(page.text)
    for sent in sentences[:max_sentences]:
        for link_text, link_page in page.links.items():
            if link_text in sent:
                samples.append({
                    "sentence": sent,
                    "mention": link_text,
                    "entity": link_page.title
                })
    return samples

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    seeds = ["বঙ্গবন্ধু শেখ মুজিবুর রহমান", "মুক্তিযুদ্ধ", "শেখ হাসিনা", "রবীন্দ্রনাথ ঠাকুর"]
    all_samples = []

    for s in tqdm(seeds, desc="Extracting Wikipedia pages"):
        res = extract_entity_mentions(s)
        if res:
            all_samples.extend(res)

    with open("data/processed/wiki_entity_links.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_samples)} entity mentions to data/processed/wiki_entity_links.json")
