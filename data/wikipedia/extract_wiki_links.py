"""
Bangla Wikipedia Entity Link Extractor
Create Silver Standard entity linkings data
"""

import os
import re
import wikipediaapi
import json

from setuptools.package_index import user_agent
from tqdm import tqdm
import nltk



nltk.download('punkt')

wiki = wikipediaapi.Wikipedia(
    user_agent='WikiExtractor/1.0',
    language='bn',
    extract_format=wikipediaapi.ExtractFormat.WIKIT
)


def extract_entity_mentions(page_title, max_sentencs=30):
    page = wiki.page(page_title)

    if not page.exists():
        return None

    samples = []
    text = page.text
    sentences = nltk.sent_tokenize(text)

    for sent in sentences[:max_sentencs]:
        for link_text , link_page in page.links.items():
            if link_text in sent:
                samples.append({
                    "sentence": sent,
                    "mention": link_text,
                    "entity": link_page.title
                })
    return samples

if __name__ == "__main__":
    seeds = ["বঙ্গবন্ধু শেখ মুজিবুর রহমান", "মুক্তিযুদ্ধ", "শেখ হাসিনা", "রবীন্দ্রনাথ ঠাকুর"]

    all_samples = []

    for s in tqdm(seeds):
        all_samples.extend(extract_entity_mentions(s))

    with open("../processed/wiki_entity_links.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)


