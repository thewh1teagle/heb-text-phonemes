import csv
from pathlib import Path
import re
import hashlib
from tqdm import tqdm

src_file = Path('./HeDC4-enhanced-v3.csv')
results_file = Path("all_results.csv")

def remove_diacritics(text: str):
    return re.sub(r'[\u0590-\u05c7|]', '', text)

def hash_text(text: str) -> str:
    full = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
    return full[:6]

hashes = set()
with open(results_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        hashes.add(row[0])


report = {'found': 0, 'not_found': 0}

totla_lines = sum(1 for _ in open(src_file))
with open(src_file, "r") as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in tqdm(reader, desc="Processing rows", total=totla_lines):
        id, text, accents, prefix_lengths, morphs = row
        text = remove_diacritics(text)
        hash = hash_text(text)
        if hash in hashes:
            report['found'] += 1
        else:
            report['not_found'] += 1

print(report)

