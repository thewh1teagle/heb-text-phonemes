import hashlib
import csv
from pathlib import Path
from tqdm import tqdm
import re

def remove_diacritics(text: str):
    return re.sub(r'[\u0590-\u05c7|]', '', text)


def hash_text(text: str) -> str:
    full = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
    return full[:6]


def load_txt_files(dir: Path) -> dict[str, str]:
    files = list(dir.glob("*.txt"))
    lines = []
    for file in tqdm(files, desc="Loading files"):
        with open(file, "r") as f:
            lines.extend(f.readlines())
    lines_with_hashes = []
    for line in tqdm(lines, desc="Hashing lines"):
        hash = hash_text(remove_diacritics(line))
        lines_with_hashes.append({'hash': hash, 'text': line})
    return lines_with_hashes


def main():
    dir = Path("partial_results")
    target_file = Path("all_results.csv")
    lines = load_txt_files(dir)
    with open(target_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "text"])
        for line in tqdm(lines, desc="Writing results"):
            writer.writerow([line['hash'], line['text'].strip()])

if __name__ == "__main__":
    main()