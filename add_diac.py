"""
wget https://huggingface.co/datasets/thewh1teagle/heb-text/resolve/main/HeDC4-enhanced-v3.csv.7z
"""
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import phonikud
from tqdm import tqdm
import hashlib
import re
import csv

def remove_diacritics(text: str):
    return re.sub(r'[\u0590-\u05c7|]', '', text)

def hash_text(text: str) -> str:
    full = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
    return full[:6]

def get_hash_set(file):
    hashes = set()
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hashes.add(line)
    return hashes


found_hashes = get_hash_set("found.txt")
model = AutoModel.from_pretrained("thewh1teagle/phonikud", trust_remote_code=True)
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("thewh1teagle/phonikud")
model.to("cuda:0")
model.eval()


rows_count = sum(1 for _ in open("data.csv"))


to_diac = []
with open("data.csv", "r") as in_file:
    reader = csv.reader(in_file, delimiter='\t')
    next(reader)
    for row in tqdm(reader, desc="Checking rows", total=rows_count):
        id, text, accents, prefix_lengths, morphs = row
        text = remove_diacritics(text)
        hash = hash_text(text)
        if hash in found_hashes:
            continue
        to_diac.append(text)

print(f'Found {len(to_diac)} rows to add diacritics')

with open("with_diac.txt", "w") as out_file:
    # add diacritics in batch of 200
    batch_size = 50
    for i in tqdm(range(0, len(to_diac), batch_size), desc="Adding diacritics"):
        try:
            batch = to_diac[i:i+batch_size]
            predicted = model.predict(batch, tokenizer)[0]
            for prediction in predicted:
                out_file.write(prediction + '\n')
        except Exception as e:
            print(f"Error adding diacritics: {e}")
            continue

# text = "כמה אתה חושב שזה יעלה לי? אני מגיע לשם רק בערב.."
# predicted = model.predict([text], tokenizer)[0]
# # with diacritics
# print(predicted)
# # phonemes
# phonemes = phonikud.phonemize(predicted)
# print(phonemes)