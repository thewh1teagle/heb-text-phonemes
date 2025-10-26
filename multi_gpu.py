import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import csv, hashlib, re, os
import torch

def remove_diacritics(t): return re.sub(r'[\u0590-\u05c7|]', '', t)
def hash_text(t): return hashlib.sha1(t.strip().encode()).hexdigest()[:6]

def worker(gpu_id, tasks, output_path):
    # Load model/tokenizer per worker
    tokenizer = AutoTokenizer.from_pretrained("thewh1teagle/phonikud")
    model = AutoModel.from_pretrained("thewh1teagle/phonikud", trust_remote_code=True)
    model.to(f"cuda:{gpu_id}")
    model.eval()

    batch_size = 10
    # ✅ Progress bar per GPU
    pbar = tqdm(total=len(tasks), position=gpu_id, desc=f"GPU{gpu_id}")

    with open(output_path, "w") as out:
        try:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
                preds = model.predict(batch, tokenizer)[0]
                for p in preds:
                    out.write(p + "\n")

                pbar.update(len(batch))
        except Exception as e:
            print(f"[GPU{gpu_id}] Error processing batch: {e}")
            continue

    pbar.close()
    print(f"GPU{gpu_id} ✅ finished")

def run_parallel(texts):
    num_gpus = torch.cuda.device_count()
    chunks = [texts[i::num_gpus] for i in range(num_gpus)]

    procs = []
    for gpu in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(gpu, chunks[gpu], f"out_gpu{gpu}.txt")
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

def get_hash_set(file):
    hashes = set()
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hashes.add(line)
    return hashes

if __name__ == "__main__":
    found_hashes = get_hash_set("found.txt")

    texts_to_process = []
    total_lines = sum(1 for _ in open("data.csv"))
    with open("data.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in tqdm(reader, desc="Checking rows", total=total_lines):
            id, text, accents, prefix_lengths, morphs = row
            t = remove_diacritics(text)
            if hash_text(t) not in found_hashes:
                texts_to_process.append((t))

    print("Total to process:", len(texts_to_process))
    run_parallel(texts_to_process)
