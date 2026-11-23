"""Fallback for prepare_dataset.py: fetch a BEIR dataset from its Hugging Face mirror (BeIR/<name>).

The BEIR zip server (public.ukp.informatik.tu-darmstadt.de) was unreachable on 2026-11-22. The
mirror ships the corpus as parquet, so this rewrites it as the usual datasets/<name>/corpus.jsonl
plus datasets/<name>/qrels/*.tsv. Queries are not needed (the target files hold the questions).

    .venv/bin/python -I prepare_dataset_hf.py hotpotqa msmarco
"""
import json
import os
import shutil
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

for name in sys.argv[1:]:
    out = os.path.join("datasets", name)
    tmp = os.path.join(out, "_hf")
    os.makedirs(os.path.join(out, "qrels"), exist_ok=True)
    for split in ("train", "dev", "test"):
        path = hf_hub_download(f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset", local_dir=tmp)
        shutil.copy(path, os.path.join(out, "qrels", f"{split}.tsv"))
    path = hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset", local_dir=tmp)
    n = 0
    with open(os.path.join(out, "corpus.jsonl.part"), "w") as f:
        for batch in pq.ParquetFile(path).iter_batches(columns=["_id", "title", "text"], batch_size=100_000):
            for doc in batch.to_pylist():
                # same key order and separators as the BEIR zips (retriever_mvp._line_id relies on '"_id": "' first)
                f.write(json.dumps({"_id": str(doc["_id"]), "title": doc["title"] or "", "text": doc["text"] or ""}) + "\n")
                n += 1
    os.replace(os.path.join(out, "corpus.jsonl.part"), os.path.join(out, "corpus.jsonl"))
    shutil.rmtree(tmp)
    print(f"{name}: {n} docs -> {out}/corpus.jsonl, qrels train/dev/test")
