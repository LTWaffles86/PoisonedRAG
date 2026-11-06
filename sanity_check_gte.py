"""Sanity check (E0b): our encoder path reproduces reference gte-small embeddings.

retriever_mvp.py embeds with its own length-sorted batching and calls the SentenceTransformer
forward directly. This compares that path with SentenceTransformer.encode (the model card's way)
on the target questions, their attack texts and their ground-truth docs, and checks that the
embeddings have unit norm (so dot product == cosine similarity).
"""
import argparse
import os

import torch

from retriever_mvp import doc_to_text, encode, load_corpus, load_qrels, load_targets
from src.utils import load_models

p = argparse.ArgumentParser()
p.add_argument("--n_targets", type=int, default=30)
p.add_argument("--adv_per_query", type=int, default=5)
p.add_argument("--tol", type=float, default=1e-4)
p.add_argument("--out", default="experiments/sanity/gte_small_encode_check.txt")
args = p.parse_args()

targets = load_targets("nq", args.n_targets)
qrels = load_qrels("nq", "test")
corpus = load_corpus("nq", {d for t in targets for d in qrels.get(t["id"], ())})
groups = {
    "questions": [t["question"] for t in targets],
    "attack texts": [t["question"] + "." + a for t in targets for a in t["adv_texts"][:args.adv_per_query]],
    "ground-truth docs": [doc_to_text(d) for d in corpus.values()],
}

model, _, tokenizer, get_emb = load_models("gte-small")
model.eval().cuda()
dtype = next(model.parameters()).dtype

lines = [f"Sanity check: gte-small harness encoder vs SentenceTransformer.encode (nq, {args.n_targets} targets)",
         f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, model dtype {dtype}", "",
         f"{'texts':<18} {'n':>4} {'max_abs_diff':>12} {'max_|norm-1|':>12}"]
max_diff = max_norm_err = 0.0
for name, texts in groups.items():
    ours = encode(texts, model, tokenizer, get_emb, 64)
    ref = torch.from_numpy(model.encode(texts, batch_size=64, normalize_embeddings=True))
    diff = (ours - ref).abs().max().item()
    norm_err = (ours.norm(dim=-1) - 1).abs().max().item()
    max_diff, max_norm_err = max(max_diff, diff), max(max_norm_err, norm_err)
    lines.append(f"{name:<18} {len(texts):>4} {diff:>12.2e} {norm_err:>12.2e}")

passed = dtype == torch.float32 and max_diff < args.tol and max_norm_err < args.tol
lines += ["",
          f"max |ours - reference| = {max_diff:.2e}, max |norm - 1| = {max_norm_err:.2e} (tolerance {args.tol:.0e})",
          f"RESULT: {'PASS' if passed else 'FAIL'}"]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-2:]))
print(f"Full table: {args.out}")
