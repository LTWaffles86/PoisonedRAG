"""Sanity check: cached-score reproduction.

retriever_mvp.py (full mode) merges the authors' cached clean scores (results/beir_results,
computed with torch 1.13) with attack-text scores computed in our environment. That merge is only
valid if our environment reproduces the cached scores. This script re-encodes each target query and
its cached top-k documents and compares the recomputed dot products with the cached ones.
"""
import argparse
import os

import torch

from retriever_mvp import doc_to_text, encode, load_corpus, load_targets
from src.utils import load_json, load_models

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="nq")
p.add_argument("--retriever", default="contriever")
p.add_argument("--n_targets", type=int, default=30)
p.add_argument("--top_k", type=int, default=5)
p.add_argument("--tol", type=float, default=1e-2)
p.add_argument("--out", default="experiments/sanity/score_reproduction.txt")
args = p.parse_args()

targets = load_targets(args.dataset, args.n_targets)
cached = load_json(f"results/beir_results/{args.dataset}-{args.retriever}.json")
ranked = {t["id"]: sorted(cached[t["id"]].items(), key=lambda x: x[1], reverse=True)[:args.top_k]
          for t in targets}
corpus = load_corpus(args.dataset, {d for r in ranked.values() for d, _ in r})

model, _, tokenizer, get_emb = load_models(args.retriever)
model.eval().cuda()

lines = [f"Sanity check: cached-score reproduction ({args.dataset}, {args.retriever}, "
         f"{args.n_targets} targets x top-{args.top_k})",
         f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}", "",
         f"{'query':<9} {'rank':>4} {'doc_id':<12} {'cached':>9} {'recomputed':>10} {'abs_diff':>9}"]
max_diff, order_ok = 0.0, 0
for t in targets:
    qid = t["id"]
    doc_ids = [d for d, _ in ranked[qid]]
    q_emb = encode([t["question"]], model, tokenizer, get_emb, 1)[0]
    d_emb = encode([doc_to_text(corpus[d]) for d in doc_ids], model, tokenizer, get_emb, 16)
    recomputed = (d_emb @ q_emb).tolist()
    for r, ((d, s_cached), s_new) in enumerate(zip(ranked[qid], recomputed), 1):
        diff = abs(s_cached - s_new)
        max_diff = max(max_diff, diff)
        lines.append(f"{qid:<9} {r:>4} {d:<12} {s_cached:>9.5f} {s_new:>10.5f} {diff:>9.2e}")
    same_order = sorted(range(len(recomputed)), key=lambda i: -recomputed[i]) == list(range(len(recomputed)))
    order_ok += same_order
    if not same_order:
        lines.append(f"{qid:<9} ORDER CHANGED: recomputed order {sorted(range(len(recomputed)), key=lambda i: -recomputed[i])}")

passed = max_diff < args.tol and order_ok == len(targets)
lines += ["",
          f"max |cached - recomputed| = {max_diff:.2e} (tolerance {args.tol:.0e})",
          f"top-{args.top_k} order identical for {order_ok}/{len(targets)} queries",
          f"RESULT: {'PASS' if passed else 'FAIL'}"]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[-3:]))
print(f"Full table: {args.out}")
