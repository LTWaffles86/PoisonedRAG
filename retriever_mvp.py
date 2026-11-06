"""Retriever-only PoisonedRAG harness (no LLM).

Reproduces the retrieval half of main.py: inject N attack texts per target question,
retrieve top-k, and report poisoned-retrieval metrics plus a per-query trace.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import transformers

from src.attack import Attacker
from src.utils import load_json, load_models, setup_seeds

DATA_DIR = "datasets"
RUNS_DIR = os.path.join("experiments", "runs")
EMB_CACHE_DIR = os.path.join("cache", "emb")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Retriever-only PoisonedRAG harness")
    p.add_argument("--dataset", default="nq")
    p.add_argument("--split", default="test")
    p.add_argument("--retriever", default="contriever", help="contriever | ance | gte-small")
    p.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--adv_per_query", type=int, default=5, help="N: attack texts injected per target question")
    p.add_argument("--M", type=int, default=10, help="target questions per iteration")
    p.add_argument("--repeat_times", type=int, default=3, help="iterations; total targets = M * repeat_times")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--attack", default="LM_targeted", choices=["LM_targeted", "none"])
    p.add_argument("--corpus_size", default="full",
                   help="'full' uses the cached full-corpus results; an integer encodes a random subsample live")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--run_name", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------- data loading

def load_targets(dataset, n):
    targets = list(load_json(f"results/adv_targeted_results/{dataset}.json").values())
    assert n <= len(targets), f"only {len(targets)} target questions available"
    return targets[:n]


def load_qrels(dataset, split):
    qrels = {}
    with open(os.path.join(DATA_DIR, dataset, "qrels", f"{split}.tsv")) as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def _line_id(line):
    # corpus.jsonl lines start with {"_id": "<id>", ...}; avoids json-parsing 2.6M lines
    start = line.index('"_id": "') + 8
    return line[start:line.index('"', start)]


def load_corpus(dataset, ids):
    """Load only the documents whose ids are in `ids`."""
    ids = set(ids)
    corpus = {}
    with open(os.path.join(DATA_DIR, dataset, "corpus.jsonl")) as f:
        for line in f:
            if _line_id(line) in ids:
                doc = json.loads(line)
                corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}
    missing = ids - corpus.keys()
    assert not missing, f"{len(missing)} doc ids not found in corpus, e.g. {next(iter(missing))}"
    return corpus


def all_corpus_ids(dataset):
    with open(os.path.join(DATA_DIR, dataset, "corpus.jsonl")) as f:
        return [_line_id(line) for line in f]


def doc_to_text(doc):
    # Same formatting as src/contriever_src/beir_utils.py DenseEncoderModel.encode_corpus
    return doc["title"] + " " + doc["text"] if doc["title"] else doc["text"]


# ---------------------------------------------------------------- encoding

def encode(texts, model, tokenizer, get_emb, batch_size, desc=None):
    """Embed texts in batches on the GPU; returns a CPU tensor in input order.
    Sorting by length keeps padding (and time) low. CPU output lets full NQ (4+ GB) fit."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = None
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(texts), batch_size)):
        idx = order[start:start + batch_size]
        inp = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            e = get_emb(model, inp).float().cpu()
        if out is None:
            out = torch.empty(len(texts), e.shape[1])
        out[idx] = e
        if desc and (b % 200 == 0 or b == n_batches - 1):
            print(f"  [{desc}] batch {b + 1}/{n_batches}", flush=True)
    return out


# ---------------------------------------------------------------- clean retrieval backends

class CachedFullCorpus:
    """Clean top-k from results/beir_results (precomputed by the authors over the full corpus)."""

    def __init__(self, args, targets, qrels):
        suffix = "-cos" if args.score_function == "cos_sim" else ""
        path = f"results/beir_results/{args.dataset}-{args.retriever}{suffix}.json"
        assert os.path.exists(path), (
            f"{path} not found. Full-corpus mode needs cached results; use --corpus_size N instead.")
        cached = load_json(path)
        self.ranked = {t["id"]: sorted(cached[t["id"]].items(), key=lambda x: x[1], reverse=True)
                       for t in targets}
        needed = {d for t in targets for d, _ in self.ranked[t["id"]][:args.top_k]}
        needed |= {d for t in targets for d in qrels.get(t["id"], ())}
        self.corpus = load_corpus(args.dataset, needed)
        self.size = 2681468 if args.dataset == "nq" else None
        self.top_k = args.top_k

    def topk(self, qid, q_emb):
        return self.ranked[qid][:self.top_k]

    def gt_rank(self, qid, q_emb, gt_ids):
        for r, (d, _) in enumerate(self.ranked[qid], 1):
            if d in gt_ids:
                return r
        return None  # not in cached top-100


class SubsampledCorpus:
    """Ground-truth docs of the targets plus a seeded random sample, encoded live and searched exactly."""

    def __init__(self, args, targets, qrels, model, tokenizer, get_emb):
        size = int(args.corpus_size)
        gt = sorted({d for t in targets for d in qrels.get(t["id"], ())})
        assert size >= len(gt), f"--corpus_size must be at least {len(gt)} (the targets' ground-truth docs)"
        gt_set = set(gt)
        pool = [d for d in all_corpus_ids(args.dataset) if d not in gt_set]
        sample = random.Random(args.seed).sample(pool, size - len(gt))
        self.ids = gt + sample
        self.corpus = load_corpus(args.dataset, self.ids)
        self.size = len(self.ids)
        self.top_k = args.top_k
        self.cos = args.score_function == "cos_sim"

        os.makedirs(EMB_CACHE_DIR, exist_ok=True)
        cache = os.path.join(EMB_CACHE_DIR, f"{args.dataset}-{args.retriever}-{size}-seed{args.seed}.pt")
        if os.path.exists(cache):
            print(f"Loading cached corpus embeddings from {cache}")
            self.emb = torch.load(cache)  # scored on CPU: full NQ does not fit next to the model on a 6 GB GPU
        else:
            print(f"Encoding {size} documents (this is the slow step; cached afterwards)")
            t0 = time.time()
            texts = [doc_to_text(self.corpus[d]) for d in self.ids]
            # fp32 on purpose: fp16 matmuls are ~8x slower than fp32 on GTX 16xx cards
            self.emb = encode(texts, model, tokenizer, get_emb, args.batch_size, desc="corpus")
            torch.save(self.emb, cache)
            print(f"Encoded in {time.time() - t0:.0f}s, saved to {cache}")
        if self.cos:
            self.emb = torch.nn.functional.normalize(self.emb, dim=-1)

    def _scores(self, q_emb):
        return self.emb @ q_emb

    def topk(self, qid, q_emb):
        s, i = self._scores(q_emb).topk(self.top_k)
        return [(self.ids[j], v) for j, v in zip(i.tolist(), s.tolist())]

    def gt_rank(self, qid, q_emb, gt_ids):
        scores = self._scores(q_emb)
        gt_idx = [j for j, d in enumerate(self.ids) if d in gt_ids]
        if not gt_idx:
            return None
        best = scores[gt_idx].max()
        return int((scores > best).sum().item()) + 1


# ---------------------------------------------------------------- run

class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stdout, name)


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def run(args):
    if args.run_name is None:
        args.run_name = (f"{args.dataset}-{args.retriever}-{args.corpus_size}-{args.attack}"
                         f"-k{args.top_k}-N{args.adv_per_query}")
    out_dir = os.path.join(RUNS_DIR, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, "run.txt"))
    print(f"Args: {vars(args)}")
    t_start = time.time()
    setup_seeds(args.seed)

    n_targets = args.M * args.repeat_times
    targets = load_targets(args.dataset, n_targets)
    qrels = load_qrels(args.dataset, args.split)

    model, c_model, tokenizer, get_emb = load_models(args.retriever)
    model.eval().cuda()
    c_model.eval().cuda()

    if args.corpus_size == "full":
        backend = CachedFullCorpus(args, targets, qrels)
    else:
        backend = SubsampledCorpus(args, targets, qrels, c_model, tokenizer, get_emb)
    print(f"Knowledge base: {backend.size} clean documents ({args.corpus_size})")

    attacker = None
    if args.attack != "none":
        attacker = Attacker(SimpleNamespace(attack_method=args.attack, adv_per_query=args.adv_per_query,
                                            eval_dataset=args.dataset, score_function=args.score_function),
                            model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

    def embed(texts, enc):
        e = encode(texts, enc, tokenizer, get_emb, args.batch_size)
        return torch.nn.functional.normalize(e, dim=-1) if args.score_function == "cos_sim" else e

    trace_path = os.path.join(out_dir, "trace.jsonl")
    trace_f = open(trace_path, "w")
    per_query = []
    for it in range(args.repeat_times):
        batch = targets[it * args.M:(it + 1) * args.M]
        print(f"######## Iter {it + 1}/{args.repeat_times}: targets {it * args.M}-{(it + 1) * args.M - 1} ########")

        # All M*N attack texts of this iteration are in the knowledge base at once (as in main.py),
        # so a query can also retrieve attack texts aimed at other targets.
        adv_texts, adv_owner, adv_j = [], [], []
        if attacker is not None:
            groups = attacker.get_attack([{"query": t["question"], "id": t["id"]} for t in batch])
            for t, g in zip(batch, groups):
                for j, text in enumerate(g):
                    adv_texts.append(text)
                    adv_owner.append(t["id"])
                    adv_j.append(j)
            adv_embs = embed(adv_texts, c_model)

        q_embs = embed([t["question"] for t in batch], model)

        for t, q_emb in zip(batch, q_embs):
            qid = t["id"]
            gt_ids = qrels.get(qid, set())
            cands = [{"id": d, "score": float(s), "source": "clean", "is_gt": d in gt_ids,
                      "text": doc_to_text(backend.corpus[d])} for d, s in backend.topk(qid, q_emb)]
            if adv_texts:
                adv_scores = (adv_embs @ q_emb).tolist()
                for text, owner, j, s in zip(adv_texts, adv_owner, adv_j, adv_scores):
                    cands.append({"id": f"adv:{owner}:{j}", "score": s,
                                  "source": "adv_own" if owner == qid else "adv_other",
                                  "is_gt": False, "text": text})
            top = sorted(cands, key=lambda c: c["score"], reverse=True)[:args.top_k]
            for r, c in enumerate(top, 1):
                c["rank"] = r

            n_own = sum(c["source"] == "adv_own" for c in top)
            n_other = sum(c["source"] == "adv_other" for c in top)
            first_adv = next((c["rank"] for c in top if c["source"] == "adv_own"), None)
            rec = {
                "iter": it, "id": qid, "question": t["question"],
                "correct_answer": t["correct answer"], "target_answer": t["incorrect answer"],
                "gt_doc_ids": sorted(gt_ids),
                "gt_rank_clean": backend.gt_rank(qid, q_emb, gt_ids),
                "n_adv_own_in_topk": n_own, "n_adv_other_in_topk": n_other,
                "first_adv_rank": first_adv,
                "gt_in_topk": any(c["is_gt"] for c in top),
                "top_k": [{**c, "text": c["text"][:300]} for c in top],
            }
            trace_f.write(json.dumps(rec) + "\n")
            per_query.append(rec)
            print(f"  {qid}: own adv in top-{args.top_k} = {n_own}, other adv = {n_other}, "
                  f"gt in top-k = {rec['gt_in_topk']}, gt clean rank = {rec['gt_rank_clean']}")
    trace_f.close()

    n_own = np.array([r["n_adv_own_in_topk"] for r in per_query], dtype=float)
    precision = n_own / args.top_k
    recall = n_own / args.adv_per_query
    # src.utils.f1_score uses np.divide(where=...) without `out`, which leaves garbage when P+R == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    first_ranks = [r["first_adv_rank"] for r in per_query if r["first_adv_rank"] is not None]
    metrics = {
        "n_targets": len(per_query),
        "poison_retrieval_hit_rate": float(np.mean(n_own > 0)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "f1_per_iter": [float(f1[i * args.M:(i + 1) * args.M].mean()) for i in range(args.repeat_times)],
        "clean_hit_rate": float(np.mean([r["gt_in_topk"] for r in per_query])),
        "mean_first_adv_rank": float(np.mean(first_ranks)) if first_ranks else None,
        "adv_other_in_topk_total": int(sum(r["n_adv_other_in_topk"] for r in per_query)),
    }
    summary = {
        "run_name": args.run_name,
        "args": vars(args),
        "metrics": metrics,
        "knowledge_base_size": backend.size,
        "wall_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n======== Summary ========")
    for k, v in metrics.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
    print(f"Outputs: {out_dir}/{{trace.jsonl,summary.json,run.txt}}")
    return summary


if __name__ == "__main__":
    run(parse_args())
