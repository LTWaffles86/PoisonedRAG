#!/usr/bin/env bash
# Reproduces every number in experiments/EXPERIMENT_PROTOCOL.md. Run from PoisonedRAG/ after setup.
#   bash baseline.sh          # E0 + E1 (about 2 minutes)
#   bash baseline.sh --e2     # also the 100k-document extension run (about 30 minutes the first time on a GTX 1660)
#   bash baseline.sh --e3     # also gte-small at 100k and full NQ (about 2.5 hours the first time, ~10 GB RAM)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "== E0: sanity check (cached-score reproduction)"
$PY sanity_check.py

echo "== E1: replication on full NQ, 30 targets"
$PY retriever_mvp.py --attack none --run_name E1-nq-contriever-full-clean
$PY retriever_mvp.py --run_name E1-nq-contriever-full-attack

echo "== E1b: replication with the paper's 100 targets"
$PY retriever_mvp.py --repeat_times 10 --attack none --run_name E1b-nq-contriever-full-clean-100targets
$PY retriever_mvp.py --repeat_times 10 --run_name E1b-nq-contriever-full-attack-100targets

if [[ " $* " == *" --e2 "* ]]; then
  echo "== E2: knowledge base of 100k documents (slow the first time; embeddings are cached in cache/emb/)"
  $PY retriever_mvp.py --corpus_size 100000 --run_name E2-nq-contriever-100k-attack
  $PY retriever_mvp.py --corpus_size 100000 --attack none --run_name E2-nq-contriever-100k-clean
fi

if [[ " $* " == *" --e3 "* ]]; then
  echo "== E3: retriever gte-small (cosine) at 100k docs (same KB as E2) and full NQ (same KB as E1)"
  $PY sanity_check_gte.py
  G="--retriever gte-small --score_function cos_sim"
  $PY retriever_mvp.py $G --corpus_size 100000 --run_name E3-nq-gte-small-100k-attack
  $PY retriever_mvp.py $G --corpus_size 100000 --attack none --run_name E3-nq-gte-small-100k-clean
  # full NQ: 2.5 h to encode the first time, cached in cache/emb/ (4 GB)
  $PY retriever_mvp.py $G --corpus_size 2681468 --run_name E3-nq-gte-small-full-attack
  $PY retriever_mvp.py $G --corpus_size 2681468 --attack none --run_name E3-nq-gte-small-full-clean
fi

echo
echo "== Results (F1 should be about 0.95 for 30 targets, 0.96 for 100; clean runs must be 0.0)"
for f in experiments/runs/*/summary.json; do
  $PY -c "import json,sys; s=json.load(open(sys.argv[1])); m=s['metrics']; print(f\"{s['run_name']:<45} F1={m['f1']:.3f}  hit_rate={m['poison_retrieval_hit_rate']:.2f}  clean_hit_rate={m['clean_hit_rate']:.2f}\")" "$f"
done
