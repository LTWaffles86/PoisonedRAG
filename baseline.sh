#!/usr/bin/env bash
# Reproduces every number in experiments/EXPERIMENT_PROTOCOL.md. Run from PoisonedRAG/ after setup.
# Embeddings are cached in cache/emb/, so a second run of any flag takes minutes.
#   bash baseline.sh          # E0 + E1 (about 2 minutes)
#   bash baseline.sh --e2     # + KB-size sweep, Contriever, NQ (10k/100k/500k; about 2.5 h the first time)
#   bash baseline.sh --e3     # + gte-small, NQ (10k/100k/500k/full; about 3.5 h the first time, ~10 GB RAM)
#   bash baseline.sh --e4     # + ANCE, NQ (10k/100k/500k; about 3 h the first time)
#   bash baseline.sh --e5     # + HotpotQA: Contriever full, all 3 retrievers at 10k/100k/500k
#   bash baseline.sh --e6     # + MS-MARCO: same as E5
# Flags combine: bash baseline.sh --e2 --e3 --e4 --e5 --e6
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
MSMARCO_SPLIT=train  # the msmarco target questions are in train qrels (as in main.py), not dev or test

# pair EXP DATASET RETRIEVER CORPUS_SIZE LABEL: one attack run and one clean run on the same knowledge base
pair() {
  local o="--dataset $2 --retriever $3 --corpus_size $4"
  [[ $3 == gte-small ]] && o+=" --score_function cos_sim"
  [[ $2 == msmarco ]] && o+=" --split $MSMARCO_SPLIT"
  $PY retriever_mvp.py $o --run_name "$1-$2-$3-$5-attack"
  $PY retriever_mvp.py $o --attack none --run_name "$1-$2-$3-$5-clean"
}
# sweep EXP DATASET RETRIEVER: the 10k / 100k / 500k knowledge bases
sweep() {
  pair "$1" "$2" "$3" 10000 10k
  pair "$1" "$2" "$3" 100000 100k
  pair "$1" "$2" "$3" 500000 500k
}
has() { [[ " $ARGS " == *" $1 "* ]]; }
ARGS="$*"

echo "== E0: sanity check (cached-score reproduction)"
$PY sanity_check.py

echo "== E1: replication on full NQ, 30 targets"
pair E1 nq contriever full full

echo "== E1b: replication with the paper's 100 targets"
$PY retriever_mvp.py --repeat_times 10 --attack none --run_name E1b-nq-contriever-full-clean-100targets
$PY retriever_mvp.py --repeat_times 10 --run_name E1b-nq-contriever-full-attack-100targets

if has --e2; then
  echo "== E2: KB-size sweep, Contriever, NQ"
  sweep E2 nq contriever
fi

if has --e3; then
  echo "== E3: gte-small (cosine), NQ"
  $PY sanity_check_encoder.py --retriever gte-small
  sweep E3 nq gte-small
  pair E3 nq gte-small 2681468 full  # 2.5 h to encode the first time, 4 GB cache
fi

if has --e4; then
  echo "== E4: ANCE (dot product), NQ"
  $PY sanity_check_encoder.py --retriever ance
  sweep E4 nq ance
fi

for spec in "--e5 E5 hotpotqa test" "--e6 E6 msmarco $MSMARCO_SPLIT"; do
  read -r flag exp ds split <<< "$spec"
  if has "$flag"; then
    echo "== $exp: $ds"
    $PY sanity_check.py --dataset "$ds" --out "experiments/sanity/score_reproduction_$ds.txt"
    pair "$exp" "$ds" contriever full full
    $PY retriever_mvp.py --dataset "$ds" --split "$split" --repeat_times 10 --attack none --run_name "$exp-$ds-contriever-full-clean-100targets"
    $PY retriever_mvp.py --dataset "$ds" --split "$split" --repeat_times 10 --run_name "$exp-$ds-contriever-full-attack-100targets"
    for r in contriever gte-small ance; do sweep "$exp" "$ds" "$r"; done
  fi
done

echo
echo "== Results (clean runs must have F1 = 0)"
$PY summarize_runs.py
