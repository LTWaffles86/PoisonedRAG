# Reproducing our results (ECE 424 project, retriever component)

This reproduces the **retrieval** result of PoisonedRAG (Zou et al., USENIX Security 2025): on NQ with
the Contriever retriever, injecting 5 attack texts per target question puts them into the top-5
retrieved documents with **F1 = 0.96** (paper Table 1, black-box). It also replicates HotpotQA and MS-MARCO, and runs our
extension: knowledge bases of 10k/100k/500k docs with three retrievers (Contriever, ANCE, gte-small). There is no LLM in this part; that's a separate component.

## 1. Requirements
- Linux with an NVIDIA GPU (we used a GTX 1660, 6 GB) and a recent driver
- About 6.5 GB of disk for the three datasets, 2.5 GB for the Python environment, 1.5 GB for the models,
  and up to 17 GB for the embedding cache (`cache/emb/`) if you run every flag
- About 10 GB of free RAM for the full-NQ gte-small run
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## 2. Setup (about 5 minutes)
All commands run from the `PoisonedRAG/` directory.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements-mvp.txt

.venv/bin/python prepare_dataset.py   # NQ, HotpotQA, MS-MARCO from the BEIR server
# If the BEIR server is unreachable (it was on 2026-11-22), use the Hugging Face mirror instead:
.venv/bin/python -I prepare_dataset_hf.py nq hotpotqa msmarco
```

## 3. Reproduce everything

```bash
bash baseline.sh          # E0 + E1, about 2 minutes
bash baseline.sh --e2     # + Contriever on NQ at 10k/100k/500k docs (about 2.5 h the first time)
bash baseline.sh --e3     # + gte-small on NQ at 10k/100k/500k and full (about 3.5 h, ~10 GB RAM)
bash baseline.sh --e4     # + ANCE on NQ at 10k/100k/500k (about 3 h)
bash baseline.sh --e5     # + HotpotQA: replication + all 3 retrievers at 10k/100k/500k
bash baseline.sh --e6     # + MS-MARCO: same (uses the train qrels, where its target questions are)
```
Flags combine. Everything at once took about 15 h on a GTX 1660. Embeddings are cached, so reruns take minutes.
The script ends with `summarize_runs.py`, which writes `experiments/results.md`, `report/results_table.tex` and
`report/fig_f1_vs_kb_size.{pdf,png}`.

The script stops at the first error.

## 4. Step by step

| Step | Command | What you should see | Output |
|---|---|---|---|
| E0 sanity check | `.venv/bin/python sanity_check.py` | `RESULT: PASS` (max diff about 7e-07) | `experiments/sanity/score_reproduction.txt` |
| E1 clean (null control) | `.venv/bin/python retriever_mvp.py --attack none --run_name E1-nq-contriever-full-clean` | `f1: 0.0`, `clean_hit_rate: 0.2` | `experiments/runs/E1-nq-contriever-full-clean/` |
| E1 attack | `.venv/bin/python retriever_mvp.py --run_name E1-nq-contriever-full-attack` | `f1: 0.9533`, `poison_retrieval_hit_rate: 1.0` | `experiments/runs/E1-nq-contriever-full-attack/` |
| E1b attack, 100 targets | `.venv/bin/python retriever_mvp.py --repeat_times 10 --run_name E1b-nq-contriever-full-attack-100targets` | `f1: 0.958` | `experiments/runs/E1b-.../` |
| E2 100k docs | `.venv/bin/python retriever_mvp.py --corpus_size 100000 --run_name E2-nq-contriever-100k-attack` | `f1: 1.0` (about 30 min the first time) | `experiments/runs/E2-nq-contriever-100k-attack/` |
| E0b gte-small check | `.venv/bin/python sanity_check_encoder.py --retriever gte-small` | `RESULT: PASS` (max diff about 1e-07) | `experiments/sanity/encoder_check_gte-small.txt` |
| E0c ANCE check | `.venv/bin/python sanity_check_encoder.py --retriever ance` | `RESULT: PASS` (max diff about 1e-06) | `experiments/sanity/encoder_check_ance.txt` |
| E0 HotpotQA / MS-MARCO | `.venv/bin/python sanity_check.py --dataset hotpotqa --out experiments/sanity/score_reproduction_hotpotqa.txt` | `RESULT: PASS` | same folder |
| E3 gte-small 100k | `.venv/bin/python retriever_mvp.py --retriever gte-small --score_function cos_sim --corpus_size 100000 --run_name E3-nq-gte-small-100k-attack` | `f1: 0.9867` (about 6 min the first time) | `experiments/runs/E3-nq-gte-small-100k-attack/` |
| E3 gte-small full NQ | same with `--corpus_size 2681468 --run_name E3-nq-gte-small-full-attack` | `f1: 0.9333` (about 2 h 40 min the first time) | `experiments/runs/E3-nq-gte-small-full-attack/` |

Each run folder has `summary.json` (metrics, arguments, versions), `trace.jsonl` (the top-5 for every
question, with each entry labelled clean, own attack text, or another target's attack text) and `run.txt` (console log).
Results are deterministic, so you should get the same numbers.

## 5. Expected results

| Run | Targets | Knowledge base | Our F1 | Paper F1 |
|---|---|---|---|---|
| E1 attack | 30 | full NQ (2,681,468 docs) | 0.953 | 0.96 |
| E1b attack | 100 | full NQ | 0.958 | 0.96 |
| E1 clean | 30 | full NQ | 0.000 | - |
| E2 attack | 30 | 100,000 docs | 1.000 | - (our extension) |
| E3 attack, gte-small | 30 | 100,000 docs | 0.987 | - (our extension) |
| E3 clean, gte-small | 30 | 100,000 docs | 0.000 | - |
| E3 attack, gte-small | 30 | full NQ | 0.933 | - (paper: Contriever 0.96) |
| E3 clean, gte-small | 30 | full NQ | 0.000 | - |
| E5 attack, Contriever | 100 | full HotpotQA (5,233,329 docs) | 0.998 | 1.0 |
| E6 attack, Contriever | 100 | full MS-MARCO (8,841,823 docs) | 0.894 | 0.89 |

The full sweep (every retriever, dataset and KB size) is in `experiments/results.md`. Every clean run has F1 = 0.

## 6. Where to look
- **Sanity check and its explanation:** `experiments/EXPERIMENT_PROTOCOL.md`, section "Sanity check (E0)"; raw table in `experiments/sanity/score_reproduction.txt`.
- **Checkable trace:** the first line of `experiments/runs/E1-nq-contriever-full-attack/trace.jsonl` (question `test1`), explained in the protocol.
- **Protocol, deviations from the paper, run log, planned next runs:** `experiments/EXPERIMENT_PROTOCOL.md`.
- **Report draft (LaTeX):** `report/kb_embedding_sections.tex`, with `report/results_table.tex` and `report/fig_f1_vs_kb_size.pdf`.
- **Code:** `retriever_mvp.py` (harness), `sanity_check.py`, `sanity_check_encoder.py` and `summarize_runs.py`. They reuse the authors' `src/attack.py` and `src/utils.py`.

## 7. Troubleshooting
- **Dataset download fails or is blocked:** use `prepare_dataset_hf.py` (above). It writes the same `datasets/<name>/corpus.jsonl` and `qrels/`.
- **CUDA out of memory:** add `--batch_size 16` (this only matters for E2).
- **The first run pauses before printing anything:** it is downloading `facebook/contriever` (about 440 MB) or `thenlper/gte-small` (about 70 MB) or ANCE (about 500 MB) from Hugging Face.
- **`Contriever LOAD REPORT ... pooler ... UNEXPECTED`:** expected and harmless. Contriever doesn't use BERT's pooler layer.
