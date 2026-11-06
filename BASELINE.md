# Reproducing our results (ECE 424 project, retriever component)

This reproduces the **retrieval** result of PoisonedRAG (Zou et al., USENIX Security 2025): on NQ with
the Contriever retriever, injecting 5 attack texts per target question puts them into the top-5
retrieved documents with **F1 = 0.96** (paper Table 1, black-box). It also includes the first run of our
extension (a smaller knowledge base) and a second retriever, gte-small (E3). There is no LLM in this part; that's a separate component.

## 1. Requirements
- Linux with an NVIDIA GPU (we used a GTX 1660, 6 GB) and a recent driver
- About 4 GB of disk: 1.5 GB for NQ, about 2.5 GB for the Python environment, about 0.5 GB for the model
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## 2. Setup (about 5 minutes)
All commands run from the `PoisonedRAG/` directory.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements-mvp.txt

mkdir -p datasets
curl -L -o datasets/nq.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip -q datasets/nq.zip -d datasets && rm datasets/nq.zip
```

## 3. Reproduce everything

```bash
bash baseline.sh          # E0 + E1, about 2 minutes
bash baseline.sh --e2     # also the 100k-document extension run, about 30 more minutes the first time
bash baseline.sh --e3     # also gte-small at 100k and full NQ, about 2.5 more hours the first time (needs ~10 GB free RAM)
```

The script stops at the first error and ends by printing one line per run.

## 4. Step by step

| Step | Command | What you should see | Output |
|---|---|---|---|
| E0 sanity check | `.venv/bin/python sanity_check.py` | `RESULT: PASS` (max diff about 7e-07) | `experiments/sanity/score_reproduction.txt` |
| E1 clean (null control) | `.venv/bin/python retriever_mvp.py --attack none --run_name E1-nq-contriever-full-clean` | `f1: 0.0`, `clean_hit_rate: 0.2` | `experiments/runs/E1-nq-contriever-full-clean/` |
| E1 attack | `.venv/bin/python retriever_mvp.py --run_name E1-nq-contriever-full-attack` | `f1: 0.9533`, `poison_retrieval_hit_rate: 1.0` | `experiments/runs/E1-nq-contriever-full-attack/` |
| E1b attack, 100 targets | `.venv/bin/python retriever_mvp.py --repeat_times 10 --run_name E1b-nq-contriever-full-attack-100targets` | `f1: 0.958` | `experiments/runs/E1b-.../` |
| E2 100k docs | `.venv/bin/python retriever_mvp.py --corpus_size 100000 --run_name E2-nq-contriever-100k-attack` | `f1: 1.0` (about 30 min the first time) | `experiments/runs/E2-nq-contriever-100k-attack/` |
| E0b gte-small check | `.venv/bin/python sanity_check_gte.py` | `RESULT: PASS` (max diff about 1e-07) | `experiments/sanity/gte_small_encode_check.txt` |
| E3 gte-small 100k | `.venv/bin/python retriever_mvp.py --retriever gte-small --score_function cos_sim --corpus_size 100000 --run_name E3-nq-gte-small-100k-attack` | `f1: 0.9867` (about 6 min the first time) | `experiments/runs/E3-nq-gte-small-100k-attack/` |
| E3 gte-small full NQ | same with `--corpus_size 2681468 --run_name E3-nq-gte-small-full-attack` | see protocol (about 2.5 h the first time) | `experiments/runs/E3-nq-gte-small-full-attack/` |

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

## 6. Where to look
- **Sanity check and its explanation:** `experiments/EXPERIMENT_PROTOCOL.md`, section "Sanity check (E0)"; raw table in `experiments/sanity/score_reproduction.txt`.
- **Checkable trace:** the first line of `experiments/runs/E1-nq-contriever-full-attack/trace.jsonl` (question `test1`), explained in the protocol.
- **Protocol, deviations from the paper, run log, planned next runs:** `experiments/EXPERIMENT_PROTOCOL.md`.
- **Code:** `retriever_mvp.py` (harness) and `sanity_check.py`. They reuse the authors' `src/attack.py` and `src/utils.py`.

## 7. Troubleshooting
- **NQ download fails or is blocked:** the zip is also linked from the [BEIR repo](https://github.com/beir-cellar/beir). Unpack it so that `datasets/nq/corpus.jsonl` exists.
- **CUDA out of memory:** add `--batch_size 16` (this only matters for E2).
- **The first run pauses before printing anything:** it is downloading `facebook/contriever` (about 440 MB) or `thenlper/gte-small` (about 70 MB) from Hugging Face.
- **`Contriever LOAD REPORT ... pooler ... UNEXPECTED`:** expected and harmless. Contriever doesn't use BERT's pooler layer.
