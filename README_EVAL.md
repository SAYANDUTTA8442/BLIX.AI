# Blix Evaluation — Replication Guide

This document describes how to reproduce the benchmark results from the paper.

## Requirements

**Hardware** (recommended): NVIDIA RTX 4090 (24 GB VRAM), 64 GB RAM, 200 GB SSD.
**Expected runtime**: < 12 hours for 200 samples × 5 seeds × 4 datasets × 5 profiles on the recommended hardware.

**Software**: Python 3.10+, Ollama (for local LLM serving).

## 1. Installation

```bash
git clone https://github.com/SAYANDUTTA8442/blix.ai
cd blix.ai

# Core dependencies
pip install -e .

# Evaluation dependencies
pip install -e ".[eval]"

# Ollama LLM backend
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b   # default model used in paper
```

## 2. Optional: NLI Faithfulness Metrics

NLI metrics require `transformers` and a GPU:

```bash
pip install transformers torch accelerate
# The harness will auto-download roberta-large-mnli on first run
```

If `transformers` is not installed, NLI columns (`entailment_score`, `hallucination_rate`) will be `NaN` — all other metrics are unaffected.

## 3. Running the Full Benchmark

```bash
bash run_benchmark.sh
```

Or directly:

```bash
python eval_harness.py \
    --datasets hotpotqa,locomo,narrativeqa,streamingqa \
    --samples 200 \
    --seeds 42,43,44,45,46 \
    --profiles full,no_graph,no_adma,both \
    --output results/ \
    --nli-metrics \
    --profile-memory \
    --visualize \
    --verbose
```

### Resuming interrupted runs

```bash
python eval_harness.py --resume ...  # skips existing seed/profile CSVs
```

## 4. Output Structure

```
results/
├── hotpotqa/
│   ├── seed_42/full.csv  no_graph.csv  no_adma.csv  both.csv
│   ├── aggregate_summary.json   (mean ± std + bootstrap p-values)
│   └── ablation_table.md        (paper-ready markdown table)
├── locomo/ narrativeqa/ streamingqa/
├── combined_summary.md          (main paper table)
├── combined_summary.tex         (LaTeX version)
└── figures/
    ├── learning_curve.png
    ├── latency_analysis.png
    └── graph_density_growth.png
```

## 5. CSV Schema

Each `<profile>.csv` contains one row per test query with 27 columns:

| Column | Description |
|--------|-------------|
| `question_id` | Index in test split |
| `hit_1/5/10` | Hit Rate@k (answer in top-k retrieved) |
| `mrr` | Mean Reciprocal Rank |
| `ndcg` | NDCG@10 |
| `rouge_l` | ROUGE-L F1 |
| `bleu_4` | BLEU-4 |
| `faithfulness_bert` | Cosine similarity (sentence-transformers) |
| `entailment_score` | NLI entailment probability (roberta-large-mnli) |
| `hallucination_rate` | 1 − entailment_score |
| `latency` | Seconds per query (wall clock) |
| `tokens_per_sec` | Generation throughput |
| `vram_peak_gb` | Peak VRAM (CUDA) per query |
| `ram_peak_gb` | Peak RAM (psutil) per query |
| `node_count` | HGSHM graph nodes at query time |
| `graph_density` | edges / nodes |
| `consolidation_rate` | clusters / nodes |
| `policy_divergence` | KL(Beta(α,β) ‖ Beta(1,1)) |
| `policy_switching` | Policy changes / queries so far |
| `cache_hit_rate` | Policy cache hits / total requests |
| `state_consistency` | Weighted correctness (StreamingQA only) |

## 6. Ablation Profiles

| Profile | HGSHM | Graph | ADMA |
|---------|-------|-------|------|
| `full` | ✅ | ✅ | ✅ |
| `no_graph` | ✅ | ❌ | ✅ |
| `no_adma` | ✅ | ✅ | ❌ |
| `both` | ✅ | ❌ | ❌ |
| `rag` | ❌ | ❌ | ❌ |

## 7. Statistical Significance

All comparisons use **paired bootstrap** (10,000 resamples) on Hit@5.
Markers: † p<0.10, * p<0.05, ** p<0.01, *** p<0.001.

## 8. Dataset Notes

- **HotpotQA**: `distractor` config, validation split. Temporal split 60/20/20.
- **LoCoMo**: `multi_hop` config, test split. Temporal split 60/20/20.
- **NarrativeQA**: test split, split by `story_id` to prevent cross-story leakage.
- **StreamingQA**: test split. Temporal split 60/20/20. State consistency metric enabled.

## 9. Compute Budget

| Component | Approximate time |
|-----------|-----------------|
| Data loading (4 datasets) | 2–5 min |
| Adaptation phase (train) per seed per profile | 3–8 min |
| Test evaluation (200 samples) per profile | 10–30 min |
| Full run (5 seeds × 5 profiles × 4 datasets) | 6–12 hours |

For a quick sanity check, run with `--samples 20 --seeds 42 --profiles full,rag`.
