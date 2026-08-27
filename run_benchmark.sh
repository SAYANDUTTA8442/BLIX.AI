#!/bin/bash
# run_benchmark.sh — Cloud execution wrapper for Blix ablation benchmark
# Runs in a detached tmux session so SSH disconnections don't kill the job.
#
# Usage:
#   bash run_benchmark.sh                    # all defaults
#   bash run_benchmark.sh --no-rag          # skip RAG baseline
#   bash run_benchmark.sh --samples 500     # larger dataset

set -euo pipefail

SESSION="blix_eval"
LOGFILE="benchmark.log"

# Kill any previous session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

CMD="python eval_harness.py \
    --datasets hotpotqa,locomo,narrativeqa,streamingqa \
    --samples 200 \
    --seeds 42,43,44,45,46 \
    --profiles full,no_graph,no_adma,both \
    --output results/ \
    --nli-metrics \
    --profile-memory \
    --visualize \
    --verbose \
    $@ \
    2>&1 | tee $LOGFILE"

tmux new-session -d -s "$SESSION" "bash -c '$CMD'; echo 'DONE' >> $LOGFILE"

echo "Benchmark started in tmux session: $SESSION"
echo "Monitor progress : tmux attach -t $SESSION"
echo "Tail logs        : tail -f $LOGFILE"
echo "Detach from tmux : Ctrl-B then D"
