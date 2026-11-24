#!/usr/bin/env bash
# Level-3 runs: llama3.2 3B and llama3 8B via Ollama (needs NEO4J_PASSWORD). Also dumps the 77 round-1 questions.
set -x
P=${PYTHON:-python}
$P pipeline.py --ds test --levels L1,L2,L3a --ctx fiche_cluster --tag test_L1-L2-L3a > results/log_L3a.txt 2>&1
$P pipeline.py --ds test --levels L1,L2,L3a,L3b --ctx fiche_cluster --tag test_L1-L2-L3a-L3b --dump-questions results/round1_questions.json > results/log_L3b.txt 2>&1
$P pipeline.py --ds test --levels L1,L2,L3b --ctx fiche_cluster --tag test_L1-L2-L3b > results/log_L3b_only.txt 2>&1
$P context_exp.py results/round1_questions.json results/context_exp.json > results/log_ctx.txt 2>&1
