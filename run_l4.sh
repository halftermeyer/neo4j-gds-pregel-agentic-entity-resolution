#!/usr/bin/env bash
# Second chain: trivial prior, level 4 (Gemini), context comparison for level 4.
set -x
P=${PYTHON:-python}
$P pipeline.py --ds test --levels L1,L2,PRIOR --ctx fiche_cluster --tag test_L1-L2-PRIOR > results/log_prior.txt 2>&1
$P pipeline.py --ds test --levels L1,L2,L4 --ctx fiche_cluster --tag test_L1-L2-L4 > results/log_L4.txt 2>&1
$P context_exp.py results/round1_questions.json results/context_exp_L4.json L4 > results/log_ctx_L4.txt 2>&1
echo CHAIN_DONE
