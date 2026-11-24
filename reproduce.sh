#!/usr/bin/env bash
# Core reproduction: data, candidate edges, classifiers, Pregel equivalence check, baselines,
# and the local cascades without generative models (rules, L1, L1+L2, L1+L2+"always A").
# Needs: NEO4J_PASSWORD (and optionally NEO4J_URI, NEO4J_USER, NEO4J_DATABASE), Ollama with
# nomic-embed-text, and the myns-er-pregel plugin deployed (see README). About 5 minutes.
set -euo pipefail
: "${NEO4J_PASSWORD:?export NEO4J_PASSWORD first}"
P=${PYTHON:-python}
mkdir -p results
$P gen_data.py                       # dev (seed 1) + test (seed 2), 300 persons each
$P features.py                       # normalisation, blocking, SPK_ER_CANDIDATE edges, rule decisions
$P -c "from common import driver; from models import embed_all, train; d=driver(); embed_all(d); train(d)"
$P check_pregel.py                   # procedure vs Python reference: identical partitions
$P baselines.py                      # WCC vs constrained Pregel on score thresholds
$P pipeline.py --ds test --levels "" --tag test_rules
$P pipeline.py --ds test --levels L1 --tag test_L1
$P pipeline.py --ds test --levels L1,L2 --tag test_L1-L2
$P pipeline.py --ds test --levels L1,L2,PRIOR --ctx fiche_cluster --tag test_L1-L2-PRIOR
$P tables.py
