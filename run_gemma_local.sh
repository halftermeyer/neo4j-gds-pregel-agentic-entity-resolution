#!/usr/bin/env bash
# gemma4:31b via Ollama, sequential calls. OLLAMA_URL defaults to a dedicated server on port 11435
# started with OLLAMA_CONTEXT_LENGTH=8192 (see README); override OLLAMA_URL to use another one.
set -x
P=${PYTHON:-python}
export OLLAMA_URL=${OLLAMA_URL:-http://localhost:11435}
$P context_exp.py results/round1_questions.json results/context_exp_gemma_local.json GEMMA31_LOCAL > results/log_ctx_gemma_local.txt 2>&1
$P context_exp.py results/round1_questions.json results/context_exp_gemma_local_think.json GEMMA31_LOCAL_THINK pair > results/log_ctx_gemma_local_think.txt 2>&1
$P pipeline.py --ds test --levels L1,L2,L3c --ctx pair --tag test_L1-L2-L3c > results/log_L3c.txt 2>&1
$P pipeline.py --ds test --levels L1,L2,L3cT --ctx pair --tag test_L1-L2-L3cT > results/log_L3cT.txt 2>&1
echo CHAIN_DONE
