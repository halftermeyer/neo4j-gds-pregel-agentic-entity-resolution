# Agentic entity resolution with a constrained GDS Pregel clustering

A proof of concept for resolving person records from several sources (CRM, KYC, legacy) in Neo4j:

1. **Cypher** normalises the records, blocks candidate pairs and scores them. A rule gives each candidate edge a decision: MATCH, AMBIGUOUS or NO. Two records with different birth dates or national ids always get NO.
2. **A custom GDS Pregel procedure** (`myns.erCluster`, Java) clusters records over the MATCH edges. Unlike WCC, it refuses any merge that would put two different birth dates or national ids in one cluster. It also returns the AMBIGUOUS edges whose answer would change a cluster: the "decisive" questions.
3. **A Python loop** turns the decisive edges into multiple-choice questions ("is this cluster the same person as A, B, none of them, or unsure?") and sends them up a **cascade of deciders**:
   - L1: calibrated gradient-boosting classifier on the edge features;
   - L2: L1 plus local embeddings;
   - L3: local LLMs through Ollama (llama3.2 3B, llama3 8B, gemma4 31B);
   - L4: a hosted LLM (Gemini).

   Each level answers only when confident and passes the rest up. Decisions are written back on the edges with their provenance, the graph is re-projected, and the loop runs until no decisive question is left.

Everything is measured against a synthetic dataset with known truth (fixed seeds), so precision and recall are exact.

The full write-up, with all tables, limits and discussion, is in **[REPORT.md](REPORT.md) (in French)**. All numbers come from the JSON files in [`results/`](results/). Each configuration was run **once**: differences of a few points between neighbouring configurations are not significant.

## Key results (test set: 300 persons, 646 records)

| Configuration | Pair precision | Pair recall | F1 | Exact entities /300 | Wall time |
|---|---|---|---|---|---|
| Rules only (Pregel over rule MATCH edges) | 1.000 | 0.412 | 0.583 | 154 | 0.4 s |
| + L1 calibrated classifier | 1.000 | 0.606 | 0.754 | 203 | 52 s |
| + L1 + L2 embeddings | 1.000 | 0.614 | 0.760 | 205 | 54 s |
| + L1 + L2 + "always pick the best-scored option" (trivial baseline) | 0.848 | 0.899 | 0.873 | 230 | 51 s |
| + L1 + L2 + llama3.2 3B (local) | 0.794 | 0.810 | 0.802 | 198 | 147 s |
| + L1 + L2 + llama3 8B (local) | 0.824 | 0.883 | 0.853 | 219 | 186 s |
| + L1 + L2 + gemma4 31B (local, no thinking) | **0.934** | **0.856** | **0.894** | **242** | 526 s |
| + L1 + L2 + gemma4 31B (local, thinking) | 0.919 | 0.843 | 0.879 | 236 | 4 800 s |
| + L1 + L2 + Gemini pro (hosted) | 0.930 | 0.806 | 0.864 | 235 | 1 074 s |

What the runs show:

- **The profile constraint is where the graph pays off.** On the same edges (score ≥ 0.75), WCC reaches 0.40 pair precision with 39 clusters that mix different birth dates or ids. The constrained Pregel clustering reaches 0.84 with none.
- **Rules + Pregel + a calibrated classifier keep precision at 1.00** and lift recall from 0.41 to 0.61, all local, in under a minute.
- **Small local LLMs (3B, 8B) do worse than the trivial baseline.** They almost never abstain and merge people who should stay apart.
- **A local 31B model (gemma4) matches Gemini pro's precision** with better recall, without sending records off the machine. Thinking mode does not help and is 5 to 9 times slower.
- **About a third of the residual questions are undecidable from the records**: the name alone matches, with no corroborating field. Gemini abstains on all of them; the other models guess some of them.
- **No generative model keeps precision at 1.00.** Going up the cascade trades precision for recall.

## Repository layout

| Path | Role |
|---|---|
| `gen_data.py` | Synthetic records with known truth: `dev` (seed 1) and `test` (seed 2), 300 persons each. Variations: typos, phonetic variants, initials, compound first names, nicknames, swapped first/last names, case, missing birth dates, reformatted or moved addresses. Hard negatives: relatives and homonyms. |
| `features.py` | Normalisation, blocking, candidate edges with features, score and rule decision, all in Cypher (APOC double metaphone, Jaro-Winkler, nickname dictionary as nodes). |
| `pregel/src/main/java/myns/er/ConstrainedClustering.java` | The `myns.erCluster` procedure (GDS Pregel, constrained clustering). |
| `pregel/build.sh` | Compiles the procedure against the GDS jar of your Neo4j install and deploys it. |
| `graph_ops.py` | GDS projection, procedure call, WCC, and a Python reference of the Pregel semantics. |
| `check_pregel.py` | Checks the procedure against the Python reference (identical partitions). |
| `models.py` | L1/L2 classifiers, Ollama calls (L3), Gemini/Gemma API calls (L4). |
| `pipeline.py` | The outer loop and cascade. Writes `results/<tag>.json`. |
| `context_exp.py` | Same 77 hard questions, three prompt contexts, per model. |
| `baselines.py` | WCC vs constrained Pregel at several score thresholds. |
| `residual_analysis.py` | How much of the residual is decidable at all. |
| `tables.py` | Prints the report tables from `results/`. |
| `reproduce.sh` | Core reproduction (no generative model). |
| `run_llm.sh`, `run_gemma_local.sh`, `run_l4.sh` | LLM runs (levels 3 and 4). |
| `results/` | The JSON results behind every number in the report. |

## Prerequisites

Versions used for the reported runs. Other versions may work, but the GDS internal Pregel API changes between releases.

- **Neo4j 2026.10.2 Enterprise**, self-managed (custom procedures cannot be installed on Aura). Plugins:
  - Graph Data Science 2026.10.2;
  - APOC core 2026.10.2.
- **A JDK 21** (`javac`, `jar`) to build the procedure.
- **Python 3.11** with `pip install -r requirements.txt`.
- **Ollama** with `nomic-embed-text`. It is needed even for the core run, because L2 and the classifier training use embeddings. For level 3, also pull `llama3.2`, `llama3` and `gemma4:31b` (19 GB; about 26 GB of RAM with an 8k context).
- Optional, for level 4: a Google AI Studio key in `GEMINI_API_KEY`. **Level 4 sends the residual records to Google.** They are synthetic here.

The code runs on a Mac (Apple Silicon) for the reported timings; nothing in it is Mac-specific.

## Setup

### 1. Neo4j configuration

In `conf/neo4j.conf`, allow the GDS, APOC and `myns` namespaces and make Cypher 25 the default language (the queries use Cypher 25 functions such as `string.join` and `coll.distinct`):

```properties
dbms.security.procedures.unrestricted=apoc.*,gds.*,myns.*
dbms.security.procedures.allowlist=apoc.*,gds.*,myns.*
db.query.default_language=CYPHER_25
```

Then create the database (any name works, see `NEO4J_DATABASE` below):

```cypher
CREATE DATABASE `entity-resolution` IF NOT EXISTS;
```

### 2. Build and deploy the Pregel procedure

```bash
cd pregel
DBMS_HOME=/path/to/neo4j NAME=myns-er-pregel ./build.sh --deploy
cd ..
```

`DBMS_HOME` is the Neo4j home that contains `plugins/`, `lib/` and `conf/`. For Neo4j Desktop it looks like `~/Library/Application Support/neo4j-desktop/Application/relate-data-neo4j-local/dbmss/dbms-<id>`.

Restart Neo4j, then check that the procedure is there:

```cypher
SHOW PROCEDURES YIELD name WHERE name = 'myns.erCluster' RETURN name;
```

If it is missing, look for `IllegalAccess` or `myns` in `logs/debug.log`.

Without a GDS licence, the procedure runs with concurrency 4, the maximum allowed.

### 3. Environment

```bash
export NEO4J_PASSWORD=...                     # never stored in a file by this code
export NEO4J_URI=bolt://localhost:7687        # default
export NEO4J_USER=neo4j                       # default; also used as the GDS catalog user
export NEO4J_DATABASE=entity-resolution       # default
export OLLAMA_URL=http://localhost:11434      # default
```

## Reproduce

### Core run, no generative model (about 5 minutes)

```bash
./reproduce.sh
```

This script:
1. generates and loads the data;
2. builds the candidate edges;
3. computes the embeddings and trains L1/L2 on `dev`;
4. checks the procedure against the Python reference;
5. runs the WCC baselines;
6. runs the rules, L1, L1+L2 and L1+L2+trivial-baseline cascades on `test`;
7. prints the tables.

Expected output:
- data: 649 `dev` records and 646 `test` records;
- `check_pregel.py`: identical partitions (495 and 491 clusters);
- metrics matching the first four rows of the results table above.

The core run is deterministic. It was re-run from a clean clone of this repository, against a freshly regenerated dataset, and gave the same metrics and the same number of calls per level as the committed `results/`, with identical baselines; only the timings differed. Runs that include an LLM are not guaranteed to repeat exactly.

### Level 3: local LLMs

```bash
./run_llm.sh          # llama3.2 3B and llama3 8B; also writes results/round1_questions.json
```

`run_llm.sh` must run first: it dumps the 77 round-1 questions that `context_exp.py` and `residual_analysis.py` reuse.

For gemma4 31B, start a dedicated Ollama server with a small context first. By default, Ollama reserved a 262k-token context and 44 GB of memory for this model; with 8k it needs about 26 GB.

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Then, in another shell:

```bash
./run_gemma_local.sh  # three prompt contexts, thinking mode, then the two cascades (about 1.5 h)
```

Calls to Ollama are sequential. `OLLAMA_NUM_PARALLEL=4` brought no gain without thinking mode and about 17 % with it, on the MLX runner of an Apple Silicon Mac.

### Level 4: hosted models (sends records to Google)

```bash
export GEMINI_API_KEY=...
./run_l4.sh           # trivial baseline, Gemini pro cascade, Gemini on the three prompt contexts
python context_exp.py results/round1_questions.json results/context_exp_gemma_api.json GEMMA31_API,GEMMA26_API
```

### Analysis and tables

```bash
python residual_analysis.py
python tables.py
```

`residual_analysis.py` rebuilds the round-1 graph state itself.

## Data written to Neo4j, and cleanup

Everything this code writes uses dedicated names:

- nodes `:SpikeERRecord` and `:SpikeERNickname`;
- relationships `:SPK_ER_CANDIDATE`;
- constraint `spk_er_rid` and index `spk_er_nick`;
- GDS in-memory graphs `spk_er_g`, `spk_er_wcc` and `spk_er_base` (dropped after use).

To remove it all:

```cypher
MATCH (n:SpikeERRecord) DETACH DELETE n;
MATCH (n:SpikeERNickname) DETACH DELETE n;
DROP CONSTRAINT spk_er_rid IF EXISTS;
DROP INDEX spk_er_nick IF EXISTS;
```

## Known limits

All of these are discussed in REPORT.md.

- **Synthetic data**, generated for this PoC. The proportions of variations, relatives and homonyms are choices, so the results do not transfer as is to real data.
- **Slight leakage on the test set.** The rule layer was adjusted after looking at errors on `test`. The classifiers are trained on `dev` only, and the LLMs are not tuned. A third, never-inspected seed would give a clean number.
- **One run per configuration.** LLMs at temperature 0 are not fully deterministic, and the cascade amplifies differences.
- **Global state in the master.** The profile constraint applies to the whole cluster, so the union-find lives in `masterCompute`. It is correct and deterministic but not pure message passing, and it was not measured beyond a few hundred nodes.
- **Not done:** companies, a real cross-encoder at level 2, and an ONNX model embedded in the Pregel procedure.
