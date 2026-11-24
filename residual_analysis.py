"""How decidable is the round-1 residual? For each of the 77 questions L1+L2 could not settle,
check whether the record pairs between the subject cluster and the candidate cluster carry any
corroborating evidence beyond the name (same dob, nid, e-mail, phone, or address JW >= 0.9).
Without corroboration, a name-only match versus a homonym is not decidable from the records.
Rebuilds the round-1 graph state itself (reset + Pregel), as context_exp.py does."""
import json
from collections import Counter, defaultdict

from common import driver, run
from graph_ops import pregel_cluster
from pipeline import reset

d = json.load(open("results/round1_questions.json"))
drv = driver()
reset(drv, d["ds"])
clusters, *_ = pregel_cluster(drv, d["ds"], write=True)
assert clusters == d["clusters"], "round-1 clustering not reproduced"
rows = run(drv, """MATCH (a:SpikeERRecord {ds:'test'})-[r:SPK_ER_CANDIDATE]-(b:SpikeERRecord)
                   RETURN a.cluster AS ca, b.cluster AS cb,
                          r.f_dob = 1 OR r.f_nid = 1 OR r.f_email = 1 OR r.f_phone = 1 OR r.f_addr >= 0.9 AS corro""")
corro = defaultdict(bool)
for r in rows:
    corro[(r["ca"], r["cb"])] |= bool(r["corro"])
qcls = {}
for it in d["questions"]:
    q = it["q"]
    opts = {o["letter"]: o["cluster"] for o in q["options"]}
    if it["truth"] == "NONE":
        cls = "gold NONE"
    else:
        cls = "gold option, corroborated" if corro[(q["x"], opts[it["truth"]])] else "gold option, name only"
    qcls[q["x"]] = cls
out = {"distribution": Counter(qcls.values())}
json.dump(qcls, open("results/residual_classes.json", "w"), indent=1)
for f in ("context_exp", "context_exp_L4", "context_exp_gemma_api", "context_exp_gemma_local", "context_exp_gemma_local_think"):
    c = json.load(open(f"results/{f}.json"))
    for k, v in c.items():
        if k == "gold_distribution":
            continue
        t = defaultdict(Counter)
        for r in v["rows"]:
            cls = qcls[r["x"]]
            t[cls]["n"] += 1
            t[cls]["unsure" if r["answer"] == "UNSURE" else ("correct" if r["answer"] == r["truth"] else "wrong")] += 1
        out[f"{f}:{k}" if f.startswith("context_exp_gemma") else k] = {cls: dict(cn) for cls, cn in t.items()}
json.dump(out, open("results/residual_analysis.json", "w"), indent=1, default=dict)
print(json.dumps(out, indent=1, default=dict))
