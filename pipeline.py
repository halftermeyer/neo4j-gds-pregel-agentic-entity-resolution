"""Outer loop: Pregel clustering -> decisive ambiguous edges -> multiple-choice questions ->
decision cascade -> decisions written on edges with provenance -> re-project -> until convergence.

Usage: python pipeline.py --ds test --levels L1,L2,L3a,L3b --ctx fiche_cluster --tag full
Writes results/<tag>.json (metrics, calls, latencies, per-question log).
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from common import driver, run
from graph_ops import pregel_cluster, wcc
from models import FEATS, FEATS2, GEMINI_MODEL, ask_gemini, ask_llm, edge_matrix, load

LLM = {"L3a": "llama3.2:latest", "L3b": "llama3:latest", "L3c": "gemma4:31b", "L3cT": "gemma4:31b"}
THINK = {"L3c": False, "L3cT": True}
THRESH = {"L1": (0.90, 0.10), "L2": (0.90, 0.10)}
CARD = """n.first_name + ' ' + n.last_name + coalesce(', born ' + toString(n.dob), '') +
          coalesce(', ' + n.address, '') + coalesce(', ' + n.email, '') + coalesce(', tel ' + n.phone, '') +
          ' [' + n.source + ']'"""


# ---------------------------------------------------------------- setup / state
def reset(drv, ds):
    run(drv, """MATCH (a:SpikeERRecord {ds:$ds})-[r:SPK_ER_CANDIDATE]->()
                SET r.decision = r.rule_decision, r.decided_by = 'rule', r.rationale = 'rule layer',
                    r.w = (r.rule_decision + 1) * 10 + r.score,
                    r.model = null, r.p = null, r.round = null, r.ctx = null, r.no_shared = null, r.reopened = null""",
        ds=ds)
    run(drv, "MATCH (n:SpikeERRecord {ds:$ds}) REMOVE n.cluster", ds=ds)


def edge_features(drv, pairs):
    rows = run(drv, f"""UNWIND $pairs AS p
        MATCH (a:SpikeERRecord {{rid: p[0]}})-[r:SPK_ER_CANDIDATE]-(b:SpikeERRecord {{rid: p[1]}})
        RETURN p[0] AS a, p[1] AS b, r {{{', '.join('.' + f for f in FEATS2)}}} AS f""", pairs=pairs)
    return {(r["a"], r["b"]): r["f"] for r in rows}


def cluster_cards(drv, ds, cids, k=6):
    rows = run(drv, f"""MATCH (n:SpikeERRecord {{ds:$ds}}) WHERE n.cluster IN $cids
        WITH n.cluster AS c, n ORDER BY n.rid
        WITH c, collect(n) AS ns
        RETURN c, size(ns) AS size,
               coll.distinct([n IN ns | n.first_name + ' ' + n.last_name])[..{k}] AS names,
               coll.distinct([n IN ns WHERE n.dob IS NOT NULL | n.dob])[..3] AS dobs,
               coll.distinct([n IN ns WHERE n.address IS NOT NULL | n.address])[..3] AS addresses,
               coll.distinct([n IN ns WHERE n.email IS NOT NULL | n.email])[..2] AS emails,
               coll.distinct([n IN ns WHERE n.phone IS NOT NULL | n.phone])[..2] AS phones,
               coll.distinct([n IN ns | n.source]) AS sources,
               COUNT {{ MATCH (x:SpikeERRecord)-[r:SPK_ER_CANDIDATE {{decision: 1}}]->(y:SpikeERRecord)
                       WHERE x.cluster = c AND y.cluster = c AND x.ds = $ds }} AS confirmed_links""",
               ds=ds, cids=cids)
    return {r["c"]: r for r in rows}


def record_cards(drv, rids):
    rows = run(drv, f"MATCH (n:SpikeERRecord) WHERE n.rid IN $rids RETURN n.rid AS rid, {CARD} AS card", rids=rids)
    return {r["rid"]: r["card"] for r in rows}


def fmt_cluster(c, core=False):
    s = f"cluster of {c['size']} record(s): names {c['names']}"
    s += f"; dates of birth {[str(d) for d in c['dobs']] or 'unknown'}"
    if c["addresses"]:
        s += f"; addresses {c['addresses']}"
    if c["emails"]:
        s += f"; emails {c['emails']}"
    if c["phones"]:
        s += f"; phones {c['phones']}"
    s += f"; sources {c['sources']}"
    if core:
        s += f"; confirmed internal links {c['confirmed_links']}"
    return s


# ---------------------------------------------------------------- questions
def build_questions(decisive, clusters, skip_sigs, members):
    """Group decisive edges by cluster pair; one multiple-choice question per 'cluster to attach'
    (the smaller side), up to 2 options; each cluster appears in at most one question per round."""
    size = Counter(clusters.values())
    pair_edges = defaultdict(list)
    for r in decisive:
        c1, c2 = r["cluster"], r["otherCluster"]
        a, b = (r["rid"], r["other"]) if c1 < c2 else (r["other"], r["rid"])
        pair_edges[(min(c1, c2), max(c1, c2))].append((a, b, r["score"]))
    cand = defaultdict(list)
    for (c1, c2), es in pair_edges.items():
        best = max(e[2] for e in es)
        cand[c1].append((c2, best))
        cand[c2].append((c1, best))
    locked, qs = set(), []
    for x in sorted(cand, key=lambda c: (size[c], c)):
        if x in locked:
            continue
        opts = []
        for o, best in sorted(cand[x], key=lambda t: (-t[1], t[0])):
            if o in locked or o == x:
                continue
            key = (min(x, o), max(x, o))
            if skip_sigs.get(key) == (members[key[0]], members[key[1]]):
                continue
            opts.append(o)
            if len(opts) == 2:
                break
        if not opts:
            continue
        locked.add(x)
        locked.update(opts)
        qs.append({"x": x, "options": [{"letter": "AB"[i], "cluster": o,
                                         "edges": pair_edges[(min(x, o), max(x, o))]} for i, o in enumerate(opts)]})
    return qs


def truth_answer(q, truth_major):
    tx = truth_major[q["x"]]
    for o in q["options"]:
        if truth_major[o["cluster"]] == tx:
            return o["letter"]
    return "NONE"


# ---------------------------------------------------------------- levels
def level_clf(model, feats, q, ef, hi, lo):
    ps = {}
    for o in q["options"]:
        X = edge_matrix([ef[(a, b)] for a, b, _ in o["edges"]], feats)
        ps[o["letter"]] = float(model.predict_proba(X)[:, 1].max())
    best = max(ps, key=ps.get)
    others = [p for k, p in ps.items() if k != best]
    if ps[best] >= hi and all(p < lo for p in others):
        return best, f"p={ps}", ps
    if all(p < lo for p in ps.values()):
        return "NONE", f"p={ps}", ps
    return "UNSURE", f"p={ps}", ps


def llm_prompt(drv, ds, q, ctx, cards_cache):
    cids = [q["x"]] + [o["cluster"] for o in q["options"]]
    cc = cluster_cards(drv, ds, cids)
    # subject record: X-side endpoint of the best decisive edge
    best_edge = max((e for o in q["options"] for e in o["edges"]), key=lambda e: e[2])
    xset = set(r for r, c in cards_cache["clusters"].items() if c == q["x"])
    x_rid = best_edge[0] if best_edge[0] in xset else best_edge[1]
    need = [x_rid]
    opt_rid = {}
    for o in q["options"]:
        e = max(o["edges"], key=lambda e: e[2])
        orid = e[1] if e[0] in xset else e[0]
        opt_rid[o["letter"]] = orid
        need.append(orid)
    rc = record_cards(drv, need)
    if ctx == "pair":
        text = f"Subject record: {rc[x_rid]}\n" + "\n".join(
            f"Option {o['letter']}: record {rc[opt_rid[o['letter']]]}" for o in q["options"])
        text += "\nWhich option is the same person as the subject?"
    elif ctx == "fiche_cluster":
        text = f"Subject record: {rc[x_rid]}\n" + "\n".join(
            f"Option {o['letter']}: {fmt_cluster(cc[o['cluster']])}" for o in q["options"])
        text += "\nEach option is a group of records already believed to be one person. Which option is the same person as the subject?"
    else:
        text = f"Subject: {fmt_cluster(cc[q['x']], core=True)}\n" + "\n".join(
            f"Option {o['letter']}: {fmt_cluster(cc[o['cluster']], core=True)}" for o in q["options"])
        text += ("\nEach group is a set of records already believed to be one person (confirmed internal links "
                 "are rule- or analyst-confirmed). Which option is the same person as the subject group?")
    return text


# ---------------------------------------------------------------- metrics
def metrics(pred, truth, strong):
    """Pairwise P/R/F1, exact-cluster rate, and number of predicted clusters mixing strong ids."""
    def c2(n):
        return n * (n - 1) // 2
    pc, tc, both = Counter(pred.values()), Counter(truth.values()), Counter((pred[r], truth[r]) for r in pred)
    tp = sum(c2(n) for n in both.values())
    pp = sum(c2(n) for n in pc.values())
    tpairs = sum(c2(n) for n in tc.values())
    p = tp / pp if pp else 1.0
    rcl = tp / tpairs if tpairs else 1.0
    pm, tm = defaultdict(set), defaultdict(set)
    for r in pred:
        pm[pred[r]].add(r)
        tm[truth[r]].add(r)
    pred_sets = set(frozenset(s) for s in pm.values())
    exact = sum(1 for s in tm.values() if frozenset(s) in pred_sets)
    conflict = 0
    for s in pm.values():
        dobs = {strong[r][0] for r in s if strong[r][0]}
        nids = {strong[r][1] for r in s if strong[r][1]}
        if len(dobs) > 1 or len(nids) > 1:
            conflict += 1
    return {"pair_precision": round(p, 4), "pair_recall": round(rcl, 4),
            "pair_f1": round(2 * p * rcl / (p + rcl), 4) if p + rcl else 0,
            "exact_entities": exact, "true_entities": len(tm), "exact_rate": round(exact / len(tm), 4),
            "pred_clusters": len(pm), "clusters_with_strong_id_conflict": conflict}


def load_truth(drv, ds):
    rows = run(drv, "MATCH (n:SpikeERRecord {ds:$ds}) RETURN n.rid AS rid, n.truth_eid AS t, n.dob AS dob, n.nid AS nid", ds=ds)
    return {r["rid"]: r["t"] for r in rows}, {r["rid"]: (r["dob"], r["nid"]) for r in rows}


# ---------------------------------------------------------------- main loop
def write_decision(drv, q, ans, level, model, rationale, p, rnd, ctx, prof):
    upd = []
    for o in q["options"]:
        dec = 1 if ans == o["letter"] else -1
        shared = len(prof[q["x"]] & prof[o["cluster"]])
        for a, b, _ in o["edges"]:
            upd.append({"a": a, "b": b, "d": dec, "shared": shared})
    run(drv, """UNWIND $u AS u
        MATCH (a:SpikeERRecord {rid: u.a})-[r:SPK_ER_CANDIDATE]-(b:SpikeERRecord {rid: u.b})
        SET r.decision = u.d, r.w = (u.d + 1) * 10 + r.score, r.decided_by = $level, r.model = $model,
            r.rationale = $rat, r.p = $p, r.round = $rnd, r.ctx = $ctx,
            r.no_shared = CASE WHEN u.d = -1 THEN u.shared ELSE null END""",
        u=upd, level=level, model=model, rat=rationale, p=json.dumps(p) if p else None, rnd=rnd, ctx=ctx)


def cluster_profiles(clusters, strong):
    prof = defaultdict(set)
    for r, c in clusters.items():
        d, n = strong[r]
        if d:
            prof[c].add(("dob", d))
        if n:
            prof[c].add(("nid", n))
    return prof


def compatible(p1, p2):
    for t1, v1 in p1:
        for t2, v2 in p2:
            if t1 == t2 and v1 != v2:
                return False
    return True


def reopen_no(drv, ds, clusters, prof):
    """Cache rule: a model NO is kept unless the two clusters now share a strong identifier value
    they did not share when the NO was given (new strong clue) -> reopen as AMBIGUOUS."""
    rows = run(drv, """MATCH (a:SpikeERRecord {ds:$ds})-[r:SPK_ER_CANDIDATE]->(b)
                       WHERE r.decision = -1 AND r.decided_by <> 'rule'
                       RETURN a.rid AS a, b.rid AS b, r.no_shared AS s""", ds=ds)
    reopen = []
    for r in rows:
        ca, cb = clusters[r["a"]], clusters[r["b"]]
        if ca == cb:
            continue
        shared = len(prof[ca] & prof[cb])
        if shared > (r["s"] or 0) and compatible(prof[ca], prof[cb]):
            reopen.append({"a": r["a"], "b": r["b"]})
    if reopen:
        run(drv, """UNWIND $u AS u MATCH (a:SpikeERRecord {rid: u.a})-[r:SPK_ER_CANDIDATE]->(b:SpikeERRecord {rid: u.b})
                    SET r.decision = 0, r.w = 10 + r.score, r.decided_by = 'reopened', r.reopened = coalesce(r.reopened, 0) + 1""",
            u=reopen)
    return len(reopen)


def pipeline(drv, ds, levels, ctx, max_rounds=30, dump_questions=None):
    reset(drv, ds)
    truth, strong = load_truth(drv, ds)
    mdl = load()
    calls, lat, log, rounds_info = Counter(), defaultdict(list), [], []
    skip_sigs = {}
    remaining_by_level = Counter()
    reopened_total = 0
    t_start = time.time()
    for rnd in range(1, max_rounds + 1):
        clusters, decisive, refused, moot, st = pregel_cluster(drv, ds)
        prof = cluster_profiles(clusters, strong)
        if rnd > 1:
            n_re = reopen_no(drv, ds, clusters, prof)
            reopened_total += n_re
            if n_re:
                clusters, decisive, refused, moot, st = pregel_cluster(drv, ds)
        members = defaultdict(set)
        for r, c in clusters.items():
            members[c].add(r)
        members = {c: frozenset(s) for c, s in members.items()}
        truth_major = {c: Counter(truth[r] for r in s).most_common(1)[0][0] for c, s in members.items()}
        qs = build_questions(decisive, clusters, skip_sigs, members) if levels else []
        rounds_info.append({"round": rnd, **st, "questions": len(qs)})
        if not qs:
            break
        ef = edge_features(drv, [[a, b] for q in qs for o in q["options"] for a, b, _ in o["edges"]])
        cache = {"clusters": clusters}
        dumped = []
        for q in qs:
            ans, level_used = "UNSURE", None
            for lv in levels:
                t0 = time.time()
                if lv == "PRIOR":   # trivial reference: always the top-scored option
                    ans, rat, p, model = "A", "prior: highest-score option", None, "prior"
                elif lv in ("L1", "L2"):
                    hi, lo = THRESH[lv]
                    ans, rat, p = level_clf(mdl["l1" if lv == "L1" else "l2"], FEATS if lv == "L1" else FEATS2,
                                            q, ef, hi, lo)
                    model = "HistGB" + ("+nomic-embed" if lv == "L2" else "")
                else:
                    if dump_questions is not None and rnd == 1 and not dumped_has(dumped, q):
                        dumped.append({"q": q, "truth": truth_answer(q, truth_major)})
                    text = llm_prompt(drv, ds, q, ctx, cache)
                    if lv == "L4":
                        ans, rat, dt = ask_gemini(text, q["options"])
                        p, model = None, GEMINI_MODEL
                    else:
                        ans, rat, dt = ask_llm(LLM[lv], text, q["options"], think=THINK.get(lv))
                        p, model = None, LLM[lv]
                calls[lv] += 1
                lat[lv].append(time.time() - t0)
                if ans != "UNSURE":
                    level_used = lv
                    break
                remaining_by_level[lv] += 1
            gold = truth_answer(q, truth_major)
            log.append({"round": rnd, "x": q["x"], "options": [o["cluster"] for o in q["options"]],
                        "answer": ans, "level": level_used, "truth": gold, "correct": ans == gold if ans != "UNSURE" else None})
            if ans == "UNSURE":
                for o in q["options"]:
                    key = (min(q["x"], o["cluster"]), max(q["x"], o["cluster"]))
                    skip_sigs[key] = (members[key[0]], members[key[1]])
                continue
            write_decision(drv, q, ans, level_used, model, rat, p, rnd, ctx, prof)
        if dump_questions is not None and rnd == 1:
            with open(dump_questions, "w") as fh:
                json.dump({"ds": ds, "clusters": clusters, "questions": dumped}, fh)
    final_clusters, decisive, refused, moot, st = pregel_cluster(drv, ds)
    elapsed = time.time() - t_start
    res = {"ds": ds, "levels": levels, "ctx": ctx, "rounds": rounds_info, "elapsed_s": round(elapsed, 1),
           "calls": dict(calls), "latency_ms": {k: {"median": round(1000 * float(np.median(v)), 1),
                                                     "p95": round(1000 * float(np.percentile(v, 95)), 1),
                                                     "total_s": round(float(np.sum(v)), 1)} for k, v in lat.items()},
           "unsure_after_level": dict(remaining_by_level), "reopened": reopened_total,
           "final_refused_match_edges": len(refused), "final_decisive_unresolved": len(decisive),
           "metrics": metrics(final_clusters, truth, strong)}
    acc = defaultdict(lambda: [0, 0])
    for e in log:
        if e["level"]:
            acc[e["level"]][0] += int(e["correct"])
            acc[e["level"]][1] += 1
    res["answers_correct_by_level"] = {k: f"{c}/{n}" for k, (c, n) in acc.items()}
    res["answers_by_kind"] = {f"{k[0]}:{k[1]}": v for k, v in Counter((e["level"] or "unresolved", "NONE" if e["answer"] == "NONE" else ("UNSURE" if e["answer"] == "UNSURE" else "OPTION")) for e in log).items()}
    # same final MATCH decisions, transitive closure (WCC) instead of constrained clustering
    res["wcc_on_final_match_edges"] = metrics(wcc(drv, ds, "r.decision = 1"), truth, strong)
    res["log"] = log
    return res


def dumped_has(dumped, q):
    return any(d["q"]["x"] == q["x"] for d in dumped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="test")
    ap.add_argument("--levels", default="L1,L2,L3a,L3b")
    ap.add_argument("--ctx", default="fiche_cluster", choices=["pair", "fiche_cluster", "cluster_cluster"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dump-questions", default=None)
    a = ap.parse_args()
    levels = [l for l in a.levels.split(",") if l]
    drv = driver()
    res = pipeline(drv, a.ds, levels, a.ctx, dump_questions=a.dump_questions)
    os.makedirs("results", exist_ok=True)
    tag = a.tag or f"{a.ds}_{'-'.join(levels) or 'rules'}_{a.ctx}"
    with open(f"results/{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    show = {k: v for k, v in res.items() if k != "log"}
    print(json.dumps(show, indent=1, default=str))


if __name__ == "__main__":
    main()
