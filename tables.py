"""Print the markdown tables of REPORT.md from results/*.json (all numbers come from executed runs)."""
import json
import os
from collections import Counter

R = "results"


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


RUNS = [("Règles seules (Pregel sur MATCH)", "test_rules"), ("+ L1 classifieur", "test_L1"),
        ("+ L1 + L2 embeddings", "test_L1-L2"), ("+ L1 + L2 + « toujours A » (référence triviale)", "test_L1-L2-PRIOR"),
        ("+ L1 + L2 + L3a llama3.2 3B", "test_L1-L2-L3a"), ("+ L1 + L2 + L3a + L3b llama3 8B", "test_L1-L2-L3a-L3b"),
        ("+ L1 + L2 + L3b llama3 8B", "test_L1-L2-L3b"), ("+ L1 + L2 + L4 Gemini (hébergé)", "test_L1-L2-L4"),
        ("+ L1 + L2 + L3c gemma4 31B local, sans réflexion", "test_L1-L2-L3c"),
        ("+ L1 + L2 + L3cT gemma4 31B local, avec réflexion", "test_L1-L2-L3cT")]


def main():
    print("| Configuration | P paire | R paire | F1 | entités exactes /300 | clusters à id. fort mixte | appels par niveau | questions sans réponse finale | durée totale |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, tag in RUNS:
        r = load(tag)
        if not r:
            continue
        m = r["metrics"]
        unres = sum(1 for e in r["log"] if e["answer"] == "UNSURE")
        print(f"| {label} | {m['pair_precision']:.3f} | {m['pair_recall']:.3f} | {m['pair_f1']:.3f} | {m['exact_entities']} | "
              f"{m['clusters_with_strong_id_conflict']} | {r['calls'] or '-'} | {unres} | {r['elapsed_s']} s |")
    print()
    print("| Configuration | réponses justes par niveau (question à choix vs vérité) | latence médiane / p95 par appel |")
    print("|---|---|---|")
    for label, tag in RUNS:
        r = load(tag)
        if not r or not r["calls"]:
            continue
        lat = ", ".join(f"{k}: {v['median']:.0f} / {v['p95']:.0f} ms" for k, v in r["latency_ms"].items())
        print(f"| {label} | {r['answers_correct_by_level']} | {lat} |")
    print()
    # cascade curve: questions entering / settled at each step (first round set and all rounds)
    print("| Run | Marche | questions reçues | tranchées | justes | restent (UNSURE) |")
    print("|---|---|---|---|---|---|")
    for label, tag in RUNS[1:]:
        r = load(tag)
        if not r:
            continue
        settled = Counter(e["level"] for e in r["log"] if e["level"])
        good = Counter(e["level"] for e in r["log"] if e["level"] and e["correct"])
        for lv in r["levels"]:
            n_in = r["calls"].get(lv, 0)
            print(f"| {tag} | {lv} | {n_in} | {settled.get(lv, 0)} | {good.get(lv, 0)} | {r['unsure_after_level'].get(lv, 0)} |")
    print()
    for f in ("context_exp", "context_exp_L4", "context_exp_gemma_api", "context_exp_gemma_local", "context_exp_gemma_local_think"):
        c = load(f)
        if not c:
            continue
        print(f"{f}: gold {c.get('gold_distribution')}")
        print("| Contexte / modèle | justes | fausses (dont fusion à tort) | UNSURE | couverture | exactitude sur tranchées | latence médiane |")
        print("|---|---|---|---|---|---|---|")
        for k, v in c.items():
            if k == "gold_distribution":
                continue
            print(f"| {k} | {v.get('correct', 0)} | {v.get('wrong', 0)} ({v.get('wrong_merge', 0)}) | {v.get('unsure', 0)} | "
                  f"{v['coverage']} | {v['accuracy_on_decided']} | {v['median_ms']} ms |")
        print()
    b = load("baselines_score_threshold")
    if b:
        print("| Seuil score | arêtes | WCC P / R / F1 | WCC clusters mixtes | Pregel contraint P / R / F1 | Pregel clusters mixtes |")
        print("|---|---|---|---|---|---|")
        for t, v in b.items():
            w, p = v["wcc"], v["pregel_constrained"]
            print(f"| {t} | {v['edges']} | {w['pair_precision']:.3f} / {w['pair_recall']:.3f} / {w['pair_f1']:.3f} | "
                  f"{w['clusters_with_strong_id_conflict']} | {p['pair_precision']:.3f} / {p['pair_recall']:.3f} / {p['pair_f1']:.3f} | "
                  f"{p['clusters_with_strong_id_conflict']} |")


if __name__ == "__main__":
    main()
