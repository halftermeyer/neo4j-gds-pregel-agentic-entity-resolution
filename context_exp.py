"""Same task, three prompt contexts: the round-1 questions that L1+L2 could not settle
(dumped by pipeline.py --dump-questions), asked to each local LLM with each context.
Graph state = round 1 (rules-only clustering), rebuilt here before asking."""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import driver
from graph_ops import pregel_cluster
from models import GEMINI_MODEL, ask_gemini, ask_llm
from pipeline import LLM, llm_prompt, reset


API = {"L4": GEMINI_MODEL, "GEMMA31_API": "gemma-4-31b-it", "GEMMA26_API": "gemma-4-26b-a4b-it"}
LOCAL = dict(LLM, GEMMA31_LOCAL="gemma4:31b", GEMMA31_LOCAL_THINK="gemma4:31b")
THINK = {"GEMMA31_LOCAL": False, "GEMMA31_LOCAL_THINK": True}


def main(path, out, levels=("L3a", "L3b"), contexts=("pair", "fiche_cluster", "cluster_cluster")):
    d = json.load(open(path))
    drv = driver()
    reset(drv, d["ds"])
    clusters, *_ = pregel_cluster(drv, d["ds"], write=True)
    assert clusters == {k: v for k, v in d["clusters"].items()}, "round-1 clustering not reproduced"
    res = {}
    for ctx in contexts:
        for lv in levels:
            stats, lat, rows = Counter(), [], []
            texts = []
            for item in d["questions"]:
                q = item["q"]
                for o in q["options"]:
                    o["edges"] = [tuple(e) for e in o["edges"]]
                texts.append(llm_prompt(drv, d["ds"], q, ctx, {"clusters": clusters}))

            def call(i):
                q, t0 = d["questions"][i]["q"], time.time()
                if lv in API:
                    a, r, _ = ask_gemini(texts[i], q["options"], model=API[lv])
                else:
                    a, r, _ = ask_llm(LOCAL[lv], texts[i], q["options"], think=THINK.get(lv))
                return a, r, time.time() - t0
            # hosted API: 8 calls in flight; local Ollama: sequential (one GPU)
            with ThreadPoolExecutor(max_workers=8 if lv in API else 1) as ex:
                answers = list(ex.map(call, range(len(texts))))
            for item, (ans, reason, dt) in zip(d["questions"], answers):
                q = item["q"]
                lat.append(dt)
                gold = item["truth"]
                if ans == "UNSURE":
                    stats["unsure"] += 1
                elif ans == gold:
                    stats["correct"] += 1
                else:
                    stats["wrong"] += 1
                    stats["wrong_merge" if ans != "NONE" else "wrong_none"] += 1
                rows.append({"x": q["x"], "answer": ans, "truth": gold, "reason": reason})
            n = len(d["questions"])
            decided = stats["correct"] + stats["wrong"]
            res[f"{ctx}/{lv}"] = {"n": n, **stats, "accuracy_on_decided": round(stats["correct"] / decided, 3) if decided else None,
                                  "coverage": round(decided / n, 3), "median_ms": round(1000 * sorted(lat)[len(lat) // 2]),
                                  "rows": rows}
            print(ctx, lv, {k: v for k, v in res[f"{ctx}/{lv}"].items() if k != "rows"}, flush=True)
    gold = Counter(i["truth"] for i in d["questions"])
    res["gold_distribution"] = dict(gold)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("L3a", "L3b"),
         tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("pair", "fiche_cluster", "cluster_cluster"))
