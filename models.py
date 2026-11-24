"""Decision levels.

L1  feature classifier (sklearn gradient boosting) on the Cypher features (incl. diminutive dictionary)
L2  L1 features + cosine of local nomic-embed-text embeddings (Ollama). Not a cross-encoder:
    no torch / sentence-transformers in this environment (stated in the report).
L3a llama3.2 3B via Ollama, L3b llama3 8B via Ollama (generative, multiple choice).
"""
import json
import pickle
import time
import urllib.request

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from common import run

import os
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
FEATS = ["f_jw_last", "f_jw_first", "f_phon_last", "f_nick", "f_ini", "f_prefix", "f_first", "f_name",
         "f_swapped", "f_dob", "f_nid", "f_email", "f_phone", "f_addr", "f_pc", "f_same_src", "f_first_ini", "score"]
FEATS2 = FEATS + ["f_emb"]


def ollama(path, payload, timeout=120):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def record_text(n):
    parts = [f"{n.get('first_name') or ''} {n.get('last_name') or ''}".strip()]
    if n.get("dob"):
        d = str(n["dob"])
        parts.append(f"born {d[:4]}-{d[4:6]}-{d[6:]}")
    for k in ("address", "email", "phone"):
        if n.get(k):
            parts.append(str(n[k]))
    return ", ".join(parts)


def embed_all(drv):
    rows = run(drv, "MATCH (n:SpikeERRecord) WHERE n.emb IS NULL RETURN n.rid AS rid, n {.*} AS n")
    t0 = time.time()
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        out = ollama("/api/embed", {"model": "nomic-embed-text",
                                    "input": ["search_document: " + record_text(r["n"]) for r in chunk]})
        run(drv, "UNWIND $rows AS r MATCH (n:SpikeERRecord {rid: r.rid}) SET n.emb = r.emb",
            rows=[{"rid": r["rid"], "emb": e} for r, e in zip(chunk, out["embeddings"])])
    run(drv, """MATCH (a:SpikeERRecord)-[r:SPK_ER_CANDIDATE]->(b)
                SET r.f_emb = vector.similarity.cosine(a.emb, b.emb)""")
    return len(rows), time.time() - t0


def edge_matrix(rows, feats):
    return np.array([[float(r[f]) if r.get(f) is not None else 0.0 for f in feats] for r in rows])


def train(drv):
    """Train L1/L2 on the dev AMBIGUOUS edges (truth labels used for training only)."""
    rows = run(drv, f"""MATCH (a:SpikeERRecord {{ds:'dev'}})-[r:SPK_ER_CANDIDATE]->(b)
                        WHERE r.rule_decision = 0
                        RETURN r {{{', '.join('.' + f for f in FEATS2)}}} AS f, a.truth_eid = b.truth_eid AS y""")
    X1 = edge_matrix([r["f"] for r in rows], FEATS)
    X2 = edge_matrix([r["f"] for r in rows], FEATS2)
    y = np.array([1 if r["y"] else 0 for r in rows])
    def gb():   # regularised GB, probabilities calibrated by 5-fold isotonic regression (dev only)
        return CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4,
                                                                     min_samples_leaf=20, random_state=0),
                                      method="isotonic", cv=5)
    m1 = gb().fit(X1, y)
    m2 = gb().fit(X2, y)
    with open("models.pkl", "wb") as fh:
        pickle.dump({"l1": m1, "l2": m2}, fh)
    return len(rows), int(y.sum())


def load():
    with open("models.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------- generative levels ----------------
SYSTEM = ("You are a careful entity-resolution analyst for a bank's customer master data (France). "
          "Decide whether records describe the same real person. Consider typos, accents, case, inverted "
          "first/last names, initials, French diminutives and nicknames, compound first names, people who "
          "moved. Relatives often share last name, address and landline: a different first name (e.g. Jean vs "
          "Jeanne, Michel vs Michèle) or a different date of birth means a different person. "
          "If the evidence is insufficient, answer UNSURE rather than guess.")


def parse_json(txt):
    """Last {...} object in the text (models may wrap JSON in prose or code fences)."""
    import re
    try:
        return json.loads(txt)
    except Exception:
        objs = re.findall(r"\{[^{}]*\}", txt, flags=re.S)
        if not objs:
            raise
        return json.loads(objs[-1])


def ask_llm(model, question_text, options, think=None):
    letters = [o["letter"] for o in options]
    prompt = (question_text + "\n\nAnswer with JSON: {\"answer\": one of " +
              json.dumps(letters + ["NONE", "UNSURE"]) +
              ", \"reason\": short justification}. NONE = the subject matches none of the options.")
    t0 = time.time()
    payload = {"model": model, "stream": False, "format": "json",
               "options": {"temperature": 0, "seed": 0, "num_predict": 120},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}]}
    if think is not None:   # reasoning models (gemma4): explicit on/off; thinking needs a larger budget
        payload["think"] = think
        if think:
            payload.pop("format")
            payload["options"]["num_predict"] = 4096
    out = ollama("/api/chat", payload, timeout=600)
    dt = time.time() - t0
    txt = out["message"]["content"]
    try:
        js = parse_json(txt)
        ans = str(js.get("answer", "UNSURE")).strip().upper()
        reason = str(js.get("reason", ""))[:300]
    except Exception:
        ans, reason = "UNSURE", "unparseable: " + txt[:100]
    if ans not in letters + ["NONE", "UNSURE"]:
        ans = "UNSURE"
    return ans, reason, dt


# ---------------- level 4: hosted large LLM (optional; data leaves the machine) ----------------
GEMINI_MODEL = "gemini-pro-latest"


def ask_gemini(question_text, options, model=GEMINI_MODEL):
    import os
    letters = [o["letter"] for o in options]
    prompt = (question_text + "\n\nAnswer with JSON: {\"answer\": one of " +
              json.dumps(letters + ["NONE", "UNSURE"]) +
              ", \"reason\": short justification}. NONE = the subject matches none of the options.")
    if model.startswith("gemma"):   # Gemma on this API: no system instruction, no JSON mode
        body = {"contents": [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
                "generationConfig": {"temperature": 0}}
    else:
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    t0 = time.time()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            break
        except Exception as e:   # rate limit / transient
            if attempt == 3:
                return "UNSURE", f"api error: {e}"[:200], time.time() - t0
            time.sleep(5 * (attempt + 1))
    dt = time.time() - t0
    try:
        parts = out["candidates"][0]["content"]["parts"]
        txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        js = parse_json(txt)
        ans = str(js.get("answer", "UNSURE")).strip().upper()
        reason = str(js.get("reason", ""))[:300]
    except Exception:
        ans, reason = "UNSURE", "unparseable"
    if ans not in letters + ["NONE", "UNSURE"]:
        ans = "UNSURE"
    return ans, reason, dt
