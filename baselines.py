"""Baselines on the test set: WCC on candidate edges with score >= T, versus the constrained
Pregel clustering on exactly the same edge set (edge treated as MATCH). Read-only on the data
(uses its own projection names)."""
import json

from common import driver, run
from pipeline import load_truth, metrics

PROJ = """
MATCH (a:SpikeERRecord {ds: $ds})
OPTIONAL MATCH (a)-[r:SPK_ER_CANDIDATE]->(b:SpikeERRecord) WHERE r.score >= $t
WITH gds.graph.project($g, a, b,
       {sourceNodeProperties: a {.profile}, targetNodeProperties: CASE WHEN b IS NULL THEN null ELSE b {.profile} END,
        relationshipType: 'CANDIDATE', relationshipProperties: CASE WHEN r IS NULL THEN null ELSE {w: 20 + r.score} END},
       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.relationshipCount AS rels
"""


def main(ds="test"):
    drv = driver()
    truth, strong = load_truth(drv, ds)
    out = {}
    for t in (0.70, 0.75, 0.80, 0.85, 0.90):
        g = "spk_er_base"
        run(drv, "CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=g)
        rels = run(drv, PROJ, ds=ds, t=t, g=g)[0]["rels"]
        w = run(drv, "CALL gds.wcc.stream($g) YIELD nodeId, componentId RETURN gds.util.asNode(nodeId).rid AS rid, componentId AS c", g=g)
        p = run(drv, """CALL myns.erCluster($g) YIELD kind, nodeId, cluster WHERE kind = 'node'
                        RETURN gds.util.asNode(nodeId).rid AS rid, cluster AS c""", g=g)
        run(drv, "CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=g)
        out[str(t)] = {"edges": rels // 2,
                       "wcc": metrics({r["rid"]: r["c"] for r in w}, truth, strong),
                       "pregel_constrained": metrics({r["rid"]: r["c"] for r in p}, truth, strong)}
        print(t, json.dumps(out[str(t)]))
    json.dump(out, open("results/baselines_score_threshold.json", "w"), indent=1)


if __name__ == "__main__":
    main()
