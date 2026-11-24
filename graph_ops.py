"""GDS projection, Pregel constrained clustering, WCC baseline, and a Python reference
of the Pregel semantics (for the equivalence check)."""
import time
from collections import defaultdict

from common import USER, run

GRAPH = "spk_er_g"

PROJECT = """
MATCH (a:SpikeERRecord {ds: $ds})
OPTIONAL MATCH (a)-[r:SPK_ER_CANDIDATE]->(b:SpikeERRecord) WHERE r.decision >= 0
WITH gds.graph.project($g, a, b,
       {sourceNodeProperties: a {.profile}, targetNodeProperties: CASE WHEN b IS NULL THEN null ELSE b {.profile} END,
        relationshipType: 'CANDIDATE', relationshipProperties: CASE WHEN r IS NULL THEN null ELSE r {.w} END},
       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS rels
"""


def drop(drv, name=GRAPH):
    run(drv, "CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=name)


def project(drv, ds, name=GRAPH):
    drop(drv, name)
    return run(drv, PROJECT, ds=ds, g=name)[0]


def pregel_cluster(drv, ds, write=True):
    """Returns (cluster map rid->cluster, decisive edges, refused, moot, stats)."""
    t0 = time.time()
    proj = project(drv, ds)
    t1 = time.time()
    rows = run(drv, """
      CALL myns.erCluster($g, 100, 4, $user) YIELD kind, nodeId, cluster, otherNodeId, otherCluster, score, iterations, converged, merges
      WITH kind, nodeId, cluster, otherNodeId, otherCluster, score, iterations, converged, merges
      MATCH (n) WHERE id(n) = nodeId
      OPTIONAL MATCH (m) WHERE id(m) = otherNodeId
      RETURN kind, n.rid AS rid, cluster, m.rid AS other, otherCluster, score, iterations, converged, merges""", g=GRAPH, user=USER)
    t2 = time.time()
    drop(drv)
    clusters = {r["rid"]: r["cluster"] for r in rows if r["kind"] == "node"}
    decisive = [r for r in rows if r["kind"] == "decisive"]
    refused = [r for r in rows if r["kind"] == "refused"]
    moot = [r for r in rows if r["kind"] == "moot"]
    if write:
        run(drv, "UNWIND $rows AS r MATCH (n:SpikeERRecord {rid: r.rid}) SET n.cluster = r.c",
            rows=[{"rid": k, "c": v} for k, v in clusters.items()])
    st = {"project_s": round(t1 - t0, 3), "pregel_s": round(t2 - t1, 3),
          "iterations": rows[0]["iterations"] if rows else 0, "converged": rows[0]["converged"] if rows else True,
          "merges": rows[0]["merges"] if rows else 0, "nodes": proj["nodes"], "rels": proj["rels"],
          "decisive": len(decisive), "refused": len(refused), "moot": len(moot)}
    return clusters, decisive, refused, moot, st


def wcc(drv, ds, where):
    """WCC baseline on candidate edges selected by a Cypher predicate over r."""
    drop(drv, "spk_er_wcc")
    run(drv, f"""
      MATCH (a:SpikeERRecord {{ds: $ds}})
      OPTIONAL MATCH (a)-[r:SPK_ER_CANDIDATE]->(b:SpikeERRecord) WHERE {where}
      WITH gds.graph.project('spk_er_wcc', a, b, {{}}, {{undirectedRelationshipTypes: ['*']}}) AS g
      RETURN g.nodeCount""", ds=ds)
    rows = run(drv, """CALL gds.wcc.stream('spk_er_wcc') YIELD nodeId, componentId
                       RETURN gds.util.asNode(nodeId).rid AS rid, componentId""")
    drop(drv, "spk_er_wcc")
    return {r["rid"]: r["componentId"] for r in rows}


# ---------- Python reference of the Pregel semantics (executable spec) ----------
TYPE_DIV = 10**12


def compatible(a, b):
    for x in a:
        for y in b:
            if x > 0 and y > 0 and x // TYPE_DIV == y // TYPE_DIV and x != y:
                return False
    return True


def reference_cluster(drv, ds):
    nodes = run(drv, "MATCH (n:SpikeERRecord {ds:$ds}) RETURN id(n) AS id, n.rid AS rid, n.profile AS p", ds=ds)
    edges = run(drv, """MATCH (a:SpikeERRecord {ds:$ds})-[r:SPK_ER_CANDIDATE]->(b) WHERE r.decision >= 0
                        RETURN id(a) AS a, id(b) AS b, r.decision AS d, r.score AS s""", ds=ds)
    ids = sorted(n["id"] for n in nodes)
    rid = {n["id"]: n["rid"] for n in nodes}
    parent = {i: i for i in ids}
    prof = {n["id"]: sorted(set(x for x in n["p"] if x > 0)) for n in nodes}
    nbr = defaultdict(list)
    for e in edges:
        if e["d"] == 1:
            nbr[e["a"]].append((e["b"], e["s"]))
            nbr[e["b"]].append((e["a"], e["s"]))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    rounds = 0
    while True:
        rounds += 1
        props = []
        for v in ids:
            rv = find(v)
            best = None
            for u, s in nbr[v]:
                ru = find(u)
                if ru == rv or not compatible(prof[rv], prof[ru]):
                    continue
                if best is None or s > best[1] or (s == best[1] and u < best[0]):
                    best = (u, s)
            if best:
                props.append((v, best[0], best[1]))
        props.sort(key=lambda p: (-p[2], min(p[0], p[1]), max(p[0], p[1])))
        applied = 0
        for v, u, s in props:
            ra, rb = find(v), find(u)
            if ra == rb or not compatible(prof[ra], prof[rb]):
                continue
            root, child = min(ra, rb), max(ra, rb)
            parent[child] = root
            prof[root] = sorted(set(prof[ra]) | set(prof[rb]))
            applied += 1
        if applied == 0:
            break
    label = {}
    for v in ids:
        label.setdefault(find(v), min(w for w in ids if find(w) == find(v)))
    return {rid[v]: label[find(v)] for v in ids}, rounds
