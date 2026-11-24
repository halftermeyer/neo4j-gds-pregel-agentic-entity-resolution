"""Equivalence check: Pregel procedure vs Python reference (graph_ops.reference_cluster)
on the rules-only state of dev and test. Read-only except n.cluster is not written."""
from common import driver
from graph_ops import pregel_cluster, reference_cluster

drv = driver()
for ds in ("dev", "test"):
    cl, dec, ref, moot, st = pregel_cluster(drv, ds, write=False)
    rc, rounds = reference_cluster(drv, ds)
    print(ds, "identical partition:", cl == rc, "| clusters", len(set(cl.values())), "| stats", st, "| ref rounds", rounds)
