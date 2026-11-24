package myns.er;

// Constrained entity-resolution clustering with GDS Pregel.
//
// Input projection (undirected), built by the caller:
//   nodes   : records, numeric property `profile` (long[]): strong identifiers encoded as
//             type * 1e12 + value (type 1 = date of birth yyyymmdd, type 2 = national-id hash).
//             Value 0 = nothing (padding, ignored).
//   rels    : candidate pairs, property `w` = (decision + 1) * 10 + score, score in [0,1].
//             decision 1 = MATCH (w in [20,21]), 0 = AMBIGUOUS (w in [10,11]).
//             NO edges (-1) are not projected.
//
// Semantics (synchronous rounds = supersteps):
//   each vertex proposes its best MATCH edge towards another cluster whose profile is compatible
//   with its own cluster's profile (vertex-parallel); the master then applies proposals in
//   descending score order (tie-break on original ids), re-checking compatibility against the
//   merged profiles (union-find). A merge is refused when two clusters hold different values for
//   the same strong-identifier type. Unlike WCC, a MATCH edge does not force a merge.
//   Stops when a round applies no merge.
//
// Output rows (kind):
//   node      : nodeId, cluster (min original id of members)
//   decisive  : AMBIGUOUS edge between two different, compatible clusters (an answer changes a cluster)
//   refused   : MATCH edge left between two clusters because their profiles conflict
//   moot      : AMBIGUOUS edge that cannot change anything (same cluster, or conflicting clusters)
//
// Classloader rules: everything public, nested types public static final, no records.

import org.neo4j.gds.RelationshipType;
import org.neo4j.gds.api.DatabaseId;
import org.neo4j.gds.api.Graph;
import org.neo4j.gds.api.GraphStore;
import org.neo4j.gds.api.nodeproperties.ValueType;
import org.neo4j.gds.api.properties.nodes.NodePropertyValues;
import org.neo4j.gds.beta.pregel.Element;
import org.neo4j.gds.beta.pregel.Messages;
import org.neo4j.gds.beta.pregel.Pregel;
import org.neo4j.gds.beta.pregel.PregelComputation;
import org.neo4j.gds.beta.pregel.PregelConfig;
import org.neo4j.gds.beta.pregel.PregelConfigImpl;
import org.neo4j.gds.beta.pregel.PregelResult;
import org.neo4j.gds.beta.pregel.PregelSchema;
import org.neo4j.gds.beta.pregel.context.ComputeContext;
import org.neo4j.gds.beta.pregel.context.MasterComputeContext;
import org.neo4j.gds.core.concurrency.DefaultPool;
import org.neo4j.gds.core.loading.CatalogRequest;
import org.neo4j.gds.core.loading.GraphStoreCatalog;
import org.neo4j.gds.mem.MemoryEstimateDefinition;
import org.neo4j.gds.mem.MemoryEstimations;
import org.neo4j.gds.progress.tracking.ProgressTracker;
import org.neo4j.gds.termination.TerminationFlag;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.*;
import java.util.stream.Stream;

public class ConstrainedClustering {

    @Context public GraphDatabaseService db;

    public static final long TYPE_DIV = 1_000_000_000_000L;

    /** Two sorted profiles conflict if they hold different values for the same type. */
    public static boolean compatible(long[] a, long[] b) {
        for (long x : a) {
            if (x <= 0) continue;
            long tx = x / TYPE_DIV;
            for (long y : b) {
                if (y <= 0) continue;
                if (y / TYPE_DIV == tx && y != x) return false;
            }
        }
        return true;
    }

    public static long[] union(long[] a, long[] b) {
        TreeSet<Long> s = new TreeSet<>();
        for (long x : a) if (x > 0) s.add(x);
        for (long y : b) if (y > 0) s.add(y);
        long[] r = new long[s.size()]; int i = 0;
        for (long x : s) r[i++] = x;
        return r;
    }

    public static final class Computation implements PregelComputation<PregelConfig> {
        public final int[][] matchNbr;      // MATCH neighbours (mapped ids)
        public final double[][] matchScore;
        public final long[] origId;
        public final int[] parent;          // union-find, only written by master
        public final long[][] profile;      // profile of each root, only written by master
        public final int[] propTarget;      // proposal written by vertex v in compute
        public final double[] propScore;
        public long merges = 0, refusedAtMaster = 0, rounds = 0;
        public boolean stable = false;

        public Computation(int[][] matchNbr, double[][] matchScore, long[][] nodeProfile, long[] origId) {
            this.matchNbr = matchNbr; this.matchScore = matchScore; this.origId = origId;
            int n = matchNbr.length;
            parent = new int[n]; profile = new long[n][];
            propTarget = new int[n]; propScore = new double[n];
            for (int i = 0; i < n; i++) { parent[i] = i; profile[i] = union(nodeProfile[i], new long[0]); propTarget[i] = -1; }
        }

        public int find(int x) {   // read-only find (no path compression) so compute() is race-free
            while (parent[x] != x) x = parent[x];
            return x;
        }

        @Override public PregelSchema schema(PregelConfig c) {
            return PregelSchema.from(new Element("proposal", ValueType.LONG));
        }

        @Override public void compute(ComputeContext<PregelConfig> ctx, Messages messages) {
            int v = (int) ctx.nodeId();
            int rv = find(v);
            int best = -1; double bestScore = -1; long bestTie = Long.MAX_VALUE;
            int[] nb = matchNbr[v];
            for (int i = 0; i < nb.length; i++) {
                int u = nb[i];
                int ru = find(u);
                if (ru == rv) continue;
                if (!compatible(profile[rv], profile[ru])) continue;
                double s = matchScore[v][i];
                long tie = origId[u];
                if (s > bestScore || (s == bestScore && tie < bestTie)) { best = u; bestScore = s; bestTie = tie; }
            }
            propTarget[v] = best; propScore[v] = bestScore;
            ctx.setNodeValue("proposal", best < 0 ? -1L : origId[best]);
            // A vertex with no viable proposal can never get one again: merges only remove
            // cross-cluster edges and only grow profiles (compatibility is monotone decreasing).
            if (best < 0) ctx.voteToHalt();
        }

        @Override public boolean masterCompute(MasterComputeContext<PregelConfig> ctx) {
            rounds++;
            int n = parent.length;
            List<long[]> props = new ArrayList<>();   // {v, u}
            for (int v = 0; v < n; v++) if (propTarget[v] >= 0) props.add(new long[]{v, propTarget[v]});
            final double[] ps = propScore;
            props.sort((a, b) -> {
                int c = Double.compare(ps[(int) b[0]], ps[(int) a[0]]);
                if (c != 0) return c;
                long a1 = Math.min(origId[(int) a[0]], origId[(int) a[1]]), b1 = Math.min(origId[(int) b[0]], origId[(int) b[1]]);
                if (a1 != b1) return Long.compare(a1, b1);
                long a2 = Math.max(origId[(int) a[0]], origId[(int) a[1]]), b2 = Math.max(origId[(int) b[0]], origId[(int) b[1]]);
                return Long.compare(a2, b2);
            });
            long applied = 0;
            for (long[] p : props) {
                int ra = find((int) p[0]), rb = find((int) p[1]);
                if (ra == rb) continue;
                if (!compatible(profile[ra], profile[rb])) { refusedAtMaster++; continue; }
                int root = Math.min(ra, rb), child = Math.max(ra, rb);
                parent[child] = root;
                profile[root] = union(profile[ra], profile[rb]);
                applied++;
            }
            Arrays.fill(propTarget, -1);
            merges += applied;
            stable = (applied == 0);
            return stable;   // stop when a round changed nothing
        }

        @Override public MemoryEstimateDefinition estimateDefinition(boolean async) { return MemoryEstimations::empty; }
    }

    public static final class Row {
        public String kind; public long nodeId; public long cluster;
        public Long otherNodeId; public Long otherCluster; public Double score;
        public long iterations; public boolean converged; public long merges;
    }

    @Procedure(name = "myns.erCluster", mode = Mode.READ)
    @Description("Constrained ER clustering along MATCH edges (profile conflicts refuse merges); returns clusters and decisive AMBIGUOUS edges.")
    public Stream<Row> erCluster(@Name("graphName") String graphName,
                                 @Name(value = "maxIterations", defaultValue = "100") long maxIterations,
                                 @Name(value = "concurrency", defaultValue = "4") long concurrency,
                                 @Name(value = "user", defaultValue = "neo4j") String user) {
        GraphStore gs = GraphStoreCatalog.get(CatalogRequest.of(user, DatabaseId.of(db.databaseName())), graphName).graphStore();
        Graph graph = gs.getGraph(gs.relationshipTypes(), Optional.of("w"));
        int n = (int) graph.nodeCount();
        long[] origId = new long[n];
        for (int v = 0; v < n; v++) origId[v] = graph.toOriginalNodeId(v);

        NodePropertyValues pv = graph.nodeProperties("profile");
        long[][] nodeProfile = new long[n][];
        for (int v = 0; v < n; v++) { long[] p = pv.longArrayValue(v); nodeProfile[v] = p == null ? new long[0] : p; }

        List<int[]> mN = new ArrayList<>(); List<double[]> mS = new ArrayList<>();
        int[][] matchNbr = new int[n][]; double[][] matchScore = new double[n][];
        List<long[]> ambig = new ArrayList<>();    // {v,u} with v<u
        List<Double> ambigScore = new ArrayList<>();
        List<long[]> match = new ArrayList<>(); List<Double> matchSc = new ArrayList<>();
        for (int s = 0; s < n; s++) {
            final int src = s;
            List<Integer> nbs = new ArrayList<>(); List<Double> scs = new ArrayList<>();
            graph.forEachRelationship(s, 0.0, (a, b, w) -> {
                int t = (int) b;
                if (w >= 20.0) {
                    nbs.add(t); scs.add(w - 20.0);
                    if (src < t) { match.add(new long[]{src, t}); matchSc.add(w - 20.0); }
                } else if (w >= 10.0) {
                    if (src < t) { ambig.add(new long[]{src, t}); ambigScore.add(w - 10.0); }
                }
                return true;
            });
            matchNbr[s] = nbs.stream().mapToInt(Integer::intValue).toArray();
            matchScore[s] = scs.stream().mapToDouble(Double::doubleValue).toArray();
        }

        PregelConfig config = new PregelConfigImpl.Builder()
            .maxIterations((int) maxIterations).concurrency((int) concurrency).build();
        Computation comp = new Computation(matchNbr, matchScore, nodeProfile, origId);
        PregelResult res = Pregel.create(graph, config, comp, DefaultPool.INSTANCE,
                                         ProgressTracker.NULL_TRACKER, TerminationFlag.RUNNING_TRUE).run();

        // cluster label = min original id among members
        long[] label = new long[n];
        Arrays.fill(label, Long.MAX_VALUE);
        int[] root = new int[n];
        for (int v = 0; v < n; v++) { root[v] = comp.find(v); label[root[v]] = Math.min(label[root[v]], origId[v]); }

        long its = res.ranIterations(); boolean conv = comp.stable; long merges = comp.merges;
        List<Row> rows = new ArrayList<>(n + ambig.size());
        for (int v = 0; v < n; v++) {
            Row r = new Row(); r.kind = "node"; r.nodeId = origId[v]; r.cluster = label[root[v]];
            r.iterations = its; r.converged = conv; r.merges = merges; rows.add(r);
        }
        for (int i = 0; i < ambig.size(); i++) {
            int a = (int) ambig.get(i)[0], b = (int) ambig.get(i)[1];
            int ra = root[a], rb = root[b];
            Row r = new Row();
            r.kind = (ra != rb && compatible(comp.profile[ra], comp.profile[rb])) ? "decisive" : "moot";
            r.nodeId = origId[a]; r.cluster = label[ra]; r.otherNodeId = origId[b]; r.otherCluster = label[rb];
            r.score = ambigScore.get(i); r.iterations = its; r.converged = conv; r.merges = merges; rows.add(r);
        }
        for (int i = 0; i < match.size(); i++) {
            int a = (int) match.get(i)[0], b = (int) match.get(i)[1];
            if (root[a] == root[b]) continue;
            Row r = new Row(); r.kind = "refused";
            r.nodeId = origId[a]; r.cluster = label[root[a]]; r.otherNodeId = origId[b]; r.otherCluster = label[root[b]];
            r.score = matchSc.get(i); r.iterations = its; r.converged = conv; r.merges = merges; rows.add(r);
        }
        return rows.stream();
    }
}
