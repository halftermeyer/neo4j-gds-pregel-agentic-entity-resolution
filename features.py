"""Normalisation, blocking and CANDIDATE edges, all string work done in Cypher.

Creates:
  (:SpikeERNickname {nick, full})                diminutive dictionary (partial, see nicknames below)
  (:SpikeERRecord)-[:SPK_ER_CANDIDATE {score, decision, f_*}]->(:SpikeERRecord)
decision: 1 MATCH, 0 AMBIGUOUS, -1 NO (rule-based, provenance decided_by='rule').
"""
import time

from common import driver, run

# Level-1 dictionary: common diminutives only. Deliberately partial versus the generator
# (Babeth, Fanfan, Nanard, Bébert, Bob, Riri, Titi, Lolo, Pierrot, Polo, Toine, Titine,
# Loulou, Fifine, Sacha, Patou, Jojo, Margot, Tony, Lisa, Fanny, Guigui, Doudou, Lolotte are absent).
DICT = {
    "alex": ["alexandre", "alexandra"], "nico": ["nicolas"], "chris": ["christophe", "christine", "christian"],
    "fred": ["frederic", "frederique"], "cathy": ["catherine"], "cathie": ["catherine"], "phil": ["philippe"],
    "dom": ["dominique"], "isa": ["isabelle"], "steph": ["stephane", "stephanie"], "ben": ["benjamin"],
    "tom": ["thomas"], "seb": ["sebastien"], "val": ["valerie"], "pat": ["patrick", "patricia"],
    "manu": ["emmanuel", "emmanuelle"], "jo": ["joseph", "josephine"], "charlie": ["charles", "charlotte"],
    "vic": ["victor"], "gaby": ["gabriel", "gabrielle"], "matt": ["mathieu"], "oli": ["olivier"],
    "dany": ["daniel", "danielle"], "eddy": ["edouard"], "jb": ["jeanbaptiste"],
}

NORMALISE = r"""
MATCH (n:SpikeERRecord)
WITH n,
     [t IN apoc.text.split(toLower(coalesce(n.first_name,'')), '[^\\p{L}]+') WHERE t <> '' | apoc.text.clean(t)] AS ft,
     [t IN apoc.text.split(toLower(coalesce(n.last_name,'')),  '[^\\p{L}]+') WHERE t <> '' | apoc.text.clean(t)] AS lt,
     trim(coalesce(n.first_name,'')) AS fr
WITH n, ft, lt,
     // initials-only first name: J. / J.-P. / J.P. / JP (2 upper-case letters)
     (size(ft) > 0 AND all(t IN ft WHERE size(t) = 1)) OR (size(ft) = 1 AND size(fr) = 2 AND fr = toUpper(fr)) AS ini
SET n.n_first = string.join(ft, ''), n.n_last = string.join(lt, ''),
    n.n_ftok = ft, n.n_ltok = lt, n.n_first_ini = ini,
    n.n_finit = CASE WHEN ini AND size(ft) = 1 AND size(ft[0]) = 2 THEN ft[0]
                     ELSE string.join([t IN ft | left(t,1)], '') END,
    n.n_lphon = apoc.text.doubleMetaphone(string.join(lt, '')),
    n.n_keys = coll.distinct([t IN ft + lt WHERE size(t) >= 3 | apoc.text.doubleMetaphone(t)])
WITH n,
     toLower(apoc.text.clean(coalesce(n.address,''))) AS dummy,
     apoc.text.split(toLower(coalesce(n.address,'')), '[^\\p{L}0-9]+') AS at
WITH n, [t IN at WHERE t <> '' | apoc.text.clean(t)] AS at
WITH n, [t IN at | CASE t WHEN 'r' THEN 'rue' WHEN 'av' THEN 'avenue' WHEN 'ave' THEN 'avenue'
                          WHEN 'bd' THEN 'boulevard' WHEN 'boul' THEN 'boulevard' WHEN 'pl' THEN 'place'
                          WHEN 'ch' THEN 'chemin' WHEN 'all' THEN 'allee' ELSE t END] AS at
SET n.n_addr = CASE WHEN size(at) = 0 THEN null ELSE string.join([t IN at WHERE NOT t =~ '[0-9]{5}'], ' ') END,
    n.n_pc = head([t IN at WHERE t =~ '[0-9]{5}']),
    n.n_phone = n.phone, n.n_email = toLower(n.email),
    n.profile = [x IN [CASE WHEN n.dob IS NULL THEN null ELSE 1000000000000 + n.dob END,
                       CASE WHEN n.nid IS NULL THEN null ELSE 2000000000000 + n.nid END] WHERE x IS NOT NULL] + [0]
"""

# Blocking: share a phonetic key of any name token (length >= 3), or same dob + same last-name initial.
BLOCK = """
MATCH (a:SpikeERRecord)
UNWIND a.n_keys AS k
MATCH (b:SpikeERRecord)
WHERE b.ds = a.ds AND a.rid < b.rid AND k IN b.n_keys
WITH DISTINCT a, b
RETURN count(*) AS pairs
"""

# Pair features as a reusable Cypher fragment over variables a, b (records).
# Name compatibility is evaluated in both orientations (direct, swapped first/last).
PAIR_FEATURES = r"""
WITH a, b,
  // first-name compatibility between token lists x and y, returns a class score
  [o IN [[a.n_first, a.n_ftok, a.n_first_ini, a.n_finit, b.n_first, b.n_ftok, b.n_first_ini, b.n_finit, a.n_last, b.n_last, false],
         [a.n_last, a.n_ltok, size(a.n_ltok) > 0 AND all(t IN a.n_ltok WHERE size(t) = 1), string.join([t IN a.n_ltok | left(t,1)], ''), b.n_first, b.n_ftok, b.n_first_ini, b.n_finit, a.n_first, b.n_last, true]] |
   {f1: o[0], ft1: o[1], i1: o[2], in1: o[3], f2: o[4], ft2: o[5], i2: o[6], in2: o[7], l1: o[8], l2: o[9], sw: o[10]}] AS orients
UNWIND orients AS o
WITH a, b, o,
     1.0 - apoc.text.jaroWinklerDistance(o.l1, o.l2) AS jw_last,
     CASE WHEN o.f1 = '' OR o.f2 = '' THEN 0.0 ELSE 1.0 - apoc.text.jaroWinklerDistance(o.f1, o.f2) END AS jw_first,
     apoc.text.doubleMetaphone(o.l1) = apoc.text.doubleMetaphone(o.l2) AS phon_last,
     EXISTS { MATCH (k:SpikeERNickname) WHERE (k.nick = o.f1 AND k.full = o.f2) OR (k.nick = o.f2 AND k.full = o.f1) } AS nick,
     EXISTS { MATCH (k1:SpikeERNickname {nick: o.f1}), (k2:SpikeERNickname {nick: o.f2}) WHERE k1.full = k2.full } AS nick2,
     (o.i1 OR o.i2) AND o.in1 <> '' AND o.in2 <> '' AND
       (o.in1 = o.in2 OR (o.i1 AND NOT o.i2 AND size(o.in1) = 1 AND left(o.in2,1) = o.in1)
                      OR (o.i2 AND NOT o.i1 AND size(o.in2) = 1 AND left(o.in1,1) = o.in2)) AS ini_ok,
     size(o.ft1) <> size(o.ft2) AND size(o.ft1) > 0 AND size(o.ft2) > 0 AND
       (o.ft1[0..size(o.ft2)] = o.ft2 OR o.ft2[0..size(o.ft1)] = o.ft1) AS prefix_ok
WITH a, b, o, jw_last, jw_first, phon_last, nick OR nick2 AS nick, ini_ok, prefix_ok,
     CASE WHEN o.f1 = o.f2 AND o.f1 <> '' THEN 1.0
          WHEN jw_first >= 0.93 AND abs(size(o.f1) - size(o.f2)) <= 1 THEN 0.92
          WHEN nick OR nick2 THEN 0.9
          WHEN prefix_ok THEN 0.75
          WHEN jw_first >= 0.85 AND NOT (o.i1 OR o.i2) THEN jw_first
          WHEN ini_ok THEN 0.7
          ELSE jw_first * 0.6 END AS f_first
WITH a, b, o, jw_last, jw_first, phon_last, nick, ini_ok, prefix_ok, f_first,
     0.5 * CASE WHEN phon_last AND jw_last < 0.9 THEN 0.9 ELSE jw_last END + 0.5 * f_first AS name_sim
ORDER BY name_sim DESC
WITH a, b, head(collect({jw_last: jw_last, jw_first: jw_first, phon_last: phon_last, nick: nick, ini_ok: ini_ok,
                         prefix_ok: prefix_ok, f_first: f_first, name_sim: name_sim, swapped: o.sw})) AS best
WITH a, b, best,
     CASE WHEN a.dob IS NULL OR b.dob IS NULL THEN 0 WHEN a.dob = b.dob THEN 1 ELSE -1 END AS dob,
     CASE WHEN a.nid IS NULL OR b.nid IS NULL THEN 0 WHEN a.nid = b.nid THEN 1 ELSE -1 END AS nid,
     CASE WHEN a.n_email IS NULL OR b.n_email IS NULL THEN 0 WHEN a.n_email = b.n_email THEN 1 ELSE -1 END AS email,
     CASE WHEN a.n_phone IS NULL OR b.n_phone IS NULL THEN 0 WHEN a.n_phone = b.n_phone THEN 1 ELSE -1 END AS phone,
     CASE WHEN a.n_addr IS NULL OR b.n_addr IS NULL THEN -1.0 ELSE 1.0 - apoc.text.jaroWinklerDistance(a.n_addr, b.n_addr) END AS addr,
     CASE WHEN a.n_pc IS NULL OR b.n_pc IS NULL THEN 0 WHEN a.n_pc = b.n_pc THEN 1 ELSE -1 END AS pc
WITH a, b, {
  f_jw_last: best.jw_last, f_jw_first: best.jw_first, f_phon_last: CASE WHEN best.phon_last THEN 1 ELSE 0 END,
  f_nick: CASE WHEN best.nick THEN 1 ELSE 0 END, f_ini: CASE WHEN best.ini_ok THEN 1 ELSE 0 END,
  f_prefix: CASE WHEN best.prefix_ok THEN 1 ELSE 0 END, f_first: best.f_first, f_name: best.name_sim,
  f_swapped: CASE WHEN best.swapped THEN 1 ELSE 0 END, f_dob: dob, f_nid: nid, f_email: email, f_phone: phone,
  f_addr: addr, f_pc: pc, f_same_src: CASE WHEN a.source = b.source THEN 1 ELSE 0 END,
  f_first_ini: CASE WHEN a.n_first_ini OR b.n_first_ini THEN 1 ELSE 0 END
} AS f
"""

# Rule layer (fixed a priori, not tuned on test):
#   NO     : strong-id conflict (dob or nid differ), or name_sim < 0.80
#   MATCH  : same nid; or name_sim >= 0.90 and first-name exact/near and (same dob or same email)
#   else   : AMBIGUOUS
RULE = """
WITH a, b, f,
     0.55 * f.f_name + 0.15 * CASE f.f_dob WHEN 1 THEN 1.0 WHEN 0 THEN 0.5 ELSE 0.0 END
       + 0.10 * CASE WHEN f.f_addr < 0 THEN 0.5 ELSE f.f_addr END
       + 0.10 * CASE f.f_email WHEN 1 THEN 1.0 WHEN 0 THEN 0.5 ELSE 0.0 END
       + 0.10 * CASE f.f_phone WHEN 1 THEN 1.0 WHEN 0 THEN 0.5 ELSE 0.0 END AS score
WITH a, b, f, score,
     CASE WHEN f.f_dob = -1 OR f.f_nid = -1 THEN -1
          WHEN f.f_nid = 1 THEN 1
          WHEN f.f_name >= 0.90 AND f.f_first >= 0.92 AND (f.f_dob = 1 OR f.f_email = 1) THEN 1
          WHEN f.f_name >= 0.75 THEN 0
          WHEN f.f_jw_last >= 0.92 AND (f.f_dob = 1 OR f.f_email = 1 OR f.f_phone = 1 OR f.f_addr >= 0.9) THEN 0
          ELSE -1 END AS decision
"""

CREATE_EDGES = """
MATCH (a:SpikeERRecord)
CALL (a) {
  UNWIND a.n_keys AS k
  MATCH (b:SpikeERRecord) WHERE b.ds = a.ds AND a.rid < b.rid AND k IN b.n_keys RETURN b
  UNION
  MATCH (b:SpikeERRecord) WHERE b.ds = a.ds AND a.rid < b.rid AND
        (b.dob = a.dob OR b.n_phone = a.n_phone OR b.n_email = a.n_email) RETURN b
}
WITH DISTINCT a, b
""" + PAIR_FEATURES + RULE + """
WHERE f.f_name >= 0.70 OR f.f_nid = 1 OR f.f_dob = 1
   OR (f.f_jw_last >= 0.92 AND (f.f_email = 1 OR f.f_phone = 1 OR f.f_addr >= 0.9))
CREATE (a)-[r:SPK_ER_CANDIDATE]->(b)
SET r += f, r.score = score, r.decision = decision, r.rule_decision = decision,
    r.decided_by = 'rule', r.rationale = 'rule layer', r.w = (decision + 1) * 10 + score
RETURN decision, count(*) AS c ORDER BY decision
"""


def main():
    drv = driver()
    t0 = time.time()
    run(drv, "MATCH (:SpikeERRecord)-[r:SPK_ER_CANDIDATE]->() DELETE r")
    run(drv, "MATCH (k:SpikeERNickname) DELETE k")
    run(drv, "UNWIND $rows AS r UNWIND r.full AS f CREATE (:SpikeERNickname {nick: r.nick, full: f})",
        rows=[{"nick": k, "full": v} for k, v in DICT.items()])
    run(drv, "CREATE INDEX spk_er_nick IF NOT EXISTS FOR (k:SpikeERNickname) ON (k.nick)")
    run(drv, NORMALISE)
    print("normalised", round(time.time() - t0, 2), "s")
    print("blocked pairs", run(drv, BLOCK))
    t1 = time.time()
    print(run(drv, CREATE_EDGES))
    print("edges", round(time.time() - t1, 2), "s")
    # blocking recall vs truth (evaluation only)
    print(run(drv, """
      MATCH (a:SpikeERRecord), (b:SpikeERRecord) WHERE a.ds = b.ds AND a.rid < b.rid AND a.truth_eid = b.truth_eid
      RETURN a.ds AS ds, count(*) AS true_pairs,
             sum(CASE WHEN EXISTS {(a)-[:SPK_ER_CANDIDATE]-(b)} THEN 1 ELSE 0 END) AS with_edge"""))


if __name__ == "__main__":
    main()
