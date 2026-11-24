"""Synthetic person records with known truth, loaded as :SpikeERRecord.

Two datasets with different seeds: ds='dev' (classifier training, threshold choice)
and ds='test' (everything reported). The pipeline never reads `truth_eid`.

Usage: python gen_data.py [--entities 300]
"""
import argparse
import random
import unicodedata

from common import driver, run

FIRST = [
    "Jean", "Jeanne", "Jean-Pierre", "Jean-Paul", "Jean-Marc", "Jean-Baptiste", "Jean-Philippe",
    "Marie", "Marie-Claire", "Marie-Thérèse", "Anne-Sophie", "Anne", "Pierre", "Paul", "Michel",
    "Michèle", "Nicolas", "Nicole", "Christophe", "Christine", "Christian", "Frédéric", "Frédérique",
    "Alexandre", "Alexandra", "Catherine", "Philippe", "François", "Françoise", "Dominique",
    "Daniel", "Danielle", "Laurent", "Laurence", "Stéphane", "Stéphanie", "Guillaume", "Isabelle",
    "Élisabeth", "Thomas", "Julien", "Julie", "Antoine", "Sébastien", "Valérie", "Patrick",
    "Patricia", "Emmanuel", "Emmanuelle", "Robert", "Bernard", "Joseph", "Joséphine", "Charles",
    "Charlotte", "Louis", "Louise", "Henri", "Victor", "Gabriel", "Gabrielle", "Mathieu",
    "Olivier", "Sylvain", "Sylvie", "Thierry", "Benjamin", "Édouard", "Hélène", "Margaux",
]
LAST = [
    "Dupont", "Martin", "Bernard", "Dubois", "Durand", "Lefebvre", "Leroy", "Moreau", "Simon",
    "Laurent", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier", "Morel",
    "Girard", "André", "Mercier", "Blanc", "Guérin", "Boyer", "Garnier", "Chevalier", "François",
    "Legrand", "Gauthier", "Perrin", "Robin", "Clément", "Morin", "Nicolas", "Henry", "Roussel",
    "Mathieu", "Gautier", "Masson", "Marchand", "Duval", "Denis", "Lemaire", "Faure", "Rousseau",
    "Bonnet", "Lambert", "Fontaine", "Rivière", "Brunet", "Renaud", "Thibault", "Arnaud",
    "Rolland", "Perret", "Delaunay", "Rey", "Benoît", "Hamon", "Charpentier", "Pasquier",
]
# Diminutives used by the generator. The level-1 dictionary (nicknames.py) covers only part
# of them: the rest needs "world knowledge" (by construction; stated in the report).
NICK = {
    "Alexandre": ["Alex"], "Alexandra": ["Alex", "Sacha"], "Nicolas": ["Nico"],
    "Christophe": ["Chris"], "Christine": ["Chris", "Titine"], "Christian": ["Chris"],
    "Frédéric": ["Fred"], "Frédérique": ["Fred"], "Catherine": ["Cathy", "Cathie"],
    "Philippe": ["Phil"], "François": ["Fanfan"], "Françoise": ["Fanfan"],
    "Dominique": ["Dom"], "Élisabeth": ["Babeth", "Lisa"], "Isabelle": ["Isa"],
    "Stéphane": ["Steph"], "Stéphanie": ["Steph", "Fanny"], "Benjamin": ["Ben"],
    "Thomas": ["Tom"], "Antoine": ["Toine", "Tony"], "Sébastien": ["Seb"], "Valérie": ["Val"],
    "Patrick": ["Pat"], "Patricia": ["Pat", "Patou"], "Emmanuel": ["Manu"],
    "Emmanuelle": ["Manu"], "Robert": ["Bob", "Bébert"], "Bernard": ["Nanard"],
    "Joseph": ["Jo", "Jojo"], "Joséphine": ["Jo", "Fifine"], "Charles": ["Charlie"],
    "Charlotte": ["Charlie", "Lolotte"], "Louis": ["Loulou"], "Henri": ["Riri"], "Victor": ["Vic"],
    "Gabriel": ["Gaby"], "Gabrielle": ["Gaby"], "Mathieu": ["Matt"], "Olivier": ["Oli"],
    "Thierry": ["Titi"], "Daniel": ["Dany"], "Danielle": ["Dany"], "Laurent": ["Lolo"],
    "Laurence": ["Lolo"], "Pierre": ["Pierrot"], "Paul": ["Polo"], "Édouard": ["Eddy", "Doudou"],
    "Margaux": ["Margot"], "Jean-Baptiste": ["JB"], "Guillaume": ["Guigui"],
}
# Confusable first names for relatives (hard negatives)
CONFUSABLE = [("Jean", "Jeanne"), ("Jean-Pierre", "Jeanne"), ("Jean-Pierre", "Jean-Paul"),
              ("Michel", "Michèle"), ("Daniel", "Danielle"), ("Frédéric", "Frédérique"),
              ("Christophe", "Christine"), ("Christian", "Christine"), ("Louis", "Louise"),
              ("Emmanuel", "Emmanuelle"), ("Gabriel", "Gabrielle"), ("Joseph", "Joséphine"),
              ("Patrick", "Patricia"), ("Alexandre", "Alexandra"), ("Nicolas", "Nicole"),
              ("François", "Françoise"), ("Julien", "Julie"), ("Jean-Marc", "Jean-Paul")]
STREETS = ["rue de la Paix", "avenue Victor Hugo", "boulevard Voltaire", "rue du Moulin",
           "place de la République", "rue Pasteur", "avenue Jean Jaurès", "rue des Lilas",
           "chemin des Vignes", "rue de la Gare", "boulevard Saint-Michel", "rue Nationale",
           "allée des Tilleuls", "rue du Général Leclerc", "avenue de la Liberté", "rue Gambetta"]
CITIES = [("75002", "Paris"), ("69003", "Lyon"), ("13001", "Marseille"), ("31000", "Toulouse"),
          ("33000", "Bordeaux"), ("59000", "Lille"), ("44000", "Nantes"), ("67000", "Strasbourg"),
          ("35000", "Rennes"), ("06000", "Nice"), ("92100", "Boulogne-Billancourt"), ("78000", "Versailles")]
ABBR = {"rue": ["r.", "R"], "avenue": ["av.", "ave", "AV"], "boulevard": ["bd", "boul."],
        "place": ["pl."], "chemin": ["ch."], "allée": ["all."]}
PHONETIC = [("ont", "ond"), ("and", "ant"), ("vre", "vres"), ("ault", "aut"), ("ph", "f"),
            ("eau", "o"), ("y", "i"), ("ier", "iez"), ("ll", "l"), ("nn", "n"), ("tt", "t"),
            ("ss", "s"), ("ert", "ere"), ("et", "ais")]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class Gen:
    def __init__(self, seed, n_entities, ds):
        self.r = random.Random(seed)
        self.n = n_entities
        self.ds = ds
        self.entities = []
        self.records = []
        self.used_nid = set()
        self.used_email = set()

    def dob(self, lo=1940, hi=2002):
        r = self.r
        return r.randint(lo, hi) * 10000 + r.randint(1, 12) * 100 + r.randint(1, 28)

    def nid(self):
        while True:
            x = self.r.randint(10**8, 10**9 - 1)
            if x not in self.used_nid:
                self.used_nid.add(x)
                return x

    def address(self):
        r = self.r
        pc, city = r.choice(CITIES)
        return {"num": r.randint(1, 120), "street": r.choice(STREETS), "pc": pc, "city": city}

    def phone(self):
        return "0" + str(self.r.choice([1, 6, 7])) + "".join(str(self.r.randint(0, 9)) for _ in range(8))

    def new_entity(self, first=None, last=None, addr=None, phone=None, kind="base", parent=None):
        e = {"eid": f"{self.ds}-E{len(self.entities):04d}", "first": first or self.r.choice(FIRST),
             "last": last or self.r.choice(LAST), "dob": self.dob(), "nid": self.nid(),
             "addr": addr or self.address(), "phone": phone or self.phone(), "kind": kind,
             "parent": parent}
        e["email_user"] = strip_accents(f"{e['first']}.{e['last']}").lower().replace(" ", "").replace("-", "")
        e["email_dom"] = self.r.choice(["gmail.com", "orange.fr", "free.fr", "yahoo.fr", "laposte.net"])
        while f"{e['email_user']}@{e['email_dom']}" in self.used_email:   # e-mails are unique per person
            e["email_user"] += str(self.r.randint(0, 9))
        self.used_email.add(f"{e['email_user']}@{e['email_dom']}")
        self.entities.append(e)
        return e

    def build_entities(self):
        r = self.r
        while len(self.entities) < self.n:
            x = r.random()
            if x < 0.62 or len(self.entities) < 10:
                self.new_entity()
            elif x < 0.85:   # relative: same last name + address (+ landline), confusable or same first name
                base = r.choice(self.entities)
                pair = [p for p in CONFUSABLE if base["first"] in p]
                if pair and r.random() < 0.7:
                    p = r.choice(pair)
                    first = p[1] if p[0] == base["first"] else p[0]
                else:
                    first = base["first"] if r.random() < 0.5 else r.choice(FIRST)   # father / son
                self.new_entity(first=first, last=base["last"], addr=dict(base["addr"]),
                                phone=base["phone"] if r.random() < 0.4 else None,
                                kind="relative", parent=base["eid"])
            else:            # homonym: same name, unrelated person
                base = r.choice(self.entities)
                self.new_entity(first=base["first"], last=base["last"], kind="homonym", parent=base["eid"])

    # ---- record-level variations ----
    def typo(self, s):
        r = self.r
        if len(s) < 4:
            return s
        i = r.randint(1, len(s) - 2)
        op = r.choice(["sub", "del", "swap", "dup"])
        if op == "sub":
            return s[:i] + r.choice("aeioutnrsld") + s[i + 1:]
        if op == "del":
            return s[:i] + s[i + 1:]
        if op == "swap":
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:i] + s[i] + s[i:]

    def phonetic(self, s):
        opts = [(a, b) for a, b in PHONETIC if a in s.lower()]
        if not opts:
            return self.typo(s)
        a, b = self.r.choice(opts)
        idx = s.lower().rfind(a)
        return s[:idx] + b + s[idx + len(a):]

    def vary_first(self, first, ops):
        r = self.r
        parts = first.split("-")
        x = r.random()
        if x < 0.10 and first in NICK:
            ops.append("nickname")
            return r.choice(NICK[first])
        if x < 0.20:
            ops.append("initials")
            if len(parts) > 1:
                return r.choice([".".join(p[0] for p in parts) + ".", "-".join(p[0] + "." for p in parts),
                                 "".join(p[0] for p in parts), parts[0][0] + "."])
            return parts[0][0] + "."
        if x < 0.27 and len(parts) > 1:
            ops.append("first_part_only")
            return parts[0]
        if x < 0.31 and len(parts) > 1:
            ops.append("composed_space")
            return " ".join(parts)
        if x < 0.36:
            ops.append("first_typo")
            return self.typo(first)
        return first

    def vary_last(self, last, ops):
        x = self.r.random()
        if x < 0.10:
            ops.append("last_typo")
            return self.typo(last)
        if x < 0.18:
            ops.append("last_phonetic")
            return self.phonetic(last)
        return last

    def fmt_address(self, a, ops):
        r = self.r
        street = a["street"]
        if r.random() < 0.4:
            w = street.split(" ")[0]
            if w in ABBR:
                street = r.choice(ABBR[w]) + street[len(w):]
                ops.append("addr_abbrev")
        s = f"{a['num']}{r.choice([' ', ', ', ' '])}{street}"
        if r.random() < 0.8:
            s += f" {a['pc']}"
        s += f" {a['city']}"
        if r.random() < 0.3:
            s = s.upper()
        return s

    def record(self, e, src, idx):
        r = self.r
        ops = []
        first = e["first"] if idx == 0 and r.random() < 0.7 else self.vary_first(e["first"], ops)
        last = e["last"] if idx == 0 and r.random() < 0.7 else self.vary_last(e["last"], ops)
        if r.random() < 0.3:
            first, last = strip_accents(first), strip_accents(last)
            ops.append("no_accents")
        if r.random() < 0.2:
            last = last.upper()
            ops.append("upper_last")
        if r.random() < 0.05:
            first, last = last, first
            ops.append("inverted")
        p_dob = {"KYC": 0.95, "CRM": 0.4, "LEGACY": 0.6}[src]
        p_nid = {"KYC": 0.8, "CRM": 0.0, "LEGACY": 0.2}[src]
        dob = e["dob"] if r.random() < p_dob else None
        if dob and r.random() < 0.015:       # day/month swap typo on a true match (recall cost)
            y, m, d = dob // 10000, dob // 100 % 100, dob % 100
            if d <= 12 and d != m:
                dob = y * 10000 + d * 100 + m
                ops.append("dob_typo")
        nid = e["nid"] if r.random() < p_nid else None
        x = r.random()
        if x < 0.60:
            addr = self.fmt_address(e["addr"], ops)
        elif x < 0.82:
            addr = self.fmt_address(self.address(), ops)
            ops.append("moved")
        else:
            addr = None
        email = f"{e['email_user']}@{e['email_dom']}" if src == "CRM" and r.random() < 0.6 else None
        phone = e["phone"] if r.random() < {"KYC": 0.3, "CRM": 0.6, "LEGACY": 0.4}[src] else None
        return {"rid": f"{self.ds}-R{len(self.records):05d}", "ds": self.ds, "source": src,
                "first_name": first, "last_name": last, "dob": dob, "nid": nid, "address": addr,
                "email": email, "phone": phone, "truth_eid": e["eid"], "truth_kind": e["kind"],
                "variation": ops}

    def build(self):
        self.build_entities()
        r = self.r
        for e in self.entities:
            k = r.choices([1, 2, 3, 4], weights=[30, 35, 25, 10])[0]
            srcs = [r.choice(["CRM", "KYC", "LEGACY"]) for _ in range(k)]
            for i, s in enumerate(srcs):
                self.records.append(self.record(e, s, i))
        r.shuffle(self.records)
        return self.records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=300)
    a = ap.parse_args()
    drv = driver()
    run(drv, "MATCH (n:SpikeERRecord) DETACH DELETE n")
    run(drv, "CREATE CONSTRAINT spk_er_rid IF NOT EXISTS FOR (n:SpikeERRecord) REQUIRE n.rid IS UNIQUE")
    for ds, seed in (("dev", 1), ("test", 2)):
        recs = Gen(seed, a.entities, ds).build()
        run(drv, """UNWIND $rows AS row
                    CREATE (n:SpikeERRecord) SET n = row""", rows=recs)
        ents = len({x["truth_eid"] for x in recs})
        print(f"{ds}: {len(recs)} records, {ents} entities")
    print(run(drv, """MATCH (n:SpikeERRecord) UNWIND n.variation AS v
                      RETURN n.ds AS ds, v, count(*) AS c ORDER BY ds, c DESC"""))


if __name__ == "__main__":
    main()
