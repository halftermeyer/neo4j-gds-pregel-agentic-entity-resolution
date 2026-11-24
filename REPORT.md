# PoC résolution d'entités : Pregel contraint + cascade de décideurs

Date : 2026-11-23, complété le 2026-11-24 (série Gemma). Neo4j 2026.10.2 Enterprise + GDS 2026.10.2 (local, base `entity-resolution`), Ollama local, Gemini et Gemma via l'API Google.
Tous les chiffres ci-dessous viennent d'exécutions réelles (fichiers `results/*.json`). Chaque configuration a été exécutée **une seule fois** : pas d'intervalle de confiance, et les écarts de quelques points entre configurations voisines ne sont pas significatifs.

## En bref

1. **La contrainte de profil fonctionne, et c'est là que le graphe apporte le plus.** Sur exactement les mêmes arêtes (score ≥ 0,75), WCC atteint une précision par paire de 0,40 avec 39 clusters qui mélangent des dates de naissance ou des identifiants différents. Le clustering Pregel contraint atteint 0,84 avec 0 cluster mixte, pour un rappel équivalent (0,85 contre 0,91).
2. **Le pari « un classifieur sur features couvre l'essentiel » tient en partie.**
   - Les règles Cypher seules donnent une précision de 1,00 pour un rappel de 0,41.
   - Le classifieur L1 tranche à peu près la moitié des questions décisives, avec 91 à 97,5 % de réponses justes selon l'exécution (272 sur 279 quand il est seul). Le rappel monte à 0,61, la précision reste à 1,00, pour environ 50 s de calcul en tout.
   - Il reste environ 140 questions sur 646 fiches.
3. **Les embeddings locaux (L2) n'apportent rien** : 3 questions tranchées sur 243.
4. **Les petits LLM génératifs locaux (3B, 8B) sont nuisibles ici.**
   - Sur le même résidu, llama3.2 3B a 40 % de réponses justes et llama3 8B 64 %, alors que la référence triviale « toujours l'option la mieux scorée » en a 74 %.
   - Ils ne s'abstiennent presque jamais et fusionnent des personnes différentes : 10 cas « aucun » sur 11 sont faux.
   - La précision par paire tombe à 0,79–0,82.
4 bis. **Un modèle local de 31B (gemma4:31b, Ollama, sans mode réflexion) rejoint Gemini pro.** Ce point révise la conclusion initiale « le local ne suffit pas ».
   - Dans la cascade : précision 0,934, rappel 0,856, F1 0,894, 242 entités exactes sur 300. C'est la meilleure exécution du PoC sur ces trois mesures, à précision égale à Gemini pro (0,930, rappel 0,806). Une seule exécution : l'écart de rappel avec Gemini n'est pas établi.
   - Il s'abstient davantage que llama3 (27 % du résidu laissé UNSURE dans la cascade) et se trompe beaucoup moins sur les cas « aucun » (2 fausses sur 11 contre 10 pour llama3 8B). Il s'abstient moins que Gemini sur les cas « nom seul » (11 UNSURE sur 25 contre 25) et en tranche donc certains au hasard de l'a priori.
   - Coût : 2,4 s médian par appel sur un Mac (MLX), cascade complète en 526 s contre 1 074 s pour Gemini, **sans que les fiches sortent de la machine**.
   - Le mode réflexion (`think`) ne l'améliore pas (précision 0,919) et coûte 5 à 9 fois plus de temps (cascade de 80 min, p95 de 96 s par appel).
5. **Le gros LLM hébergé (Gemini, `gemini-pro-latest`) est le seul qui sache s'abstenir.**
   - Il répond à environ 50 % du résidu, avec 80 à 90 % de réponses justes quand il répond. Précision finale 0,93, rappel 0,81.
   - Il s'abstient sur 23 à 25 des 25 questions où seul le nom concorde, qui sont indécidables à partir des fiches.
   - Coûts : environ 4 s par appel, et les fiches sortent de la machine.
6. **Les trois contextes de prompt ne se distinguent pas de façon nette** avec 77 questions. Pour Gemini, les résultats vont de 86 à 90 % de réponses justes sur les questions tranchées, ce qui reste dans le bruit. Pour llama3 8B, le contexte « fiche contre cluster » fait le moins bien (65 %, contre 75 à 77 % pour les deux autres).

Ce qu’il faut retenir pour un contexte bancaire : **règles + Pregel contraint + classifieur calibré** donnent une précision de 1,00 avec un rappel de 0,61, entièrement en local et auditable. Le résidu se partage en deux :
- environ un tiers est **indécidable** sans donnée supplémentaire et relève d'un humain ou d'une collecte ;
- le reste peut être monté à un modèle génératif, à condition qu'il **sache dire UNSURE**. Les modèles locaux de 3B et 8B testés ne le savent pas ; **un 31B local (gemma4) le sait suffisamment** pour atteindre la précision de Gemini pro (0,93) sans faire sortir les données. Aucun ne garde la précision de 1,00 : monter au LLM reste un arbitrage précision contre rappel.

## Ce qui a été construit

| Fichier | Rôle |
|---|---|
| `gen_data.py` | Jeu synthétique avec vérité connue : `dev` (graine 1) et `test` (graine 2), 300 personnes chacun, 649 et 646 fiches, labels `SpikeERRecord`. |
| `features.py` | Normalisation, blocage et arêtes `SPK_ER_CANDIDATE` avec features, score et code de décision par règle, **en Cypher** (APOC doubleMetaphone, Jaro-Winkler, dictionnaire de diminutifs en nœuds `SpikeERNickname`). |
| `pregel/…/ConstrainedClustering.java` | Procédure `myns.erCluster` (GDS Pregel) : clustering contraint par profils `long[]`, qui renvoie les clusters, les arêtes ambiguës décisives, les MATCH refusés et les arêtes ambiguës sans effet (« moot »). |
| `graph_ops.py` | Projection GDS, appel de la procédure, WCC, **référence Python** de la sémantique Pregel. |
| `check_pregel.py` | Équivalence Pregel contre référence : partition **identique** sur dev et test. |
| `models.py` | L1 et L2 (gradient boosting calibré), L3 Ollama, L4 Gemini. |
| `pipeline.py` | Boucle externe : Pregel, questions à choix, cascade, décisions écrites avec leur provenance, nouvelle projection, jusqu'à convergence. |
| `context_exp.py` | Les trois contextes de prompt sur les mêmes 77 questions. |
| `baselines.py` | WCC contre Pregel contraint sur les arêtes au-dessus d'un seuil. |
| `residual_analysis.py` | Part du résidu décidable ou non. |
| `tables.py` | Génère les tableaux de ce rapport à partir de `results/`. |

### 1. Jeu synthétique

Variations injectées : fautes de frappe, variantes phonétiques (Dupont/Dupond, -ault/-aut), initiales (J.P., J.-P., JP, J.), premier élément seul d'un prénom composé, diminutifs, inversion prénom/nom (5 %), majuscules, accents retirés, date absente (selon la source : KYC 95 % présente, CRM 40 %, LEGACY 60 %), inversion jour/mois de la date (1,5 %), adresses abrégées ou en majuscules, déménagements (22 %), sources CRM, KYC et LEGACY.

Négatifs difficiles :
- **apparentés (23 %)** : même nom, même adresse, parfois même téléphone fixe, prénom confondable (Jean/Jeanne, Michel/Michèle, Daniel/Danielle…) ou identique (père et fils) ;
- **homonymes (15 %)** : même prénom et même nom, personne différente.

**Biais assumé :** le générateur utilise environ 50 diminutifs, et le dictionnaire de L1 n'en couvre qu'une partie (Babeth, Nanard, Bébert, Lolo, Riri… n'y sont pas). Ce trou dans la « connaissance du monde » existe **par construction**.

### 2. Normalisation, blocage et score (Cypher)

- **Tokens** nettoyés (accents, casse), initiales, `doubleMetaphone` par token.
- **Blocage :** clé phonétique commune, ou même date, téléphone ou e-mail. Une arête est créée si la similarité de nom est ≥ 0,70 ou s'il y a un indice fort. **Rappel du blocage sur test : 505 paires vraies sur 515 (98 %).**
- **Features :** Jaro-Winkler sur nom et prénom, concordance phonétique, diminutif du dictionnaire, initiales compatibles, prénom préfixe, orientation inversée, date, identifiant, e-mail, téléphone, Jaro-Winkler sur l'adresse normalisée, code postal.
- **Règles fixées a priori :**
  - NO si la date ou l'identifiant diffère ;
  - MATCH si l'identifiant est égal, ou si la similarité de nom est ≥ 0,90, le prénom exact ou presque, et la date ou l'e-mail égal ;
  - AMBIGUOUS si la similarité de nom est ≥ 0,75 ou si le nom est identique avec un indice corroborant ;
  - NO sinon.
- **Résultat sur test :** 192 MATCH (tous justes), 1 248 AMBIGUOUS (308 vrais, 940 faux), 2 354 NO (dont 5 vrais, presque tous dus à une inversion jour/mois de la date).

### 3. Procédure Pregel `myns.erCluster`

- **Entrée :**
  - projection non orientée ;
  - sur chaque nœud, `profile: long[]` codé `type·10¹² + valeur` (type 1 : date, type 2 : hash de l'identifiant) ;
  - sur chaque arête, `w = (décision+1)·10 + score`, ce qui fait passer décision et score dans un seul double.
- **Super-étape :**
  - **côté sommet, en parallèle :** chaque sommet propose sa meilleure arête MATCH vers un autre cluster dont le profil est compatible ;
  - **côté master :** les propositions sont appliquées par score décroissant, avec départage total sur les identifiants. La compatibilité est revérifiée sur le profil fusionné (union-find), et une fusion est refusée si un même type porte deux valeurs.
  - Arrêt quand une ronde ne fusionne plus rien. La compatibilité ne fait que décroître au fil des fusions, ce qui garantit la terminaison.
- **Choix d'architecture à connaître :** la contrainte porte sur le cluster entier, pas sur le voisinage. L'état global (union-find et profils) vit donc dans `masterCompute`, et la partie vertex-centric se limite au calcul des propositions. C'est correct et déterministe, mais ce n'est pas du Pregel « pur message ».
- **Sortie :**
  - `node` : le cluster de chaque fiche ;
  - `decisive` : les arêtes ambiguës entre deux clusters différents et compatibles, dont la réponse changerait un cluster ;
  - `moot` : les arêtes ambiguës dans un même cluster ou entre clusters en conflit, inutiles à poser ;
  - `refused` : les MATCH bloqués par un conflit.
- **Validation :** partition identique à la référence Python sur dev et test (`check_pregel.py`). Sur test, environ 0,04 s de projection et 0,04 s de Pregel pour 646 nœuds et 2 880 relations.
- **Écart mineur :** `ranIterations` vaut 2 alors que la référence compte 3 rondes, dont la dernière vide. C'est une question de convention de comptage, sans effet sur la partition.

### 4. Boucle externe

- **À chaque tour :**
  1. Projection, puis Pregel, puis écriture de `n.cluster`.
  2. Les arêtes décisives sont regroupées par paire de clusters, et une **question à choix** est construite par cluster à rattacher : le plus petit, avec au plus deux options A et B triées par meilleur score, plus « aucun » et « UNSURE ».
  3. Chaque cluster figure dans une seule question par tour.
- **Fiches de cluster en Cypher :** variantes de nom `[..6]`, dates `[..3]`, adresses `[..3]`, e-mails, téléphones, sources, nombre de liens confirmés.
- **Décisions écrites sur les arêtes :** `decision`, `decided_by` (règle, L1, L2, L3a, L3b, L4), `model`, `rationale`, `p` (probabilités du classifieur), `round`, `ctx`.
- **Cache :**
  - un MATCH est définitif ;
  - un NO donné par un modèle est rouvert seulement si les deux clusters partagent désormais une valeur d'identifiant fort qu'ils ne partageaient pas (2 réouvertures dans les exécutions L3a et L3a+L3b, 0 ailleurs) ;
  - une question UNSURE n'est reposée que si la composition des clusters a changé.
- **Convergence :** atteinte en 7 à 17 tours selon la configuration.

### 5. Décideurs

- **L1 :** `HistGradientBoosting` régularisé, probabilités **calibrées** par régression isotonique en validation croisée à 5 plis. Entraînement sur les 1 552 arêtes ambiguës de **dev** (311 positives). Seuils fixés a priori : ≥ 0,9 pour MATCH, < 0,1 pour NO, sinon UNSURE.
  - Pour une question, p(option) est le maximum sur les arêtes décisives qui la relient au cluster à rattacher.
  - Sans calibration, le même modèle donnait 93 % de réponses justes à ces seuils (surconfiance). Après calibration, au niveau des arêtes du test : 99,2 % de MATCH justes et 98,7 % de NO justes. Au niveau des questions : 91 à 97,5 % selon l'exécution.
- **L2 :** L1 plus le cosinus des embeddings `nomic-embed-text` (Ollama, environ 12 s pour 1 295 fiches). **Ce n'est pas un cross-encoder** : ni torch ni sentence-transformers ne sont installés, et le cross-encoder n'a donc pas été testé.
- **L3a :** llama3.2 3B. **L3b :** llama3 8B. Tous deux via Ollama, avec une température de 0, une sortie JSON et le même prompt système.
- **L3c / L3cT :** `gemma4:31b` (Ollama, environ 19 Go, quantifié 4 bits), sans et avec le mode réflexion. Serveur Ollama dédié avec un contexte limité à 8 192 jetons : par défaut, Ollama réservait un contexte de 262 k jetons et 44 Go de mémoire pour ce modèle ; à 8 k, 26 Go. En mode réflexion, la sortie JSON contrainte est désactivée (sinon la réflexion consomme le budget et la réponse est vide) et la réponse est extraite du texte.
- **L4 :** `gemini-pro-latest` via l'API, même prompt, température de 0, sortie JSON. **Les fiches sortent de la machine.** Ici elles sont synthétiques ; en contexte bancaire, c'est rédhibitoire sans accord.
- **PRIOR :** référence triviale qui répond toujours A, c'est-à-dire l'option au meilleur score.

## Résultats (jeu test, 300 personnes, 646 fiches)

### Qualité finale selon la hauteur de la cascade

| Configuration | P paire | R paire | F1 | entités exactes /300 | clusters à id. fort mixte | appels par niveau | réponses UNSURE en bout de cascade* | durée |
|---|---|---|---|---|---|---|---|---|
| Règles seules (Pregel sur MATCH) | 1.000 | 0.412 | 0.583 | 154 | 0 | - | - | 0.4 s |
| + L1 classifieur | 1.000 | 0.606 | 0.754 | 203 | 0 | L1 523 | 244 | 52 s |
| + L1 + L2 embeddings | 1.000 | 0.614 | 0.760 | 205 | 0 | L1 518, L2 243 | 240 | 54 s |
| + L1 + L2 + « toujours A » (référence triviale) | 0.848 | 0.899 | 0.873 | 230 | 0 | L1 274, L2 139, PRIOR 136 | 0 | 51 s |
| + L1 + L2 + L3a llama3.2 3B | 0.794 | 0.810 | 0.802 | 198 | 0 | L1 286, L2 144, L3a 141 | 9 | 147 s |
| + L1 + L2 + L3a + L3b llama3 8B | 0.791 | 0.825 | 0.808 | 202 | 0 | L1 279, L2 139, L3a 136, L3b 9 | 0 | 140 s |
| + L1 + L2 + L3b llama3 8B | 0.824 | 0.883 | 0.853 | 219 | 0 | L1 281, L2 140, L3b 137 | 1 | 186 s |
| + L1 + L2 + L4 Gemini (hébergé) | 0.930 | 0.806 | 0.864 | 235 | 0 | L1 411, L2 210, L4 207 | 104 | 1 074 s |
| + L1 + L2 + L3c gemma4 31B local, sans réflexion | 0.934 | 0.856 | 0.894 | 242 | 0 | L1 355, L2 184, L3c 181 | 49 | 526 s |
| + L1 + L2 + L3cT gemma4 31B local, avec réflexion | 0.919 | 0.843 | 0.879 | 236 | 0 | L1 363, L2 185, L3cT 182 | 63 | 4 800 s |

\* Une question restée UNSURE est reposée quand la composition des clusters change : ce sont des réponses, pas des questions distinctes.

Le F1 le plus élevé revient à « toujours A » (0,873), parce que dans ce résidu la vraie réponse est l'option A dans 78 % des cas. **Le F1 n'est pas la bonne métrique pour une banque :** une fusion à tort coûte plus cher qu'un doublon. À précision 1,00, le meilleur rappel est celui de L1+L2 (0,61). Si une précision de 0,93 est acceptable, Gemini monte le rappel à 0,81, et gemma4 31B local à 0,86.

### Courbe « ce qui reste à chaque marche »

Ce sont les questions décisives cumulées sur tous les tours. Les questions ne sont pas exactement les mêmes d'une exécution à l'autre, car les réponses des étages hauts changent les clusters des tours suivants.

| Exécution | Marche | reçues | tranchées | justes | restent |
|---|---|---|---|---|---|
| L1 seul | L1 | 523 | 279 | 272 (97,5 %) | 244 |
| L1+L2 | L1 → L2 | 518 → 243 | 275 → 3 | 268 → 2 | 240 |
| L1+L2+PRIOR | L1 → L2 → PRIOR | 274 → 139 → 136 | 135 → 3 → 136 | 127 → 2 → 100 (74 %) | 0 |
| L1+L2+L3a+L3b | L3a → L3b | 136 → 9 | 127 → 9 | 55 (43 %) → 6 | 0 |
| L1+L2+L3b | L3b | 137 | 136 | 87 (64 %) | 1 |
| L1+L2+L4 | L4 | 207 | 103 | 82 (80 %) | 104 |
| L1+L2+L3c | L3c | 181 | 132 | 99 (75 %) | 49 |
| L1+L2+L3cT | L3cT | 182 | 119 | 86 (72 %) | 63 |

Sur les questions mal tranchées, gemma4 sans réflexion fait 23 fusions à tort et 10 « aucun » à tort ; Gemini 14 et 7. La précision par paire reste pourtant comparable (0,934 contre 0,930). Explication probable, non vérifiée : ces fusions à tort touchent de petits clusters et coûtent peu de paires. Il s'agit d'une seule exécution.

Avant toute question, les règles ont tranché 192 MATCH et 2 354 NO, et Pregel a écarté 63 arêtes ambiguës sans effet dès le premier tour.

**Latences par appel (médiane / p95) :**
- L1 : environ 80 / 165 ms, dont l'essentiel est la requête Cypher des features ; la prédiction elle-même prend moins de 1 ms ;
- L3a : 630 à 700 / 1 000 ms ;
- L3b : 900 à 1 000 / 1 100 à 1 700 ms ;
- L3c (gemma4 31B, sans réflexion) : 2 400 / 3 900 ms ;
- L3cT (avec réflexion) : 13 300 / 96 000 ms ;
- L4 : 4 000 / 8 300 ms.

**Parallélisme local :** avec `OLLAMA_NUM_PARALLEL=4`, quatre requêtes simultanées sur gemma4 31B n'ont apporté aucun gain sans réflexion et environ 17 % avec réflexion : le moteur MLX d'Ollama traite les requêtes quasiment l'une après l'autre sur ce Mac. Les appels locaux sont donc séquentiels. Les appels à l'API, eux, partent par 8.

### Même tâche, trois contextes de prompt

Il s'agit des 77 questions du premier tour que ni L1 ni L2 n'ont tranchées. La bonne réponse est A pour 60 d'entre elles, B pour 6 et « aucun » pour 11. L'état du graphe est celui du premier tour, reconstruit à l'identique (vérifié par assertion).

| Contexte / modèle | justes | fausses (dont fusion à tort) | UNSURE | couverture | justes / tranchées | latence médiane |
|---|---|---|---|---|---|---|
| fiche contre fiche / llama3.2 | 21 | 26 (22) | 30 | 0,61 | 0,45 | 622 ms |
| fiche contre cluster / llama3.2 | 29 | 44 (43) | 4 | 0,95 | 0,40 | 634 ms |
| cluster contre cluster / llama3.2 | 35 | 33 (32) | 9 | 0,88 | 0,52 | 924 ms |
| fiche contre fiche / llama3 8B | 59 | 18 (17) | 0 | 1,00 | 0,77 | 808 ms |
| fiche contre cluster / llama3 8B | 50 | 27 (26) | 0 | 1,00 | 0,65 | 960 ms |
| cluster contre cluster / llama3 8B | 57 | 19 (19) | 1 | 0,99 | 0,75 | 1 120 ms |
| fiche contre fiche / Gemini | 37 | 5 (5) | 35 | 0,55 | 0,88 | 4,9 s |
| fiche contre cluster / Gemini | 36 | 4 (3) | 37 | 0,52 | 0,90 | 4,7 s |
| cluster contre cluster / Gemini | 36 | 6 (5) | 35 | 0,55 | 0,86 | 4,5 s |
| fiche contre fiche / gemma4 31B local | 45 | 16 (10) | 16 | 0,79 | 0,74 | 2,3 s |
| fiche contre cluster / gemma4 31B local | 47 | 17 (10) | 13 | 0,83 | 0,73 | 2,9 s |
| cluster contre cluster / gemma4 31B local | 45 | 17 (8) | 15 | 0,81 | 0,73 | 3,4 s |
| fiche contre fiche / gemma4 31B local, réflexion | 42 | 10 (4) | 25 | 0,68 | 0,81 | 11,5 s |
| fiche contre fiche / Gemma 4 31B API | 44 | 8 (6) | 25 | 0,68 | 0,85 | 17 s* |
| fiche contre cluster / Gemma 4 31B API | 42 | 9 (6) | 26 | 0,66 | 0,82 | 20 s* |
| cluster contre cluster / Gemma 4 31B API | 42 | 6 (4) | 29 | 0,62 | 0,88 | 24 s* |
| fiche contre fiche / Gemma 4 26B-A4B API | 41 | 5 (4) | 31 | 0,60 | 0,89 | 18 s* |
| fiche contre cluster / Gemma 4 26B-A4B API | 46 | 10 (10) | 21 | 0,73 | 0,82 | 22 s* |
| cluster contre cluster / Gemma 4 26B-A4B API | 42 | 6 (6) | 29 | 0,62 | 0,88 | 24 s* |
| référence « toujours A » | 60 | 17 | 0 | 1,00 | 0,78 | 0 |

\* Latences de l'API Gemma mesurées avec 8 appels simultanés et des reprises sur erreur de quota : elles reflètent la file d'attente du service gratuit, pas le modèle.

Conclusions :
- **Aucun modèle local ne bat la référence triviale.**
- Llama3.2 a un fort biais vers l'option B : dans la cascade, il répond B 74 fois, quand la bonne réponse est B dans seulement 12 cas.
- Pour llama3 8B, ajouter les fiches de cluster (contexte « fiche contre cluster ») **dégrade** les résultats, alors que « cluster contre cluster » avec noyau confirmé revient au niveau de « fiche contre fiche ».
- Pour Gemini et Gemma, aucune différence significative entre les trois contextes.
- **Même modèle, deux hébergements :** Gemma 4 31B via l'API s'abstient davantage (25 UNSURE contre 16) et a plus de réponses justes sur les questions tranchées (0,85 contre 0,74) que la version locale quantifiée en 4 bits sans réflexion. La version locale avec réflexion s'en rapproche (25 UNSURE, 0,81). L'écart peut venir de la quantification, du mode réflexion activé par défaut côté API, ou du prompt (pas d'instruction système ni de JSON contraint côté API pour Gemma). Ces causes n'ont pas été séparées.
- **Écart entre l'expérience isolée et la cascade :** sur ces 77 questions, gemma4 local sans réflexion n'a que 74 % de réponses justes parmi celles qu'il tranche, moins que « toujours A » (78 %), alors que dans la cascade il donne la meilleure exécution. Explications probables, non vérifiées : dans la cascade, les questions des tours suivants portent sur des clusters plus gros et donc plus riches en indices ; et ses « aucun » à tort ne coûtent que du rappel. Le chiffre de cascade ne doit pas être lu comme une supériorité du modèle sur la question isolée.

### Quelle part du résidu est décidable ? (`residual_analysis.py`)

Sur ces 77 questions, on vérifie si les paires de fiches entre le cluster à rattacher et le bon cluster portent un indice autre que le nom (date, identifiant, e-mail, téléphone, ou adresse avec Jaro-Winkler ≥ 0,9).

| Catégorie (n) | llama3 8B fiche contre fiche | llama3 8B cluster contre cluster | Gemini fiche contre fiche | Gemini fiche contre cluster |
|---|---|---|---|---|
| Bonne option avec indice corroborant (41) | 38 justes, 3 fausses | 38 justes, 2 fausses, 1 UNSURE | 34 justes, 4 fausses, 3 UNSURE | 33 justes, 2 fausses, 6 UNSURE |
| Bonne option, nom seul (25) | 20 justes, 5 fausses | 19 justes, 6 fausses | **25 UNSURE** | 23 UNSURE, 1 juste, 1 fausse |
| Bonne réponse « aucun » (11) | **10 fausses** (fusion), 1 juste | **11 fausses** | 3 justes, 1 fausse, 7 UNSURE | 2 justes, 1 fausse, 8 UNSURE |

Mêmes catégories, contexte fiche contre fiche, pour la série Gemma :

| Catégorie (n) | gemma4 31B local | gemma4 31B local, réflexion | Gemma 4 31B API | Gemma 4 26B-A4B API |
|---|---|---|---|---|
| Bonne option avec indice corroborant (41) | 34 justes, 7 fausses | 33 justes, 3 fausses, 5 UNSURE | 34 justes, 4 fausses, 3 UNSURE | 32 justes, 3 fausses, 6 UNSURE |
| Bonne option, nom seul (25) | 7 justes, 7 fausses, 11 UNSURE | 5 justes, 6 fausses, 14 UNSURE | 6 justes, 2 fausses, 17 UNSURE | 8 justes, 1 fausse, 16 UNSURE |
| Bonne réponse « aucun » (11) | 4 justes, 2 fausses, 5 UNSURE | 4 justes, 1 fausse, 6 UNSURE | 4 justes, 2 fausses, 5 UNSURE | 1 juste, 1 fausse, 9 UNSURE |

Lecture :
- Environ un tiers du résidu (25 questions sur 77) repose sur le nom seul. Ce sont des homonymes ou des fiches identiques sans aucun indice, et **aucun décideur ne peut les trancher sans donnée supplémentaire**.
- Les « justes » de llama3 8B sur ces 25 questions relèvent de la chance que donne l'a priori (A est souvent juste), et il fusionne presque tous les cas où la bonne réponse est « aucun ».
- Gemini se comporte comme on l'attend d'un analyste : il s'abstient sur ce qui est indécidable et tranche ce qui est corroboré, avec 83 à 87 % de réponses justes.
- La série Gemma est entre les deux : elle ne fusionne presque plus les cas « aucun », mais tranche encore 8 à 14 des 25 questions « nom seul », dont la moitié faussement pour le modèle local.

### Baseline WCC contre clustering contraint, sur les mêmes arêtes (`baselines.py`)

Toutes les arêtes candidates de score ≥ T sont traitées comme des MATCH, y compris celles en conflit de date.

| Seuil score | arêtes | WCC P / R / F1 | clusters WCC mixtes | Pregel contraint P / R / F1 | clusters Pregel mixtes |
|---|---|---|---|---|---|
| 0,70 | 982 | 0,263 / 0,950 / 0,411 | 54 | 0,797 / 0,872 / 0,833 | 0 |
| 0,75 | 632 | 0,400 / 0,909 / 0,555 | 39 | 0,839 / 0,847 / 0,843 | 0 |
| 0,80 | 331 | 0,805 / 0,664 / 0,728 | 5 | 0,947 / 0,662 / 0,779 | 0 |
| 0,85 | 155 | 1,000 / 0,338 / 0,505 | 0 | 1,000 / 0,338 / 0,505 | 0 |
| 0,90 | 43 | 1,000 / 0,085 / 0,157 | 0 | 1,000 / 0,085 / 0,157 | 0 |

C'est le résultat le plus net du PoC : dès que le seuil laisse passer des arêtes douteuses, la fermeture transitive enchaîne les personnes, et la contrainte de profil le bloque.

À noter :
- Pregel contraint seul au seuil 0,75 (F1 0,84) **fait mieux en F1** que la cascade L1+L2 (0,76), mais avec une précision de 0,84.
- **Dans la boucle, la contrainte agit en amont.** Les paires en conflit ne deviennent jamais des questions. Le WCC calculé sur les MATCH finaux de la cascade donne donc la même partition que Pregel dans toutes les exécutions. Le gain se mesure en questions évitées, pas en clusters différents.

## Limites et points non faits

- **Fuite légère sur le test.** Pendant la mise au point, j'ai inspecté des erreurs des règles sur `test` et corrigé la couche Cypher : fautes de frappe sur les prénoms, inversion avec initiale, blocage par date, téléphone et e-mail, e-mails rendus uniques dans le générateur. Les classifieurs sont entraînés uniquement sur `dev`, et les LLM ne sont pas ajustés. Pour un chiffre propre, il faudrait une troisième graine jamais regardée.
- **Données synthétiques générées par moi.** Les proportions de variations, d'homonymes et d'apparentés sont des choix. Le trou dans le dictionnaire de diminutifs est volontaire. Ces résultats ne se transposent pas tels quels à des données réelles.
- **Une seule exécution par configuration.** Les LLM à température 0 ne sont pas parfaitement déterministes, et la cascade amplifie les écarts : L1 reçoit entre 274 et 286 questions selon l'exécution.
- **Cross-encoder (niveau 2) non testé** : ni torch ni sentence-transformers dans l'environnement. Il est remplacé par un bi-encodeur `nomic-embed-text`, qui ne sert à rien ici.
- **Gemini ne fait sortir que le résidu**, mais il le fait sortir. Il est utilisé uniquement parce que l'utilisateur a fourni une clé et que les données sont synthétiques.
- **Les sociétés n'ont pas été traitées.**
- **Gemma 4 et Gemini n'ont pas été appelés dans les mêmes conditions** : JSON contraint et instruction système pour Gemini et pour Ollama, ni l'un ni l'autre pour Gemma via l'API (non pris en charge), et mode réflexion inconnu côté API.
- **Pregel :** la contrainte étant globale au cluster, le master fait l'union-find. Au-delà de quelques millions de nœuds, l'étape master séquentielle deviendrait le goulot. Ce point n'a pas été mesuré.

## Pistes de suite (non testées)

- **Troisième graine jamais regardée**, pour retirer la fuite légère sur le test, et plusieurs exécutions par configuration pour avoir des intervalles.
- **Modèle ONNX embarqué dans la procédure Pregel.** Charger un classifieur (L1 exporté en ONNX, ou un petit cross-encoder) dans `myns.erCluster` et scorer les fusions directement dans `masterCompute`, sur le cluster fusionné plutôt que sur la paire. Cela éviterait l'aller-retour Python à chaque tour. Faisabilité discutée, **ni construite ni mesurée** : à vérifier le chargement d'onnxruntime dans le classloader de Neo4j et le coût par fusion.
- **Cross-encoder** au niveau 2 (torch absent de l'environnement).
- **Sociétés** (SIREN, raisons sociales, adresses de siège).
- **Isoler l'écart Gemma local / API** : même prompt, même mode réflexion, quantification 8 bits.

## Reproduire

Voir le [README](README.md) (en anglais) : prérequis, installation du plugin, `./reproduce.sh` pour la partie locale sans LLM génératif, puis les scripts des niveaux 3 et 4.

Données en base : nœuds `SpikeERRecord` et `SpikeERNickname`, relations `SPK_ER_CANDIDATE`, contrainte `spk_er_rid`, index `spk_er_nick`. Plugin : `plugins/myns-er-pregel.jar` (procédure `myns.erCluster`).
