# Plan technique — idées RapidRAW/Furka pour Libre DT-Lab

## Contexte

RapidRAW v1.7.0 (« Furka ») apporte deux familles d'améliorations :

1. **Démosaïcage GPU** : RCD pour Bayer avec une étape de *debias* du vert
   interpolé, interpolation résiduelle du rouge/bleu, et Markesteijn 3-pass
   pour X-Trans.
2. **Moteur tonal Furka** : base edge-aware calculée en stops (log2 de la
   luminance), sliders Highlights/Shadows/Whites/Blacks retravaillés, contraste
   pivoté sur la médiane de l'image, Blacks après Contrast, color grading en
   OKLab, sharpening sans halo.

Libre DT-Lab possède déjà RCD (CPU + OpenCL), Markesteijn 1/3-pass, VNG,
LMMSE, un module Contrast & Texture à base d'EIGF, et une basecurve
scene-referred. L'objectif est de reprendre les **idées** les plus utiles,
pas de copier le code RapidRAW.

## Contraintes

- **Licence** : RapidRAW est AGPL-3.0, Libre DT-Lab/darktable est GPL-3.0.
  Ne pas copier-coller de code RapidRAW. Réimplémenter les algorithmes dans le
  style et l'architecture de darktable, à partir des descriptions et des
  publications. Documenter clairement toute inspiration.
- **CPU + OpenCL** : garder la parité numérique et la gestion des tuiles.
- **Non-régression** : les nouveaux comportements doivent être optionnels ou
  désactivés par défaut ; ne pas changer le rendu des modules existants.
- **Performance** : attention aux buffers plein cadre et aux passes
  supplémentaires.
- **Validation** : images synthétiques (rampe, damier, textures fines) +
  images réelles (neige, ciel/arête, ombres profondes), CPU vs OpenCL,
  plein cadre vs tuiles.
- **Référence anti-halo** : `src/iop/contrast.c` actuel est la référence.
  Ses correctifs (guide luma du profil de travail, base EIGF basse fréquence
  dédiée aux Highlights/Shadows) ne doivent pas être remplacés sans A/B
  mesuré.

## État actuel de Libre DT-Lab

### Démosaïcage

- `src/iop/demosaic.c` : module principal, méthodes Bayer/X-Trans.
- `src/iop/demosaicing/rcd.c` : RCD CPU (tiles `DT_RCD_TILESIZE`,
  `RCD_BORDER = 10`, `RCD_MARGIN = 9`), étapes 0 à 4.
- `data/kernels/demosaic_rcd.cl` : RCD OpenCL :
  `rcd_populate`, `rcd_step_1`, `rcd_step_2`, `rcd_step_3`,
  `rcd_step_4_0`, `rcd_step_4_1`, `rcd_step_4_2`, `rcd_step_4_3`,
  `rcd_write_output`.
- `process_rcd_cl()` dans `rcd.c` alloue `cfa`, `rgb0`, `rgb1`, `rgb2`,
  `VH_dir`, `PQ_dir`, `VP_diff`, `HQ_diff`, puis enchaîne les kernels.
- **Il n'y a pas d'étape de debias du vert.**
- Markesteijn 3-pass existe déjà (`demosaic_markesteijn.cl`).

### Tonal — état actuel de `contrast.c` (« Tone & Texture »)

- **Guide luma** : `compute_working_luma()` utilise les coefficients
  `luma_r/g/b` du profil de travail (matrice RGB->XYZ), et non la norme
  euclidienne RGB. C'est le correctif principal contre les halos aux
  frontières de teintes saturées (feuillage vert / ciel bleu).
- **Cinq échelles EIGF** : décomposition multi-échelle, détail en log2 (EV),
  protection d'arête par échelle (`f_mult_*`), protection du bruit,
  contraste global pondéré par la CSF autour du gris moyen (0.1845).
- **Base basse fréquence dédiée** : `compute_smoothed_base_luminance()`
  calcule une base EIGF à géométrie fixe (≈5 % de diamètre, feathering 7,
  3 itérations), **découplée** des curseurs spatiaux. Le terme
  Highlights/Shadows (`sh_term`) est évalué sur `luminance_base`, pas sur la
  luminance du pixel : le décalage tonal est quasi constant localement, donc
  le contraste local est préservé.
- **OpenCL** : `kernel_luminance` reçoit les coefficients luma ; `dev_base`
  et `fast_eigf_surface_blur_cl` reproduisent la même base qu'en CPU.
- **Résultat** : plus de halos sur les images de test de l'auteur.
- `src/iop/basecurve.c` : basecurve scene-referred (display, kinematic,
  dynamic, cinematic DRT).
- Modules darktable existants : `toneequal`, `atrous`, `filmicrgb`, `agx`,
  `colorbalancergb`.

---

## Phase 1 — Debias RCD (recommandé en premier)

### Objectif

Supprimer le damier vert résiduel dans les zones claires et uniformes
(neige, ciel surexposé) produit par l'interpolation RCD du vert.

### Algorithme RapidRAW

Constantes :

- `DEBIAS_RADIUS = 3`
- `DEBIAS_NORM = 1 / 64`
- `DEBIAS_LIMIT = 0.04`

Le filtre est un noyau binomial 7 taps : `[1, 6, 15, 20, 15, 6, 1]`
(somme 64).

Passe horizontale, pour chaque pixel `(x, y)` :

```text
modulated = 0
level = 0
weight = 1
pour k = 0..6 :
  xk = clamp(x + k - 3)
  g = green(xk, y)
  modulated += weight * (FC(xk, y) == GREEN ? -g : g)
  level     += weight * g
  weight    *= (6 - k) / (k + 1)
```

Passe verticale, uniquement aux sites non-verts :

```text
modulated = 0
level = 0
weight = 1
pour k = 0..6 :
  yk = clamp(y + k - 3)
  modulated += weight * debias_h_modulated(x, yk)
  level     += weight * debias_h_level(x, yk)
  weight    *= (6 - k) / (k + 1)

limit = DEBIAS_LIMIT * level * DEBIAS_NORM
green(x, y) -= clamp(2 * modulated * DEBIAS_NORM, -limit, limit)
```

Le signe alterné (`-g` aux sites verts, `+g` ailleurs) est essentiel : il
mesure le biais haute fréquence du motif CFA.

### Implémentation CPU (`src/iop/demosaicing/rcd.c`)

Point d'insertion : **après STEP 3** (vert interpolé) et **avant STEP 4**
(rouge/bleu).

1. Allouer un scratch de 2 floats par pixel de tuile :
   `modulated` et `level` horizontaux.
   Taille : `2 * DT_RCD_TILESIZE * DT_RCD_TILESIZE * sizeof(float)`.
2. Passe horizontale sur une région sûre (marge 3), avec coordonnées
   clampées aux bords de tuile.
3. Passe verticale sur les sites `FC(row, col, filters) != GREEN`, avec
   coordonnées clampées.
4. Mettre à jour `rgb[1][indx]`.
5. Libérer le scratch.

Remarques :

- `RCD_BORDER = 10` et `RCD_MARGIN = 9` sont supérieurs au rayon 3 ; la zone
  de sortie valide ne devrait pas être affectée par les bords de tuile.
- Calculer les bords avec des coordonnées clampées est acceptable si la zone
  écrite est à l'intérieur du halo. Vérifier l'absence de coutures.
- Ne pas modifier le vert aux sites CFA verts : la passe verticale les saute.
- Optionnel : rendre le debias désactivable par un paramètre caché ou une
  constante de compilation pour comparer.

### Implémentation OpenCL (`data/kernels/demosaic_rcd.cl`)

Ajouter deux kernels :

- `rcd_debias_h(global float *rgb1, global float *mod, global float *level,
  const int w, const int height, const unsigned int filters)`
- `rcd_debias_v(global float *rgb1, global float *mod, global float *level,
  const int w, const int height, const unsigned int filters)`

Détails :

- `rcd_debias_h` : une invocation par pixel, écrit `mod[idx]` et `level[idx]`.
- `rcd_debias_v` : une invocation par pixel ; `if (FC(row, col, filters) == GREEN) return;`
  puis convolue verticalement `mod`/`level`, calcule `limit`, et soustrait de
  `rgb1[idx]`.
- Utiliser `FC` et `GREEN` comme les kernels existants.
- Scratch : réutiliser `VP_diff` et `HQ_diff` (déjà alloués et non utilisés
  avant `rcd_step_4_0`), ou allouer deux buffers dédiés si l'on préfère une
  séparation stricte. Les réutiliser évite ~2 × width × height × 4 octets.
- Ordre d'enqueue dans `process_rcd_cl()` :
  `rcd_step_3` → `rcd_debias_h` → `rcd_debias_v` → `rcd_step_4_0` → ...
- Ajouter `kernel_rcd_debias_h` et `kernel_rcd_debias_v` dans
  `dt_iop_demosaic_global_data_t`, les créer dans `init_global()` et les
  libérer dans `cleanup_global()` (`src/iop/demosaic.c`).

### Validation

- Image synthétique : zone plate très claire + texture fine ; mesurer
  l'amplitude du damier vert avant/après.
- Rampe de gris et mire de couleur : vérifier l'absence de dérive.
- Comparaison plein cadre vs traitement par tuiles : pas de couture.
- CPU vs OpenCL : écart maximal attendu < 1e-5.
- Images réelles : neige, ciel, surfaces uniformes, hautes lumières.
- Vérifier que le debias ne mange pas le détail vert fin (mesurer le MTF ou
  comparer visuellement à 200 %).

### Avantages

- Petit changement, ciblé, fort impact sur un artefact connu.
- Faible risque de régression.
- CPU et OpenCL, architecture existante respectée.
- Pas de nouveau paramètre utilisateur nécessaire.

### Inconvénients / risques

- Peut légèrement adoucir le détail vert si le rayon/la limite sont trop
  agressifs.
- Coût mémoire/bande passante (une passe horizontale + une verticale).
- Ne corrige pas la fausse couleur rouge/bleu (Phase 2).
- Nécessite de valider les bords de tuile et la parité CPU/OpenCL.

### Effort estimé

- 1 à 2 jours : implémentation CPU, kernels OpenCL, intégration, tests.

---

## Phase 2 — Interpolation résiduelle R/B

### Objectif

Réduire les fausses couleurs et le moiré en rouge/bleu en interpolant les
résidus guidés par le vert, au lieu d'une simple interpolation directionnelle
des différences de couleur.

### Algorithme RapidRAW (résumé)

Après le vert (et le debias), pour chaque site non-vert :

1. `rb_prepare` : accumule `g`, `x` (valeur CFA), `g²`, `x²`, `g·x`,
   ainsi que des produits passe-haut.
2. `rb_stats` : somme horizontale sur un rayon de 4 (9 taps).
3. `rb_coef` : somme verticale, puis régression linéaire locale
   (`color_coef`) avec poids d'erreur.
4. `rb_average` : moyenne horizontale des coefficients.
5. `rb_tentative` : moyenne verticale et estimation R/B préliminaire.
6. `rb_residual` : interpolation des résidus sur un voisinage 3×3, avec une
   carte de confiance `crit`.
7. `rb_smooth` : lissage gaussien horizontal de `crit` (sigma 1.5, rayon 5).
8. `rb_out` : lissage gaussien vertical, normalisation, combinaison des
   résidus, remplacement des sites R/B connus par les valeurs CFA d'origine.

Paramètres RapidRAW : `RB_RADIUS = 4`, `RB_SIGMA = 1.5`, `RB_AREA = 81`,
`RB_SHOT = 1e-4`, `RB_EPS = 1e-5`, `RB_TINY = 1e-20`.

### Implémentation proposée

Ne pas remplacer le RCD par défaut. Ajouter une méthode ou une case à cocher
« RCD residual R/B » :

- **Prototype CPU d'abord** : réutiliser les buffers de `rcd.c` ou ajouter des
  buffers dédiés. Le plus simple est de dupliquer la logique R/B de RapidRAW
  en C, avec des boucles et des buffers plein cadre, pour valider la qualité.
- **OpenCL ensuite** : découper en kernels comme RapidRAW, ou fusionner les
  passes pour limiter les buffers.
- **Marge de tuile** : la fenêtre totale est d'au moins
  `RB_RADIUS + 5 + 3 ≈ 12–16` pixels. `RCD_BORDER = 10` est insuffisant :
  prévoir une marge dédiée (par exemple 20) ou un traitement plein cadre pour
  cette méthode.
- **Mémoire** : prévoir plusieurs buffers plein cadre (état, stats, coefs,
  tentative, résidu, lissé). Sur CPU, attention au pic mémoire ; sur OpenCL,
  attention aux allocations.

### Validation

- Mires de haute fréquence (tissus, grilles, plumes), texte, damiers.
- Mesurer les fausses couleurs (écarts chroma sur zones neutres).
- Comparer à RCD standard, RCD dual, LMMSE, AMaZE.
- Vérifier les bords de tuile et la parité CPU/OpenCL.
- Profiler le temps de traitement et la mémoire.

### Avantages

- Meilleure suppression des fausses couleurs et du moiré.
- Meilleur rendu des textures fines.
- Réutilise l'idée du guidage par le vert, déjà présente dans RCD.

### Inconvénients / risques

- Complexité élevée : plusieurs passes, buffers, régression locale.
- Coût CPU/GPU et mémoire important.
- Risque d'artefacts si les paramètres ou la marge de tuile sont mal réglés.
- Peut faire doublon avec RCD dual / LMMSE déjà présents.
- Validation longue.

### Effort estimé

- 1 à 2 semaines pour un prototype CPU + OpenCL solide.

---

## Phase 3 — Base tonale Furka / contraste

### Objectif

Comparer les idées Furka à l'implémentation actuelle de `contrast.c` et
n'ajouter que des améliorations **optionnelles**, sans régresser la protection
anti-halo déjà validée.

### Ce que fait déjà `contrast.c` (référence à préserver)

- Guide luma = luma du profil de travail (`luma_r/g/b`), pas la norme
  euclidienne RGB. C'est le correctif principal contre les halos aux
  frontières de teintes saturées.
- Cinq échelles EIGF, détail en log2 (EV), protection d'arête par échelle
  (`f_mult_*`), protection du bruit, contraste global pondéré par la CSF
  autour du gris moyen.
- Base basse fréquence dédiée pour Highlights/Shadows :
  `compute_smoothed_base_luminance`, géométrie fixe (≈5 % de diamètre,
  feathering 7, 3 itérations), indépendante des curseurs spatiaux.
- Le terme `sh_term` est évalué sur `luminance_base`, pas sur la luminance
  du pixel, ce qui conserve le contraste local.
- OpenCL : `kernel_luminance` avec les coefficients luma, `dev_base` et
  `fast_eigf_surface_blur_cl` avec les mêmes paramètres.
- Résultat : plus de halos sur les images de test de l'auteur.

### Idées Furka qui restent candidates

- Pivot du contraste sur la médiane/midtone de l'image au lieu de 0.1845.
- Blacks après Contrast (mais `contrast.c` n'a pas de blacks ; à évaluer
  dans un autre module).
- Base calculée directement en log2 luma avec pondération de gamme.
- Comportement Highlights/Whites/Shadows plus élaboré.
- Color grading OKLab (module séparé).

### Implémentation proposée

Principe : **ne rien remplacer par défaut**. Toute idée Furka doit être un
mode optionnel, désactivé par défaut, avec A/B.

1. **Curseur de percentile (18–60 %)** (le plus petit candidat)
   - `18 %` = gris moyen fixe 0.1845 / comportement historique.
   - `50 %` = médiane exacte de l'image.
   - `60 %` = 60ᵉ percentile de la distribution log2 de la luma.
   - Entre 18 et 50 : interpolation smoothstep entre le gris moyen et le
     percentile demandé ; au-delà de 50 : percentile direct.
   - Histogramme log2 échantillonné (1 pixel sur 16), mis en cache
     preview/full ; le changement de curseur ne relance pas le scan.
   - CPU et OpenCL reçoivent le même histogramme et le même pivot.
   - Risque : change le rendu des images sombres/claires ; doit rester opt-in.

2. **Blacks après Contrast**
   - Ne pas ajouter un curseur Blacks dans `contrast.c` sans étude.
   - Évaluer d'abord si le module `blacks` existant ou `basecurve.c` couvre
     déjà le besoin. Sinon, prototype dans un module séparé.

3. **Base log2 / pondération de gamme**
   - Expérimental uniquement : ajouter un mode `base_log` qui calcule
     `L = log2(luma)` et applique EIGF sur `L`, avec un poids de gamme
     `exp(-0.5*((guide-L)/sigma)^2)` ou un seuil.
   - Comparer à la base actuelle sur les images de halo de l'auteur.
   - Si le gain n'est pas mesurable, **garder la base actuelle**.

4. **Highlights/Whites/Shadows**
   - Ne pas modifier `sh_term` par défaut. Toute variante Furka doit être
     un mode optionnel et comparée au rendu actuel.

5. **OKLab**
   - Traiter dans un module de color grading séparé, pas dans
     `contrast.c`.

### Invariants anti-halo (non négociables)

- `compute_working_luma` garde les coefficients du profil de travail.
- `compute_smoothed_base_luminance` garde sa géométrie fixe et reste
  découplée des curseurs spatiaux.
- `luminance_base` ne sert qu'au terme Highlights/Shadows ; le contraste
  local reste basé sur les cinq échelles EIGF et le détail log2.
- CPU/OpenCL doivent rester alignés : mêmes coefficients luma, mêmes
  paramètres de blur, même base.
- Avec Highlights/Shadows à 0, la sortie doit rester identique au
  comportement actuel.
- Aucun changement ne doit augmenter les halos sur les images de référence.

### Validation

- Utiliser les images de halo de l'auteur (arête contre ciel, foliage/ciel,
  neige, ombres profondes) et comparer A/B.
- Mesurer le profil de halo le long des arêtes, le clipping 0/1, la
  conservation du contraste local et les dérives de couleur.
- Comparer au rendu actuel de `contrast.c` ; critère : halo non pire,
  clipping non pire, détail local conservé.
- Vérifier preview/full/export et CPU/OpenCL.
- Ne pas comparer uniquement à `toneequal`/`filmicrgb`/`agx` ; le rendu
  actuel de `contrast.c` est la référence.

### Avantages (si mode optionnel bien isolé)

- Pivot percentile plus robuste sur les images sombres/claires, avec un
  réglage continu entre gris moyen (18 %) et 60ᵉ percentile.
- Possibilité de tester la base log2 sans risquer la régression.
- Réutilise l'infrastructure EIGF existante.

### Inconvénients / risques

- Toute modification de la base peut réintroduire des halos.
- Le pivot percentile change le rendu et l'historique visuel.
- Blacks n'est pas dans ce module ; risque de duplication.
- La base log2 peut ne pas battre la base linéaire + EIGF actuelle.
- Coût de calcul et complexité CPU/OpenCL.

### Effort estimé

- Curseur de percentile optionnel : 2 à 4 jours.
- Base log2 optionnelle : 1 à 2 semaines.
- Module séparé : 2 à 4 semaines.
- Ne pas commencer par Phase 3 si Phase 1/2 ne sont pas stabilisées.

---

## Option — un module « Basic » à la Furka, et le rôle de `exposure`

### Question

Peut-on mettre l'idée Furka dans le module `exposure`, qui deviendrait le
réglage de base ?

### Réponse courte

Oui pour l'idée, non pour `exposure` tel quel.

### Pourquoi pas dans `exposure`

- **Position pipeline** : `exposure` est à 12/21/28 selon l'ordre IOP, avant
  `colorin` (27/28), `toneequal` (21.5/24/28) et `contrast` (27.5/28.5). Les
  Highlights/Shadows/Whites/Blacks créatifs doivent être appliqués après la
  balance des blancs/color calibration, près des tone mappers, pas avant.
- **Rôle du module** : `exposure` est correctif (exposition, black level,
  deflicker, spot exposure, compensation de biais/highlight preservation).
  Le transformer en module tonal créatif mélange les responsabilités et peut
  casser les presets, l'historique et les workflows de deflicker.
- **Doublon** : `contrast.c` (Tone & Texture) a déjà Highlights/Shadows et une
  base anti-halo à la bonne position. Ajouter la même chose dans `exposure`
  créerait un double traitement et des conflits d'ordre.
- **Sémantique darktable** : « basic » est un groupe de modules, pas un
  module. `exposure` n'est pas le panneau Basic de RapidRAW.

### Options

**Option A (recommandée)** : étendre `contrast.c` / Tone & Texture avec des
modes Furka **optionnels** (curseur de percentile 18–60 %, base log2 expérimentale,
éventuellement Whites/Blacks), en conservant les invariants anti-halo.
C'est le meilleur rapport risque/gain.

**Option B** : créer un nouveau module `basic_tone` / `furka_basic` placé à
la position de `contrast` (après `colorin`, avant `agx`/`filmicrgb`), avec
exposure, contrast, highlights, shadows, whites, blacks. Il réutilise
`fast_eigf_surface_blur` et la base anti-halo. C'est le plus proche du panneau
Basic de RapidRAW, mais c'est un vrai module (GUI, introspection, presets,
OpenCL, ordre IOP) et il faut éviter le double tone mapping avec
`toneequal`/`contrast`/`basecurve`/`agx`/`filmicrgb`.

**Option C (non recommandée)** : ajouter les contrôles tonals dans `exposure`.
Cela peut sembler simple, mais c'est au mauvais endroit du pipeline et cela
duplique `contrast.c`.

### Avantages / inconvénients

Option A :
- Avantages : petit périmètre, position correcte, base anti-halo déjà validée,
  pas de nouveau module.
- Inconvénients : `contrast.c` devient plus chargé ; les ajouts doivent rester
  optionnels.

Option B :
- Avantages : un seul panneau « Basic » comme RapidRAW, position pipeline
  correcte.
- Inconvénients : travail important, risque de doublon avec les modules
  existants, interactions à définir avec les tone mappers.

Option C :
- Avantages : interface unique.
- Inconvénients : mauvais placement pipeline, duplication, risque de
  régression.

### Recommandation

Ne pas transformer `exposure` en module Basic. Commencer par l'Option A
(modes optionnels dans Tone & Texture), puis éventuellement l'Option B si le
besoin d'un panneau Basic unifié est confirmé.

---

## Statut d'implémentation

- **Option A — curseur de percentile** : implémenté dans `src/iop/contrast.c`
  et `data/kernels/contrast.cl` derrière `pivot_percentile`
  (`18..60`, défaut `18`). `18 %` reproduit le gris moyen historique,
  `50 %` la médiane, `60 %` le 60ᵉ percentile, avec interpolation smoothstep
  entre 18 et 50. L'histogramme log2 échantillonné (1 pixel sur 16) est mis
  en cache preview/full, en CPU et OpenCL ; changer le curseur ne relance pas
  le scan. Le tiling OpenCL est désactivé quand le pivot est actif. Le
  comportement par défaut reste strictement celui d'avant.
- **À faire ensuite** : tests visuels A/B sur les images de halo, mesure du
  clipping et du contraste local, puis éventuellement base log2 optionnelle.

---

## Ordre recommandé

1. **Phase 1 — Debias RCD** : petit, mesurable, faible risque.
2. **Phase 2 — Residual R/B** : seulement si la Phase 1 est validée et si les
   comparaisons montrent un gain réel par rapport à RCD dual/LMMSE.
3. **Phase 3 — Base tonale Furka** : commencer par un prototype optionnel,
   mesurer les halos et le clipping avant toute généralisation.
4. **Option Basic/Furka** : modes optionnels dans Tone & Texture d'abord ;
   nouveau module Basic seulement si nécessaire.
5. **OKLab / sharpening** : à traiter séparément, après stabilisation.

## Fichiers à inspecter en priorité

- `src/iop/demosaicing/rcd.c`
- `data/kernels/demosaic_rcd.cl`
- `src/iop/demosaic.c` (global data, init/cleanup, enqueue)
- `src/iop/contrast.c`
- `src/iop/exposure.c` (comprendre son rôle et son ordre IOP)
- `src/iop/basecurve.c`
- `src/common/fast_guided_filter.h`
- `src/iop/toneequal.c`, `src/iop/atrous.c` (pour comparaison)

## Prompt pour une nouvelle session dans Libre DT-Lab

```text
Tu travailles dans /home/cc/darktable/libre-dt-lab, le fork darktable
« Libre DT-Lab ». Lis d'abord dev-doc/plan-rcd-furka.md.

Nous commençons par la Phase 1 : debias RCD. Inspecte :
- src/iop/demosaicing/rcd.c
- data/kernels/demosaic_rcd.cl
- src/iop/demosaic.c

Ne copie pas de code AGPL de RapidRAW : réimplémente l'algorithme dans le
style GPL de darktable. Avant de modifier, résume le pipeline RCD actuel et
propose les points d'insertion exacts CPU et OpenCL, avec les buffers scratch
et la gestion des bords de tuile. Ensuite implémente CPU + OpenCL, puis
valide avec des tests de non-régression (plein cadre vs tuiles, CPU vs GPU,
zones claires uniformes, textures fines). Demande confirmation si une
ambiguïté bloque.

Pour toute intervention ultérieure sur contrast.c, préserve impérativement
les invariants anti-halo du plan : guide luma du profil de travail, base EIGF
basse fréquence dédiée aux Highlights/Shadows avec géométrie fixe, détail
local basé sur les cinq échelles, et parité CPU/OpenCL.
```
