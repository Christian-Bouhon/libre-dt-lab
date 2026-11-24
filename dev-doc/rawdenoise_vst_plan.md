# Plan : débruitage raw par profils + VST

- **Statut** : Phase de réflexion
- **Date** : 2026-11-23
- **Portée** : fork personnel `libre-dt-lab`
- **Objectif** : améliorer le débruitage raw classique pour se rapprocher de
  PureRaw 3, tout en restant déterministe et non destructif.
- **Décision actuelle** : ne pas implémenter maintenant. Le module
  `denoiseprofile` modifié donne déjà de bons résultats. Reprendre seulement si
  la valeur ajoutée est démontrée.

---

## 1. Contexte

### Ce qui fonctionne déjà

- **Neural restore + Okeinla** fournit un débruitage IA cuit dans un DNG/TIFF,
  reproductible, proche de PureRaw.
- **`denoiseprofile`** (RGB, profils `noiseprofiles.json`) fonctionne bien après
  les modifications locales.
- **`rawdenoise`** classique est déterministe, CPU-only, non destructif, mais
  utilise un modèle de bruit fixe et un seuil manuel.

### Pourquoi ce projet est différé

- PureRaw 3 repose sur un réseau qui fait débruitage + dématriçage conjoint.
  Un débruiteur CFA classique, même avec VST, ne l'égalera pas sur les cas
  extrêmes.
- `denoiseprofile` couvre déjà une partie du besoin.
- La valeur ajoutée du VST rawdenoise doit être prouvée avant un chantier de
  plusieurs semaines.

### Décisions déjà prises

- **Pas d'IOP IA Okeinla** : la reproductibilité du fichier cuit est prioritaire.
- **Okeinla reste dans neural restore** comme seul chemin IA.
- **Pas de synchronisation upstream** : trop de divergence.
- **CPU-only** pour `rawdenoise` : pas de port OpenCL.

---

## 2. Objectif et non-objectifs

### Objectif

Améliorer `rawdenoise` avec :

- les profils de bruit `noiseprofiles.json` ;
- une transformée d'Anscombe généralisée (VST) par canal CFA ;
- des seuils wavelet en unités de bruit ;
- des courbes automatiques selon l'ISO ;
- un mode `legacy` pour préserver les anciens edits.

### Non-objectifs

- Remplacer `denoiseprofile`.
- Remplacer Okeinla / neural restore.
- Ajouter un IOP IA.
- Suivre upstream ou porter OpenCL.
- Modifier le format des profils `noiseprofiles.json`.

---

## 3. État actuel du code

| Fichier | Rôle |
|---|---|
| `src/iop/rawdenoise.c` | Module v2, `sqrt` + `dwt_denoise` + `noise_all` fixe |
| `src/common/dwt.c` | `dwt_denoise()`, seuillage soft par bande |
| `src/common/noiseprofiles.c/.h` | Matching et interpolation des profils |
| `src/iop/denoiseprofile.c` | Référence : profils `a`/`b`, VST, auto-profil, presets de courbes |
| `src/develop/noise_generator.h` | Bruit synthétique pour les tests |
| `src/libs/neural_restore.c` | Chemin IA, Okeinla |
| `src/common/iop_order.c` | `rawdenoise` après `temperature` |

### Modèle Okeinla

- Modèle : `rawdenoise-okeinla`
- Entrée : `[1, 4, 512, 512]` Bayer packé RGGB
- Sortie : `[1, 3, 1024, 1024]` caméra RGB
- Profil : `wb_norm: none`, `output_scale: match_gain`,
  `bayer_orientation: force_rggb`, `edge_pad: mirror_cropped`
- Licence : GPL-3.0-only, NAFNet MIT, datasets CC BY-SA 4.0
- Référence : https://huggingface.co/AyedaOk/okeinla

---

## 4. Conception proposée

### 4.1 Deux méthodes

Params v3 :

- `method` : `legacy` ou `profiled`
- `profile_auto`
- `profile_name`
- `profile_a[3]`, `profile_b[3]`
- `profile_strength`
- `bias` / `shadows`
- `curve_preset`
- champs existants : `threshold`, `x`, `y`

Migration v2 → v3 :

- `method = legacy`
- profil désactivé
- seuil et courbes copiés
- **ancien rendu inchangé**

### 4.2 Profils

- `dt_noiseprofile_get_matching()`
- `dt_noiseprofile_interpolate()`
- logique de `denoiseprofile.c` (`dt_iop_denoiseprofile_get_auto_profile`)
- **stocker `a`/`b` résolus dans les params**, pas seulement le nom du profil.

Fallback : profil générique ou `a`/`b` nuls, avec message clair.

### 4.3 Balance des blancs et CFA

- `rawdenoise` est après `temperature`.
- Récupérer la WB via `piece->pipe->dsc.temperature.coeffs`.
- Diviser chaque site par le coefficient du canal.
- Mapping :
  - site R → profil R
  - site B → profil B
  - sites G1/G2 → profil G
- Gérer la parité du ROI via `roi_in->x/y` et `FC()`.

### 4.4 VST

Pour chaque canal CFA :

```text
x = valeur_CFA / wb[canal]
y = 2 * sqrt(max(x + b, 0) / a)
```

Appliquer `dwt_denoise()` dans le domaine VST.

Inverse non biaisée (référence `backtransform_v2`) :

```text
z = (y + sqrt(y² + bias)) / (4 / sqrt(a))
x = a * z² / 4 - b
valeur_CFA = x * wb[canal]
```

Puis clip [0,1].

### 4.5 Seuils wavelet

```text
seuil[bande] = strength * force[canal][bande] * sigma_bande
```

- `sigma_bande` calibré sur bruit blanc dans le domaine VST.
- `force` vient des courbes existantes.
- Presets repris de `denoiseprofile` :
  - `preserve details` : `0.5 - 0.45*t`
  - `high ISO` : `0.9 - 0.8*smoothstep(t)`
  - `smooth`
  - `custom`

Auto-sélection selon ISO et `a`, surcharge manuelle possible.

### 4.6 Détail

MVP :

- VST + profils par canal + courbes par bande.
- `strength`, `bias`, `profile_strength`.
- Onglets R/G/B existants.

Plus tard :

- wavelet edge-aware ;
- blend de détail haute fréquence ;
- passe de détail guidée.

### 4.7 Legacy

- `legacy` conserve le code actuel.
- `profiled` est le nouveau moteur.
- Anciens edits migrés en `legacy`.
- Bascule manuelle legacy → profiled avec avertissement explicite.
- Presets migrés via `legacy_params`.
- Ne jamais supprimer l'entrée d'historique : sans migration,
  `develop.c:2659` la supprime.

---

## 5. Phases

### Phase 0 — Validation de la valeur ajoutée (gate)

- VST sur zone plate : variance ≈ 1.
- Bruit synthétique `a*x + b` : retrouver le signal.
- Mapping CFA/WB : vérifier sur plusieurs canaux.
- Comparer avec `denoiseprofile` sur une image dématriçée.
- Estimer le gain de détail vs `rawdenoise` actuel.

**Go/no-go** : poursuivre seulement si le gain visuel et la robustesse sont
démontrés.

### Phase 1 — Params, migration, UI

- params v3, `legacy_params` v2 → v3 ;
- combo méthode ;
- combo profil auto/générique/profils ;
- affichage ISO détecté ;
- stockage `a`/`b` dans les params ;
- tests de migration.

### Phase 2 — Moteur VST

- préconditionnement VST par canal CFA ;
- `dwt_denoise` dans le domaine VST ;
- inverse non biaisée ;
- WB et clip ;
- calibration `sigma_bande` ;
- auto-courbes ;
- `strength`, `bias`, `profile_strength`.

### Phase 3 — Tests et documentation

- variance VST ;
- bruit synthétique ;
- image d'intégration raw ;
- non-régression legacy ;
- absence de profil / non-raw / X-Trans ;
- documentation du workflow.

### Phase 4 — Option détail avancé

- edge-aware ;
- détail recovery ;
- à ne faire que si la Phase 3 montre un manque.

---

## 6. Validation sans raw multi-ISO

Aucun raw de test multi-ISO n'est disponible.

- Bruit synthétique avec variance `a*x + b` connue.
- Zones plates pour vérifier la variance VST.
- Image d'intégration raw existante pour la non-régression.
- Comparaison avec `denoiseprofile` sur une image dématriçée.
- Si des raws réels sont fournis plus tard, affiner la calibration.

---

## 7. Risques

| Risque | Mitigation |
|---|---|
| Profils RGB mal adaptés au CFA | Phase 0, facteur de calibration |
| WB incorrecte | Réutiliser `dsc.temperature.coeffs`, tests |
| Inverse VST biaisée | Port de `backtransform_v2`, tests |
| Gain DWT mal calibré | Mesure sur bruit blanc |
| Anciens edits cassés | Mode `legacy` + migration |
| Pas de raws de test | Validation synthétique |
| Performance CPU | Mesurer, garder legacy |
| Attente PureRaw irréaliste | Okeinla reste le chemin IA |

---

## 8. Critères d'acceptation

- Les edits v2 rendent strictement comme avant.
- Aucun module supprimé de l'historique.
- Moteur `profiled` déterministe CPU.
- Profil auto + interpolation fonctionnels.
- `a`/`b` stockés dans les params.
- Variance VST ≈ 1 sur zone plate.
- Pas de biais visible clair/foncé.
- Détail préservé au moins aussi bien que le mode legacy.
- Pas de crash sans profil / non-raw / X-Trans.

---

## 9. Questions ouvertes

- Plan complet VST ou MVP d'abord ?
- `profiled` par défaut pour les nouveaux edits ?
- Edge-aware dès la v1 ou plus tard ?
- CPU-only confirmé ?
- La valeur ajoutée vs `denoiseprofile` est-elle suffisante ? (gate Phase 0)

---

## 10. Checklist de reprise

1. Relire ce document.
2. Vérifier l'état de `src/iop/rawdenoise.c` et `denoiseprofile.c`.
3. Exécuter la Phase 0.
4. Décider go/no-go.
5. Si go : Phase 1, puis Phase 2.
6. Si no-go : archiver ce plan.

---

## 11. Références

- Okeinla : https://huggingface.co/AyedaOk/okeinla
- `onnx/model.json` : adapter `rawnind-bayer-v1`
- Export darktable : `src/rawnind/tools/export_darktable_model.py`
- RawNIND : https://github.com/trougnouf/rawnind_jddc
- NAFNet : https://github.com/megvii-research/NAFNet
- Profils : `data/noiseprofiles.json`
- VST de référence : `src/iop/denoiseprofile.c`
- DWT : `src/common/dwt.c`
- Bruit synthétique : `src/develop/noise_generator.h`
- Upstream OpenCL rawdenoise (non suivi) : commit `51027032c1`
