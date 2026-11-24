# AI Assistant v2 — plan

- **Statut** : implémenté (catalogue, activation/revert, stats, aperçu optionnel et adaptateurs cloud) ; benchmark qwen3.6 à documenter
- **Date** : 2026-11-24
- **Portée** : fork personnel `libre-dt-lab`
- **Objectif** : améliorer l'analyse et les corrections de l'assistant darkroom en
  lui donnant un catalogue de modules extensible, la possibilité d'activer des
  modules absents de l'historique, des statistiques locales et des hints
  déterministes.

---

## 1. Décisions validées

- L'assistant peut **activer des modules** absents de l'historique, avec
  indication claire et revert.
- La liste des modules autorisés est gérée dans **Préférences → Assistant**,
  sous forme de liste avec ajout/retrait.
- La liste est **préremplie avec les modules v1**.
- Pas de colonne `State` dans le tableau de résultats. Un tag discret
  `· enable` apparaît sur la ligne et un résumé indique le nombre de modules à
  activer.
- Les **statistiques locales + hints déterministes** sont inclus dès la v1.
- Le tone mapper de référence est celui de `plugins/darkroom/workflow`, avec
  priorité au tone mapper déjà actif.
- Premier test de qualité avec **qwen3.6**, puis ajout des adaptateurs
  **Claude/Gemini** (API natives, texte et aperçu optionnel).
- **Aucune image n'est envoyée par défaut**. Le contexte reste du texte et des
  nombres ; l'aperçu est une option explicite (voir §8 et `AI_ASSISTANT.md`).

---

## 2. Problème initial (traité dans ce plan)

L'assistant ne proposait que des changements sur les modules déjà activés dans
l'historique. Il ne connaissait ni les nouveaux modules du fork (3DCF, Tone &
Texture, ACES 2.0, Color Harmonizer, RGB Primaries) ni leur sémantique. Il ne
recevait aucune statistique d'image : le modèle ne pouvait pas réellement
analyser l'exposition, le clipping ou le contraste.

La qualité dépendait aussi fortement du modèle local utilisé.

---

## 3. Architecture cible

```text
data/ai_assistant_modules.json          catalogue des modules et sliders
src/common/assistant/assistant_catalog.c/.h   chargement/validation du catalogue
src/common/assistant/assistant_context.c/.h   contexte texte + modules + stats
src/common/assistant/assistant_actions.c/.h   autorisation, activation,
                                              application, revert
src/common/assistant/assistant_stats.c/.h     capture 512 px, percentiles,
                                              clipping, hints
src/common/assistant/assistant_image.c/.h     aperçu JPEG 768 px optionnel
                                              en mémoire
src/libs/ai_assistant.c                 panneau darkroom
src/gui/preferences_assistant.c         préférences, liste des modules
src/libs/workflow_helpers.c/.h          helper find/load module partagé
```

---

## 4. Catalogue de modules

Fichier installé dans `<datadir>/libre-dt-lab/ai_assistant_modules.json`.

Format :

```json
{
  "version": 1,
  "modules": [
    {
      "operation": "3dcf",
      "name": "3D Colorimetric Film",
      "description": "spectral film tone mapping; contrast, gamma, vibrance, highlights",
      "role": "tone_mapper",
      "exclusive_role": true,
      "addable": true,
      "params": [
        {
          "name": "contrast",
          "description": "global contrast",
          "max_delta": 0.5
        }
      ]
    }
  ]
}
```

### Modules v1

| Module | Sliders principaux |
|---|---|
| `3dcf` | `contrast`, `gamma`, `vibrance`, `input_exposure`, `contrast_pivot`, `toe_power`, `shoulder_power`, `hl_desaturation`, `hl_detail_recovery` |
| `contrast` | `global_scale`, `micro_scale`, `fine_scale`, `local_scale`, `broad_scale`, `coarse_scale`, `gain_highlights`, `gain_shadows`, `colorful_contrast`, `contrast_balance`, `color_balance` |
| `aces20` | `exposure_ev`, `peak_luminance` |
| `colorharmonizer` | `anchor_hue`, `pull_strength`, `neutral_protection`, `pull_width`, `smoothing` |
| `primaries` | `red_hue`, `red_purity`, `green_hue`, `green_purity`, `blue_hue`, `blue_purity`, `achromatic_tint_hue`, `achromatic_tint_purity` |
| `exposure` | `exposure`, `black` |
| `toneequal` | les 9 bandes tonales |
| `agx` | `look_brightness`, `look_saturation`, `curve_contrast_around_pivot` |
| `filmicrgb` | `contrast`, `saturation`, `balance` |
| `colorbalancergb` | `contrast`, `vibrance`, `saturation_global` |
| `sharpen` | `amount`, `threshold` |
| `diffuse` | `sharpness` |

Le catalogue est la partie « apprentissage » : on peut ajouter un module ou
modifier ses sliders sans recompiler. La validation reste faite par
introspection.

---

## 5. Préférences → Assistant

Nouvelle section **Modules utilisables par l'assistant** :

- liste des modules autorisés avec rôle ;
- bouton **Ajouter un module** listant les modules du catalogue non autorisés ;
- croix pour retirer un module ;
- bouton **Réinitialiser** (revient aux modules v1) ;
- stockage dans `plugins/assistant/modules`, séparateur `|` ;
- préremplissage avec les modules v1.

L'assistant ne peut proposer que des modules présents dans cette liste.

---

## 6. Activer un module absent de l'historique

`src/libs/workflow_helpers.c` contient déjà :

- `_find_module()` : cherche dans `dev->iop` puis `dev->alliop` ;
- `_load_module()` : charge le module SO, l'insère dans `dev->iop` et
  l'iop_order, initialise le GUI.

À factoriser dans un helper partagé, puis à utiliser par l'assistant.

### Application

1. Sauvegarder pour chaque module touché :
   - le blob complet de `params` ;
   - l'état `enabled` ;
   - le fait qu'il était déjà dans l'historique.
2. Si le module est désactivé :
   - `dt_iop_reload_defaults()` ;
   - appliquer les paramètres proposés ;
   - `enabled = TRUE`.
3. `dt_dev_add_history_item(dev, module, TRUE)`.

### Revert

- module déjà présent : restaurer le blob et l'état `enabled`, puis ajouter
  l'entrée d'historique ;
- module chargé par l'assistant : `dt_dev_module_remove()` ;
- aucun revert partiel silencieux.

### Garde-fous

- un seul tone mapper actif (`exclusive_role`) ;
- maximum 5 changements ;
- maximum 2 modules nouvellement activés ;
- pas de masques, courbes, blending, ordre, white balance.

---

## 7. Tone mapper de référence

Lire `plugins/darkroom/workflow` :

| Préférence | Op |
|---|---|
| `scene-referred (3DCF)` | `3dcf` |
| `scene-referred (ACES 2.0)` | `aces20` |
| `scene-referred (AgX)` | `agx` |
| `scene-referred (filmic)` | `filmicrgb` |
| `scene-referred (sigmoid)` | `sigmoid` |
| `scene-referred (basecurve)` | `basecurve` |
| `none` | aucun |

Règle : si un tone mapper est déjà actif dans l'historique, on règle celui-là.
Sinon, on utilise celui des préférences. On n'en active jamais un deuxième.

---

## 8. Statistiques locales et hints

Dans le worker, exporter l'image à **512 px en mémoire** via
`dt_imageio_export_with_flags`, avec un format de capture inspiré de
`neural_restore.c`. Aucun fichier n'est écrit et cette capture de statistiques
reste locale : elle n'est jamais envoyée.

L'**aperçu optionnel** est distinct : une case « envoyer l'aperçu » génère un
JPEG **768 px** du rendu darkroom courant, en mémoire, sans métadonnées. Il
n'est envoyé que si elle est cochée et que le modèle choisi accepte les images ;
sinon l'assistant continue en texte seul et journalise le repli. Voir
`AI_ASSISTANT.md`.

Calculs :

- percentiles de luminance : p1, p5, p25, p50, p75, p95, p99 ;
- pourcentage de clipping bas/haut, par canal ;
- moyenne de luminance ;
- `contrast_spread = p95 - p5`.

Hints :

- clipping haut ou p99 > 0.99 → exposition négative / highlights ;
- p50 < 0.12 → exposition positive ;
- clipping bas ou p5 < 0.02 → relever les ombres ;
- `contrast_spread` trop faible → augmenter le contraste ;
- `contrast_spread` trop élevé → réduire le contraste.

Le contexte JSON contient les stats et les hints. Le prompt impose de les
utiliser pour justifier chaque changement.

---

## 9. Tableau de résultats

Toujours 5 colonnes :

```text
[Apply] | Module | Slider | Value | Why
```

- tag `· enable` sur le module s'il sera activé ;
- résumé au-dessus du tableau : « 3 changements · 1 module à activer » ;
- tooltip du bouton Apply indiquant l'activation.

---

## 10. Modèles

1. Test avec **qwen3.6** (local) :
   - portrait, paysage, backlight ;
   - JSON valide, latence, pertinence.
2. Si insuffisant : DeepSeek texte (déjà supporté).
3. Ajouter les adaptateurs **Anthropic** et **Gemini** (API natives) :
   - endpoints, clés, modèles dans les préférences ;
   - même plan JSON et mêmes validations ;
   - texte par défaut, aperçu 768 px optionnel si le modèle accepte les images.

Par défaut, aucune image ne quitte la machine. Une image ne part que si
l'utilisateur coche « envoyer l'aperçu » ; elle est alors réduite à 768 px,
générée en mémoire et sans métadonnées.

---

## 11. Phases

| Phase | Contenu | Effort |
|---|---|---|
| 0 | Spike activation/retrait module + revert blob + capture stats 512 px + test qwen3.6 | 4–5 jours |
| 1 | Catalogue JSON, modules v1, préférences liste d'ajout, available_modules, tag enable, apply/revert | 2 semaines |
| 2 | Stats + hints + prompt few-shot | 1 semaine |
| 3 | Adaptateurs Claude/Gemini (texte + aperçu optionnel) + benchmark modèles | 1 semaine |

**Total v1 : 4 à 5 semaines** à temps partiel.

---

## 12. Phase 0 — détail

1. Vérifier le chargement d'un module absent de l'historique via
   `dt_iop_load_module`/`dt_iop_get_module_so`.
2. Activer le module, ajouter l'entrée d'historique, vérifier le rendu.
3. Sauvegarder/restaurer le blob de paramètres et l'état `enabled`.
4. Tester `dt_dev_module_remove()` pour un module ajouté.
5. Implémenter une capture 512 px et calculer les percentiles/clipping.
6. Tester qwen3.6 sur trois images (portrait, paysage, backlight) et noter :
   - JSON valide ;
   - latence ;
   - pertinence des changements.
7. Décider si le prompt/ modèle doit être ajusté avant la Phase 1.

---

## 13. Critères d'acceptation

- On peut ajouter/retirer un module dans Préférences → Assistant.
- L'assistant ne propose que les modules autorisés.
- Il peut activer 3dcf, Tone & Texture, etc., avec indication et revert.
- Un seul tone mapper à la fois.
- Les stats locales sont présentes dans le contexte et influencent les
  propositions.
- Par défaut, aucune image n'est envoyée ; l'aperçu 768 px est opt-in et
  retombe en texte seul si le modèle ou le fournisseur le refuse.
- Le revert restaure exactement les paramètres et l'état d'activation
  précédents.
- Le test qwen3.6 est documenté.

---

## 14. Risques

- L'activation de modules modifie l'historique : revert complet obligatoire.
- Conflits de tone mappers : validation par rôle.
- `primaries` et `colorharmonizer` sont créatifs : deltas conservateurs.
- La capture 512 px doit rester dans le worker, jamais sur le thread GUI.
- qwen3.6 est gros (36B) : latence et VRAM à mesurer.
- Un petit modèle restera limité ; Claude/Gemini (texte ou vision) seront
  probablement meilleurs pour l'analyse.
