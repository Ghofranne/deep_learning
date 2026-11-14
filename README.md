# deep_learning

Projet de groupe du module **Deep Learning** : détecter les objets inattendus et dangereux sur la
route (animaux, rochers, cônes, pneus perdus, véhicules inhabituels) à partir du dataset
[RoadAnomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/) (Lis et al., ICCV 2019).

Le dépôt contient les notebooks (exécutables sur Google Colab), le code partagé et une
application web qui regroupe les objectifs des membres du groupe.

## Objectifs

| N° | Objectif | Membre | État |
|---|---|---|---|
| 1 | Segmentation des anomalies au pixel près | *à compléter* | terminé |
| 2 | Robustesse des décisions de conduite aux conditions météo | *à compléter* | intégré |
| 3 | Détection, comptage et catégorisation d'obstacles | Ghofrane | intégré |
| 4 | Segmentation d'instance des anomalies (Mask R-CNN) | Membre 4 | intégré |
| 5 | Classification du type d'anomalie (ResNet18) | Membre 5 | intégré |
| 6 | Segmentation d'instance des anomalies (Mask R-CNN) | Membre 6 | intégré |

## Objectif 1 — Segmentation des anomalies

Pour chaque pixel d'une image de route, décider s'il appartient à une anomalie (segmentation
sémantique binaire). Cinq approches ont été comparées sur les mêmes 60 images, en validation
croisée en 5 plis, stratifiée par catégorie et groupée par source (pour éviter la fuite de
données entre images presque identiques).

| Modèle | Approche | AP | IoU |
|---|---|---|---|
| **C. DeepLabV3+ (ResNet34, ImageNet)** | segmentation supervisée, transfer learning | **0,830** | **0,659** |
| B. U-Net (ResNet34, ImageNet) | segmentation supervisée, transfer learning | 0,798 | 0,642 |
| D2. SegFormer Cityscapes + max-logit | détection hors distribution, sans entraînement | 0,423 | 0,282 |
| A. U-Net entraîné de zéro | CNN de référence | 0,237 | 0,200 |
| E1. Autoencodeur convolutif (erreur L1) | erreur de reconstruction | 0,206 | 0,119 |

AP = précision moyenne (hasard : 0,098). IoU calculée sur l'ensemble des pixels, au seuil choisi
sur la validation.

**À retenir :**

- le pré-entraînement sur ImageNet triple l'IoU (0,21 → 0,64) ;
- U-Net et DeepLabV3+ ne diffèrent pas significativement en IoU (test de Wilcoxon, p = 0,31) ;
- DeepLabV3+ est plus stable d'un pli à l'autre et généralise mieux aux anomalies jamais vues
  (pneus : AP 0,573 contre 0,124) : c'est le modèle déployé ;
- point faible commun : les petits objets (moins de 15 % détectés).

## Objectif 2 — Robustesse des décisions de conduite à la météo

Un classifieur ResNet18 prédit une manœuvre parmi « éviter à gauche », « éviter à droite » et
« freiner ». Dans l'onglet 2 de l'application, une image peut être évaluée sous plusieurs météo
simulées : brouillard, pluie, nuit, soleil, flou de vitesse et neige. L'interface compare les
probabilités et la décision avant/après transformation et affiche une carte Grad-CAM.

Le paquet du modèle indique une accuracy de 0,650 par beau temps et 0,640 sous météo défavorable.
Les résultats complets et par pli sont dans `application/modeles/objectif_2/resultats/` lorsqu'ils
sont disponibles localement. Il s'agit d'une étude de faisabilité sur peu d'images, pas d'un système
de conduite autonome ni d'un dispositif de sécurité.

## Objectif 3 — Détection et comptage d'obstacles

L'onglet 3 utilise le modèle YOLO M1 à trois classes (`animal`, `cone`, `autre`) pour dessiner des
boîtes, compter les détections et fournir un niveau d'alerte indicatif. Il permet aussi de comparer
les détections sur une image d'origine et sa version dégradée synthétiquement (faible luminosité,
brouillard, flou, bruit ou pluie), et d'afficher EigenCAM ou Grad-CAM. Ces essais image par image ne
remplacent pas les mesures mAP/recall sur les images annotées du notebook ; les cartes CAM restent
des visualisations indicatives. Le seuil de confiance du modèle est défini dans
`application/modeles/objectif_3/config.json`.

Le fichier `model.pt` est un poids local d'environ 6,5 Mo et reste ignoré par Git ; il doit être
fourni avec l'application ou téléchargé séparément. Les performances comparatives et les métriques
de test n'étaient pas précisées dans les fichiers fournis.

## Objectif 4 — Segmentation d'instance

Mask R-CNN détecte chaque anomalie, superpose son masque et indique son score de confiance.
Le notebook `04_segmentation_instance.ipynb` peut réutiliser un checkpoint existant ou entraîner
le modèle si `ENTRAINER = True`. Il exporte `model.pt` et `config.json` directement dans
`application/modeles/objectif_4/`.

## Structure du dépôt

```
├── notebooks/
│   ├── 01_exploration_dataset.ipynb      exploration et statistiques du dataset
│   ├── 02_pretraitement.ipynb            masques, 5 plis, augmentation, Dataset PyTorch
│   ├── 03a_modeles_supervises.ipynb      U-Net de zéro, U-Net et DeepLabV3+ pré-entraînés
│   ├── 03b_cityscapes_incertitude.ipynb  SegFormer Cityscapes + scores d'incertitude
│   ├── 03c_autoencodeur.ipynb            autoencodeur convolutif
│   ├── 04_benchmark.ipynb                comparaison des modèles de segmentation sémantique
│   ├── 04_segmentation_instance.ipynb    entraînement et déploiement Mask R-CNN (objectif 4)
│   └── 05_deploiement.ipynb              entraînement final, export ONNX, application
├── src/
│   └── roadanomaly_lib.py                chargement, découpage et prétraitement partagés
├── application/                          application web Flask (une page par objectif)
│   ├── app.py                            serveur Flask
│   ├── objectifs/objectif_1.py … objectif_6.py (objectifs 1 à 6 intégrés)
│   ├── templates/, static/               pages HTML, thème et JavaScript
│   ├── modeles/objectif_N/               modèles (non versionnés, voir plus bas)
│   ├── GUIDE_INTEGRATION.md              comment ajouter son objectif
│   └── requirements.txt
└── resultats/                            figures et tableaux produits par les notebooks
```

## Reproduire les résultats (Google Colab)

1. Télécharger le dataset RoadAnomaly et le compresser en `RoadAnomaly.zip`.
2. Ouvrir les notebooks dans l'ordre sur Colab (*Fichier → Importer un notebook*), GPU T4 activé
   à partir du notebook 03a.
3. Fichiers à envoyer dans la première cellule de chaque notebook :

| Notebook | Fichiers à envoyer | Produit |
|---|---|---|
| 01, 02 | `RoadAnomaly.zip` | `resultats_etape2.zip` (contient `roadanomaly_lib.py`) |
| 03a, 03b, 03c | `RoadAnomaly.zip`, `roadanomaly_lib.py` | `resultats_etape3*.zip` |
| 04_benchmark | les deux précédents + les trois `resultats_etape3*.zip` | `resultats_etape4.zip` |
| 04_segmentation_instance | `RoadAnomaly.zip` pour entraîner (`ENTRAINER = True`) | `application/modeles/objectif_4/model.pt` et `config.json` |
| 05 | `RoadAnomaly.zip`, `roadanomaly_lib.py`, `resultats_etape4.zip` | modèle ONNX (`model.onnx`, `config.json`) |

## Lancer l'application

```
cd application
pip install -r requirements.txt
python app.py
```

puis ouvrir http://127.0.0.1:5000. Le modèle de l'objectif 1 (`model.onnx` et `config.json`,
produits par le notebook 05) doit être placé dans `application/modeles/objectif_1/`.
Pour activer l'objectif 2, placer également son fichier `model.pt` dans
`application/modeles/objectif_2/` ; son `config.json` est déjà fourni.
Le modèle YOLO `application/modeles/objectif_3/model.pt` est également requis pour l'objectif 3 ;
son `config.json` est fourni.
Le notebook `notebooks/04_segmentation_instance.ipynb` prépare le modèle de l'objectif 4 dans
`application/modeles/objectif_4/`. Les modèles des objectifs 5 et 6 sont fournis dans leurs
dossiers `application/modeles/objectif_N/`.

**Les modèles ne sont pas dans le dépôt** (fichiers lourds) : ils sont disponibles dans les
*Releases* du dépôt, ou à régénérer avec le notebook 05.

## Données

Le dataset RoadAnomaly n'est pas inclus : ses images appartiennent à leurs auteurs et sont
fournies pour la recherche. Le télécharger depuis la page de l'EPFL et citer :

> K. Lis, K. Nakka, P. Fua, M. Salzmann. *Detecting the Unexpected via Image Resynthesis.*
> ICCV 2019.