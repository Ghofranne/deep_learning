"""Objectif 6 — Segmentation d'instance des anomalies routières (Mask R-CNN).

Modèle  : Mask R-CNN ResNet50-FPN entraîné sur le dataset RoadAnomaly.
Fichier : modeles/objectif_6/mask_rcnn_final.pth
Classes : 1 classe d'objet (anomalie) + background  →  num_classes = 2.

Pour chaque image, le modèle détecte les anomalies, dessine leurs boîtes
englobantes et superpose leurs masques de segmentation d'instance.
"""
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .commun import dossier_modele

DOSSIER = dossier_modele(6)

# Seuil de confiance par défaut pour filtrer les détections
_SEUIL_CONF = 0.5

# Couleur des masques et boîtes (cyan, en RGB)
_COULEUR_MASQUE = np.array([0, 200, 255], dtype=np.uint8)
_COULEUR_BOITE  = (0, 200, 255)

INFO = {
    "bouton": "Mask R-CNN",
    "titre": "Segmentation d'instance des anomalies",
    "membre": "Membre 6",
    "description": (
        "Détecte et segmente chaque anomalie présente sur la route "
        "(animaux, obstacles, cônes…) à l'aide de Mask R-CNN "
        "(ResNet50-FPN) entraîné sur le dataset RoadAnomaly."
    ),
    "libelle_resultat": "Anomalies segmentées",
    "pastilles": ["Modèle : Mask R-CNN ResNet50-FPN", "Tâche : segmentation d'instance"],
}
PRET = True
FICHIERS_REQUIS = ("mask_rcnn_final.pth",)


def charger():
    """Charge le Mask R-CNN depuis mask_rcnn_final.pth (num_classes=2)."""
    # Construire l'architecture standard avec les mêmes têtes que lors de l'entraînement
    modele = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    # Remplacer la tête de classification par celle à 2 classes (background + anomalie)
    in_features_box = modele.roi_heads.box_predictor.cls_score.in_features
    modele.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=2)

    in_features_mask = modele.roi_heads.mask_predictor.conv5_mask.in_channels
    modele.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced=256, num_classes=2
    )

    state = torch.load(DOSSIER / "mask_rcnn_final.pth", map_location="cpu", weights_only=False)
    modele.load_state_dict(state)
    modele.eval()
    return modele


def predire(image, modele):
    """Détecte et segmente les anomalies sur l'image.

    image  : tableau numpy RGB, valeurs 0-255, forme (H, W, 3).
    modele : Mask R-CNN renvoyé par charger().
    Retourne (image_resultat, texte_markdown).
    """
    # Convertir en tenseur float [0, 1]
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        sorties = modele(tensor)[0]

    boites   = sorties["boxes"].numpy()
    scores   = sorties["scores"].numpy()
    masques  = sorties["masks"].numpy()   # shape : (N, 1, H, W)

    # Filtrer par seuil de confiance
    garder = scores >= _SEUIL_CONF
    boites  = boites[garder]
    scores  = scores[garder]
    masques = masques[garder]

    # Visualisation : superposer masques et boîtes sur l'image originale
    resultat = image.copy()
    epaisseur = max(2, image.shape[1] // 400)

    for i, (boite, score, masque) in enumerate(zip(boites, scores, masques)):
        # Masque binaire (seuil 0.5)
        masque_bin = (masque[0] >= 0.5)

        # Colorer le masque avec transparence
        overlay = resultat.copy()
        overlay[masque_bin] = (
            overlay[masque_bin].astype(np.float32) * 0.4
            + _COULEUR_MASQUE.astype(np.float32) * 0.6
        ).astype(np.uint8)
        resultat = overlay

        # Dessiner la boîte englobante
        x1, y1, x2, y2 = map(int, boite)
        cv2.rectangle(resultat, (x1, y1), (x2, y2), _COULEUR_BOITE, epaisseur)

        # Étiquette avec le score
        etiquette = f"anomalie {score:.2f}"
        (tw, th), _ = cv2.getTextSize(etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(y1 - th - 6, 0)
        cv2.rectangle(resultat, (x1, y0), (x1 + tw + 6, y0 + th + 6), _COULEUR_BOITE, -1)
        cv2.putText(
            resultat, etiquette, (x1 + 3, y0 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

    # Résumé Markdown
    n = len(scores)
    if n == 0:
        texte = (
            "**Aucune anomalie détectée** "
            f"(seuil de confiance : {_SEUIL_CONF:.0%})"
        )
    else:
        lignes = "\n".join(
            f"- Anomalie {i + 1} : {s:.1%}" for i, s in enumerate(scores)
        )
        texte = (
            f"**{n} anomalie(s) détectée(s)** "
            f"(seuil : {_SEUIL_CONF:.0%})\n\n"
            f"{lignes}"
        )

    return resultat, texte
