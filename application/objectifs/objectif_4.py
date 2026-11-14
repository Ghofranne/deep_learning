"""Objectif 4 — segmentation d'instance des anomalies routières (Mask R-CNN)."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .commun import dossier_modele

DOSSIER_APPLICATION = dossier_modele(4)
DOSSIER_NOTEBOOK = Path(__file__).resolve().parents[2] / "modeles" / "objectif_4"
FICHIERS_REQUIS = ("model.pt", "config.json")
DOSSIER = next(
    (
        dossier
        for dossier in (DOSSIER_APPLICATION, DOSSIER_NOTEBOOK)
        if all((dossier / nom).is_file() for nom in FICHIERS_REQUIS)
    ),
    DOSSIER_APPLICATION,
)

INFO = {
    "bouton": "Segmentation d'instance",
    "titre": "Segmentation d'instance des anomalies",
    "membre": "Membre 4",
    "description": (
        "Mask R-CNN repère chaque anomalie routière, trace son contour et "
        "affiche une boîte ainsi qu'un score de confiance."
    ),
    "libelle_resultat": "Anomalies segmentées",
}
PRET = True

_COULEUR_MASQUE = np.array([249, 87, 56], dtype=np.uint8)
_COULEUR_BOITE = (190, 45, 20)
_SEUIL_DEFAUT = 0.5


def _creer_modele():
    modele = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=400,
        max_size=700,
    )
    in_features_box = modele.roi_heads.box_predictor.cls_score.in_features
    modele.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=2)
    in_features_mask = modele.roi_heads.mask_predictor.conv5_mask.in_channels
    modele.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced=256, num_classes=2
    )
    return modele


def charger():
    """Charge une seule fois les poids exportés par 04_segmentation_instance.ipynb."""
    config = json.loads((DOSSIER / "config.json").read_text(encoding="utf-8"))
    seuil = float(config.get("seuil_confiance", _SEUIL_DEFAUT))
    if not 0 <= seuil <= 1:
        raise ValueError("Le seuil de confiance de l'objectif 4 doit être compris entre 0 et 1.")

    modele = _creer_modele()
    etat = torch.load(DOSSIER / "model.pt", map_location="cpu", weights_only=True)
    modele.load_state_dict(etat)
    modele.eval()
    return {"modele": modele, "config": config, "seuil": seuil}


def pastilles(modele):
    config = modele["config"]
    resultats = [config.get("modele", "Mask R-CNN ResNet50-FPN")]
    if "mIoU" in config:
        resultats.append(f"mIoU : {float(config['mIoU']):.3f}".replace(".", ","))
    resultats.append(f"Seuil : {modele['seuil']:.0%}")
    return resultats


@torch.inference_mode()
def predire(image, modele):
    """Segmente les anomalies et renvoie l'image annotée avec un résumé Markdown."""
    hauteur, largeur = image.shape[:2]
    tenseur = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .float()
        .div(255)
    )
    sortie = modele["modele"]([tenseur])[0]
    boites = sortie["boxes"].detach().cpu().numpy()
    scores = sortie["scores"].detach().cpu().numpy()
    masques = sortie["masks"].detach().cpu().numpy()
    classes = sortie["labels"].detach().cpu().numpy()

    garder = (scores >= modele["seuil"]) & (classes == 1)
    boites, scores, masques = boites[garder], scores[garder], masques[garder]
    resultat = np.ascontiguousarray(image.copy())
    epaisseur = max(2, largeur // 400)

    for numero, (boite, score, masque) in enumerate(zip(boites, scores, masques), start=1):
        zone = masque[0] >= 0.5
        resultat[zone] = (
            resultat[zone].astype(np.float32) * 0.4
            + _COULEUR_MASQUE.astype(np.float32) * 0.6
        ).astype(np.uint8)

        x1, y1, x2, y2 = np.rint(boite).astype(int)
        x1, x2 = np.clip([x1, x2], 0, largeur - 1)
        y1, y2 = np.clip([y1, y2], 0, hauteur - 1)
        cv2.rectangle(resultat, (x1, y1), (x2, y2), _COULEUR_BOITE, epaisseur)
        etiquette = f"Anomalie {numero} · {score:.0%}"
        cv2.putText(
            resultat,
            etiquette,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _COULEUR_BOITE,
            2,
            cv2.LINE_AA,
        )

    if len(scores) == 0:
        texte = f"**Aucune anomalie détectée** (seuil : {modele['seuil']:.0%})."
    else:
        lignes = "\n".join(
            f"- Anomalie {i} : confiance {score:.1%}"
            for i, score in enumerate(scores, start=1)
        )
        texte = (
            f"**{len(scores)} anomalie(s) détectée(s)** "
            f"(seuil : {modele['seuil']:.0%}).\n\n{lignes}"
        )
    return resultat, texte
