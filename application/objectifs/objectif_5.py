"""Objectif 5 — Classification du type d'anomalie routière (ResNet18).

Modèle : ResNet18 entraîné sur le dataset RoadAnomaly.
Fichier de poids : modeles/objectif_5/resnet18_types.pth
Classes (3 sorties de la couche fc) :
  0 → animaux   (animals)
  1 → obstacles (rochers, hérissons tchèques, débris…)
  2 → cônes     (cones)
  ⚠ Si ton entraînement utilisait un ordre différent, ajuste la liste CLASSES ci-dessous.
"""
import cv2
import numpy as np
import torch
import torchvision

from .commun import dossier_modele

DOSSIER = dossier_modele(5)

# --- Classes dans l'ordre de la couche fc du modèle (fc.weight shape : [3, 512]) ---
# Vérifie l'ordre utilisé lors de l'entraînement et ajuste si nécessaire.
CLASSES = ["animaux", "obstacles", "cônes"]

# Normalisation ImageNet standard (même que torchvision)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

INFO = {
    "bouton": "Classification",
    "titre": "Classification du type d'anomalie",
    "membre": "Membre 5",
    "description": (
        "Identifie le type d'anomalie présente sur la route "
        "(animaux, obstacles, cônes) à l'aide d'un ResNet18 entraîné "
        "sur le dataset RoadAnomaly."
    ),
    "libelle_resultat": "Type d'anomalie prédit",
    "pastilles": ["Modèle : ResNet18", f"Classes : {', '.join(CLASSES)}"],
}
PRET = True
FICHIERS_REQUIS = ("resnet18_types.pth",)


def charger():
    """Charge le ResNet18 (3 classes) depuis resnet18_types.pth."""
    poids = DOSSIER / "resnet18_types.pth"
    modele = torchvision.models.resnet18(num_classes=len(CLASSES))
    state = torch.load(poids, map_location="cpu", weights_only=False)
    modele.load_state_dict(state)
    modele.eval()
    return modele


def predire(image, modele):
    """Classifie le type d'anomalie sur l'image.

    image  : tableau numpy RGB, valeurs 0-255, forme (H, W, 3).
    modele : ResNet18 renvoyé par charger().
    Retourne (image_resultat, texte_markdown).
    """
    # Prétraitement : redimensionner à 224×224, normaliser
    img_224 = cv2.resize(image, (224, 224))
    x = torch.from_numpy(img_224).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x - _MEAN) / _STD

    with torch.no_grad():
        probabilites = modele(x).softmax(dim=1)[0].numpy()

    idx_predit = int(probabilites.argmax())
    classe_predite = CLASSES[idx_predit]
    confiance = float(probabilites[idx_predit])

    # Résumé Markdown : classe prédite + tableau de probabilités
    lignes = "\n".join(
        f"- {'**' if i == idx_predit else ''}{cls}{'**' if i == idx_predit else ''}"
        f" : {probabilites[i]:.1%}"
        for i, cls in enumerate(CLASSES)
    )
    texte = (
        f"**Classe prédite : {classe_predite}** ({confiance:.1%})\n\n"
        f"{lignes}"
    )
    return image, texte
