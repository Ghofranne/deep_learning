"""Objectif 2 — robustesse des décisions de conduite aux conditions météo."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .commun import dossier_modele, en_base64

INFO = {
    "bouton": "Météo & conduite",
    "titre": "Robustesse de la conduite à la météo",
    "membre": "Membre 2",
    "description": (
        "Le modèle prédit une manœuvre de conduite et permet de comparer sa décision "
        "sur l'image d'origine et sous différentes conditions météo simulées."
    ),
}
PRET = True
GABARIT = "objectif_2.html"
FICHIERS_REQUIS = ("model.pt", "config.json")
CHAMPS_ANALYSE = ("meteo", "intensite")

DOSSIER = dossier_modele(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

METEO = ("aucune", "brouillard", "pluie", "nuit", "soleil", "flou de vitesse", "neige")


def meteo_brouillard(img, intensite, rng):
    h = img.shape[0]
    voile = (intensite * (0.3 + 0.5 * (1 - np.linspace(0, 1, h))))[:, None, None]
    out = img.astype(np.float32) * (1 - voile) + np.array([215, 215, 220], np.float32) * voile
    return cv2.GaussianBlur(out, (0, 0), 0.5 + 1.5 * intensite).clip(0, 255).astype(np.uint8)


def meteo_pluie(img, intensite, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32) * (0.85 - 0.2 * intensite) + np.array([0, 5, 15], np.float32) * intensite
    layer = np.zeros((h, w), np.float32)
    n = int(250 + 900 * intensite)
    longueur = int(8 + 10 * intensite)
    angle = rng.uniform(-0.3, 0.3)
    for x, y in zip(rng.integers(0, w, n), rng.integers(0, h, n)):
        cv2.line(
            layer,
            (int(x), int(y)),
            (int(x + longueur * np.sin(angle)), int(y + longueur * np.cos(angle))),
            1.0,
            1,
        )
    layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None] * (0.5 + 0.4 * intensite)
    out = out * (1 - layer) + 200 * layer
    return cv2.GaussianBlur(out, (0, 0), 0.6 + intensite).clip(0, 255).astype(np.uint8)


def meteo_nuit(img, intensite, rng):
    h, w = img.shape[:2]
    out = (img.astype(np.float32) / 255.0) ** (1.5 + 1.5 * intensite) * (0.75 - 0.35 * intensite)
    yy, xx = np.mgrid[0:h, 0:w]
    phares = np.exp(
        -(((xx - w / 2) / (0.35 * w)) ** 2 + ((yy - h) / (0.45 * h)) ** 2)
    )[..., None]
    out = out * (1 + 1.5 * phares) * np.array([0.85, 0.9, 1.1], np.float32)
    out += rng.normal(0, 0.02 + 0.03 * intensite, out.shape)
    return (out * 255).clip(0, 255).astype(np.uint8)


def meteo_soleil(img, intensite, rng):
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.0, 0.3) * h
    yy, xx = np.mgrid[0:h, 0:w]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / w
    flare = np.exp(-(distance / (0.15 + 0.25 * intensite)) ** 2)[..., None] * (0.6 + 0.3 * intensite)
    out = img.astype(np.float32)
    moyenne = out.mean()
    out = (out - moyenne) * (1 - 0.4 * intensite) + moyenne + 30 * intensite
    out = out * (1 - flare) + np.array([255, 240, 200], np.float32) * flare
    return out.clip(0, 255).astype(np.uint8)


def meteo_flou(img, intensite, rng):
    taille = int(5 + 14 * intensite) | 1
    noyau = np.zeros((taille, taille), np.float32)
    if rng.random() < 0.5:
        noyau[taille // 2, :] = 1.0
    else:
        noyau[:, taille // 2] = 1.0
    return cv2.filter2D(img, -1, noyau / taille)


def meteo_neige(img, intensite, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32) * (0.8 - 0.1 * intensite) + 40 * intensite
    layer = np.zeros((h, w), np.float32)
    n = int(300 + 1200 * intensite)
    for x, y, rayon in zip(rng.integers(0, w, n), rng.integers(0, h, n), rng.integers(1, 3, n)):
        cv2.circle(layer, (int(x), int(y)), int(rayon), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None]
    return (out * (1 - layer) + 245 * layer).clip(0, 255).astype(np.uint8)


TRANSFORMATIONS_METEO = {
    "brouillard": meteo_brouillard,
    "pluie": meteo_pluie,
    "nuit": meteo_nuit,
    "soleil": meteo_soleil,
    "flou de vitesse": meteo_flou,
    "neige": meteo_neige,
}


class SpatialHead(nn.Module):
    def __init__(self, canaux):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(canaux * 8, 3)

    def forward(self, x):
        return self.fc(self.dropout(self.pool(x).flatten(1)))


class Net(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        return self.head(self.features(x))


def make_model():
    resnet = torchvision.models.resnet18(weights=None)
    features = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )
    return Net(features, SpatialHead(512))


def charger():
    cfg = json.loads((DOSSIER / "config.json").read_text(encoding="utf-8"))
    model = make_model()
    state = torch.load(DOSSIER / "model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return {"cfg": cfg, "model": model, "height": int(cfg["H"]), "width": int(cfg["W"])}


def pastilles(modele):
    cfg = modele["cfg"]
    return [
        cfg.get("modele", "ResNet18"),
        f"Accuracy beau temps : {cfg['acc_beau_temps']}",
        f"Accuracy météo : {cfg['acc_mauvais_temps']}",
    ]


def tensor_image(img):
    x = torch.from_numpy(img[None]).float().div(255).permute(0, 3, 1, 2)
    return ((x - MEAN) / STD).to(DEVICE)


@torch.no_grad()
def probabilites(img, modele):
    logits = modele["model"](tensor_image(img)).float()
    return F.softmax(logits, dim=1)[0].cpu().numpy()


def gradcam(img, modele):
    x = tensor_image(img)
    model = modele["model"]
    with torch.enable_grad():
        features = model.features(x)
        logits = model.head(features)
        score = logits[0, logits.argmax(1)[0]]
        gradients = torch.autograd.grad(score, features)[0]
    cam = F.relu((gradients.mean((2, 3), keepdim=True) * features).sum(1))[0]
    cam = (cam / (cam.max() + 1e-8)).detach().float().cpu().numpy()
    cam = cv2.resize(cam, (modele["width"], modele["height"]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.cvtColor(
        cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return (0.55 * img + 0.45 * heat).astype(np.uint8)


def analyser(modele, img=None, seuil=None, ident=None, meteo="aucune", intensite=0.8):
    if img is None:
        return None
    if meteo not in METEO:
        raise ValueError(f"Météo inconnue. Choix possibles : {', '.join(METEO)}.")
    try:
        intensite = float(intensite)
    except (TypeError, ValueError) as exc:
        raise ValueError("L'intensité météo doit être un nombre entre 0 et 1.") from exc
    if not np.isfinite(intensite) or not 0 <= intensite <= 1:
        raise ValueError("L'intensité météo doit être comprise entre 0 et 1.")

    debut = time.perf_counter()
    h, w = modele["height"], modele["width"]
    original = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    probs_beau = probabilites(original, modele)
    image_meteo = (
        original
        if meteo == "aucune"
        else TRANSFORMATIONS_METEO[meteo](original, intensite, np.random.default_rng(0))
    )
    probs = probs_beau if meteo == "aucune" else probabilites(image_meteo, modele)
    classes = modele["cfg"]["CLASS_NAMES"]
    decision_beau = int(probs_beau.argmax())
    decision = int(probs.argmax())
    probabilites_beau = [
        {"nom": classes[i], "pourcentage": round(float(probs_beau[i]) * 100, 1)}
        for i in range(len(classes))
    ]
    probabilites_meteo = [
        {"nom": classes[i], "pourcentage": round(float(probs[i]) * 100, 1)}
        for i in range(len(classes))
    ]

    return {
        "meteo": meteo,
        "intensite": round(intensite, 2),
        "decision": classes[decision],
        "decision_beau": classes[decision_beau],
        "meme_decision": decision == decision_beau,
        "probabilites": probabilites_meteo,
        "probabilites_beau": probabilites_beau,
        "original": en_base64(original),
        "image_meteo": en_base64(image_meteo),
        "gradcam": en_base64(gradcam(image_meteo, modele)),
        "ms": round((time.perf_counter() - debut) * 1000),
    }
