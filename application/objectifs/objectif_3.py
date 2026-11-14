"""Objectif 3 : détection, robustesse et explicabilité des obstacles routiers."""
import html
import json
import time
from collections import Counter
from threading import Lock

import cv2
import numpy as np

from .commun import dossier_modele, en_base64

DOSSIER = dossier_modele(3)


def _lire_config():
    chemin = DOSSIER / "config.json"
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


_CFG = _lire_config()

INFO = {
    "bouton": "Détection",
    "titre": "Détection et comptage d'obstacles",
    "membre": "Ghofrane",
    "description": (
        "Détectez et comptez les obstacles, simulez cinq conditions dégradées et visualisez "
        "EigenCAM ou Grad-CAM."
    ),
    "libelle_resultat": "Obstacles détectés",
}
PRET = True
FICHIERS_REQUIS = ("config.json", _CFG.get("model_file", "model.pt"))
CHAMPS_ANALYSE = ("perturbation", "niveau", "explication")

COULEURS = {
    "animal": (255, 200, 0),
    "cone": (255, 90, 0),
    "autre": (0, 200, 255),
    "anomalie": (255, 40, 40),
}
_VERROU = Lock()


class _Detecteur:
    def __init__(self, cfg):
        import torch

        self.cfg = cfg
        self.noms = cfg.get("class_names", ["anomalie"])
        self.cuda = torch.cuda.is_available()
        poids = DOSSIER / cfg["model_file"]
        if cfg["model_type"] == "yolo":
            from ultralytics import YOLO

            self.type, self.modele = "yolo", YOLO(str(poids))
            self.device = torch.device("cuda" if self.cuda else "cpu")
        elif cfg["model_type"] == "fasterrcnn":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            self.torch, self.type = torch, "fasterrcnn"
            self.device = torch.device("cuda" if self.cuda else "cpu")
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                min_size=int(cfg.get("frcnn_min_size", 720)),
                max_size=int(cfg.get("frcnn_max_size", 1280)),
                box_score_thresh=0.001,
                box_detections_per_img=100,
            )
            model.roi_heads.box_predictor = FastRCNNPredictor(
                model.roi_heads.box_predictor.cls_score.in_features,
                2,
            )
            model.load_state_dict(torch.load(poids, map_location=self.device))
            self.modele = model.to(self.device).eval()
        else:
            raise ValueError(f"model_type inconnu dans config.json : {cfg['model_type']}")

    def detecter(self, image_rgb, conf):
        if self.type == "yolo":
            result = self.modele.predict(
                image_rgb[..., ::-1],
                conf=conf,
                iou=0.6,
                imgsz=int(self.cfg.get("imgsz", 640)),
                max_det=300,
                agnostic_nms=True,
                device=0 if self.cuda else "cpu",
                verbose=False,
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
        else:
            tensor = (
                self.torch.from_numpy(np.ascontiguousarray(image_rgb))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .to(self.device)
            )
            with self.torch.no_grad():
                output = self.modele([tensor])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            keep = scores >= conf
            boxes, scores = boxes[keep], scores[keep]
            classes = np.zeros(len(scores), dtype=int)

        order = np.argsort(-scores)
        names = [
            self.noms[int(class_id)] if int(class_id) < len(self.noms) else str(int(class_id))
            for class_id in classes[order]
        ]
        return boxes[order], scores[order], names


def charger():
    """Appelée une seule fois au démarrage : charge le modèle décrit par config.json."""
    return _Detecteur(_lire_config())


def pastilles(modele):
    """Petites étiquettes sous le titre de la page."""
    cfg = modele.cfg
    return [
        f"Modèle : {cfg.get('model_name', cfg['model_type'])}",
        f"Classes : {', '.join(modele.noms)}",
        f"Seuil de confiance : {cfg['conf']:.2f}".replace(".", ","),
    ]


def niveau_alerte(boxes, largeur, hauteur):
    """Règle d'alerte indicative basée sur le nombre et la taille des obstacles."""
    if len(boxes) == 0:
        return "Aucun"
    aire_max = max(
        100 * (x2 - x1) * (y2 - y1) / (largeur * hauteur)
        for x1, y1, x2, y2 in boxes
    )
    if len(boxes) >= 4 or aire_max >= 5:
        return "Élevé"
    if len(boxes) >= 2 or aire_max >= 1:
        return "Moyen"
    return "Faible"


def _annoter(image, boxes, scores, noms):
    resultat = np.ascontiguousarray(image.copy())
    epaisseur = max(2, image.shape[1] // 400)
    for (x1, y1, x2, y2), score, nom in zip(boxes, scores, noms):
        couleur = COULEURS.get(nom, (255, 40, 40))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(resultat, p1, p2, couleur, epaisseur)
        etiquette = f"{nom} {score:.2f}"
        (texte_largeur, texte_hauteur), _ = cv2.getTextSize(
            etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        y0 = max(p1[1] - texte_hauteur - 6, 0)
        cv2.rectangle(
            resultat,
            (p1[0], y0),
            (p1[0] + texte_largeur + 6, y0 + texte_hauteur + 6),
            couleur,
            -1,
        )
        cv2.putText(
            resultat,
            etiquette,
            (p1[0] + 3, y0 + texte_hauteur + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return resultat


def predire(image, modele):
    """Renvoie l'image annotée et un résumé Markdown des obstacles détectés."""
    conf = float(modele.cfg["conf"])
    hauteur, largeur = image.shape[:2]
    with _VERROU:
        boxes, scores, noms = modele.detecter(image, conf)
    resultat = _annoter(image, boxes, scores, noms)
    nombre = len(scores)
    comptage = Counter(noms)
    lignes = "\n".join(f"- {classe} : {quantite}" for classe, quantite in comptage.most_common())
    if not lignes:
        lignes = "- aucun obstacle détecté"
    texte = (
        f"**{nombre} obstacle(s) détecté(s)** · niveau d'alerte : "
        f"**{niveau_alerte(boxes, largeur, hauteur)}**\n\n"
        f"{lignes}\n\n"
        f"Seuil de confiance : {conf:.2f}".replace(".", ",")
        + " (règle d'alerte indicative)."
    )
    return resultat, texte


_NIVEAUX = {
    "faible luminosité": (0.8, 0.6, 0.45, 0.3, 0.2),
    "brouillard": (0.15, 0.3, 0.45, 0.6, 0.75),
    "flou de mouvement": (5, 9, 15, 25, 41),
    "bruit gaussien": (5, 10, 20, 35, 50),
    "pluie simulée": (200, 500, 1000, 2000, 3500),
}


def degrader(image, perturbation, niveau):
    """Applique une perturbation synthétique issue de la partie robustesse du notebook."""
    if perturbation == "aucune":
        return image.copy()
    if perturbation not in _NIVEAUX:
        raise ValueError("Perturbation inconnue.")
    if not 1 <= niveau <= 5:
        raise ValueError("Le niveau de perturbation doit être compris entre 1 et 5.")

    valeur = _NIVEAUX[perturbation][niveau - 1]
    x = image.astype(np.float32)
    hauteur, largeur = x.shape[:2]
    rng = np.random.default_rng(0)
    if perturbation == "faible luminosité":
        x = 255.0 * (x / 255.0) ** 1.3 * valeur
    elif perturbation == "brouillard":
        alpha = np.clip(
            valeur * (1.2 - np.linspace(0, 1, hauteur))[:, None, None],
            0,
            0.95,
        )
        x = x * (1 - alpha) + 215.0 * alpha
    elif perturbation == "flou de mouvement":
        noyau = np.zeros((int(valeur), int(valeur)), np.float32)
        noyau[int(valeur) // 2, :] = 1.0 / valeur
        x = cv2.filter2D(x, -1, noyau)
    elif perturbation == "bruit gaussien":
        x = x + rng.normal(0, valeur, x.shape)
    else:
        pluie = np.zeros((hauteur, largeur), np.uint8)
        nombre_traits = max(1, int(valeur * (largeur * hauteur) / (1280 * 720)))
        for _ in range(nombre_traits):
            x0 = int(rng.integers(0, largeur))
            y0 = int(rng.integers(0, hauteur))
            longueur = int(rng.integers(12, 30))
            cv2.line(pluie, (x0, y0), (x0 - longueur // 4, y0 + longueur), 255, 1)
        pluie = cv2.GaussianBlur(pluie, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
        x = x * 0.85
        x = x * (1 - 0.6 * pluie) + 220.0 * 0.6 * pluie
        x = cv2.GaussianBlur(x, (3, 3), 0)
    return np.clip(x, 0, 255).astype(np.uint8)


def _couche_cam(net):
    couches = net.model
    source = couches[-1].f
    return couches[source[0] if isinstance(source, (list, tuple)) else source]


def _entree_cam(image, taille):
    hauteur, largeur = image.shape[:2]
    largeur_modele = int(taille)
    hauteur_modele = max(32, int(round(hauteur * largeur_modele / largeur / 32)) * 32)
    entree = cv2.resize(image, (largeur_modele, hauteur_modele)).astype(np.float32) / 255.0
    return entree


def _normaliser_cam(cam, taille):
    cam = cv2.resize(cam.astype(np.float32), taille, interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-9)


def _carte_eigen(modele, image):
    import torch

    net = modele.modele.model.to(modele.device).float().eval()
    entree = torch.from_numpy(_entree_cam(image, modele.cfg.get("imgsz", 640)))
    entree = entree.permute(2, 0, 1)[None].to(modele.device)
    activations = {}
    hook = _couche_cam(net).register_forward_hook(
        lambda *args: activations.__setitem__("a", args[-1].detach())
    )
    try:
        with torch.no_grad():
            net(entree)
    finally:
        hook.remove()

    activation = activations["a"][0].cpu().numpy()
    canaux, hauteur, largeur = activation.shape
    matrice = np.nan_to_num(activation.reshape(canaux, -1).T)
    matrice -= matrice.mean(0)
    _, _, vecteurs = np.linalg.svd(matrice, full_matrices=False)
    projection = matrice @ vecteurs[0]
    norme = np.linalg.norm(activation.reshape(canaux, -1), axis=0)
    if np.corrcoef(projection, norme)[0, 1] < 0:
        projection = -projection
    return _normaliser_cam(
        np.clip(projection.reshape(hauteur, largeur), 0, None),
        (image.shape[1], image.shape[0]),
    )


def _carte_grad(modele, image):
    import torch

    net = modele.modele.model.to(modele.device).float().eval()
    entree = torch.from_numpy(_entree_cam(image, modele.cfg.get("imgsz", 640)))
    entree = entree.permute(2, 0, 1)[None].to(modele.device).requires_grad_(True)
    activations = {}

    def capturer(*args):
        output = args[-1]
        output.retain_grad()
        activations["a"] = output

    hook = _couche_cam(net).register_forward_hook(capturer)
    try:
        with torch.enable_grad():
            sortie = net(entree)
            predictions = sortie[0] if isinstance(sortie, (list, tuple)) else sortie
            if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
                raise RuntimeError("La sortie du modèle YOLO ne permet pas de calculer Grad-CAM.")
            scores = predictions[0, 4:, :]
            if scores.numel() == 0:
                raise RuntimeError("Le modèle YOLO n'a fourni aucun score pour Grad-CAM.")
            net.zero_grad(set_to_none=True)
            scores.max().backward()
    finally:
        hook.remove()

    activation = activations["a"]
    gradient = activation.grad
    if gradient is None:
        raise RuntimeError("Le gradient n'a pas été calculé pour la couche Grad-CAM.")
    cam = torch.relu((gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(1))[0]
    return _normaliser_cam(
        cam.detach().cpu().numpy(),
        (image.shape[1], image.shape[0]),
    )


def _superposer_cam(image, chaleur):
    couleurs = cv2.applyColorMap((chaleur * 255).astype(np.uint8), cv2.COLORMAP_JET)
    couleurs = cv2.cvtColor(couleurs, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.5, couleurs, 0.5, 0)


def _resume_detection(boxes, noms, largeur, hauteur):
    comptage = Counter(noms)
    if comptage:
        details = ", ".join(
            f"{html.escape(nom)} : {quantite}" for nom, quantite in comptage.most_common()
        )
    else:
        details = "aucun obstacle détecté"
    return (
        f"{len(boxes)} obstacle(s), {details}, alerte "
        f"{niveau_alerte(boxes, largeur, hauteur)}"
    )


def analyser(modele, img=None, seuil=None, ident=None, perturbation="aucune",
             niveau="1", explication="aucune"):
    """Compare une image avant/après perturbation et renvoie les cartes CAM demandées."""
    del seuil, ident
    if img is None:
        raise ValueError("Aucune image reçue.")
    try:
        niveau = int(niveau)
    except (TypeError, ValueError) as erreur:
        raise ValueError("Le niveau de perturbation doit être un entier de 1 à 5.") from erreur
    if explication not in {"aucune", "eigencam", "gradcam", "les_deux"}:
        raise ValueError("Méthode d'explicabilité inconnue.")
    if explication != "aucune" and modele.type != "yolo":
        raise ValueError("EigenCAM et Grad-CAM sont disponibles uniquement pour le modèle YOLO.")
    if perturbation != "aucune" and perturbation not in _NIVEAUX:
        raise ValueError("Perturbation inconnue.")
    if perturbation != "aucune" and not 1 <= niveau <= 5:
        raise ValueError("Le niveau de perturbation doit être compris entre 1 et 5.")

    image_perturbee = degrader(img, perturbation, niveau)
    conf = float(modele.cfg["conf"])
    hauteur, largeur = img.shape[:2]
    debut = time.time()
    with _VERROU:
        boxes_orig, scores_orig, noms_orig = modele.detecter(img, conf)
        boxes, scores, noms = modele.detecter(image_perturbee, conf)
        images_cam = {}
        if explication in {"eigencam", "les_deux"}:
            images_cam["eigencam"] = en_base64(
                _superposer_cam(image_perturbee, _carte_eigen(modele, image_perturbee))
            )
        if explication in {"gradcam", "les_deux"}:
            images_cam["gradcam"] = en_base64(
                _superposer_cam(image_perturbee, _carte_grad(modele, image_perturbee))
            )

    image_annotee = _annoter(image_perturbee, boxes, scores, noms)
    image_reference = (
        en_base64(_annoter(img, boxes_orig, scores_orig, noms_orig))
        if perturbation != "aucune"
        else None
    )
    texte = (
        f"<p><strong>Image analysée :</strong> {html.escape(perturbation)}"
        + (f" · niveau {niveau}/5" if perturbation != "aucune" else "")
        + f"</p><p><strong>Image d'origine :</strong> "
        f"{_resume_detection(boxes_orig, noms_orig, largeur, hauteur)}."
        f"<br><strong>Image analysée :</strong> "
        f"{_resume_detection(boxes, noms, largeur, hauteur)}.</p>"
        f"<p>Seuil de confiance : {conf:.2f}.</p>"
    )
    if images_cam:
        texte += (
            "<p>Les cartes CAM sont des visualisations indicatives des activations et des gradients "
            "du détecteur, pas une preuve de causalité.</p>"
        )
    return {
        "image": en_base64(image_annotee),
        "image_reference": image_reference,
        "texte": texte,
        **images_cam,
        "ms": round((time.time() - debut) * 1000),
    }
