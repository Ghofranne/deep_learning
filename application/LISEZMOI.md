# Application — Détection d'anomalies routières (version Flask)

Une application web en **Flask** avec une barre de 6 boutons en haut de la page. Chaque bouton ouvre
l'interface d'un objectif, dans la même fenêtre : segmentation sémantique, robustesse météo,
détection d'obstacles, segmentation d'instance, classification et segmentation d'instance.

## Installation (une seule fois)

1. Copiez **`model.onnx`** et **`config.json`** (produits par le notebook 05, les mêmes que pour
   l'application Gradio) dans le dossier **`modeles/objectif_1/`**.
2. Pour l'objectif 2, placez les poids **`model.pt`** dans **`modeles/objectif_2/`**. Le fichier
   `config.json` est déjà fourni.
3. Le modèle YOLO de l'objectif 3 est attendu dans **`modeles/objectif_3/model.pt`** ; son
   `config.json` est fourni.
4. Le notebook `../notebooks/04_segmentation_instance.ipynb` exporte le modèle de l'objectif 4 dans
   **`modeles/objectif_4/`**. Il peut aussi réutiliser le checkpoint déjà placé dans
   `../modeles/objectif_4/`.
5. Les modèles des objectifs 5 et 6 sont fournis dans leurs dossiers respectifs.
6. Ouvrez un terminal dans ce dossier et tapez :

   ```
   pip install -r requirements.txt
   ```

## Lancement

```
python app.py
```

puis ouvrez **http://127.0.0.1:5000** dans le navigateur. Pour arrêter : **Ctrl + C** dans le terminal.

## Utilisation (objectif 1)

- Déposez une photo (clic, glisser-déposer ou Ctrl + V) ou cliquez sur un exemple : l'analyse se lance.
- Le curseur **Seuil** change la sensibilité ; l'image n'est pas réanalysée, seul le seuil est réappliqué.
- Cliquez sur une **ligne du tableau** ou directement sur une **zone de l'image** : elle est mise en
  surbrillance et agrandie à droite.

## Utilisation (objectif 2)

- Déposez une image de route ou choisissez un exemple, puis sélectionnez une météo simulée et son
  intensité.
- L'application compare les probabilités et la manœuvre prédite avec et sans météo simulée, puis
  affiche une carte Grad-CAM.
- Ce modèle expérimental n'est pas un système de conduite autonome ni un dispositif de sécurité.

## Utilisation (objectif 3)

- Déposez une photo de route ou sélectionnez un exemple, puis cliquez sur « Analyser ».
- Les boîtes annotées, le nombre d'obstacles par classe et un niveau d'alerte indicatif s'affichent.
- Choisissez une dégradation synthétique (faible luminosité, brouillard, flou de mouvement, bruit
  gaussien ou pluie simulée) et un niveau de 1 à 5 pour comparer les détections avant/après.
- Sélectionnez EigenCAM, Grad-CAM ou les deux pour afficher les cartes d'activation du modèle.
- Le seuil de confiance est configuré dans `modeles/objectif_3/config.json`.
- Ces dégradations interactives illustrent la robustesse sur une image ; elles ne calculent pas les
  métriques mAP/recall du notebook, qui exigent les images annotées de validation/test. Les cartes
  CAM sont indicatives et ne prouvent pas une causalité.

## Utilisation (objectif 4)

- Déposez une photo de route ou choisissez un exemple, puis cliquez sur « Analyser ».
- Les masques de chaque anomalie sont superposés à l'image, avec une boîte et un score de confiance.
- Le seuil par défaut est défini dans `modeles/objectif_4/config.json`.

## Utilisation (objectifs 5 et 6)

- L'objectif 5 prédit le type d'anomalie avec ResNet18.
- L'objectif 6 détecte les anomalies et superpose leurs masques avec Mask R-CNN.

## Organisation des fichiers

```
app.py                  serveur Flask : pages, navigation, API d'analyse
objectifs/              un fichier Python par objectif (+ commun.py, outils partagés)
templates/              pages HTML (Jinja) : base, objectif 1, page standard, attente, erreur
static/                 style.css (thème) et scripts JavaScript
modeles/objectif_N/     modèle de chaque objectif
exemples/objectif_N/    images d'exemple de chaque objectif
```

## Comment ça marche

1. Au démarrage, `app.py` importe chaque fichier `objectifs/objectif_N.py` et appelle `charger()`
   une seule fois (le modèle reste en mémoire).
2. `/objectif/N` renvoie la page HTML de l'objectif N.
3. Quand on clique sur « Analyser », le navigateur envoie l'image à `/api/objectif/N/analyser`
   (requête POST). Le serveur lance le modèle et répond en JSON (images encodées en base64, chiffres,
   liste des zones). Le JavaScript met la page à jour sans la recharger.
