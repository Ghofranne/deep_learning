// Page standard (objectifs 2 à 6) : image -> predire() -> image résultat + texte.
(() => {
  const img = document.getElementById("img-resultat");
  const texte = document.getElementById("texte-resultat");
  const MESSAGE = `<p class="discret">Déposez une image ou choisissez un exemple, puis cliquez sur « Analyser ».</p>`;
  const perturbation = document.getElementById("perturbation");
  const niveau = document.getElementById("niveau-perturbation");
  const explication = document.getElementById("explication");
  const comparaison = document.getElementById("comparaison-robustesse");
  const figureReference = document.getElementById("figure-reference");
  const imageReference = document.getElementById("img-reference");
  const imagesExplicabilite = document.getElementById("images-explicabilite");
  const imagesCam = {
    eigencam: {
      image: document.getElementById("img-eigencam"),
      figure: document.getElementById("figure-eigencam"),
    },
    gradcam: {
      image: document.getElementById("img-gradcam"),
      figure: document.getElementById("figure-gradcam"),
    },
  };
  const legendeResultat = document.getElementById("legende-resultat");
  let enCours = false;
  let aRelancer = false;
  let demande = 0;

  function effacerExplications() {
    if (comparaison) comparaison.hidden = true;
    if (figureReference) figureReference.hidden = true;
    if (imageReference) afficherImage(imageReference, null);
    if (imagesExplicabilite) imagesExplicabilite.hidden = true;
    Object.values(imagesCam).forEach(({ image, figure }) => {
      if (image) afficherImage(image, null);
      if (figure) figure.hidden = true;
    });
  }

  async function analyser() {
    if (enCours) {
      aRelancer = true;
      return;
    }
    if (!Entree.source) return;
    enCours = true;
    const demandeCourante = ++demande;
    Entree.chargement(true);
    try {
      const champs = perturbation ? {
        perturbation: perturbation.value,
        niveau: niveau.value,
        explication: explication.value,
      } : {};
      const rep = await Entree.envoyer(champs);
      if (demandeCourante !== demande) return;
      if (!rep.ok) {
        afficherImage(img, null);
        effacerExplications();
        texte.innerHTML = `<div class="erreur-texte">${echapper(rep.json.erreur || "Erreur inconnue.")}</div>`;
      } else {
        afficherImage(img, rep.json.image);
        texte.innerHTML = rep.json.texte + `<p class="discret">Analyse en ${rep.json.ms} ms.</p>`;
        if (comparaison) {
          comparaison.hidden = false;
          afficherImage(imageReference, rep.json.image_reference);
          figureReference.hidden = !rep.json.image_reference;
          legendeResultat.textContent = perturbation.value === "aucune"
            ? "Détections sur l'image d'origine"
            : `Détections · ${perturbation.value}, niveau ${niveau.value}/5`;
          Object.entries(imagesCam).forEach(([methode, elements]) => {
            const image = rep.json[methode];
            afficherImage(elements.image, image);
            elements.figure.hidden = !image;
          });
          imagesExplicabilite.hidden = !rep.json.eigencam && !rep.json.gradcam;
        }
      }
    } catch (e) {
      if (demandeCourante === demande) {
        effacerExplications();
        console.error("Erreur pendant l'analyse.", e);
        texte.innerHTML = `<div class="erreur-texte">Erreur pendant l'analyse : ${echapper(e.message || e)}</div>`;
      }
    } finally {
      enCours = false;
      Entree.chargement(false);
      if (aRelancer) {
        aRelancer = false;
        analyser();
      }
    }
  }

  Entree.bouton.addEventListener("click", analyser);
  Entree.surChangement(analyser);
  Entree.surEffacement(() => {
    demande += 1;
    aRelancer = false;
    afficherImage(img, null);
    effacerExplications();
    texte.innerHTML = MESSAGE;
  });
  if (perturbation) {
    perturbation.addEventListener("change", analyser);
    explication.addEventListener("change", analyser);
    niveau.addEventListener("input", () => {
      document.getElementById("niveau-perturbation-val").textContent = `${niveau.value}/5`;
    });
    niveau.addEventListener("change", analyser);
  }
})();
