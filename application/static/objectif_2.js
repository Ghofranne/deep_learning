// Objectif 2 : compare les décisions du modèle avant et après simulation météo.
(() => {
  const meteo = document.getElementById("meteo");
  const intensite = document.getElementById("intensite");
  const intensiteVal = document.getElementById("intensite-val");
  const resume = document.getElementById("resume");
  const resultat = document.getElementById("resultat");
  const images = {
    original: document.querySelector("#img-original"),
    meteo: document.querySelector("#img-meteo"),
    gradcam: document.querySelector("#img-gradcam"),
  };
  let enCours = false;
  let aRelancer = false;
  let demande = 0;

  function afficherProbabilites(element, probabilites) {
    element.innerHTML = probabilites.map(p => {
      const valeur = Math.max(0, Math.min(100, Number(p.pourcentage) || 0));
      return `<div class="bar-proba">
        <div class="bar-proba-haut"><span>${echapper(p.nom)}</span><strong>${valeur.toFixed(1)} %</strong></div>
        <div class="bar-proba-fond"><div class="bar-proba-remplissage" style="width:${valeur}%"></div></div>
      </div>`;
    }).join("");
  }

  function afficher(r) {
    resume.hidden = true;
    resultat.hidden = false;
    document.getElementById("decision-meteo").textContent = r.decision;
    document.getElementById("decision-beau").textContent = r.decision_beau;

    const comparaison = document.getElementById("comparaison-decision");
    if (r.meteo === "aucune") {
      comparaison.textContent = "Aucune météo simulée.";
      comparaison.className = "comparaison-decision";
    } else {
      comparaison.textContent = r.meme_decision
        ? "La météo simulée ne change pas la décision."
        : "La météo simulée change la décision.";
      comparaison.className = `comparaison-decision ${r.meme_decision ? "decision-stable" : "decision-changee"}`;
    }

    for (const [image, url] of [
      [images.original, r.original],
      [images.meteo, r.image_meteo],
      [images.gradcam, r.gradcam],
    ]) {
      if (!image) throw new Error("Un emplacement d’image du résultat est absent de la page.");
      afficherImage(image, url);
    }
    document.getElementById("legende-meteo").textContent = r.meteo === "aucune"
      ? "Image utilisée pour la prédiction"
      : `Avec ${r.meteo} · intensité ${r.intensite.toFixed(1)}`;
    afficherProbabilites(document.getElementById("probabilites-beau"), r.probabilites_beau);
    afficherProbabilites(document.getElementById("probabilites-meteo"), r.probabilites);
    document.getElementById("temps-analyse").textContent = `Analyse en ${r.ms} ms.`;
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
      const rep = await Entree.envoyer({ meteo: meteo.value, intensite: intensite.value });
      if (demandeCourante !== demande) return;
      if (!rep.ok) {
        resultat.hidden = true;
        resume.hidden = false;
        resume.className = "resume vide alerte";
        resume.textContent = rep.json.erreur || "Erreur inconnue.";
      } else {
        afficher(rep.json);
      }
    } catch (e) {
      if (demandeCourante === demande) {
        console.error("Erreur pendant l'analyse de l'objectif 2.", e);
        resultat.hidden = true;
        resume.hidden = false;
        resume.className = "resume vide alerte";
        resume.textContent = `Erreur pendant l'analyse : ${e.message || e}`;
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
    resultat.hidden = true;
    resume.hidden = false;
    resume.className = "resume vide";
    resume.textContent = "Déposez une photo ou choisissez un exemple, puis cliquez sur « Analyser ».";
    Object.values(images).forEach(img => afficherImage(img, null));
  });
  meteo.addEventListener("change", analyser);
  intensite.addEventListener("input", () => {
    intensiteVal.textContent = Number(intensite.value).toFixed(1).replace(".", ",");
  });
  intensite.addEventListener("change", analyser);
})();
