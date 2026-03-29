"""Classifieur pour l'inférence en production.

Ce module est autonome : il charge les modèles directement via joblib (modèles
classiques) ou HuggingFace (transformers), sans importer le package arachne.

Dépendances requises :
  - Modèles classiques (pipeline.joblib) : joblib, scikit-learn
  - Modèles transformer (hf_model/)      : torch, transformers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from production._utils import charger_config_experience


class ClassifieurProduction:
    """Classifieur pour l'inférence en production.

    Charge un modèle sauvegardé depuis un répertoire d'expérience Arachne.
    Le type de modèle est détecté automatiquement :

    - ``model/pipeline.joblib`` → pipeline sklearn, chargé via joblib
    - ``model/hf_model/``       → modèle HuggingFace, chargé via transformers

    Ce module est autonome : il n'importe pas le package arachne.

    Args:
        config: Dictionnaire de configuration de production.
                Doit contenir ``modele.repertoire_experience``.

    Exemple:
        >>> config = {"modele": {"repertoire_experience": "models/tfidf_logistic_..."}}
        >>> clf = ClassifieurProduction(config)
        >>> labels = clf.predire(["col1 | col2 | val1 | val2"])
        >>> probs  = clf.predire_probabilites(["col1 | col2"])
    """

    def __init__(self, config: dict) -> None:
        repertoire = Path(config["modele"]["repertoire_experience"])
        repertoire_modele = repertoire / "model"

        if not repertoire_modele.exists():
            raise FileNotFoundError(
                f"Dossier model/ introuvable dans {repertoire}. "
                "Vérifiez que save_model: true était activé lors de l'entraînement."
            )

        config_experience = charger_config_experience(repertoire)
        type_features = config_experience.get("features", {}).get("type", "tfidf")

        if type_features == "transformer_tokenizer":
            self._pipeline = self._charger_transformer(repertoire_modele)
            self._type = "transformer"
            # Labels lus depuis la config sauvegardée — aucune dépendance envers arachne
            self._classes_transformer: list[str] = config_experience.get("data", {}).get(
                "labels", ["batiment", "vehicule", "sinistre", "autre"]
            )
        else:
            self._pipeline = self._charger_classique(repertoire_modele)
            self._type = "classique"

    @staticmethod
    def _charger_classique(repertoire_modele: Path):
        """Charge un pipeline sklearn depuis pipeline.joblib.

        Args:
            repertoire_modele: Dossier contenant pipeline.joblib.

        Retours:
            Pipeline sklearn restauré.
        """
        import joblib
        chemin = repertoire_modele / "pipeline.joblib"
        if not chemin.exists():
            raise FileNotFoundError(f"pipeline.joblib introuvable dans {repertoire_modele}.")
        return joblib.load(chemin)

    @staticmethod
    def _charger_transformer(repertoire_modele: Path):
        """Charge un modèle HuggingFace depuis hf_model/.

        Args:
            repertoire_modele: Dossier contenant le sous-dossier hf_model/.

        Retours:
            Tuple (modele, tokeniseur, dispositif).
        """
        # Import optionnel : torch et transformers ne sont requis que pour ce type de modèle.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        chemin_hf = repertoire_modele / "hf_model"
        tokeniseur = AutoTokenizer.from_pretrained(str(chemin_hf))
        modele = AutoModelForSequenceClassification.from_pretrained(str(chemin_hf))
        dispositif = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modele.to(dispositif)
        return (modele, tokeniseur, dispositif)

    def predire(self, textes: list[str]) -> list[str]:
        """Prédit les classes pour une liste de textes prétraités.

        Args:
            textes: Textes issus de PreprocesseurProduction.transformer[_lot]().

        Retours:
            Liste des labels prédits (batiment, vehicule, sinistre, autre).
        """
        if self._type == "classique":
            return list(self._pipeline.predict(textes))
        return self._predire_transformer(textes)

    def predire_probabilites(self, textes: list[str]) -> np.ndarray:
        """Calcule les probabilités d'appartenance à chaque classe.

        Args:
            textes: Textes issus de PreprocesseurProduction.transformer[_lot]().

        Retours:
            Tableau numpy de forme (n_échantillons, n_classes).
        """
        if self._type == "classique":
            return self._pipeline.predict_proba(textes)
        return self._predire_proba_transformer(textes)

    def obtenir_classes(self) -> list[str]:
        """Retourne la liste ordonnée des classes du modèle.

        Retours:
            Liste de noms de classes.
        """
        if self._type == "classique":
            return list(self._pipeline.classes_)
        return self._classes_transformer

    def _predire_transformer(self, textes: list[str]) -> list[str]:
        """Inférence pour les modèles HuggingFace."""
        probs = self._predire_proba_transformer(textes)
        indices = np.argmax(probs, axis=1)
        classes = self.obtenir_classes()
        return [classes[i] for i in indices]

    def _predire_proba_transformer(self, textes: list[str]) -> np.ndarray:
        """Probabilités pour les modèles HuggingFace."""
        import torch

        modele, tokeniseur, dispositif = self._pipeline
        modele.eval()
        toutes_probs: list[np.ndarray] = []

        for i in range(0, len(textes), 32):
            batch = textes[i:i + 32]
            encodages = tokeniseur(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
            encodages = {k: v.to(dispositif) for k, v in encodages.items()}
            with torch.no_grad():
                sorties = modele(**encodages)
            probs = torch.softmax(sorties.logits, dim=-1).cpu().numpy()
            toutes_probs.append(probs)

        return np.vstack(toutes_probs)
