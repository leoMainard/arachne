"""Suivi des expériences : sauvegarde de la config, des métriques et des artefacts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from arachne.data.s3 import ConnecteurS3


class SuiveurExperience:
    """Gère la sauvegarde des artefacts et des métriques d'une expérience.

    Chaque expérience reçoit un identifiant unique (nom + horodatage) et
    est sauvegardée dans son propre sous-répertoire.

    Args:
        nom_experience: Nom de l'expérience (issu du YAML).
        repertoire_sortie: Répertoire racine de sauvegarde (par défaut : ``models/``).

    Exemple:
        >>> suiveur = SuiveurExperience("tfidf_logistic")
        >>> suiveur.enregistrer_config(config)
        >>> suiveur.enregistrer_metriques_test(metriques)
        >>> suiveur.finaliser()
    """

    def __init__(
        self,
        nom_experience: str,
        repertoire_sortie: Path = Path("models"),
        local: bool = True,
        connecteur_s3: "ConnecteurS3 | None" = None,
        prefixe_s3: str = "",
    ) -> None:
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.id_experience = f"{nom_experience}_{horodatage}"
        self._local = local
        self._connecteur_s3 = connecteur_s3
        self._prefixe_s3 = f"{prefixe_s3.rstrip('/')}/{self.id_experience}"

        if self._local:
            self.repertoire_experience = repertoire_sortie / self.id_experience
            self.repertoire_experience.mkdir(parents=True, exist_ok=True)
        else:
            # Répertoire fictif pour garder l'attribut accessible (plots, modèle)
            # sans créer de dossier sur le disque.
            self.repertoire_experience = repertoire_sortie / self.id_experience

        self._metriques: dict = {
            "experiment_id": self.id_experience,
            "experiment_name": nom_experience,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": None,
            "data": {},
            "cv_results": {},
            "test_metrics": {},
            "status": "en_cours",
        }

    def enregistrer_config(self, config: dict) -> None:
        """Sauvegarde la configuration YAML de l'expérience (local et/ou S3).

        Args:
            config: Dictionnaire de configuration fusionné.
        """
        contenu = yaml.dump(config, allow_unicode=True, default_flow_style=False).encode("utf-8")
        if self._local:
            self.repertoire_experience.mkdir(parents=True, exist_ok=True)
            (self.repertoire_experience / "config.yaml").write_bytes(contenu)
        if self._connecteur_s3 is not None:
            self._connecteur_s3.envoyer_objet(contenu, f"{self._prefixe_s3}/config.yaml")

    def enregistrer_info_donnees(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        distribution_classes: dict,
    ) -> None:
        """Enregistre les informations sur le jeu de données utilisé.

        Args:
            n_train: Nombre d'échantillons d'entraînement.
            n_val: Nombre d'échantillons de validation.
            n_test: Nombre d'échantillons de test.
            distribution_classes: Dictionnaire {classe: nombre d'occurrences}.
        """
        self._metriques["data"] = {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_total": n_train + n_val + n_test,
            "distribution_classes": {k: int(v) for k, v in distribution_classes.items()},
        }

    def enregistrer_resultats_cv(self, resultats_cv: dict) -> None:
        """Enregistre les résultats de la validation croisée.

        Args:
            resultats_cv: Dictionnaire avec mean_accuracy, std_accuracy, fold_scores.
        """
        self._metriques["cv_results"] = resultats_cv

    def enregistrer_metriques_test(self, metriques: dict) -> None:
        """Enregistre les métriques calculées sur le jeu de test.

        Args:
            metriques: Dictionnaire retourné par calculer_metriques().
        """
        self._metriques["test_metrics"] = metriques

    def enregistrer_duree(self, secondes: float) -> None:
        """Enregistre la durée totale d'entraînement.

        Args:
            secondes: Durée en secondes.
        """
        self._metriques["duration_seconds"] = round(secondes, 2)

    def finaliser(self) -> None:
        """Marque l'expérience comme terminée et écrit metrics.json sur le disque."""
        self._metriques["status"] = "terminee"
        self._sauvegarder_metriques()

    def obtenir_resume(self) -> dict:
        """Retourne une copie du dictionnaire de métriques de l'expérience.

        Retours:
            Dictionnaire complet des métriques et métadonnées.
        """
        return self._metriques.copy()

    def _sauvegarder_metriques(self) -> None:
        """Écrit metrics.json sur le disque local et/ou l'envoie vers S3."""
        contenu = json.dumps(self._metriques, indent=2, ensure_ascii=False).encode("utf-8")
        if self._local:
            self.repertoire_experience.mkdir(parents=True, exist_ok=True)
            (self.repertoire_experience / "metrics.json").write_bytes(contenu)
        if self._connecteur_s3 is not None:
            self._connecteur_s3.envoyer_objet(contenu, f"{self._prefixe_s3}/metrics.json")


def charger_toutes_experiences(
    repertoire_modeles: Path = Path("models"),
) -> list[dict]:
    """Charge les métriques de toutes les expériences depuis le répertoire de modèles.

    Args:
        repertoire_modeles: Répertoire racine contenant les sous-dossiers d'expériences.

    Retours:
        Liste de dictionnaires de métriques, triée par ordre chronologique inverse.
    """
    experiences: list[dict] = []
    for fichier_metriques in sorted(
        repertoire_modeles.glob("*/metrics.json"), reverse=True
    ):
        try:
            with open(fichier_metriques, encoding="utf-8") as f:
                donnees = json.load(f)
                donnees["_path"] = str(fichier_metriques.parent)
                experiences.append(donnees)
        except (json.JSONDecodeError, KeyError):
            continue
    return experiences
