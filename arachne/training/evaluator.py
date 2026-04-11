"""Calcul des métriques et génération des visualisations d'évaluation."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour la génération de fichiers PNG sans affichage
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

if TYPE_CHECKING:
    from arachne.data.s3 import ConnecteurS3


def calculer_metriques(
    y_reel: list[str],
    y_predit: list[str],
    classes: list[str],
) -> dict:
    """Calcule l'ensemble des métriques de classification.

    Args:
        y_reel: Labels réels.
        y_predit: Labels prédits par le modèle.
        classes: Liste ordonnée des classes à évaluer.

    Retours:
        Dictionnaire avec les clés : accuracy, macro_f1, weighted_f1, par_classe.
        ``par_classe`` contient precision, recall, f1 et support pour chaque classe.
    """
    rapport = classification_report(
        y_reel, y_predit, labels=classes, output_dict=True, zero_division=0
    )

    par_classe: dict[str, dict] = {}
    for label in classes:
        if label in rapport:
            par_classe[label] = {
                "precision": round(rapport[label]["precision"], 4),
                "rappel": round(rapport[label]["recall"], 4),
                "f1": round(rapport[label]["f1-score"], 4),
                "support": int(rapport[label]["support"]),
            }

    return {
        "accuracy": round(accuracy_score(y_reel, y_predit), 4),
        "macro_f1": round(f1_score(y_reel, y_predit, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_reel, y_predit, average="weighted", zero_division=0), 4),
        "par_classe": par_classe,
    }


def _generer_matrice_confusion(
    y_reel: list[str],
    y_predit: list[str],
    classes: list[str],
    titre: str,
) -> bytes:
    """Génère la matrice de confusion et retourne les bytes PNG.

    Args:
        y_reel: Labels réels.
        y_predit: Labels prédits.
        classes: Liste ordonnée des classes.
        titre: Titre du graphique.

    Retours:
        Contenu PNG encodé en bytes.
    """
    mc = confusion_matrix(y_reel, y_predit, labels=classes)
    mc_normalisee = mc.astype(float) / mc.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        mc, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[0],
    )
    axes[0].set_title(f"{titre} — Effectifs")
    axes[0].set_ylabel("Label réel")
    axes[0].set_xlabel("Label prédit")

    sns.heatmap(
        mc_normalisee, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[1],
        vmin=0, vmax=1,
    )
    axes[1].set_title(f"{titre} — Normalisée")
    axes[1].set_ylabel("Label réel")
    axes[1].set_xlabel("Label prédit")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def sauvegarder_matrice_confusion(
    y_reel: list[str],
    y_predit: list[str],
    classes: list[str],
    chemin_sortie: Path | None,
    titre: str = "Matrice de confusion",
    connecteur_s3: "ConnecteurS3 | None" = None,
    cle_s3: str = "",
) -> None:
    """Génère et sauvegarde la matrice de confusion (local et/ou S3).

    Args:
        y_reel: Labels réels.
        y_predit: Labels prédits.
        classes: Liste ordonnée des classes.
        chemin_sortie: Chemin PNG local. Passer ``None`` pour ne pas écrire sur disque.
        titre: Titre du graphique.
        connecteur_s3: Si fourni, upload le PNG vers S3.
        cle_s3: Clé S3 de destination.
    """
    png = _generer_matrice_confusion(y_reel, y_predit, classes, titre)
    if chemin_sortie is not None:
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        chemin_sortie.write_bytes(png)
    if connecteur_s3 is not None and cle_s3:
        connecteur_s3.envoyer_objet(png, cle_s3)


def _generer_graphique_metriques(metriques: dict, titre: str) -> bytes | None:
    """Génère le graphique métriques par classe et retourne les bytes PNG.

    Args:
        metriques: Dictionnaire retourné par calculer_metriques().
        titre: Titre du graphique.

    Retours:
        Contenu PNG encodé en bytes, ou None si pas de données par classe.
    """
    par_classe = metriques.get("par_classe", {})
    if not par_classe:
        return None

    classes = list(par_classe.keys())
    precision = [par_classe[c]["precision"] for c in classes]
    rappel = [par_classe[c]["rappel"] for c in classes]
    f1 = [par_classe[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    largeur = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - largeur, precision, largeur, label="Précision", color="#4C72B0")
    ax.bar(x, rappel, largeur, label="Rappel", color="#DD8452")
    ax.bar(x + largeur, f1, largeur, label="F1", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(titre)
    ax.legend()
    ax.axhline(
        y=metriques.get("accuracy", 0),
        color="red", linestyle="--", alpha=0.5,
        label=f"Accuracy ({metriques['accuracy']:.3f})",
    )
    ax.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def sauvegarder_graphique_metriques(
    metriques: dict,
    chemin_sortie: Path | None,
    titre: str = "Métriques par classe",
    connecteur_s3: "ConnecteurS3 | None" = None,
    cle_s3: str = "",
) -> None:
    """Génère et sauvegarde un graphique à barres des métriques (local et/ou S3).

    Args:
        metriques: Dictionnaire retourné par calculer_metriques().
        chemin_sortie: Chemin PNG local. Passer ``None`` pour ne pas écrire sur disque.
        titre: Titre du graphique.
        connecteur_s3: Si fourni, upload le PNG vers S3.
        cle_s3: Clé S3 de destination.
    """
    png = _generer_graphique_metriques(metriques, titre)
    if png is None:
        return
    if chemin_sortie is not None:
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        chemin_sortie.write_bytes(png)
    if connecteur_s3 is not None and cle_s3:
        connecteur_s3.envoyer_objet(png, cle_s3)
