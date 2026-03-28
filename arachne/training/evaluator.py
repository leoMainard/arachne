"""Calcul des métriques et génération des visualisations d'évaluation."""
from __future__ import annotations

from pathlib import Path

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


def sauvegarder_matrice_confusion(
    y_reel: list[str],
    y_predit: list[str],
    classes: list[str],
    chemin_sortie: Path,
    titre: str = "Matrice de confusion",
) -> None:
    """Génère et sauvegarde la matrice de confusion en PNG.

    Produit deux sous-graphiques : valeurs brutes et valeurs normalisées.

    Args:
        y_reel: Labels réels.
        y_predit: Labels prédits.
        classes: Liste ordonnée des classes.
        chemin_sortie: Chemin du fichier PNG de sortie.
        titre: Titre du graphique.
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
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sauvegarder_graphique_metriques(
    metriques: dict,
    chemin_sortie: Path,
    titre: str = "Métriques par classe",
) -> None:
    """Génère et sauvegarde un graphique à barres des métriques par classe.

    Args:
        metriques: Dictionnaire retourné par calculer_metriques().
        chemin_sortie: Chemin du fichier PNG de sortie.
        titre: Titre du graphique.
    """
    par_classe = metriques.get("par_classe", {})
    if not par_classe:
        return

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
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)
