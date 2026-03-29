"""Pipeline d'entraînement avec validation croisée optionnelle."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold, cross_val_score

from arachne.constants import LABELS
from arachne.data.loader import ChargeurDonnees
from arachne.data.preprocessing import Preprocesseur, decouper_dataset
from arachne.models import obtenir_modele
from arachne.models.classical import ClassifieurClassique
from arachne.tracking.tracker import SuiveurExperience
from arachne.training.evaluator import (
    calculer_metriques,
    sauvegarder_graphique_metriques,
    sauvegarder_matrice_confusion,
)

console = Console()


def executer_experience(config: dict) -> dict:
    """Exécute une expérience d'entraînement complète depuis une configuration YAML.

    Étapes : chargement → découpage → prétraitement → modèle →
    validation croisée → entraînement final → évaluation → sauvegarde.

    Args:
        config: Configuration fusionnée (base.yaml + experiment.yaml).

    Retours:
        Dictionnaire récapitulatif de l'expérience (métriques, chemins, etc.).
    """
    nom_exp = config.get("experiment", {}).get("name", "experience")
    description = config.get("experiment", {}).get("description", "")
    config_donnees = config["data"]
    config_preprocessing = config.get("preprocessing", {})
    config_entrainement = config.get("training", {})

    suiveur = SuiveurExperience(
        nom_experience=nom_exp,
        repertoire_sortie=Path(config.get("tracking", {}).get("output_dir", "models")),
    )
    suiveur.enregistrer_config(config)

    console.rule(f"[bold blue]{nom_exp}")
    if description:
        console.print(f"[dim]{description}[/dim]")

    # 1. Chargement des données
    console.print("\n[bold]1. Chargement des données...[/bold]")
    chargeur = ChargeurDonnees(config)
    df = chargeur.charger()
    console.print(f"   {len(df)} échantillons chargés.")
    console.print(f"   Distribution des classes :\n{df['label'].value_counts().to_string()}")

    # 2. Découpage train / val / test
    df_train, df_val, df_test = decouper_dataset(
        df,
        taille_test=config_donnees.get("test_size", 0.2),
        taille_val=config_donnees.get("val_size", 0.1),
        stratifier=config_donnees.get("stratify", True),
        graine=config_donnees.get("random_seed", 42),
    )
    console.print(
        f"\n   Découpage : train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )
    suiveur.enregistrer_info_donnees(
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
        distribution_classes=df["label"].value_counts().to_dict(),
    )

    # 3. Prétraitement des tableaux en texte
    console.print("\n[bold]2. Prétraitement des tableaux...[/bold]")
    preprocesseur = Preprocesseur.depuis_config(config_preprocessing)
    textes_train = preprocesseur.transformer_lot(df_train["table_data"].tolist())
    textes_val = preprocesseur.transformer_lot(df_val["table_data"].tolist())
    textes_test = preprocesseur.transformer_lot(df_test["table_data"].tolist())

    labels_train = df_train["label"].tolist()
    labels_val = df_val["label"].tolist()
    labels_test = df_test["label"].tolist()

    classes = sorted(config_donnees.get("labels", LABELS))

    # 4. Construction du modèle
    console.print("\n[bold]3. Construction du modèle...[/bold]")
    modele = obtenir_modele(config.get("model", {}), config.get("features", {}))

    # 5. Validation croisée (modèles classiques uniquement)
    nb_folds = config_entrainement.get("cv_folds", 5)
    if nb_folds and nb_folds > 1 and isinstance(modele, ClassifieurClassique):
        console.print(f"\n[bold]4. Validation croisée ({nb_folds} folds)...[/bold]")

        textes_tv = textes_train + textes_val
        labels_tv = labels_train + labels_val

        pipeline_cv = ClassifieurClassique.construire_pipeline(
            config.get("model", {}), config.get("features", {})
        )
        skf = StratifiedKFold(
            n_splits=nb_folds,
            shuffle=True,
            random_state=config_donnees.get("random_seed", 42),
        )
        scores_folds = cross_val_score(
            pipeline_cv, textes_tv, labels_tv,
            cv=skf,
            scoring=config_entrainement.get("scoring", "accuracy"),
            n_jobs=-1,
        )
        resultats_cv = {
            "mean_accuracy": round(float(np.mean(scores_folds)), 4),
            "std_accuracy": round(float(np.std(scores_folds)), 4),
            "fold_scores": [round(float(s), 4) for s in scores_folds],
        }
        console.print(
            f"   Accuracy CV : {resultats_cv['mean_accuracy']:.4f} "
            f"± {resultats_cv['std_accuracy']:.4f}"
        )
        suiveur.enregistrer_resultats_cv(resultats_cv)
    else:
        if nb_folds and nb_folds > 1:
            console.print(
                "\n[yellow]   Validation croisée ignorée pour les modèles transformer.[/yellow]"
            )

    # 6. Entraînement final
    console.print("\n[bold]5. Entraînement du modèle final...[/bold]")
    t0 = time.time()
    modele.entrainer(textes_train, labels_train, textes_val, labels_val)
    duree = time.time() - t0
    console.print(f"   Entraînement terminé en {duree:.1f}s")

    # 7. Évaluation sur le jeu de test
    console.print("\n[bold]6. Évaluation sur le jeu de test...[/bold]")
    y_predit = modele.predire(textes_test)
    metriques_test = calculer_metriques(labels_test, y_predit, classes)

    console.print(f"   Accuracy    : {metriques_test['accuracy']:.4f}")
    console.print(f"   Macro F1    : {metriques_test['macro_f1']:.4f}")
    console.print(f"   Weighted F1 : {metriques_test['weighted_f1']:.4f}")
    _afficher_tableau_par_classe(metriques_test["par_classe"])

    suiveur.enregistrer_metriques_test(metriques_test)
    suiveur.enregistrer_duree(duree)

    # 8. Génération des graphiques
    repertoire_plots = suiveur.repertoire_experience / "plots"
    sauvegarder_matrice_confusion(
        labels_test, y_predit, classes,
        chemin_sortie=repertoire_plots / "matrice_confusion.png",
        titre=nom_exp,
    )
    sauvegarder_graphique_metriques(
        metriques_test,
        chemin_sortie=repertoire_plots / "metriques_par_classe.png",
        titre=f"{nom_exp} — Métriques par classe",
    )

    # 9. Sauvegarde du modèle
    if config.get("tracking", {}).get("save_model", True):
        console.print("\n[bold]7. Sauvegarde du modèle...[/bold]")
        modele.sauvegarder(suiveur.repertoire_experience / "model")
        console.print(f"   Sauvegardé dans : {suiveur.repertoire_experience}")

    suiveur.finaliser()
    console.print(
        f"\n[bold green]Terminé ! Résultats sauvegardés dans : {suiveur.repertoire_experience}[/bold green]"
    )

    return suiveur.obtenir_resume()


def _afficher_tableau_par_classe(par_classe: dict) -> None:
    """Affiche un tableau Rich des métriques par classe dans le terminal.

    Args:
        par_classe: Dictionnaire {classe: {precision, rappel, f1, support}}.
    """
    tableau = Table(title="Métriques par classe", show_header=True)
    tableau.add_column("Classe", style="cyan")
    tableau.add_column("Précision", justify="right")
    tableau.add_column("Rappel", justify="right")
    tableau.add_column("F1", justify="right")
    tableau.add_column("Support", justify="right")

    for label, metriques in par_classe.items():
        tableau.add_row(
            label,
            f"{metriques['precision']:.4f}",
            f"{metriques['rappel']:.4f}",
            f"{metriques['f1']:.4f}",
            str(metriques["support"]),
        )
    console.print(tableau)
