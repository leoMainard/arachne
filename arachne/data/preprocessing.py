"""Prétraitement des tableaux : conversion matrice → représentation textuelle."""
from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from arachne.constants import LABELS


class Preprocesseur:
    """Convertit des matrices de tableaux en représentations textuelles.

    Les en-têtes sont répétés ``poids_entetes`` fois pour amplifier leur
    influence lors de la vectorisation TF-IDF.

    Args:
        lignes_entetes: Nombre de lignes traitées comme en-têtes.
        poids_entetes: Nombre de répétitions des en-têtes dans le texte final.
        max_cellules_contenu: Nombre maximum de cellules de contenu à inclure.
        longueur_max: Limite optionnelle en nombre de caractères du texte final.

    Exemple:
        >>> prep = Preprocesseur(lignes_entetes=1, poids_entetes=3)
        >>> texte = prep.transformer([["Col1", "Col2"], ["Val1", "Val2"]])
        >>> textes = prep.transformer_lot(liste_de_tableaux)
    """

    def __init__(
        self,
        lignes_entetes: int = 1,
        poids_entetes: int = 3,
        max_cellules_contenu: int = 200,
        longueur_max: int | None = None,
    ) -> None:
        self.lignes_entetes = lignes_entetes
        self.poids_entetes = poids_entetes
        self.max_cellules_contenu = max_cellules_contenu
        self.longueur_max = longueur_max

    @classmethod
    def depuis_config(cls, config_preprocessing: dict) -> "Preprocesseur":
        """Crée un Preprocesseur à partir d'un dictionnaire de configuration YAML.

        Args:
            config_preprocessing: Section ``preprocessing`` du fichier YAML.

        Retours:
            Instance configurée de Preprocesseur.
        """
        return cls(
            lignes_entetes=config_preprocessing.get("header_rows", 1),
            poids_entetes=config_preprocessing.get("header_weight", 3),
            max_cellules_contenu=config_preprocessing.get("max_content_cells", 200),
            longueur_max=config_preprocessing.get("max_length", None),
        )

    def transformer(self, tableau: list[list]) -> str:
        """Convertit un tableau (matrice 2D) en représentation textuelle pondérée.

        Args:
            tableau: Matrice 2D de valeurs (list[list[str]]).

        Retours:
            Chaîne de texte représentant le tableau, avec les en-têtes pondérés.
        """
        if not tableau:
            return ""

        nb_lignes_entetes = min(self.lignes_entetes, len(tableau))
        entetes = tableau[:nb_lignes_entetes]
        contenu = tableau[nb_lignes_entetes:]

        cellules_entetes = [
            self._nettoyer_cellule(cellule)
            for ligne in entetes
            for cellule in ligne
            if self._nettoyer_cellule(cellule)
        ]
        texte_entetes = " | ".join(cellules_entetes)

        cellules_contenu = [
            self._nettoyer_cellule(cellule)
            for ligne in contenu
            for cellule in ligne
            if self._nettoyer_cellule(cellule)
        ][:self.max_cellules_contenu]
        texte_contenu = " | ".join(cellules_contenu)

        parties = [texte_entetes] * self.poids_entetes
        if texte_contenu:
            parties.append(texte_contenu)

        texte = " ".join(p for p in parties if p)

        if self.longueur_max:
            texte = texte[:self.longueur_max]

        return texte

    def transformer_lot(self, tableaux: list[list[list]]) -> list[str]:
        """Convertit une liste de tableaux en représentations textuelles.

        Args:
            tableaux: Liste de matrices 2D.

        Retours:
            Liste de chaînes de texte, une par tableau.
        """
        return [self.transformer(tableau) for tableau in tableaux]

    @staticmethod
    def _nettoyer_cellule(valeur) -> str:
        """Normalise une cellule en chaîne de caractères propre.

        Args:
            valeur: Valeur brute de la cellule (any).

        Retours:
            Chaîne nettoyée (espaces multiples supprimés, strip).
        """
        if valeur is None:
            return ""
        texte = str(valeur).strip()
        return re.sub(r"\s+", " ", texte)


def decouper_dataset(
    df: pd.DataFrame,
    taille_test: float = 0.2,
    taille_val: float = 0.1,
    stratifier: bool = True,
    graine: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe un DataFrame en ensembles d'entraînement, validation et test.

    Args:
        df: DataFrame source avec une colonne ``label``.
        taille_test: Proportion des données pour le test (ex: 0.2 = 20%).
        taille_val: Proportion des données pour la validation (ex: 0.1 = 10%).
        stratifier: Si True, maintient la distribution des classes dans chaque split.
        graine: Graine aléatoire pour la reproductibilité.

    Retours:
        Tuple (df_train, df_val, df_test).
    """
    col_stratification = df["label"] if stratifier else None

    df_train_val, df_test = train_test_split(
        df,
        test_size=taille_test,
        stratify=col_stratification,
        random_state=graine,
    )

    taille_val_relative = taille_val / (1.0 - taille_test)
    col_strat_tv = df_train_val["label"] if stratifier else None

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=taille_val_relative,
        stratify=col_strat_tv,
        random_state=graine,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )
