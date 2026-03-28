"""Chargement des données depuis PostgreSQL ou des fichiers locaux."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arachne.constants import COLONNES_REQUISES, SourceDonnees


def _verifier_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie que le DataFrame contient les colonnes obligatoires.

    Args:
        df: DataFrame à valider.

    Retours:
        Le DataFrame inchangé si valide.

    Raises:
        ValueError: Si des colonnes obligatoires sont manquantes.
    """
    manquantes = COLONNES_REQUISES - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes dans le jeu de données : {manquantes}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return df


def _parser_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """Désérialise la colonne table_data si elle contient des chaînes JSON.

    Args:
        df: DataFrame avec une colonne table_data.

    Retours:
        DataFrame avec table_data contenant des objets list[list[str]].
    """
    def parser(valeur) -> list[list]:
        if isinstance(valeur, str):
            return json.loads(valeur)
        return valeur

    df = df.copy()
    df["table_data"] = df["table_data"].apply(parser)
    return df


class ChargeurDonnees:
    """Charge les données labellisées depuis PostgreSQL ou un fichier local.

    Args:
        config: Dictionnaire de configuration (section ``data`` du YAML).

    Exemple:
        >>> chargeur = ChargeurDonnees(config)
        >>> df = chargeur.charger()
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def charger(self) -> pd.DataFrame:
        """Charge les données selon la source définie dans la configuration.

        Retours:
            DataFrame avec colonnes ``table_data`` (list[list[str]]) et ``label`` (str).

        Raises:
            ValueError: Si la source de données est inconnue.
        """
        source = self._config["data"]["source"]

        if source == SourceDonnees.POSTGRESQL:
            config_db = self._config["data"].get("postgresql", {})
            requete = self._config["data"].get("query", None)
            return self.depuis_postgresql(config_db, requete)
        elif source == SourceDonnees.LOCAL:
            chemin = self._config["data"]["local_path"]
            return self.depuis_local(chemin)
        else:
            raise ValueError(
                f"Source de données inconnue : '{source}'. "
                f"Valeurs acceptées : {[s.value for s in SourceDonnees]}"
            )

    @staticmethod
    def depuis_postgresql(config_db: dict, requete: str | None = None) -> pd.DataFrame:
        """Charge les données depuis une base PostgreSQL.

        Args:
            config_db: Paramètres de connexion (host, port, dbname, user, password).
            requete: Requête SQL personnalisée. Par défaut : ``SELECT id, table_data, label FROM tables``.

        Retours:
            DataFrame avec colonnes ``table_data`` et ``label``.

        Raises:
            ImportError: Si psycopg2 n'est pas installé.
        """
        # Import optionnel : psycopg2 n'est requis que pour la source PostgreSQL.
        # Mettre cet import au niveau module ferait crasher le chargement si
        # l'utilisateur n'a pas installé psycopg2-binary.
        try:
            import psycopg2
        except ImportError as erreur:
            raise ImportError(
                "psycopg2 est requis pour PostgreSQL. "
                "Installez-le avec : pip install psycopg2-binary"
            ) from erreur

        if requete is None:
            requete = "SELECT id, table_data, label FROM tables"

        connexion = psycopg2.connect(**config_db)
        try:
            df = pd.read_sql(requete, connexion)
        finally:
            connexion.close()

        df = _verifier_colonnes(df)
        df = _parser_table_data(df)
        return df

    @staticmethod
    def depuis_local(chemin: str | Path) -> pd.DataFrame:
        """Charge les données depuis un fichier local (parquet, CSV, JSON).

        Args:
            chemin: Chemin vers le fichier de données.

        Retours:
            DataFrame avec colonnes ``table_data`` et ``label``.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le format de fichier n'est pas supporté.
        """
        chemin = Path(chemin)
        if not chemin.exists():
            raise FileNotFoundError(f"Fichier de données introuvable : {chemin}")

        suffixe = chemin.suffix.lower()
        if suffixe == ".parquet":
            df = pd.read_parquet(chemin)
        elif suffixe == ".csv":
            df = pd.read_csv(chemin)
        elif suffixe == ".json":
            df = pd.read_json(chemin)
        elif suffixe == ".jsonl":
            df = pd.read_json(chemin, lines=True)
        else:
            raise ValueError(
                f"Format non supporté : {suffixe}. "
                "Formats acceptés : .parquet, .csv, .json, .jsonl"
            )

        df = _verifier_colonnes(df)
        df = _parser_table_data(df)
        return df

    @staticmethod
    def exporter_parquet(df: pd.DataFrame, chemin_sortie: str | Path) -> None:
        """Sauvegarde le DataFrame en parquet pour un usage local ultérieur.

        La colonne table_data est sérialisée en JSON pour la compatibilité parquet.

        Args:
            df: DataFrame à sauvegarder.
            chemin_sortie: Chemin du fichier parquet de destination.
        """
        chemin_sortie = Path(chemin_sortie)
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        df_sauvegarde = df.copy()
        df_sauvegarde["table_data"] = df_sauvegarde["table_data"].apply(json.dumps)
        df_sauvegarde.to_parquet(chemin_sortie, index=False)
