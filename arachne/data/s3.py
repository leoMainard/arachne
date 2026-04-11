"""Connecteur S3 pour la sauvegarde et le chargement des artefacts d'expérience."""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


class ConnecteurS3:
    """Gère les transferts vers/depuis un bucket S3 (ou compatible S3 : MinIO, OVH, etc.).

    Args:
        config: Section ``stockage.s3`` du fichier YAML.
                Champs requis : ``access_key``, ``secret_key``, ``host``, ``bucket_name``.
                Champ optionnel : ``region`` (défaut : ``eu-west-3``).

    Raises:
        ImportError: Si boto3 n'est pas installé.

    Exemple:
        >>> connecteur = ConnecteurS3.depuis_config(config["stockage"])
        >>> if connecteur:
        ...     connecteur.envoyer_objet(data, "experiences/exp_123/metrics.json")
    """

    def __init__(self, config: dict) -> None:
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError as erreur:
            raise ImportError(
                "boto3 est requis pour l'intégration S3. "
                "Installez avec : uv add boto3"
            ) from erreur

        config_botocore = BotocoreConfig(region_name=config.get("region", "eu-west-3"))
        self._client = boto3.client(
            "s3",
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            endpoint_url=config["host"],
            config=config_botocore,
        )
        self.bucket_name: str = config["bucket_name"]
        logger.debug(f"Connecté au endpoint S3 : {config['host']}")
        logger.info(f"Bucket S3 actif : {self.bucket_name}")

    @classmethod
    def depuis_config(cls, config_stockage: dict) -> "ConnecteurS3 | None":
        """Instancie le connecteur si S3 est activé dans la configuration.

        Args:
            config_stockage: Section ``stockage`` du YAML.

        Retours:
            Instance de ConnecteurS3, ou None si ``stockage.s3.actif`` est False.
        """
        config_s3 = config_stockage.get("s3", {})
        if not config_s3.get("actif", False):
            return None
        return cls(config_s3)

    def envoyer_fichier(self, chemin_local: Path, cle_s3: str) -> None:
        """Upload un fichier local vers S3.

        Args:
            chemin_local: Chemin du fichier source sur le disque.
            cle_s3: Clé de destination dans le bucket.
        """
        self._client.upload_file(str(chemin_local), self.bucket_name, cle_s3)
        logger.debug(f"Envoyé : {chemin_local} → s3://{self.bucket_name}/{cle_s3}")

    def envoyer_objet(self, data: bytes, cle_s3: str) -> None:
        """Upload des données en mémoire vers S3 (sans écriture disque).

        Args:
            data: Contenu brut à uploader.
            cle_s3: Clé de destination dans le bucket.
        """
        self._client.upload_fileobj(BytesIO(data), self.bucket_name, cle_s3)
        logger.debug(
            f"Objet envoyé → s3://{self.bucket_name}/{cle_s3} ({len(data)} octets)"
        )

    def telecharger_fichier(self, cle_s3: str, chemin_local: Path) -> None:
        """Télécharge un objet S3 vers le disque local.

        Args:
            cle_s3: Clé source dans le bucket.
            chemin_local: Chemin de destination local.
        """
        chemin_local.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket_name, cle_s3, str(chemin_local))
        logger.debug(
            f"Téléchargé : s3://{self.bucket_name}/{cle_s3} → {chemin_local}"
        )

    def telecharger_objet(self, cle_s3: str) -> bytes:
        """Télécharge un objet S3 en mémoire (sans écriture disque).

        Args:
            cle_s3: Clé source dans le bucket.

        Retours:
            Contenu brut de l'objet.
        """
        buf = BytesIO()
        self._client.download_fileobj(self.bucket_name, cle_s3, buf)
        buf.seek(0)
        return buf.read()

    def envoyer_repertoire(self, chemin_local: Path, prefixe_s3: str) -> None:
        """Upload récursivement tous les fichiers d'un répertoire local vers S3.

        Typiquement utilisé pour les modèles HuggingFace (répertoire ``hf_model/``).

        Args:
            chemin_local: Répertoire source à uploader.
            prefixe_s3: Préfixe (« dossier ») de destination dans le bucket.
                        Ex : ``"arachne/experiences/exp_123/model/hf_model"``
        """
        chemin_local = Path(chemin_local)
        prefixe = prefixe_s3.rstrip("/")
        for fichier in sorted(chemin_local.rglob("*")):
            if fichier.is_file():
                chemin_relatif = fichier.relative_to(chemin_local).as_posix()
                self.envoyer_fichier(fichier, f"{prefixe}/{chemin_relatif}")
        logger.info(
            f"Répertoire envoyé : {chemin_local} → s3://{self.bucket_name}/{prefixe}/"
        )

    def telecharger_repertoire(self, prefixe_s3: str, chemin_local: Path) -> None:
        """Télécharge tous les objets d'un préfixe S3 dans un répertoire local.

        Typiquement utilisé pour récupérer un modèle HuggingFace sauvegardé.

        Args:
            prefixe_s3: Préfixe source dans le bucket.
            chemin_local: Répertoire de destination local (créé si absent).
        """
        prefixe = prefixe_s3.rstrip("/") + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefixe):
            for obj in page.get("Contents", []):
                cle = obj["Key"]
                chemin_relatif = cle[len(prefixe):]
                if chemin_relatif:
                    self.telecharger_fichier(cle, chemin_local / chemin_relatif)
        logger.info(
            f"Répertoire téléchargé : s3://{self.bucket_name}/{prefixe} → {chemin_local}"
        )
