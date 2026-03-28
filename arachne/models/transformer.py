"""Classifieur CamemBERT par fine-tuning (nécessite torch + transformers)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arachne.constants import ID_VERS_LABEL, LABEL_VERS_ID, LABELS
from arachne.models.base import ClassifieurBase

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Jeu de données PyTorch — défini au niveau du module (pas dans une méthode)
# ---------------------------------------------------------------------------

def _creer_dataset(textes: list[str], labels: list[str], tokeniseur, longueur_max: int):
    """Crée un Dataset PyTorch à partir de textes et de labels.

    Args:
        textes: Textes des tableaux.
        labels: Labels correspondants.
        tokeniseur: Tokeniseur HuggingFace.
        longueur_max: Longueur maximale de tokenisation.

    Retours:
        Instance de _DatasetTableaux.
    """
    return _DatasetTableaux(textes, labels, tokeniseur, longueur_max)


class _DatasetTableaux:
    """Dataset PyTorch pour les tableaux d'assurance.

    Args:
        textes: Textes des tableaux prétraités.
        labels: Labels de classification.
        tokeniseur: Tokeniseur HuggingFace (ex: CamemBERT).
        longueur_max: Nombre maximal de tokens par séquence.
    """

    def __init__(
        self,
        textes: list[str],
        labels: list[str],
        tokeniseur,
        longueur_max: int,
    ) -> None:
        # Import optionnel : torch n'est requis que pour les modèles transformer.
        # L'import au niveau module ferait crasher tout le package pour les
        # utilisateurs sans GPU/torch.
        import torch

        self._encodages = tokeniseur(
            textes,
            truncation=True,
            padding=True,
            max_length=longueur_max,
            return_tensors="pt",
        )
        self._labels = torch.tensor(
            [LABEL_VERS_ID.get(label, 3) for label in labels],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            **{cle: valeur[idx] for cle, valeur in self._encodages.items()},
            "labels": self._labels[idx],
        }


class ClassifieurTransformer(ClassifieurBase):
    """Classifieur CamemBERT (ou tout modèle HuggingFace de classification).

    Requiert l'installation de l'extra [transformers] :
    ``pip install arachne[transformers]``

    Args:
        config_modele: Section ``model`` du fichier YAML.
        config_features: Section ``features`` du fichier YAML.

    Exemple:
        >>> clf = ClassifieurTransformer(config_modele, config_features)
        >>> clf.entrainer(textes_train, labels_train, textes_val, labels_val)
        >>> predictions = clf.predire(textes_test)
    """

    def __init__(self, config_modele: dict, config_features: dict) -> None:
        self._config_modele = config_modele
        self._config_features = config_features
        self._modele = None
        self._tokeniseur = None
        self._dispositif = None
        self._classes: list[str] = LABELS.copy()

    def _obtenir_dispositif(self) -> "torch.device":
        """Détermine le dispositif de calcul (CPU ou GPU).

        Retours:
            torch.device approprié selon la configuration et la disponibilité.
        """
        import torch

        dispositif_cfg = self._config_modele.get("params", {}).get("device", "auto")
        if dispositif_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dispositif_cfg)

    def entrainer(
        self,
        textes_train: list[str],
        labels_train: list[str],
        textes_val: list[str] | None = None,
        labels_val: list[str] | None = None,
    ) -> None:
        """Fine-tune le modèle CamemBERT sur les données d'entraînement.

        Args:
            textes_train: Textes d'entraînement.
            labels_train: Labels d'entraînement.
            textes_val: Textes de validation (non utilisés dans cette implémentation).
            labels_val: Labels de validation (non utilisés dans cette implémentation).

        Raises:
            ImportError: Si torch ou transformers ne sont pas installés.
        """
        # Import optionnel : torch et transformers ne sont requis que pour ce modèle.
        try:
            import torch
            from torch.optim import AdamW
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers import get_linear_schedule_with_warmup
        except ImportError as erreur:
            raise ImportError(
                "torch et transformers sont requis pour les modèles transformer. "
                "Installez-les avec : pip install arachne[transformers]"
            ) from erreur

        params = self._config_modele.get("params", {})
        nom_modele = params.get("model_name", "camembert-base")
        nb_classes = params.get("num_labels", 4)
        dropout = params.get("dropout", 0.1)
        longueur_max = self._config_modele.get("max_length", 512)

        cfg_entrainement = self._config_modele.get("training", {})
        nb_epochs = cfg_entrainement.get("epochs", 5)
        taille_batch = cfg_entrainement.get("batch_size", 16)
        taux_apprentissage = cfg_entrainement.get("learning_rate", 2e-5)
        ratio_warmup = cfg_entrainement.get("warmup_ratio", 0.1)
        decroissance_poids = cfg_entrainement.get("weight_decay", 0.01)

        self._dispositif = self._obtenir_dispositif()

        self._tokeniseur = AutoTokenizer.from_pretrained(nom_modele)
        self._modele = AutoModelForSequenceClassification.from_pretrained(
            nom_modele,
            num_labels=nb_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self._modele.to(self._dispositif)

        dataset_train = _creer_dataset(textes_train, labels_train, self._tokeniseur, longueur_max)
        chargeur_train = DataLoader(dataset_train, batch_size=taille_batch, shuffle=True)

        optimiseur = AdamW(
            self._modele.parameters(),
            lr=taux_apprentissage,
            weight_decay=decroissance_poids,
        )
        nb_etapes_total = len(chargeur_train) * nb_epochs
        nb_etapes_warmup = int(nb_etapes_total * ratio_warmup)
        planificateur = get_linear_schedule_with_warmup(
            optimiseur, nb_etapes_warmup, nb_etapes_total
        )

        self._modele.train()
        for epoch in range(nb_epochs):
            perte_totale = 0.0
            for batch in chargeur_train:
                batch = {cle: val.to(self._dispositif) for cle, val in batch.items()}
                optimiseur.zero_grad()
                sorties = self._modele(**batch)
                perte = sorties.loss
                perte.backward()
                torch.nn.utils.clip_grad_norm_(self._modele.parameters(), 1.0)
                optimiseur.step()
                planificateur.step()
                perte_totale += perte.item()
            perte_moy = perte_totale / len(chargeur_train)
            print(f"Époque {epoch + 1}/{nb_epochs} — perte : {perte_moy:.4f}")

    def predire(self, textes: list[str]) -> list[str]:
        """Prédit les classes pour une liste de textes.

        Args:
            textes: Textes à classifier.

        Retours:
            Liste des labels prédits.
        """
        probabilites = self.predire_probabilites(textes)
        indices = np.argmax(probabilites, axis=1)
        return [ID_VERS_LABEL[i] for i in indices]

    def predire_probabilites(self, textes: list[str]) -> np.ndarray:
        """Calcule les probabilités d'appartenance à chaque classe.

        Args:
            textes: Textes à classifier.

        Retours:
            Tableau numpy (n_échantillons, n_classes).

        Raises:
            RuntimeError: Si le modèle n'est pas encore entraîné.
        """
        import torch

        if self._modele is None or self._tokeniseur is None:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelez entrainer() d'abord.")

        self._modele.eval()
        toutes_probabilites: list[np.ndarray] = []

        taille_batch = 32
        for i in range(0, len(textes), taille_batch):
            batch_textes = textes[i:i + taille_batch]
            encodages = self._tokeniseur(
                batch_textes,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            encodages = {cle: val.to(self._dispositif) for cle, val in encodages.items()}
            with torch.no_grad():
                sorties = self._modele(**encodages)
            probabilites = torch.softmax(sorties.logits, dim=-1).cpu().numpy()
            toutes_probabilites.append(probabilites)

        return np.vstack(toutes_probabilites)

    def obtenir_classes(self) -> list[str]:
        """Retourne les classes connues du modèle.

        Retours:
            Liste des noms de classes.
        """
        return self._classes

    def sauvegarder(self, repertoire: Path) -> None:
        """Sauvegarde le modèle HuggingFace dans un sous-répertoire hf_model.

        Args:
            repertoire: Répertoire de destination.
        """
        repertoire.mkdir(parents=True, exist_ok=True)
        if self._modele is not None:
            self._modele.save_pretrained(repertoire / "hf_model")
        if self._tokeniseur is not None:
            self._tokeniseur.save_pretrained(repertoire / "hf_model")

    @classmethod
    def charger(cls, repertoire: Path) -> "ClassifieurTransformer":
        """Charge un modèle CamemBERT depuis un répertoire sauvegardé.

        Args:
            repertoire: Répertoire contenant le sous-dossier hf_model.

        Retours:
            Instance de ClassifieurTransformer restaurée.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        chemin_hf = repertoire / "hf_model"
        instance = cls.__new__(cls)
        instance._classes = LABELS.copy()
        instance._config_modele = {}
        instance._config_features = {}
        instance._tokeniseur = AutoTokenizer.from_pretrained(str(chemin_hf))
        instance._modele = AutoModelForSequenceClassification.from_pretrained(str(chemin_hf))
        instance._dispositif = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance._modele.to(instance._dispositif)
        return instance
