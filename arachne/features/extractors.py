"""Extracteurs de features pour la classification de tableaux."""
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from arachne.constants import SEPARATEUR_CONTENU, TypeFeatures


# ---------------------------------------------------------------------------
# Mots-clés pour les features explicites
# ---------------------------------------------------------------------------

_MOTS_CLES_PAR_CLASSE: dict[str, list[str]] = {
    "batiment": [
        "batiment", "bâtiment", "immeuble", "surface", "construction", "bati",
        "local", "commune", "adresse", "m2", "m²", "logement", "ets",
        "établissement", "etablissement",
    ],
    "vehicule": [
        "vehicule", "véhicule", "immatriculation", "marque", "modele", "modèle",
        "moteur", "carburant", "km", "kilometrage", "kilométrage", "puissance",
        "cylindree", "cylindrée", "voiture", "camion", "bus", "mec",
    ],
    "sinistre": [
        "sinistre", "franchise", "indemnite", "indemnité", "garantie",
        "déclaration", "declaration", "evenement", "événement", "montant",
        "reglement", "règlement", "nature", "expertise", "recours",
    ],
}


# ---------------------------------------------------------------------------
# TF-IDF standard
# ---------------------------------------------------------------------------

def construire_vectoriseur_tfidf(params: dict) -> TfidfVectorizer:
    """Instancie un vectoriseur TF-IDF à partir des paramètres de configuration.

    Args:
        params: Dictionnaire de paramètres (section ``features.params`` du YAML).

    Retours:
        Vectoriseur TF-IDF configuré (non entraîné).
    """
    plage_ngrams = params.get("ngram_range", [1, 2])
    return TfidfVectorizer(
        max_features=params.get("max_features", 15000),
        ngram_range=tuple(plage_ngrams),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 2),
        analyzer=params.get("analyzer", "word"),
        strip_accents=params.get("strip_accents", None),
        lowercase=params.get("lowercase", True),
    )


# ---------------------------------------------------------------------------
# Double TF-IDF (en-têtes / contenu séparés)
# ---------------------------------------------------------------------------

class ExtracteurDoubleTFIDF(BaseEstimator, TransformerMixin):
    """TF-IDF indépendant sur les en-têtes et le contenu, puis concaténation.

    Attend un texte au format ``"{entetes} __CONTENU__ {contenu}"``
    produit par ``Preprocesseur`` avec ``format_sortie="separe"``.

    Args:
        params_entetes: Paramètres TF-IDF pour les en-têtes.
        params_contenu: Paramètres TF-IDF pour le contenu.
    """

    def __init__(self, params_entetes: dict, params_contenu: dict) -> None:
        self.params_entetes = params_entetes
        self.params_contenu = params_contenu
        self._tfidf_entetes: TfidfVectorizer | None = None
        self._tfidf_contenu: TfidfVectorizer | None = None

    def _separer(self, X: list[str]) -> tuple[list[str], list[str]]:
        """Sépare les textes en parties en-têtes et contenu."""
        parties = [t.split(SEPARATEUR_CONTENU, 1) for t in X]
        entetes = [p[0].strip() for p in parties]
        contenu = [p[1].strip() if len(p) > 1 else "" for p in parties]
        return entetes, contenu

    def fit(self, X: list[str], y=None) -> "ExtracteurDoubleTFIDF":
        """Entraîne les deux vectoriseurs TF-IDF.

        Args:
            X: Liste de textes avec séparateur __CONTENU__.
            y: Non utilisé.

        Retours:
            Self.
        """
        from scipy.sparse import hstack  # noqa: F401 (import check)
        entetes, contenu = self._separer(X)
        self._tfidf_entetes = construire_vectoriseur_tfidf(self.params_entetes).fit(entetes)
        self._tfidf_contenu = construire_vectoriseur_tfidf(self.params_contenu).fit(contenu)
        return self

    def transform(self, X: list[str], y=None):
        """Transforme les textes en matrice (entetes + contenu concaténés).

        Args:
            X: Liste de textes avec séparateur __CONTENU__.

        Retours:
            Matrice sparse (n_échantillons, n_features_entetes + n_features_contenu).
        """
        from scipy.sparse import hstack
        entetes, contenu = self._separer(X)
        mat_entetes = self._tfidf_entetes.transform(entetes)
        mat_contenu = self._tfidf_contenu.transform(contenu)
        return hstack([mat_entetes, mat_contenu])


# ---------------------------------------------------------------------------
# Features explicites (mots-clés)
# ---------------------------------------------------------------------------

class ExtracteurFeaturesExplicites(BaseEstimator, TransformerMixin):
    """Features binaires basées sur la présence de mots-clés métier dans le texte.

    Crée une colonne par classe (batiment, vehicule, sinistre) indiquant
    si au moins un de ses mots-clés est présent dans le texte.
    Complète le TF-IDF avec un signal symbolique fort.

    Args:
        mots_cles_par_classe: Dictionnaire ``{classe: [mots_cles]}``.
                              Utilise les mots-clés métier par défaut si None.
    """

    def __init__(self, mots_cles_par_classe: dict[str, list[str]] | None = None) -> None:
        self.mots_cles_par_classe = mots_cles_par_classe or _MOTS_CLES_PAR_CLASSE

    def fit(self, X: list[str], y=None) -> "ExtracteurFeaturesExplicites":
        """Sans entraînement nécessaire (règles déterministes).

        Retours:
            Self.
        """
        return self

    def transform(self, X: list[str], y=None):
        """Crée une matrice binaire de présence de mots-clés.

        Args:
            X: Liste de textes à analyser.

        Retours:
            Matrice sparse (n_échantillons, n_classes).
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        classes = list(self.mots_cles_par_classe.keys())
        features = np.zeros((len(X), len(classes)), dtype=np.float32)

        for i, texte in enumerate(X):
            texte_lower = texte.lower()
            for j, classe in enumerate(classes):
                if any(mot in texte_lower for mot in self.mots_cles_par_classe[classe]):
                    features[i, j] = 1.0

        return csr_matrix(features)


# ---------------------------------------------------------------------------
# Lemmatiseur français (spaCy)
# ---------------------------------------------------------------------------

class TransformeurLemmatiseur(BaseEstimator, TransformerMixin):
    """Lemmatise le texte en français via spaCy avant vectorisation TF-IDF.

    Réduit les variantes morphologiques (bâtiment/bâtiments, immeuble/immeubles)
    au même lemme, ce qui améliore la couverture du vocabulaire sur petits datasets.

    Args:
        modele_spacy: Nom du modèle spaCy à utiliser (défaut: ``fr_core_news_sm``).

    Note:
        Requiert : ``pip install spacy && python -m spacy download fr_core_news_sm``
    """

    def __init__(self, modele_spacy: str = "fr_core_news_sm") -> None:
        self.modele_spacy = modele_spacy
        self._nlp = None

    def _charger_modele(self):
        """Charge le modèle spaCy (lazy, une seule fois)."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.modele_spacy, disable=["parser", "ner"])
            except Exception as erreur:
                raise ImportError(
                    f"spaCy et le modèle '{self.modele_spacy}' sont requis pour la lemmatisation. "
                    "Installez avec : pip install spacy && python -m spacy download fr_core_news_sm"
                ) from erreur

    def fit(self, X: list[str], y=None) -> "TransformeurLemmatiseur":
        """Charge le modèle spaCy si nécessaire.

        Retours:
            Self.
        """
        self._charger_modele()
        return self

    def transform(self, X: list[str], y=None) -> list[str]:
        """Lemmatise une liste de textes.

        Args:
            X: Textes à lemmatiser.

        Retours:
            Liste de textes lemmatisés.
        """
        self._charger_modele()
        return [
            " ".join(
                token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            )
            for doc in self._nlp.pipe(X, batch_size=32)
        ]


# ---------------------------------------------------------------------------
# Registre des extracteurs
# ---------------------------------------------------------------------------

def obtenir_extracteur(config_features: dict):
    """Retourne l'extracteur de features correspondant à la configuration.

    Args:
        config_features: Section ``features`` du fichier YAML de l'expérience.

    Retours:
        Extracteur de features instancié (non entraîné).

    Raises:
        ValueError: Si le type de features est inconnu.
    """
    type_features = config_features.get("type", TypeFeatures.TFIDF)
    params = config_features.get("params", {})

    if type_features in (TypeFeatures.TFIDF, "tfidf"):
        return construire_vectoriseur_tfidf(params)

    elif type_features in (TypeFeatures.TFIDF_SEPARE, "tfidf_separe"):
        params_entetes = config_features.get("params_entetes", {"max_features": 10000, "ngram_range": [1, 2], "sublinear_tf": True})
        params_contenu = config_features.get("params_contenu", {"max_features": 5000, "ngram_range": [1, 2], "sublinear_tf": True})
        return ExtracteurDoubleTFIDF(params_entetes, params_contenu)

    elif type_features in (TypeFeatures.FEATURES_EXPLICITES, "features_explicites"):
        return FeatureUnion([
            ("tfidf", construire_vectoriseur_tfidf(params)),
            ("keywords", ExtracteurFeaturesExplicites()),
        ])

    elif type_features in (TypeFeatures.TFIDF_LEMMATISE, "tfidf_lemmatise"):
        modele_spacy = config_features.get("modele_spacy", "fr_core_news_sm")
        return Pipeline([
            ("lemmatiseur", TransformeurLemmatiseur(modele_spacy)),
            ("tfidf", construire_vectoriseur_tfidf(params)),
        ])

    else:
        raise ValueError(
            f"Type de features '{type_features}' non géré ici. "
            "La tokenisation transformer est intégrée au modèle ClassifieurTransformer."
        )
