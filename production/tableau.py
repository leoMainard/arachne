"""Classe Tableau — point d'entrée principal pour l'inférence en production."""
from __future__ import annotations

from production.classifieur import ClassifieurProduction
from production.preprocesseur import PreprocesseurProduction


class Tableau:
    """Représente un tableau extrait d'un document d'appel d'offres.

    Point d'entrée principal pour la classification en production.
    Encapsule la matrice brute et orchestre prétraitement + prédiction
    à partir d'une configuration.

    Args:
        data: Matrice 2D du tableau, chaque cellule étant une chaîne de caractères.

    Note:
        Pour classifier plusieurs tableaux avec le même modèle, instanciez
        PreprocesseurProduction et ClassifieurProduction une seule fois en
        dehors de la boucle (chargement du modèle coûteux), puis appelez
        predict() en leur passant ces instances directement.

    Exemple (usage simple) :
        >>> config = {"modele": {"repertoire_experience": "models/tfidf_logistic_..."}}
        >>> tableau = Tableau([
        ...     ["Bâtiment", "Adresse", "Surface (m²)", "Valeur assurée (€)"],
        ...     ["Mairie", "1 Place de la République", "450", "850 000"],
        ... ])
        >>> label = tableau.predict(config)
        >>> probs = tableau.predict_proba(config)

    Exemple (usage performant pour plusieurs tableaux) :
        >>> preprocesseur = PreprocesseurProduction(config)
        >>> classifieur   = ClassifieurProduction(config)
        >>> for data in liste_de_matrices:
        ...     label = Tableau(data).predict(config, preprocesseur, classifieur)
    """

    def __init__(self, data: list[list[str]]) -> None:
        self.data = data

    def predict(
        self,
        config: dict,
        preprocesseur: PreprocesseurProduction | None = None,
        classifieur: ClassifieurProduction | None = None,
    ) -> str:
        """Classifie le tableau et retourne le label prédit.

        Instancie préprocesseur et classifieur depuis la config si non fournis.
        Pour des appels répétés, passez des instances pré-construites afin
        d'éviter de recharger le modèle à chaque appel.

        Args:
            config: Dictionnaire de configuration de production.
                    Doit contenir ``modele.repertoire_experience``.
            preprocesseur: Instance pré-construite (optionnel, pour les appels en boucle).
            classifieur: Instance pré-construite (optionnel, pour les appels en boucle).

        Retours:
            Label prédit parmi : batiment, vehicule, sinistre, autre.
        """
        preprocesseur = preprocesseur or PreprocesseurProduction(config)
        classifieur = classifieur or ClassifieurProduction(config)

        texte = preprocesseur.transformer(self.data)
        return classifieur.predire([texte])[0]

    def predict_proba(
        self,
        config: dict,
        preprocesseur: PreprocesseurProduction | None = None,
        classifieur: ClassifieurProduction | None = None,
    ) -> dict[str, float]:
        """Retourne les probabilités d'appartenance à chaque classe.

        Args:
            config: Dictionnaire de configuration de production.
                    Doit contenir ``modele.repertoire_experience``.
            preprocesseur: Instance pré-construite (optionnel, pour les appels en boucle).
            classifieur: Instance pré-construite (optionnel, pour les appels en boucle).

        Retours:
            Dictionnaire {label: probabilité} trié par probabilité décroissante.
        """
        preprocesseur = preprocesseur or PreprocesseurProduction(config)
        classifieur = classifieur or ClassifieurProduction(config)

        texte = preprocesseur.transformer(self.data)
        probabilites = classifieur.predire_probabilites([texte])[0]
        classes = classifieur.obtenir_classes()

        return {
            str(label): float(prob)
            for label, prob in sorted(
                zip(classes, probabilites),
                key=lambda x: x[1],
                reverse=True,
            )
        }
