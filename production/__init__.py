"""Module de production pour l'inférence de classification de tableaux."""
from production.classifieur import ClassifieurProduction
from production.preprocesseur import PreprocesseurProduction
from production.tableau import Tableau

__all__ = ["Tableau", "PreprocesseurProduction", "ClassifieurProduction"]
