"""Module de production pour l'inférence de classification de tableaux."""
from .classifieur import ClassifieurProduction
from .preprocesseur import PreprocesseurProduction
from .tableau import Tableau

__all__ = ["Tableau", "PreprocesseurProduction", "ClassifieurProduction"]
