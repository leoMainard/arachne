"""Génère un faux jeu de données de tableaux d'assurance pour tester le pipeline.

Produit un fichier parquet avec ~300 tableaux réalistes répartis en 4 classes :
batiment, vehicule, sinistre, autre.

Utilisation :
    python scripts/generer_donnees_test.py
    python scripts/generer_donnees_test.py --output data/tables_test.parquet --nb-lignes 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console

from arachne.data.loader import ChargeurDonnees

console = Console()

# ──────────────────────────────────────────────
# Données de référence pour la génération
# ──────────────────────────────────────────────

_BATIMENTS_NOMS = [
    "Mairie centrale", "Annexe administrative", "Bibliothèque municipale",
    "Salle des fêtes", "École primaire Jules Ferry", "Gymnase municipal",
    "Piscine intercommunale", "Centre technique municipal", "Crèche Les Petits Lutins",
    "Médiathèque", "Palais des sports", "Foyer rural", "EHPAD Les Tilleuls",
    "Centre culturel", "Hôtel de ville", "École maternelle Pasteur",
    "Centre de loisirs", "Dépôt de bus", "Archives municipales",
]
_COMMUNES = [
    "Lyon", "Bordeaux", "Nantes", "Strasbourg", "Lille", "Rennes", "Grenoble",
    "Montpellier", "Toulouse", "Nice", "Clermont-Ferrand", "Dijon", "Reims",
    "Angers", "Brest", "Limoges", "Tours", "Amiens", "Caen",
]
_TYPES_CONSTRUCTION = [
    "Béton armé", "Maçonnerie traditionnelle", "Ossature bois",
    "Charpente métallique", "Mixte béton/brique",
]
_USAGES_BATIMENT = [
    "Bureaux administratifs", "Équipement sportif", "Établissement scolaire",
    "Équipement culturel", "Logement social", "Services techniques", "Accueil du public",
]


def _generer_tableau_batiment(rng: random.Random) -> list[list[str]]:
    """Génère un tableau de patrimoine bâti."""
    schema = rng.choice([
        ["Bâtiment", "Adresse", "Commune", "Surface (m²)", "Valeur assurée (€)", "Type de construction", "Usage"],
        ["Désignation", "Commune", "Année construction", "Surface (m²)", "Valeur de remplacement (€)", "Matériaux"],
        ["Réf.", "Bâtiment", "Adresse", "CP", "Ville", "Surface", "Val. assurée", "Observations"],
        ["N°", "Désignation du bien", "Localisation", "Surface totale", "Capital assuré"],
    ])
    entete = schema

    nb_lignes = rng.randint(3, 15)
    lignes = [entete]
    for _ in range(nb_lignes):
        surface = rng.randint(80, 5000)
        valeur = surface * rng.randint(800, 2500)
        commune = rng.choice(_COMMUNES)
        batiment = rng.choice(_BATIMENTS_NOMS)
        construction = rng.choice(_TYPES_CONSTRUCTION)
        annee = rng.randint(1920, 2020)
        adresse = f"{rng.randint(1, 150)} rue {rng.choice(['de la Paix', 'Victor Hugo', 'du Général de Gaulle', 'Jean Jaurès', 'de la République'])}"

        nb_cols = len(entete)
        valeurs = [batiment, adresse, commune, str(surface), f"{valeur:,}".replace(",", " "), construction, str(annee)]
        lignes.append(valeurs[:nb_cols])

    return lignes


_MARQUES = ["Renault", "Peugeot", "Citroën", "Ford", "Volkswagen", "Mercedes", "Toyota", "Iveco", "MAN"]
_MODELES = {
    "Renault": ["Master", "Trafic", "Kangoo", "Mégane", "Clio", "Scénic"],
    "Peugeot": ["Partner", "Expert", "Boxer", "308", "508", "3008"],
    "Citroën": ["Berlingo", "Jumpy", "Jumper", "C3", "C4"],
    "Ford": ["Transit", "Connect", "Focus", "Kuga"],
    "Volkswagen": ["Transporter", "Crafter", "Golf", "Passat"],
    "Mercedes": ["Sprinter", "Vito", "Classe A", "Classe C"],
    "Toyota": ["Proace", "Yaris", "Prius"],
    "Iveco": ["Daily 35S", "Daily 50C", "Eurocargo"],
    "MAN": ["TGE 3", "TGE 5", "TGE 6"],
}
_USAGES_VEHICULE = [
    "Transport de personnel", "Maintenance voirie", "Collecte des déchets",
    "Véhicule de fonction", "Transport scolaire", "Engin de chantier",
    "Police municipale", "Service incendie", "Entretien espaces verts",
]
_ENERGIES = ["Diesel", "Essence", "Électrique", "Hybride", "GNV"]


def _generer_immatriculation(rng: random.Random) -> str:
    """Génère une immatriculation française aléatoire (format SIV)."""
    lettres = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return (
        f"{rng.choice(lettres)}{rng.choice(lettres)}-"
        f"{rng.randint(100, 999)}-"
        f"{rng.choice(lettres)}{rng.choice(lettres)}"
    )


def _generer_tableau_vehicule(rng: random.Random) -> list[list[str]]:
    """Génère un tableau de flotte de véhicules."""
    schema = rng.choice([
        ["Immatriculation", "Marque", "Modèle", "Année", "Énergie", "Puissance (CV)", "Usage", "Valeur (€)"],
        ["N° parc", "Immatriculation", "Marque", "Modèle", "Mise en circulation", "Carburant", "Kilométrage", "Valeur vénale"],
        ["Réf.", "Désignation", "Immatriculation", "Marque/Modèle", "Année MEC", "Usage", "Val. assurée (€)"],
        ["ID", "Immatriculation", "Marque", "Modèle", "Année", "Type", "Puissance", "Valeur neuf", "Garanties"],
    ])
    entete = schema

    nb_lignes = rng.randint(4, 20)
    lignes = [entete]
    for i in range(nb_lignes):
        marque = rng.choice(_MARQUES)
        modele = rng.choice(_MODELES.get(marque, ["Standard"]))
        annee = rng.randint(2005, 2024)
        puissance = rng.randint(75, 280)
        valeur = rng.randint(8000, 85000)
        immat = _generer_immatriculation(rng)
        usage = rng.choice(_USAGES_VEHICULE)
        energie = rng.choice(_ENERGIES)
        km = rng.randint(5000, 250000)

        nb_cols = len(entete)
        valeurs = [immat, marque, modele, str(annee), energie, str(puissance), usage, f"{valeur:,}".replace(",", " "), str(km), str(i + 1)]
        lignes.append(valeurs[:nb_cols])

    return lignes


_NATURES_SINISTRE = [
    "Dégât des eaux", "Incendie", "Bris de glace", "Vol avec effraction",
    "Vandalisme", "Catastrophe naturelle", "Accident de la circulation",
    "Responsabilité civile", "Dommage électrique", "Tempête/grêle",
    "Effondrement partiel", "Dégât des eaux (fuite toiture)",
]
_GARANTIES = [
    "Dommages aux biens", "Responsabilité civile", "Vol/vandalisme",
    "Bris de glace", "Flotte automobile", "Protection juridique",
    "Tous risques", "Catastrophes naturelles",
]
_ETATS = ["Clos", "En cours", "Expertise en cours", "Litige", "Recours"]


def _generer_tableau_sinistre(rng: random.Random) -> list[list[str]]:
    """Génère un tableau d'historique de sinistres."""
    schema = rng.choice([
        ["N° sinistre", "Date survenance", "Nature du sinistre", "Garantie", "Montant indemnisé (€)", "Franchise (€)", "État"],
        ["Référence", "Date", "Description", "Garantie mobilisée", "Coût total", "Part assureur", "Part assuré", "Statut"],
        ["Sinistre", "Date ouverture", "Type", "Lieu", "Montant réglé", "Franchise", "Recours", "Clôture"],
        ["N°", "Année", "Nature", "Site concerné", "Indemnité versée (€)", "Observations"],
    ])
    entete = schema

    nb_lignes = rng.randint(3, 18)
    lignes = [entete]
    for _ in range(nb_lignes):
        jour = rng.randint(1, 28)
        mois = rng.randint(1, 12)
        annee = rng.randint(2018, 2024)
        date = f"{jour:02d}/{mois:02d}/{annee}"
        nature = rng.choice(_NATURES_SINISTRE)
        garantie = rng.choice(_GARANTIES)
        montant = rng.randint(500, 150000)
        franchise = rng.choice([150, 300, 500, 750, 1000, 1500])
        etat = rng.choice(_ETATS)
        ref = f"{annee}-{rng.randint(100, 999)}"
        recours = rng.choice(["Oui", "Non"])

        nb_cols = len(entete)
        valeurs = [ref, date, nature, garantie, f"{montant:,}".replace(",", " "), str(franchise), etat, recours, ""]
        lignes.append(valeurs[:nb_cols])

    return lignes


_SCHEMAS_AUTRE = [
    {
        "entete": ["Interlocuteur", "Fonction", "Téléphone", "Email", "Disponibilité"],
        "generateur": lambda rng: [
            rng.choice(["M.", "Mme"]) + " " + rng.choice(["Dupont", "Martin", "Bernard", "Petit", "Robert"]),
            rng.choice(["DGS", "DGA", "Responsable RH", "DAF", "DSI", "Élu référent"]),
            f"0{rng.randint(1,9)} {rng.randint(10,99)} {rng.randint(10,99)} {rng.randint(10,99)} {rng.randint(10,99)}",
            f"contact@mairie-{rng.randint(1000,9999)}.fr",
            rng.choice(["Lun-Ven 9h-17h", "Sur RDV", "Permanence Mardi/Jeudi"]),
        ],
    },
    {
        "entete": ["Prestataire", "Prestation", "Montant annuel HT (€)", "Durée contrat", "Renouvellement"],
        "generateur": lambda rng: [
            rng.choice(["Société Nettoyage Pro", "MaintenancePlus SARL", "BureauTique SAS", "InfoSystèmes SA"]),
            rng.choice(["Nettoyage locaux", "Maintenance ascenseurs", "Infogérance", "Gardiennage"]),
            str(rng.randint(5000, 80000)),
            rng.choice(["1 an", "3 ans", "5 ans"]),
            rng.choice(["2024", "2025", "2026", "Tacite reconduction"]),
        ],
    },
    {
        "entete": ["Effectif", "Catégorie", "Temps plein", "Temps partiel", "Contractuels", "Titulaires"],
        "generateur": lambda rng: [
            rng.choice(["Administration générale", "Technique", "Scolaire", "Animation", "Police"]),
            rng.choice(["A", "B", "C"]),
            str(rng.randint(1, 50)),
            str(rng.randint(0, 20)),
            str(rng.randint(0, 15)),
            str(rng.randint(1, 40)),
        ],
    },
    {
        "entete": ["Garantie", "Objet assuré", "Prime annuelle HT (€)", "Franchise", "Plafond (€)"],
        "generateur": lambda rng: [
            rng.choice(["RC Générale", "Dommages aux biens", "Flotte auto", "Protection juridique"]),
            rng.choice(["Ensemble du patrimoine", "Véhicules de la commune", "Agents", "Élus"]),
            str(rng.randint(1000, 50000)),
            str(rng.randint(150, 2000)),
            str(rng.randint(100000, 5000000)),
        ],
    },
]


def _generer_tableau_autre(rng: random.Random) -> list[list[str]]:
    """Génère un tableau d'une catégorie 'autre' (non pertinent pour la classification)."""
    schema = rng.choice(_SCHEMAS_AUTRE)
    entete = schema["entete"]
    generateur = schema["generateur"]

    nb_lignes = rng.randint(2, 10)
    lignes = [entete]
    for _ in range(nb_lignes):
        valeurs = generateur(rng)
        lignes.append(valeurs[:len(entete)])

    return lignes


# ──────────────────────────────────────────────
# Générateur principal
# ──────────────────────────────────────────────

_GENERATEURS = {
    "batiment": _generer_tableau_batiment,
    "vehicule": _generer_tableau_vehicule,
    "sinistre": _generer_tableau_sinistre,
    "autre": _generer_tableau_autre,
}

_DISTRIBUTION_DEFAUT = {
    "batiment": 80,
    "vehicule": 80,
    "sinistre": 80,
    "autre": 60,
}


def generer_dataset(
    distribution: dict[str, int] | None = None,
    graine: int = 42,
) -> pd.DataFrame:
    """Génère un jeu de données de tableaux d'assurance labellisés.

    Args:
        distribution: Dictionnaire {label: nombre d'exemples}. Si None, utilise
                      la distribution par défaut (80/80/80/60).
        graine: Graine aléatoire pour la reproductibilité.

    Retours:
        DataFrame avec colonnes ``id``, ``table_data`` (list[list[str]]), ``label``.
    """
    if distribution is None:
        distribution = _DISTRIBUTION_DEFAUT

    rng = random.Random(graine)
    lignes: list[dict] = []
    id_courant = 1

    for label, nb_exemples in distribution.items():
        generateur = _GENERATEURS[label]
        for _ in range(nb_exemples):
            tableau = generateur(rng)
            lignes.append({
                "id": id_courant,
                "table_data": tableau,
                "label": label,
            })
            id_courant += 1

    # Mélanger le dataset
    rng.shuffle(lignes)

    df = pd.DataFrame(lignes)
    return df


def _analyser_arguments() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Retours:
        Namespace avec les arguments parsés.
    """
    parser = argparse.ArgumentParser(
        description="Génère un faux jeu de données de tableaux d'assurance."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/tables_test.parquet"),
        help="Chemin du fichier parquet de sortie.",
    )
    parser.add_argument("--nb-batiment", type=int, default=80)
    parser.add_argument("--nb-vehicule", type=int, default=80)
    parser.add_argument("--nb-sinistre", type=int, default=80)
    parser.add_argument("--nb-autre", type=int, default=60)
    parser.add_argument("--graine", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du générateur."""
    args = _analyser_arguments()

    distribution = {
        "batiment": args.nb_batiment,
        "vehicule": args.nb_vehicule,
        "sinistre": args.nb_sinistre,
        "autre": args.nb_autre,
    }

    total = sum(distribution.values())
    console.print(f"[bold]Génération de {total} tableaux...[/bold]")
    for label, nb in distribution.items():
        console.print(f"  {label:12s} : {nb} exemples")

    df = generer_dataset(distribution, graine=args.graine)

    console.print(f"\nAperçu d'un tableau 'batiment' :")
    exemple = df[df["label"] == "batiment"].iloc[0]["table_data"]
    for ligne in exemple[:3]:
        console.print(f"  {ligne}")

    ChargeurDonnees.exporter_parquet(df, args.output)
    console.print(f"\n[green]Jeu de données sauvegardé : {args.output}[/green]")
    console.print(f"[green]Total : {len(df)} tableaux[/green]")
    console.print(f"\nDistribution :\n{df['label'].value_counts().to_string()}")
    console.print(f"\nPour tester le pipeline :")
    console.print(f"  python scripts/train.py --config configs/experiments/tfidf_logistic.yaml")


if __name__ == "__main__":
    main()
