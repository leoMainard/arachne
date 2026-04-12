# Arachne

![Arachne](arachne.png)

**Description**  
Arachne vous permet d'entraîner un modèle de classification de tableaux à partir :
* d'une source de données PostgreSQL ou locale
* d'une configuration YAML (choix de la stratégie de preprocessing, features, modèle, hyperparamètres)
* de sauvegarder les métriques, graphiques et modèles sur S3 et/ou en local
* de comparer vos expérimentations dans une interface Streamlit en local

Le format d'entrée attendu est une matrice `list[list[str]]` représentant un tableau structuré.

---

**Table des matières**
1. [Installation](#installation)
2. [Configuration](#configuration)
   - [Structure des fichiers](#structure-des-fichiers)
   - [Référence complète des paramètres](#référence-complète-des-paramètres)
3. [Data](#data)
   - [Sources de données](#sources-de-données)
   - [Format attendu](#format-attendu)
   - [Preprocessing](#preprocessing)
4. [Features](#features)
5. [Modèles](#modèles)
6. [Tracking & Stockage](#tracking--stockage)
7. [Scripts](#scripts)
8. [Dashboard](#dashboard)

---

## Installation

```bash
# Dépendances de base
uv sync
# ou
pip install -e .

# Modèles transformer (CamemBERT) — GPU recommandé
pip install -e ".[transformers]"

# LightGBM et XGBoost
pip install -e ".[gradient-boost]"

# Lemmatisation française (spaCy)
pip install -e ".[lemmatisation]"
python -m spacy download fr_core_news_sm

# Stockage S3 / MinIO
pip install -e ".[s3]"
```

---

## Configuration

### Structure des fichiers

```
configs/
├── base.yaml            ← valeurs par défaut partagées par toutes les expériences
└── experiments/
    ├── tfidf_logistic.yaml
    ├── lightgbm.yaml
    └── ...              ← un fichier par expérience
```

Les paramètres d'un fichier d'expérience **écrasent** ceux de `base.yaml` (fusion profonde). Il suffit de ne déclarer que ce qui change.

**Exemple minimal :**
```yaml
# configs/experiments/mon_experience.yaml
experiment:
  name: "mon_experience"

model:
  type: "logistic_regression"
  params:
    C: 5.0
```

### Référence complète des paramètres

```yaml
experiment:
  name: "nom_experience"          # identifiant de l'expérience (utilisé pour nommer le dossier de sortie)
  description: "..."              # description libre (affichée dans le terminal)

data:
  # Source PostgreSQL
  source: "postgresql"
  query: "SELECT table_data, label FROM ma_table"

  # Source locale (parquet)
  source: "local"
  local_path: "data/tables.parquet"

  test_size: 0.2        # proportion du jeu de test (ex : 0.2 = 20 %)
  val_size: 0.1         # proportion du jeu de validation
  stratify: true        # maintenir la distribution des classes dans chaque split
  random_seed: 42
  labels:
    - label_1
    - label_2
    - label_n

preprocessing:
  header_rows: 1              # nombre de lignes considérées comme en-têtes
  header_weight: 3            # répétitions des en-têtes dans le texte (mode "standard")
  max_content_cells: 200      # nombre maximum de cellules de contenu à inclure
  max_length: 5000            # limite optionnelle en caractères du texte final
  format_sortie: "standard"   # "standard" | "entetes_seuls" | "separe"  → voir section Data

features:
  type: "tfidf"               # voir section Features pour tous les types disponibles
  params:
    max_features: 10000
    ngram_range: [1, 2]
    sublinear_tf: true
    min_df: 1
    analyzer: "word"          # "word" | "char" | "char_wb"
    strip_accents: null       # null | "ascii" | "unicode"
    lowercase: true

model:
  type: "logistic_regression"  # voir section Modèles pour tous les types
  params:
    C: 1.0
    max_iter: 1000
    class_weight: "balanced"

training:
  cv_folds: 5                       # nombre de folds de validation croisée (0 ou 1 = désactivé)
  scoring: "accuracy"               # métrique de scoring pour la CV
  entrainer_sur_train_val: false    # si true, entraîne le modèle final sur train + val après la CV

tracking:
  save_model: true         # sauvegarder le modèle entraîné
  output_dir: "models"     # répertoire racine de sortie

stockage:
  local: true              # conserver les artefacts sur le disque local
  s3:
    actif: false           # activer l'upload vers S3
    access_key: ""
    secret_key: ""
    host: ""               # ex : "https://s3.eu-west-3.amazonaws.com" ou endpoint MinIO
    bucket_name: ""
    region: "eu-west-3"
    prefixe: "arachne/experiences/"   # préfixe commun à toutes les expériences dans le bucket
```

---

## Data

### Sources de données

| Source | Paramètre | Description |
|---|---|---|
| Local | `source: "local"` | Lit un fichier `.parquet` via `local_path` |
| PostgreSQL | `source: "postgresql"` | Exécute `query` sur la base configurée dans `.env` |

Variables d'environnement PostgreSQL (fichier `.env` à la racine) :
```bash
DB_HOST=
POSTGRES_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_SCHEMA=
```

### Format attendu

Le DataFrame doit contenir deux colonnes :

| Colonne | Type | Description |
|---|---|---|
| `table_data` | `list[list[str]]` | Tableau sous forme de matrice 2D |
| `label` | `str` | Classe du tableau |

Exemple de valeur pour `table_data` :
```python
[
    ["Véhicule", "Immatriculation", "Marque", "Puissance"],
    ["Camion 1", "AB-123-CD", "Renault", "120 cv"],
    ["Camion 2", "EF-456-GH", "Volvo",   "340 cv"],
]
```

### Preprocessing

Le preprocessing convertit chaque matrice en une chaîne de texte exploitable par les vectoriseurs. Trois stratégies sont disponibles via `preprocessing.format_sortie` :

| `format_sortie` | Comportement | Cas d'usage |
|---|---|---|
| `"standard"` | En-têtes répétés `header_weight` fois + contenu. Ex : `"Véhicule Immatriculation ... Véhicule Immatriculation ... Camion 1 AB-123-CD ..."` | Défaut, donne plus de poids aux en-têtes pour le TF-IDF |
| `"entetes_seuls"` | Uniquement les cellules d'en-tête, sans répétition ni contenu | Quand les en-têtes sont suffisamment discriminants |
| `"separe"` | `"{en-têtes} __CONTENU__ {contenu}"` | Requis pour `features.type: "tfidf_separe"` (double TF-IDF) |

---

## Features

Le bloc `features` du YAML décrit comment transformer les textes en vecteurs numériques.

### `tfidf` — TF-IDF standard

```yaml
features:
  type: "tfidf"
  params:
    max_features: 15000
    ngram_range: [1, 2]     # unigrammes + bigrammes
    sublinear_tf: true      # log(1 + tf) — atténue les fréquences élevées
    min_df: 2               # ignorer les termes qui apparaissent dans moins de 2 documents
    analyzer: "word"        # tokenisation par mot
```

Pour les n-grammes de caractères (utile sur des textes très courts ou bruités) :
```yaml
features:
  type: "tfidf"
  params:
    analyzer: "char_wb"
    ngram_range: [3, 5]
```

### `tfidf_separe` — Double TF-IDF

Applique un TF-IDF **indépendant** sur les en-têtes et sur le contenu, puis concatène les deux vecteurs. Permet de pondérer différemment les deux parties.

> Requiert `preprocessing.format_sortie: "separe"`

```yaml
preprocessing:
  format_sortie: "separe"

features:
  type: "tfidf_separe"
  params_entetes:
    max_features: 10000
    ngram_range: [1, 2]
    sublinear_tf: true
  params_contenu:
    max_features: 5000
    ngram_range: [1, 2]
    sublinear_tf: true
```

### `features_explicites` — Mots-clés + TF-IDF

Combine un TF-IDF classique avec des features binaires basées sur la présence de mots-clés métier (un bit par classe). Signal symbolique fort, particulièrement utile quand les en-têtes contiennent des termes très caractéristiques.

```yaml
features:
  type: "features_explicites"
  params:
    max_features: 15000
    ngram_range: [1, 2]
```

Les mots-clés sont définis dans `arachne/features/extractors.py` (`_MOTS_CLES_PAR_CLASSE`).

### `tfidf_lemmatise` — Lemmatisation spaCy + TF-IDF

Lemmatise le texte avant vectorisation pour réduire les variantes morphologiques (`bâtiment/bâtiments`, `sinistré/sinistres`).

> Requiert `pip install -e ".[lemmatisation]"` et `python -m spacy download fr_core_news_sm`

```yaml
features:
  type: "tfidf_lemmatise"
  modele_spacy: "fr_core_news_sm"   # ou "fr_core_news_md" pour plus de précision
  params:
    max_features: 15000
    ngram_range: [1, 2]
```

### `transformer_tokenizer` — CamemBERT

Utilisé automatiquement lorsque `model.type: "camembert"`. La tokenisation est intégrée au modèle, le bloc `features` n'est pas nécessaire.

---

## Modèles

### Modèles classiques (sklearn)

Tous les modèles classiques s'entraînent via un pipeline `Vectoriseur → Classifieur` et supportent :
- La validation croisée stratifiée (`training.cv_folds`)
- L'option `entrainer_sur_train_val` (entraînement final sur train + val après CV)

| `model.type` | Algorithme | Caractéristiques | Dépendances |
|---|---|---|---|
| `logistic_regression` | Régression logistique | Rapide, probabilités calibrées, bon point de départ | — |
| `linear_svc` | SVM linéaire | Performant sur texte, encapsulé dans `CalibratedClassifierCV` pour les probabilités | — |
| `random_forest` | Forêt aléatoire | Robuste, gère bien le déséquilibre de classes | — |
| `gradient_boosting` | HistGradientBoosting | Efficace, conversion dense automatique (incompatible sparse) | — |
| `complement_nb` | Complement Naive Bayes | Très rapide, adapté aux textes déséquilibrés | — |
| `lightgbm` | LightGBM | Gradient boosting rapide, bonnes performances sur texte sparse | `pip install -e ".[gradient-boost]"` |
| `xgboost` | XGBoost | Gradient boosting robuste, encodage des labels géré automatiquement | `pip install -e ".[gradient-boost]"` |
| `ensemble_vote` | Vote doux (LR + SVM + RF) | Chaque votant a son propre TF-IDF intégré — ne pas spécifier `features` | — |

**Paramètres courants :**

```yaml
# Régression logistique
model:
  type: "logistic_regression"
  params:
    C: 1.0                        # inverse de la régularisation
    max_iter: 1000
    class_weight: "balanced"      # compense le déséquilibre de classes
    solver: "lbfgs"

# SVM linéaire
model:
  type: "linear_svc"
  params:
    C: 1.0
    max_iter: 2000
    class_weight: "balanced"

# Forêt aléatoire
model:
  type: "random_forest"
  params:
    n_estimators: 200
    class_weight: "balanced"
    n_jobs: -1

# HistGradientBoosting
model:
  type: "gradient_boosting"
  params:
    max_iter: 200
    learning_rate: 0.1
    max_leaf_nodes: 31

# ComplementNB
model:
  type: "complement_nb"
  params:
    alpha: 1.0                    # paramètre de lissage

# LightGBM
model:
  type: "lightgbm"
  params:
    n_estimators: 300
    learning_rate: 0.05
    num_leaves: 63
    class_weight: "balanced"
    n_jobs: -1

# XGBoost
model:
  type: "xgboost"
  params:
    n_estimators: 300
    learning_rate: 0.1
    max_depth: 6

# Ensemble (vote doux)
model:
  type: "ensemble_vote"
  params:
    vote: "soft"                  # "soft" (probabilités moyennées) ou "hard" (vote majoritaire)
```

### Modèle transformer (CamemBERT)

Fine-tuning d'un modèle de langue HuggingFace. Requiert GPU pour des temps d'entraînement raisonnables.

> Requiert `pip install -e ".[transformers]"`

```yaml
features:
  type: "transformer_tokenizer"

model:
  type: "camembert"
  params:
    model_name: "camembert-base"   # tout modèle HuggingFace de classification
    num_labels: 4
    dropout: 0.1
    device: "auto"                 # "auto" | "cpu" | "cuda"
  max_length: 512
  training:
    epochs: 5
    batch_size: 16
    learning_rate: 2e-5
    warmup_ratio: 0.1
    weight_decay: 0.01
```

Le modèle est sauvegardé au format HuggingFace dans `model/hf_model/` et peut être rechargé indépendamment d'internet (utile en environnement sans accès au Hub HuggingFace).

---

## Tracking & Stockage

### Artefacts produits par expérience

Après chaque run, un répertoire `{nom}_{horodatage}/` est créé dans `output_dir` (défaut : `models/`) :

```
models/
└── tfidf_logistic_20260411_120000/
    ├── config.yaml          ← configuration complète fusionnée
    ├── metrics.json         ← métriques CV + test + métadonnées
    ├── model/
    │   ├── pipeline.joblib  ← modèle classique (sklearn)
    │   └── hf_model/        ← modèle transformer (HuggingFace)
    └── plots/
        ├── matrice_confusion.png
        └── metriques_par_classe.png
```

**Structure de `metrics.json` :**
```json
{
  "experiment_id": "tfidf_logistic_20260411_120000",
  "experiment_name": "tfidf_logistic",
  "timestamp": "2026-04-11T12:00:00",
  "duration_seconds": 4.2,
  "data": { "n_train": 210, "n_val": 30, "n_test": 60, "distribution_classes": {...} },
  "cv_results": { "mean_accuracy": 0.9714, "std_accuracy": 0.021, "fold_scores": [...] },
  "test_metrics": { "accuracy": 0.9833, "macro_f1": 0.9830, "weighted_f1": 0.9832, "par_classe": {...} },
  "status": "terminee"
}
```

### Option `entrainer_sur_train_val`

Par défaut, la CV s'effectue sur `train + val` pour estimer la vraie performance, puis le modèle final est entraîné sur `train` uniquement.

En activant `entrainer_sur_train_val: true`, le modèle final est entraîné sur `train + val` — ce qui maximise les données d'entraînement tout en conservant une évaluation honnête via la CV.

```
CV (5 folds) :  [train | val]  →  performance estimée sans biais
Modèle final :  [train + val]  →  modèle livré en production
Évaluation :    [test]         →  performance reportée
```

### Stockage S3

Configurer la section `stockage` pour activer l'upload vers S3 (AWS, MinIO, OVH, etc.) :

```yaml
stockage:
  local: false       # ne pas conserver en local
  s3:
    actif: true
    access_key: "AKIAIOSFODNN7EXAMPLE"
    secret_key: "wJalrXUtnFEMI..."
    host: "https://s3.eu-west-3.amazonaws.com"
    bucket_name: "mon-bucket-ml"
    prefixe: "arachne/experiences/"
```

| Artefact | Méthode S3 | Sérialisation |
|---|---|---|
| `config.yaml`, `metrics.json` | `envoyer_objet()` | En mémoire (`yaml.dump` / `json.dumps`) |
| `matrice_confusion.png`, `metriques_par_classe.png` | `envoyer_objet()` | En mémoire (`matplotlib` → `BytesIO`) |
| `pipeline.joblib` | `envoyer_objet()` | En mémoire (`joblib.dump` → `BytesIO`) |
| `hf_model/` | `envoyer_repertoire()` | Via répertoire temporaire auto-nettoyé |

Combinaisons possibles : `local: true` + `s3.actif: false` (défaut), `local: true` + `s3.actif: true` (sauvegarde double), `local: false` + `s3.actif: true` (S3 uniquement).

---

## Scripts

### `scripts/train.py` — Lancer un entraînement

```bash
python scripts/train.py --config configs/experiments/tfidf_logistic.yaml

# Surcharger la source de données
python scripts/train.py --config configs/experiments/tfidf_logistic.yaml --source-donnees local

# Ne pas sauvegarder le modèle (métriques et graphiques uniquement)
python scripts/train.py --config configs/experiments/tfidf_logistic.yaml --sans-sauvegarde
```

### `scripts/export_data.py` — Exporter depuis PostgreSQL

Exporte les données labellisées vers un fichier `.parquet` local, pour un usage ultérieur avec `source: "local"`.

```bash
python scripts/export_data.py \
  --host localhost --port 5432 \
  --dbname ma_base --user mon_user --password mon_mdp \
  --output data/tables.parquet

# Avec une requête personnalisée
python scripts/export_data.py \
  --dbname ma_base --user mon_user \
  --requete "SELECT table_data, label FROM schema.ma_table WHERE label IS NOT NULL" \
  --output data/tables_filtre.parquet
```

### `scripts/generer_donnees_test.py` — Générer des données synthétiques

Génère un jeu de données factice (300 tableaux répartis équitablement) pour tester le pipeline sans données réelles.

```bash
python scripts/generer_donnees_test.py
# → data/tables_test.parquet
```

---

## Dashboard

Interface Streamlit pour visualiser et comparer les expériences sauvegardées dans `models/`.

```bash
streamlit run app/app.py
```

Fonctionnalités :
- Tableau comparatif de toutes les expériences (accuracy, F1, durée)
- Métriques détaillées par classe
- Matrice de confusion et graphiques de métriques
- Filtrage par nom d'expérience
