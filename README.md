# Arachne

Table classification system for insurance tender documents.

Classifies table matrices (`list[list[str]]`) into 4 categories:
**batiment**, **vehicule**, **sinistre**, **autre**.

## Setup

```bash
pip install -e .

# For transformer models (CamemBERT):
pip install -e ".[transformers]"
```

## Workflow

### 1. Export data from PostgreSQL
```bash
python scripts/export_data.py \
  --dbname your_db \
  --user your_user \
  --output data/tables.parquet
```

### 2. Train a model
```bash
# Classical ML (fast, no GPU needed)
python scripts/train.py --config configs/experiments/tfidf_logistic.yaml

# Try other strategies
python scripts/train.py --config configs/experiments/tfidf_svm.yaml
python scripts/train.py --config configs/experiments/tfidf_random_forest.yaml

# Fine-tune CamemBERT (GPU recommended)
python scripts/train.py --config configs/experiments/camembert.yaml
```

### 3. Visualize results
```bash
streamlit run app/app.py
```

## Project structure

```
arachne/
├── configs/
│   ├── base.yaml                    # Shared defaults
│   └── experiments/                 # One file per strategy
├── data/                            # Local data cache (parquet)
├── models/                          # Saved experiments
│   └── {name}_{timestamp}/
│       ├── config.yaml
│       ├── metrics.json
│       ├── model/
│       └── plots/
├── arachne/
│   ├── config.py                    # YAML loading
│   ├── data/                        # Data loading & preprocessing
│   ├── features/                    # Feature extractors
│   ├── models/                      # Classifiers
│   ├── training/                    # Trainer & evaluator
│   └── tracking/                    # Experiment tracker
├── scripts/
│   ├── train.py                     # Training CLI
│   └── export_data.py               # PostgreSQL export
└── app/
    └── app.py                       # Streamlit dashboard
```

## Adding a new experiment

Create a YAML file in `configs/experiments/` and run:
```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

Available model types: `logistic_regression`, `linear_svc`, `random_forest`, `gradient_boosting`, `camembert`

Available feature types: `tfidf`, `transformer_tokenizer`
