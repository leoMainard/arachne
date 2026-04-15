"""Dashboard Streamlit de visualisation des expériences Arachne."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from arachne.tracking.tracker import charger_toutes_experiences

REPERTOIRE_MODELES = Path("models")

st.set_page_config(
    page_title="Arachne — Classification de tableaux",
    page_icon="🕷",
    layout="wide",
)


@st.cache_data(ttl=30)
def obtenir_experiences() -> list[dict]:
    """Charge toutes les expériences depuis le répertoire des modèles.

    Retours:
        Liste des expériences triées par date décroissante.
    """
    return charger_toutes_experiences(REPERTOIRE_MODELES)


def experiences_vers_dataframe(experiences: list[dict]) -> pd.DataFrame:
    """Convertit la liste des expériences en DataFrame pour affichage.

    Args:
        experiences: Liste de dictionnaires de métriques d'expériences.

    Retours:
        DataFrame avec les colonnes principales pour le tableau de bord.
    """
    lignes = []
    for exp in experiences:
        test = exp.get("test_metrics", {})
        cv = exp.get("cv_results", {})
        lignes.append({
            "ID": exp.get("experiment_id", ""),
            "Nom": exp.get("experiment_name", ""),
            "Date": exp.get("timestamp", "")[:19].replace("T", " "),
            "Accuracy": test.get("accuracy", None),
            "Macro F1": test.get("macro_f1", None),
            "Weighted F1": test.get("weighted_f1", None),
            "CV Accuracy": cv.get("mean_accuracy", None),
            "CV Écart-type": cv.get("std_accuracy", None),
            "Durée (s)": exp.get("duration_seconds", None),
            "Statut": exp.get("status", ""),
            "_path": exp.get("_path", ""),
        })
    return pd.DataFrame(lignes)


# ──────────────────────────────────────────────
# Barre latérale
# ──────────────────────────────────────────────

st.sidebar.title("🕷 Arachne")
page = st.sidebar.radio("Navigation", ["Vue d'ensemble", "Détails", "Comparaison"])
st.sidebar.markdown("---")
if st.sidebar.button("Actualiser les expériences"):
    st.cache_data.clear()


# ──────────────────────────────────────────────
# Page : Vue d'ensemble
# ──────────────────────────────────────────────

if page == "Vue d'ensemble":
    st.title("Vue d'ensemble des expériences")

    experiences = obtenir_experiences()
    if not experiences:
        st.warning(f"Aucune expérience trouvée dans `{REPERTOIRE_MODELES}/`. Lancez un entraînement d'abord.")
        st.code("python scripts/train.py --config configs/experiments/tfidf_logistic.yaml")
        st.stop()

    df = experiences_vers_dataframe(experiences)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expériences totales", len(df))
    meilleure_acc = df["Accuracy"].max()
    col2.metric("Meilleure accuracy", f"{meilleure_acc:.4f}" if pd.notna(meilleure_acc) else "N/A")
    meilleur_nom = df.loc[df["Accuracy"].idxmax(), "Nom"] if pd.notna(meilleure_acc) else "N/A"
    col3.metric("Meilleur modèle", meilleur_nom)

    st.markdown("---")
    st.subheader("Toutes les expériences")

    colonnes_affichage = ["Nom", "Date", "Accuracy", "Macro F1", "CV Accuracy", "CV Écart-type", "Durée (s)", "Statut"]
    st.dataframe(
        df[colonnes_affichage].style.format({
            "Accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "CV Accuracy": "{:.4f}",
            "CV Écart-type": "{:.4f}",
            "Durée (s)": "{:.1f}",
        }, na_rep="—").highlight_max(subset=["Accuracy", "Macro F1"], color="#c6efce"),
        width="stretch",
    )

    if df["Accuracy"].notna().any():
        st.subheader("Comparaison des accuracy")
        fig = px.bar(
            df.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=True),
            x="Accuracy", y="Nom", orientation="h",
            color="Accuracy", color_continuous_scale="Blues",
            text="Accuracy",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=max(300, len(df) * 50), showlegend=False)
        st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────
# Page : Détails d'une expérience
# ──────────────────────────────────────────────

elif page == "Détails":
    st.title("Détails d'une expérience")

    experiences = obtenir_experiences()
    if not experiences:
        st.warning("Aucune expérience trouvée.")
        st.stop()

    df = experiences_vers_dataframe(experiences)
    id_selectionne = st.selectbox("Sélectionner une expérience", df["ID"].tolist())

    exp = next(e for e in experiences if e["experiment_id"] == id_selectionne)
    chemin_exp = Path(exp["_path"])

    test = exp.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
    col2.metric("Macro F1", f"{test.get('macro_f1', 0):.4f}")
    col3.metric("Weighted F1", f"{test.get('weighted_f1', 0):.4f}")
    col4.metric("Durée", f"{exp.get('duration_seconds', 0):.1f}s")

    st.markdown("---")

    onglet1, onglet2, onglet3, onglet4 = st.tabs([
        "Métriques par classe", "Matrice de confusion", "Validation croisée", "Configuration"
    ])

    with onglet1:
        par_classe = test.get("par_classe", {})
        if par_classe:
            df_pc = pd.DataFrame(par_classe).T.reset_index().rename(columns={"index": "classe"})
            fig = px.bar(
                df_pc.melt(id_vars="classe", value_vars=["precision", "rappel", "f1"]),
                x="classe", y="value", color="variable", barmode="group",
                labels={"value": "Score", "classe": "Classe", "variable": "Métrique"},
                color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
            )
            fig.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                pd.DataFrame(par_classe).T.style.format(
                    "{:.4f}", subset=["precision", "rappel", "f1"]
                ),
                width="stretch",
            )

    with onglet2:
        chemin_mc = chemin_exp / "plots" / "matrice_confusion.png"
        if chemin_mc.exists():
            st.image(str(chemin_mc))
        else:
            st.info("Image de matrice de confusion introuvable.")

    with onglet3:
        cv = exp.get("cv_results", {})
        if cv:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy moyenne", f"{cv.get('mean_accuracy', 0):.4f}")
            col2.metric("Écart-type", f"{cv.get('std_accuracy', 0):.4f}")
            scores_folds = cv.get("fold_scores", [])
            if scores_folds:
                fig = px.bar(
                    x=list(range(1, len(scores_folds) + 1)),
                    y=scores_folds,
                    labels={"x": "Fold", "y": "Accuracy"},
                    title="Scores par fold de validation croisée",
                )
                fig.add_hline(
                    y=cv["mean_accuracy"], line_dash="dash", line_color="red",
                    annotation_text=f"Moyenne : {cv['mean_accuracy']:.4f}",
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Pas de validation croisée pour cette expérience.")

    with onglet4:
        chemin_config = chemin_exp / "config.yaml"
        if chemin_config.exists():
            with open(chemin_config, encoding="utf-8") as f:
                st.code(f.read(), language="yaml")
        else:
            st.json(exp)


# ──────────────────────────────────────────────
# Page : Comparaison
# ──────────────────────────────────────────────

elif page == "Comparaison":
    st.title("Comparaison d'expériences")

    experiences = obtenir_experiences()
    if len(experiences) < 2:
        st.warning("Il faut au moins 2 expériences pour effectuer une comparaison.")
        st.stop()

    df = experiences_vers_dataframe(experiences)
    ids_selectionnes = st.multiselect(
        "Sélectionner les expériences à comparer",
        df["ID"].tolist(),
        default=df["ID"].tolist()[:min(4, len(df))],
    )

    if len(ids_selectionnes) < 2:
        st.info("Sélectionnez au moins 2 expériences.")
        st.stop()

    exps_selectionnees = [e for e in experiences if e["experiment_id"] in ids_selectionnes]
    df_comparaison = df[df["ID"].isin(ids_selectionnes)]

    st.subheader("Comparaison des métriques globales")
    colonnes_metriques = ["Accuracy", "Macro F1", "Weighted F1", "CV Accuracy"]
    disponibles = [c for c in colonnes_metriques if df_comparaison[c].notna().any()]

    fig = go.Figure()
    for _, ligne in df_comparaison.iterrows():
        fig.add_trace(go.Bar(
            name=ligne["Nom"],
            x=disponibles,
            y=[ligne[m] for m in disponibles],
            text=[f"{ligne[m]:.4f}" if pd.notna(ligne[m]) else "N/A" for m in disponibles],
            textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.1], height=400)
    st.plotly_chart(fig, width="stretch")

    st.subheader("F1 par classe")
    toutes_classes: set[str] = set()
    for exp in exps_selectionnees:
        toutes_classes.update(exp.get("test_metrics", {}).get("par_classe", {}).keys())

    if toutes_classes:
        lignes_pc = []
        for exp in exps_selectionnees:
            pc = exp.get("test_metrics", {}).get("par_classe", {})
            for cls in sorted(toutes_classes):
                lignes_pc.append({
                    "Expérience": exp["experiment_name"],
                    "Classe": cls,
                    "F1": pc.get(cls, {}).get("f1", None),
                })
        df_pc = pd.DataFrame(lignes_pc)
        fig2 = px.bar(
            df_pc.dropna(subset=["F1"]),
            x="Classe", y="F1", color="Expérience", barmode="group",
        )
        fig2.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Tableau récapitulatif")
    st.dataframe(
        df_comparaison[["Nom", "Date", "Accuracy", "Macro F1", "Weighted F1", "CV Accuracy", "Durée (s)"]].style.format({
            "Accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "CV Accuracy": "{:.4f}",
            "Durée (s)": "{:.1f}",
        }, na_rep="—"),
        width="stretch",
    )
