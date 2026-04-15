"""Dashboard Gradio de visualisation des expériences Arachne."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from arachne.tracking.tracker import charger_toutes_experiences

REPERTOIRE_MODELES = Path("models")

COLS_AFFICHAGE = [
    "Nom", "Date", "Accuracy", "Macro F1", "Weighted F1",
    "CV Accuracy", "CV Écart-type", "Durée (s)", "Statut",
]


# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

def _charger_experiences() -> list[dict]:
    return charger_toutes_experiences(REPERTOIRE_MODELES)


def _vers_dataframe(experiences: list[dict]) -> pd.DataFrame:
    lignes = []
    for exp in experiences:
        test = exp.get("test_metrics", {})
        cv = exp.get("cv_results", {})
        lignes.append({
            "ID": exp.get("experiment_id", ""),
            "Nom": exp.get("experiment_name", ""),
            "Date": exp.get("timestamp", "")[:19].replace("T", " "),
            "Accuracy": test.get("accuracy"),
            "Macro F1": test.get("macro_f1"),
            "Weighted F1": test.get("weighted_f1"),
            "CV Accuracy": cv.get("mean_accuracy"),
            "CV Écart-type": cv.get("std_accuracy"),
            "Durée (s)": exp.get("duration_seconds"),
            "Statut": exp.get("status", ""),
            "_path": exp.get("_path", ""),
        })
    return pd.DataFrame(lignes) if lignes else pd.DataFrame(columns=["ID", "Nom", "Date",
        "Accuracy", "Macro F1", "Weighted F1", "CV Accuracy", "CV Écart-type", "Durée (s)",
        "Statut", "_path"])


def _graphique_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or df["Accuracy"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            annotations=[{"text": "Aucune donnée", "xref": "paper", "yref": "paper",
                          "showarrow": False, "font": {"size": 20}}],
            height=200,
        )
        return fig
    df_tri = df.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=True)
    fig = px.bar(
        df_tri, x="Accuracy", y="Nom", orientation="h",
        color="Accuracy", color_continuous_scale="Blues",
        text="Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        height=max(300, len(df_tri) * 50),
        showlegend=False,
        xaxis_range=[0, 1.05],
        margin={"l": 10, "r": 30, "t": 20, "b": 10},
    )
    return fig


# ---------------------------------------------------------------------------
# Handlers : Vue d'ensemble + rafraîchissement global
# ---------------------------------------------------------------------------

def rafraichir_tout():
    """Charge les expériences et met à jour tous les composants."""
    experiences = _charger_experiences()
    df = _vers_dataframe(experiences)

    nb = len(experiences)
    if not df.empty and df["Accuracy"].notna().any():
        best_idx = df["Accuracy"].idxmax()
        acc_str = f"{df.loc[best_idx, 'Accuracy']:.4f}"
        nom_str = df.loc[best_idx, "Nom"]
    else:
        acc_str = "—"
        nom_str = "—"

    kpi_nb = f"### {nb} expérience{'s' if nb != 1 else ''}"
    kpi_acc = f"### Meilleure accuracy\n**{acc_str}**"
    kpi_nom = f"### Meilleur modèle\n**{nom_str}**"

    df_display = df[COLS_AFFICHAGE] if not df.empty else pd.DataFrame(columns=COLS_AFFICHAGE)
    fig_overview = _graphique_overview(df)

    ids = df["ID"].tolist() if not df.empty else []
    premier_id = ids[0] if ids else None
    quatre_premiers = ids[:4] if ids else []

    return (
        experiences,
        kpi_nb,
        kpi_acc,
        kpi_nom,
        df_display,
        fig_overview,
        gr.update(choices=ids, value=premier_id),   # dropdown détails
        gr.update(choices=ids, value=quatre_premiers),  # dropdown comparaison
    )


# ---------------------------------------------------------------------------
# Handler : Détails d'une expérience
# ---------------------------------------------------------------------------

def afficher_details(id_experience: str | None, experiences: list[dict]):
    """Met à jour tous les composants de l'onglet Détails."""
    vide = ("—", "—", "—", "—", None, pd.DataFrame(), None, "—", "—", None, "")

    if not id_experience or not experiences:
        return vide

    exp = next((e for e in experiences if e["experiment_id"] == id_experience), None)
    if exp is None:
        return vide

    test = exp.get("test_metrics", {})
    cv = exp.get("cv_results", {})
    chemin_exp = Path(exp.get("_path", ""))

    acc_str = f"{test.get('accuracy', 0):.4f}" if test.get("accuracy") is not None else "—"
    f1_str = f"{test.get('macro_f1', 0):.4f}" if test.get("macro_f1") is not None else "—"
    wf1_str = f"{test.get('weighted_f1', 0):.4f}" if test.get("weighted_f1") is not None else "—"
    dur_str = f"{exp.get('duration_seconds', 0):.1f}s" if exp.get("duration_seconds") else "—"

    # Métriques par classe
    par_classe = test.get("par_classe", {})
    if par_classe:
        df_pc = pd.DataFrame(par_classe).T.reset_index().rename(columns={"index": "classe"})
        fig_pc = px.bar(
            df_pc.melt(id_vars="classe", value_vars=["precision", "rappel", "f1"]),
            x="classe", y="value", color="variable", barmode="group",
            labels={"value": "Score", "classe": "Classe", "variable": "Métrique"},
            color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
        )
        fig_pc.update_layout(yaxis_range=[0, 1.05], margin={"t": 20})
        df_pc_display = pd.DataFrame(par_classe).T.round(4).reset_index().rename(
            columns={"index": "classe"}
        )
    else:
        fig_pc = go.Figure()
        df_pc_display = pd.DataFrame()

    # Matrice de confusion
    chemin_mc = chemin_exp / "plots" / "matrice_confusion.png"
    img_mc = str(chemin_mc) if chemin_mc.exists() else None

    # Validation croisée
    cv_acc_str = f"{cv.get('mean_accuracy', 0):.4f}" if cv.get("mean_accuracy") is not None else "—"
    cv_std_str = f"{cv.get('std_accuracy', 0):.4f}" if cv.get("std_accuracy") is not None else "—"

    if cv and cv.get("fold_scores"):
        scores = cv["fold_scores"]
        fig_cv = px.bar(
            x=list(range(1, len(scores) + 1)),
            y=scores,
            labels={"x": "Fold", "y": "Accuracy"},
            title="Scores par fold de validation croisée",
        )
        fig_cv.add_hline(
            y=cv["mean_accuracy"], line_dash="dash", line_color="red",
            annotation_text=f"Moyenne : {cv['mean_accuracy']:.4f}",
        )
        fig_cv.update_layout(yaxis_range=[0, 1.05])
    else:
        fig_cv = None

    # Configuration YAML
    chemin_config = chemin_exp / "config.yaml"
    if chemin_config.exists():
        config_str = chemin_config.read_text(encoding="utf-8")
    else:
        import json
        config_str = json.dumps(exp, indent=2, ensure_ascii=False)

    return (
        acc_str, f1_str, wf1_str, dur_str,
        fig_pc, df_pc_display,
        img_mc,
        cv_acc_str, cv_std_str, fig_cv,
        config_str,
    )


# ---------------------------------------------------------------------------
# Handler : Comparaison
# ---------------------------------------------------------------------------

def comparer(ids_selectionnes: list[str], experiences: list[dict]):
    """Met à jour les graphiques de comparaison."""
    vide = (go.Figure(), go.Figure(), pd.DataFrame())

    if not ids_selectionnes or len(ids_selectionnes) < 2 or not experiences:
        return vide

    exps = [e for e in experiences if e["experiment_id"] in ids_selectionnes]
    df_all = _vers_dataframe(experiences)
    df = df_all[df_all["ID"].isin(ids_selectionnes)].copy()

    # Graphique métriques globales
    metriques = ["Accuracy", "Macro F1", "Weighted F1", "CV Accuracy"]
    disponibles = [m for m in metriques if df[m].notna().any()]

    fig_global = go.Figure()
    for _, ligne in df.iterrows():
        fig_global.add_trace(go.Bar(
            name=ligne["Nom"],
            x=disponibles,
            y=[ligne[m] for m in disponibles],
            text=[f"{ligne[m]:.4f}" if pd.notna(ligne[m]) else "N/A" for m in disponibles],
            textposition="outside",
        ))
    fig_global.update_layout(
        barmode="group",
        yaxis_range=[0, 1.15],
        height=400,
        margin={"t": 20},
    )

    # F1 par classe
    toutes_classes: set[str] = set()
    for exp in exps:
        toutes_classes.update(exp.get("test_metrics", {}).get("par_classe", {}).keys())

    lignes_f1: list[dict] = []
    for exp in exps:
        pc = exp.get("test_metrics", {}).get("par_classe", {})
        for cls in sorted(toutes_classes):
            lignes_f1.append({
                "Expérience": exp["experiment_name"],
                "Classe": cls,
                "F1": pc.get(cls, {}).get("f1"),
            })
    df_f1 = pd.DataFrame(lignes_f1)

    if not df_f1.dropna(subset=["F1"]).empty:
        fig_f1 = px.bar(
            df_f1.dropna(subset=["F1"]),
            x="Classe", y="F1", color="Expérience", barmode="group",
        )
        fig_f1.update_layout(yaxis_range=[0, 1.15], margin={"t": 20})
    else:
        fig_f1 = go.Figure()

    # Tableau récap
    df_recap = df[["Nom", "Date", "Accuracy", "Macro F1", "Weighted F1", "CV Accuracy", "Durée (s)"]].copy()

    return fig_global, fig_f1, df_recap


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="Arachne — Classification de tableaux") as demo:

    experiences_state = gr.State([])

    # En-tête
    with gr.Row():
        gr.Markdown("# 🕷 Arachne — Classification de tableaux")
        btn_refresh = gr.Button("🔄 Actualiser", scale=0, variant="secondary")

    # ──────────────────────────────────────────────
    # Onglet 1 : Vue d'ensemble
    # ──────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("Vue d'ensemble"):
            with gr.Row():
                kpi_nb = gr.Markdown("### 0 expériences")
                kpi_acc = gr.Markdown("### Meilleure accuracy\n**—**")
                kpi_nom = gr.Markdown("### Meilleur modèle\n**—**")

            gr.Markdown("---")
            gr.Markdown("### Toutes les expériences")
            df_overview = gr.Dataframe(
                headers=COLS_AFFICHAGE,
                interactive=False,
                wrap=True,
            )
            gr.Markdown("### Comparaison des accuracy")
            plot_overview = gr.Plot()

        # ──────────────────────────────────────────────
        # Onglet 2 : Détails
        # ──────────────────────────────────────────────
        with gr.Tab("Détails"):
            dd_exp = gr.Dropdown(
                label="Sélectionner une expérience",
                choices=[],
                interactive=True,
            )
            with gr.Row():
                txt_acc = gr.Textbox(label="Accuracy", interactive=False)
                txt_f1 = gr.Textbox(label="Macro F1", interactive=False)
                txt_wf1 = gr.Textbox(label="Weighted F1", interactive=False)
                txt_dur = gr.Textbox(label="Durée", interactive=False)

            gr.Markdown("---")

            with gr.Tabs():
                with gr.Tab("Métriques par classe"):
                    plot_pc = gr.Plot()
                    df_pc = gr.Dataframe(interactive=False)

                with gr.Tab("Matrice de confusion"):
                    img_mc = gr.Image(
                        label="Matrice de confusion",
                        interactive=False,
                        show_download_button=False,
                    )

                with gr.Tab("Validation croisée"):
                    with gr.Row():
                        txt_cv_acc = gr.Textbox(label="Accuracy CV", interactive=False)
                        txt_cv_std = gr.Textbox(label="Écart-type", interactive=False)
                    plot_cv = gr.Plot()

                with gr.Tab("Configuration"):
                    code_cfg = gr.Code(language="yaml", interactive=False)

        # ──────────────────────────────────────────────
        # Onglet 3 : Comparaison
        # ──────────────────────────────────────────────
        with gr.Tab("Comparaison"):
            dd_comp = gr.Dropdown(
                label="Sélectionner les expériences à comparer (minimum 2)",
                choices=[],
                multiselect=True,
                interactive=True,
            )
            gr.Markdown("### Métriques globales")
            plot_comp_global = gr.Plot()
            gr.Markdown("### F1 par classe")
            plot_comp_f1 = gr.Plot()
            gr.Markdown("### Tableau récapitulatif")
            df_comp = gr.Dataframe(interactive=False)

    # ──────────────────────────────────────────────
    # Connexion des événements
    # ──────────────────────────────────────────────

    _sorties_rafraichir = [
        experiences_state,
        kpi_nb, kpi_acc, kpi_nom,
        df_overview, plot_overview,
        dd_exp, dd_comp,
    ]

    btn_refresh.click(
        fn=rafraichir_tout,
        inputs=[],
        outputs=_sorties_rafraichir,
    )

    demo.load(
        fn=rafraichir_tout,
        inputs=[],
        outputs=_sorties_rafraichir,
    )

    _sorties_details = [
        txt_acc, txt_f1, txt_wf1, txt_dur,
        plot_pc, df_pc,
        img_mc,
        txt_cv_acc, txt_cv_std, plot_cv,
        code_cfg,
    ]

    dd_exp.change(
        fn=afficher_details,
        inputs=[dd_exp, experiences_state],
        outputs=_sorties_details,
    )

    dd_comp.change(
        fn=comparer,
        inputs=[dd_comp, experiences_state],
        outputs=[plot_comp_global, plot_comp_f1, df_comp],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
