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

# Palette cohérente
C_BLUE   = "#4C72B0"
C_GREEN  = "#55A868"
C_ORANGE = "#DD8452"
C_RED    = "#C44E52"
C_PURPLE = "#8172B2"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font=dict(family="Inter, system-ui, sans-serif", color="#2d3748"),
    margin=dict(l=16, r=16, t=28, b=16),
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── Fond général ───────────────────────────────────────────────────────── */
.gradio-container { background: #f0f2f6 !important; font-family: Inter, system-ui, sans-serif; }
footer { display: none !important; }

/* ── En-tête ────────────────────────────────────────────────────────────── */
.arachne-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 14px;
    padding: 24px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(15, 52, 96, 0.35);
    margin-bottom: 4px;
}
.arachne-header h1 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
.arachne-header .subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 13px;
    margin-top: 2px;
}

/* ── Bouton Actualiser ──────────────────────────────────────────────────── */
.btn-refresh button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s;
}
.btn-refresh button:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* ── KPI Cards (vue d'ensemble) ─────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 22px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 4px solid #4C72B0;
    text-align: center;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.12); }
.kpi-label {
    color: #718096;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.kpi-value {
    color: #1a202c;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
}
.kpi-card.green  { border-top-color: #55A868; }
.kpi-card.orange { border-top-color: #DD8452; }

/* ── Metric Cards (onglet Détails) ──────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #4C72B0;
}
.metric-card.green  { border-left-color: #55A868; }
.metric-card.orange { border-left-color: #DD8452; }
.metric-card.red    { border-left-color: #C44E52; }
.metric-label {
    color: #718096;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}
.metric-value {
    color: #1a202c;
    font-size: 1.65rem;
    font-weight: 700;
}

/* ── CV Cards (onglet Validation croisée) ───────────────────────────────── */
.cv-row {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
}
.cv-card {
    flex: 1;
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #8172B2;
    text-align: center;
}
.cv-card.std { border-left-color: #CCB974; }

/* ── Titres de sections ─────────────────────────────────────────────────── */
.section-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #4C72B0 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 2px solid #e8f0fe;
    padding-bottom: 6px;
    margin: 20px 0 10px !important;
}

/* ── Onglets ────────────────────────────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    background: transparent !important;
}
.tabs > .tab-nav > button {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #718096 !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color 0.15s, background 0.15s;
}
.tabs > .tab-nav > button:hover { color: #4C72B0 !important; background: #f0f2f6 !important; }
.tabs > .tab-nav > button.selected {
    color: #4C72B0 !important;
    border-bottom: 2px solid #4C72B0 !important;
    background: #ffffff !important;
}

/* ── Tableaux ───────────────────────────────────────────────────────────── */
.dataframe-container { border-radius: 10px !important; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
table thead tr { background: #f0f2f6 !important; }
table thead th { font-weight: 700 !important; color: #4a5568 !important; font-size: 12px !important; }

/* ── Dropdown ───────────────────────────────────────────────────────────── */
.gr-form, label { font-weight: 600 !important; color: #4a5568 !important; }
"""


# ---------------------------------------------------------------------------
# Helpers HTML
# ---------------------------------------------------------------------------

def _kpi(label: str, value: str, style: str = "") -> str:
    return f"""
    <div class="kpi-card {style}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>"""


def _metric(label: str, value: str, style: str = "") -> str:
    return f"""
    <div class="metric-card {style}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>"""


def _cv_html(acc: str, std: str) -> str:
    return f"""
    <div class="cv-row">
        <div class="cv-card">
            <div class="metric-label">Accuracy moyenne</div>
            <div class="metric-value">{acc}</div>
        </div>
        <div class="cv-card std">
            <div class="metric-label">Écart-type</div>
            <div class="metric-value">{std}</div>
        </div>
    </div>"""


def _section(titre: str) -> str:
    return f'<p class="section-title">{titre}</p>'


# ---------------------------------------------------------------------------
# Helpers données / graphiques
# ---------------------------------------------------------------------------

def _charger_experiences() -> list[dict]:
    return charger_toutes_experiences(REPERTOIRE_MODELES)


def _vers_dataframe(experiences: list[dict]) -> pd.DataFrame:
    lignes = []
    for exp in experiences:
        test = exp.get("test_metrics", {})
        cv   = exp.get("cv_results", {})
        lignes.append({
            "ID":           exp.get("experiment_id", ""),
            "Nom":          exp.get("experiment_name", ""),
            "Date":         exp.get("timestamp", "")[:19].replace("T", " "),
            "Accuracy":     test.get("accuracy"),
            "Macro F1":     test.get("macro_f1"),
            "Weighted F1":  test.get("weighted_f1"),
            "CV Accuracy":  cv.get("mean_accuracy"),
            "CV Écart-type":cv.get("std_accuracy"),
            "Durée (s)":    exp.get("duration_seconds"),
            "Statut":       exp.get("status", ""),
            "_path":        exp.get("_path", ""),
        })
    cols = ["ID", "Nom", "Date", "Accuracy", "Macro F1", "Weighted F1",
            "CV Accuracy", "CV Écart-type", "Durée (s)", "Statut", "_path"]
    return pd.DataFrame(lignes) if lignes else pd.DataFrame(columns=cols)


def _fig_overview(df: pd.DataFrame) -> go.Figure:
    if df.empty or df["Accuracy"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            **_PLOTLY_LAYOUT,
            height=200,
            annotations=[{"text": "Aucune donnée", "xref": "paper", "yref": "paper",
                           "showarrow": False, "font": {"size": 18, "color": "#a0aec0"}}],
        )
        return fig
    df_tri = df.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=True)
    fig = px.bar(
        df_tri, x="Accuracy", y="Nom", orientation="h",
        color="Accuracy", color_continuous_scale=[[0, "#bed3f3"], [1, C_BLUE]],
        text="Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=max(320, len(df_tri) * 52),
        showlegend=False,
        xaxis_range=[0, 1.1],
        coloraxis_showscale=False,
    )
    return fig


def _fig_par_classe(par_classe: dict) -> go.Figure:
    df = pd.DataFrame(par_classe).T.reset_index().rename(columns={"index": "classe"})
    df_melt = df.melt(id_vars="classe", value_vars=["precision", "rappel", "f1"])
    fig = px.bar(
        df_melt, x="classe", y="value", color="variable", barmode="group",
        labels={"value": "Score", "classe": "Classe", "variable": "Métrique"},
        color_discrete_map={"precision": C_BLUE, "rappel": C_ORANGE, "f1": C_GREEN},
    )
    fig.update_layout(**_PLOTLY_LAYOUT, yaxis_range=[0, 1.08])
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    return fig


def _fig_cv(scores: list[float], moyenne: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(range(1, len(scores) + 1)),
        y=scores,
        marker_color=[C_BLUE if s >= moyenne else C_ORANGE for s in scores],
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.add_hline(
        y=moyenne, line_dash="dash", line_color=C_RED, line_width=2,
        annotation_text=f"μ = {moyenne:.4f}",
        annotation_font_color=C_RED,
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        xaxis_title="Fold",
        yaxis_title="Accuracy",
        yaxis_range=[max(0, min(scores) - 0.05), 1.05],
        showlegend=False,
    )
    return fig


def _fig_comp_global(df: pd.DataFrame, disponibles: list[str]) -> go.Figure:
    palette = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE,
               "#64B5CD", "#CCB974", "#8172B2"]
    fig = go.Figure()
    for i, (_, ligne) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            name=ligne["Nom"],
            x=disponibles,
            y=[ligne[m] for m in disponibles],
            text=[f"{ligne[m]:.4f}" if pd.notna(ligne[m]) else "N/A" for m in disponibles],
            textposition="outside",
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        barmode="group",
        yaxis_range=[0, 1.2],
        height=420,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def _fig_f1_classe(df_f1: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df_f1.dropna(subset=["F1"]),
        x="Classe", y="F1", color="Expérience", barmode="group",
        color_discrete_sequence=[C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE],
    )
    fig.update_layout(**_PLOTLY_LAYOUT, yaxis_range=[0, 1.15],
                      legend=dict(orientation="h", y=-0.2))
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# Handler : Vue d'ensemble + rafraîchissement global
# ---------------------------------------------------------------------------

def rafraichir_tout():
    experiences = _charger_experiences()
    df = _vers_dataframe(experiences)

    nb = len(experiences)
    if not df.empty and df["Accuracy"].notna().any():
        best_idx = df["Accuracy"].idxmax()
        acc_val  = f"{df.loc[best_idx, 'Accuracy']:.4f}"
        nom_val  = df.loc[best_idx, "Nom"]
    else:
        acc_val = nom_val = "—"

    kpi_nb  = _kpi("Expériences", str(nb))
    kpi_acc = _kpi("Meilleure accuracy", acc_val, "green")
    kpi_nom = _kpi("Meilleur modèle", nom_val, "orange")

    df_display   = df[COLS_AFFICHAGE] if not df.empty else pd.DataFrame(columns=COLS_AFFICHAGE)
    fig_overview = _fig_overview(df)

    ids           = df["ID"].tolist() if not df.empty else []
    premier_id    = ids[0] if ids else None
    quatre_premiers = ids[:4] if ids else []

    return (
        experiences,
        kpi_nb, kpi_acc, kpi_nom,
        df_display,
        fig_overview,
        gr.update(choices=ids, value=premier_id),
        gr.update(choices=ids, value=quatre_premiers),
    )


# ---------------------------------------------------------------------------
# Handler : Détails
# ---------------------------------------------------------------------------

def afficher_details(id_experience: str | None, experiences: list[dict]):
    vide_html = _metric("—", "—")
    vide = (vide_html, vide_html, vide_html, vide_html,
            go.Figure(), pd.DataFrame(),
            None,
            _cv_html("—", "—"), None,
            "")

    if not id_experience or not experiences:
        return vide

    exp = next((e for e in experiences if e["experiment_id"] == id_experience), None)
    if exp is None:
        return vide

    test       = exp.get("test_metrics", {})
    cv         = exp.get("cv_results", {})
    chemin_exp = Path(exp.get("_path", ""))

    f = lambda v: f"{v:.4f}" if v is not None else "—"

    html_acc  = _metric("Accuracy",     f(test.get("accuracy")))
    html_f1   = _metric("Macro F1",     f(test.get("macro_f1")),    "green")
    html_wf1  = _metric("Weighted F1",  f(test.get("weighted_f1")), "orange")
    html_dur  = _metric("Durée",
                        f"{exp['duration_seconds']:.1f}s" if exp.get("duration_seconds") else "—",
                        "red")

    # Métriques par classe
    par_classe = test.get("par_classe", {})
    fig_pc     = _fig_par_classe(par_classe) if par_classe else go.Figure()
    df_pc      = (pd.DataFrame(par_classe).T.round(4)
                    .reset_index().rename(columns={"index": "classe"})
                  if par_classe else pd.DataFrame())

    # Matrice de confusion
    chemin_mc = chemin_exp / "plots" / "matrice_confusion.png"
    img_mc    = str(chemin_mc) if chemin_mc.exists() else None

    # Validation croisée
    cv_acc = f(cv.get("mean_accuracy"))
    cv_std = f(cv.get("std_accuracy"))
    html_cv = _cv_html(cv_acc, cv_std)
    fig_cv  = (
        _fig_cv(cv["fold_scores"], cv["mean_accuracy"])
        if cv and cv.get("fold_scores") else None
    )

    # Config
    chemin_config = chemin_exp / "config.yaml"
    config_str = (chemin_config.read_text(encoding="utf-8")
                  if chemin_config.exists()
                  else "# config.yaml introuvable")

    return (
        html_acc, html_f1, html_wf1, html_dur,
        fig_pc, df_pc,
        img_mc,
        html_cv, fig_cv,
        config_str,
    )


# ---------------------------------------------------------------------------
# Handler : Comparaison
# ---------------------------------------------------------------------------

def comparer(ids_selectionnes: list[str], experiences: list[dict]):
    vide = (go.Figure(), go.Figure(), pd.DataFrame())
    if not ids_selectionnes or len(ids_selectionnes) < 2 or not experiences:
        return vide

    exps  = [e for e in experiences if e["experiment_id"] in ids_selectionnes]
    df_all = _vers_dataframe(experiences)
    df     = df_all[df_all["ID"].isin(ids_selectionnes)].copy()

    metriques  = ["Accuracy", "Macro F1", "Weighted F1", "CV Accuracy"]
    disponibles = [m for m in metriques if df[m].notna().any()]
    fig_global  = _fig_comp_global(df, disponibles)

    toutes_classes: set[str] = set()
    for exp in exps:
        toutes_classes.update(exp.get("test_metrics", {}).get("par_classe", {}).keys())

    lignes_f1 = [
        {"Expérience": exp["experiment_name"], "Classe": cls,
         "F1": exp.get("test_metrics", {}).get("par_classe", {}).get(cls, {}).get("f1")}
        for exp in exps for cls in sorted(toutes_classes)
    ]
    df_f1  = pd.DataFrame(lignes_f1)
    fig_f1 = _fig_f1_classe(df_f1) if not df_f1.dropna(subset=["F1"]).empty else go.Figure()

    df_recap = df[["Nom", "Date", "Accuracy", "Macro F1", "Weighted F1",
                   "CV Accuracy", "Durée (s)"]].copy()
    return fig_global, fig_f1, df_recap


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Arachne — Classification de tableaux") as demo:

    experiences_state = gr.State([])

    # ── En-tête ─────────────────────────────────────────────────────────────
    with gr.Row(elem_classes="arachne-header"):
        gr.HTML("""
            <div>
                <h1>🕷 Arachne</h1>
                <div class="subtitle">Dashboard de suivi des expériences ML</div>
            </div>
        """)
        btn_refresh = gr.Button("🔄 Actualiser", elem_classes="btn-refresh", scale=0)

    # ── Onglets ──────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Vue d'ensemble ───────────────────────────────────────────────────
        with gr.Tab("📊  Vue d'ensemble"):
            with gr.Row():
                kpi_nb  = gr.HTML(_kpi("Expériences", "0"))
                kpi_acc = gr.HTML(_kpi("Meilleure accuracy", "—", "green"))
                kpi_nom = gr.HTML(_kpi("Meilleur modèle", "—", "orange"))

            gr.HTML(_section("Toutes les expériences"))
            df_overview = gr.Dataframe(
                headers=COLS_AFFICHAGE,
                interactive=False,
                wrap=False,
                elem_classes="dataframe-container",
            )
            gr.HTML(_section("Accuracy par expérience"))
            plot_overview = gr.Plot(show_label=False)

        # ── Détails ──────────────────────────────────────────────────────────
        with gr.Tab("🔍  Détails"):
            dd_exp = gr.Dropdown(
                label="Expérience",
                choices=[],
                interactive=True,
            )
            with gr.Row():
                html_acc  = gr.HTML(_metric("Accuracy", "—"))
                html_f1   = gr.HTML(_metric("Macro F1", "—", "green"))
                html_wf1  = gr.HTML(_metric("Weighted F1", "—", "orange"))
                html_dur  = gr.HTML(_metric("Durée", "—", "red"))

            with gr.Tabs():
                with gr.Tab("📈  Métriques par classe"):
                    plot_pc = gr.Plot(show_label=False)
                    df_pc   = gr.Dataframe(interactive=False,
                                           elem_classes="dataframe-container")

                with gr.Tab("🗂  Matrice de confusion"):
                    img_mc = gr.Image(
                        label=None,
                        interactive=False,
                        show_download_button=True,
                        show_label=False,
                    )

                with gr.Tab("🔁  Validation croisée"):
                    html_cv = gr.HTML(_cv_html("—", "—"))
                    plot_cv = gr.Plot(show_label=False)

                with gr.Tab("⚙️  Configuration"):
                    code_cfg = gr.Code(language="yaml", interactive=False,
                                       show_label=False)

        # ── Comparaison ──────────────────────────────────────────────────────
        with gr.Tab("⚖️  Comparaison"):
            dd_comp = gr.Dropdown(
                label="Expériences à comparer (minimum 2)",
                choices=[],
                multiselect=True,
                interactive=True,
            )
            gr.HTML(_section("Métriques globales"))
            plot_comp_global = gr.Plot(show_label=False)
            gr.HTML(_section("F1 par classe"))
            plot_comp_f1 = gr.Plot(show_label=False)
            gr.HTML(_section("Tableau récapitulatif"))
            df_comp = gr.Dataframe(interactive=False,
                                   elem_classes="dataframe-container")

    # ── Câblage des événements ───────────────────────────────────────────────
    _sorties_rafraichir = [
        experiences_state,
        kpi_nb, kpi_acc, kpi_nom,
        df_overview, plot_overview,
        dd_exp, dd_comp,
    ]

    _sorties_details = [
        html_acc, html_f1, html_wf1, html_dur,
        plot_pc, df_pc,
        img_mc,
        html_cv, plot_cv,
        code_cfg,
    ]

    btn_refresh.click(fn=rafraichir_tout, inputs=[], outputs=_sorties_rafraichir)
    demo.load(fn=rafraichir_tout, inputs=[], outputs=_sorties_rafraichir)

    dd_exp.change(fn=afficher_details,
                  inputs=[dd_exp, experiences_state],
                  outputs=_sorties_details)

    dd_comp.change(fn=comparer,
                   inputs=[dd_comp, experiences_state],
                   outputs=[plot_comp_global, plot_comp_f1, df_comp])


if __name__ == "__main__":
    demo.launch()
