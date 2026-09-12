"""
App Streamlit — Instrument de prédiction de l'Indice de Maturité de la Gouvernance (IMG)
Mémoire : Impact des Audits de Sécurité SI sur l'Amélioration de la Gouvernance

Lancer avec :  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from model_core import (
    build_markov_transition_matrix, markov_forecast_distribution, score_to_state, STATES,
    fit_stage_model, predict_stage, extrapolate_future, loocv_compare, get_candidate_models,
    build_features, collinearity_diagnostic,
)
from style import CUSTOM_CSS, MATURITY_COLORS

plt.rcParams["font.family"] = "DejaVu Sans"
INK, NAVY, SIGNAL, SAGE, AMBER, SLATE = "#0F2138", "#1E3A5F", "#2F6FED", "#178A4C", "#B8860B", "#5B6472"

st.set_page_config(page_title="IMG — Instrument prédictif", page_icon="◆", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">MÉMOIRE · GOUVERNANCE DES SYSTÈMES D'INFORMATION</div>
    <div class="hero-title">Instrument prédictif de l'Indice<br/>de Maturité de la Gouvernance</div>
    <div class="hero-sub">Modèle hybride Machine Learning — Gaussian Process Regression validé en
    Leave-One-Out Cross-Validation sur 7 algorithmes candidats, complété d'une chaîne de Markov
    pour la lecture par niveau de maturité.</div>
    <div class="ladder">
        <div class="ladder-seg" style="background:#C0392B;"></div>
        <div class="ladder-seg" style="background:#B8860B;"></div>
        <div class="ladder-seg" style="background:#178A4C;"></div>
    </div>
    <div class="ladder-label">
        <span>Inexistante / Risquée</span><span>Maîtrisée</span><span>Optimisée</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def train_models(df):
    gp_t1, scaler_t1, cols_t1 = fit_stage_model(df, "T1")
    gp_t2, scaler_t2, cols_t2 = fit_stage_model(df, "T2", extra_features=["T1"])
    P = build_markov_transition_matrix(df)
    cv = loocv_compare(df, target_col="T2")
    return gp_t1, scaler_t1, cols_t1, gp_t2, scaler_t2, cols_t2, P, cv

df = load_data()
gp_t1, scaler_t1, cols_t1, gp_t2, scaler_t2, cols_t2, P, cv_results = train_models(df)

st.markdown("""
<div class="data-flag">
    ⚠ Jeu de données actuellement <strong>synthétique</strong> (10 organisations, calibré sur les agrégats
    du mémoire). Remplacez <code>data.csv</code> par les données réelles — aucune autre modification requise.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Diagnostic de colinéarité — explique pourquoi une variable peut sembler
# ne pas influencer la prédiction (voir aussi l'expander ci-dessous)
# ---------------------------------------------------------------------------
corr_matrix, flagged_pairs = collinearity_diagnostic(df, threshold=0.7)
if flagged_pairs:
    pairs_txt = " · ".join([f"<strong>{a}</strong> ↔ <strong>{b}</strong> (r={r:+.2f})" for a, b, r in flagged_pairs])
    st.markdown(f"""
    <div class="data-flag">
        ⚠ Variables fortement corrélées dans ce jeu de données : {pairs_txt}. Le modèle ne peut pas
        distinguer leurs effets individuels — faire varier l'une d'elles seule peut avoir peu d'impact
        visible sur la prédiction. Détail dans l'expander « Diagnostic des variables » ci-dessous.
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔍 Diagnostic des variables — pourquoi certaines semblent sans effet"):
    st.markdown("""
    Chaque covariable (engagement DG, secteur, budget sécurité, suivi formel) n'influence la
    prédiction que dans la mesure où elle apporte une information que les **autres variables
    n'apportent pas déjà**. Si deux variables évoluent ensemble dans les données (une direction
    engagée alloue aussi plus de budget, par exemple), le modèle ne peut pas savoir laquelle des
    deux cause vraiment la progression — il attribue l'effet à l'une, l'autre paraît alors sans impact.
    """)
    st.markdown("**Matrice de corrélation entre covariables (jeu de données actuel) :**")
    styled_corr = corr_matrix.round(2).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
    st.dataframe(styled_corr, use_container_width=True)
    st.caption("Une corrélation proche de +1 ou −1 entre deux variables indique qu'elles se "
               "\"chevauchent\" statistiquement — c'est ce qui explique qu'une covariable puisse "
               "sembler ne pas peser dans la prédiction. Ce diagnostic doit être refait avec les "
               "données réelles : si les vraies covariables sont moins corrélées, chacune retrouvera "
               "un effet individuel plus visible.")

# ---------------------------------------------------------------------------
# Disposition en deux colonnes : saisie / résultats
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([1, 1.55], gap="large")

with col_input:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Organisation</div><div class="section-caption">Sélectionnez un cas de l\'échantillon ou saisissez un profil</div>', unsafe_allow_html=True)

    mode = st.radio("Source", ["Échantillon", "Saisie manuelle"], horizontal=True, label_visibility="collapsed")

    if mode == "Échantillon":
        org_choice = st.selectbox("Organisation", df["org_id"].tolist())
        row = df[df.org_id == org_choice].iloc[0]
        t0 = float(row.T0); secteur = row.secteur
        engagement_dg = int(row.engagement_dg); budget_secu = float(row.budget_secu)
        suivi_formel = int(row.suivi_formel)
        st.caption(f"Observé — T0 : **{t0:.0f}%** · T1 : **{row.T1:.0f}%** · T2 : **{row.T2:.0f}%**")
    else:
        t0 = st.slider("Score IMG initial — T0 (%)", 0.0, 100.0, 30.0)
        secteur = st.selectbox("Secteur", ["privé", "public"])
        engagement_dg = st.slider("Engagement Direction Générale (0–100)", 0, 100, 50)
        budget_secu = st.slider("Budget sécurité (% budget IT)", 0.0, 10.0, 2.0)
        suivi_formel = int(st.selectbox("Suivi formel", ["Oui", "Non"]) == "Oui")

    predict_btn = st.button("Calculer la trajectoire  →", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Comparaison des modèles ML ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comparaison des modèles</div><div class="section-caption">Validation Leave-One-Out — 7 algorithmes de Machine Learning</div>', unsafe_allow_html=True)

    cv_df = pd.DataFrame(cv_results).T.sort_values("MAE")
    cv_df.columns = ["MAE", "Écart-type"]

    fig_cv, ax_cv = plt.subplots(figsize=(5.4, 3.1), dpi=160)
    names = cv_df.index[::-1]
    values = cv_df["MAE"][::-1].values
    errs = cv_df["Écart-type"][::-1].values
    colors_bar = [SAGE if i == len(values) - 1 else (SIGNAL if v < 3 else AMBER) for i, v in enumerate(values)]
    ax_cv.barh(names, values, xerr=errs, color=colors_bar, height=0.6,
               error_kw={"ecolor": "#B7C0CC", "linewidth": 1.1, "capsize": 2})
    for i, v in enumerate(values):
        ax_cv.text(v + errs[i] + 0.12, i, f"{v:.2f}", va="center", fontsize=8.5, color=NAVY, weight="bold")
    ax_cv.set_xlabel("MAE (points IMG)", fontsize=9, color=SLATE)
    ax_cv.tick_params(axis="y", labelsize=8.8, colors=NAVY)
    ax_cv.tick_params(axis="x", labelsize=8, colors=SLATE)
    ax_cv.spines[["top", "right", "left"]].set_visible(False)
    ax_cv.spines["bottom"].set_color("#E2E5EA")
    ax_cv.set_xlim(0, max(values) * 1.28)
    fig_cv.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig_cv, use_container_width=True)
    plt.close(fig_cv)
    st.caption("Le Gaussian Process (vert) obtient la MAE la plus basse — retenu comme moteur de prédiction.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    if predict_btn or mode == "Échantillon":
        features = {"engagement_dg": engagement_dg, "budget_secu": budget_secu, "suivi_formel": suivi_formel,
                    "secteur_privé": 1 if secteur == "privé" else 0, "T0": t0}
        t1_pred, t1_std = predict_stage(gp_t1, scaler_t1, features, cols_t1)
        features_t2 = dict(features); features_t2["T1"] = t1_pred
        t2_pred, t2_std = predict_stage(gp_t2, scaler_t2, features_t2, cols_t2)
        future = extrapolate_future(t0, t1_pred, t2_pred, t2_std, n_periods=2)

        periods = ["T0", "T1", "T2"] + [f["period"] for f in future]
        scores = [t0, t1_pred, t2_pred] + [f["score"] for f in future]
        stds = [0, t1_std, t2_std] + [f["std"] for f in future]
        measured_flag = [True, True, True, False, False]

        # ---- Cartes métriques ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trajectoire prédite</div><div class="section-caption">Valeurs mesurées (T0–T2) puis projetées (N+1, N+2)</div>', unsafe_allow_html=True)

        mcols = st.columns(5)
        for i, (period, score, std, meas) in enumerate(zip(periods, scores, stds, measured_flag)):
            cls = "measured" if meas else "projected"
            val_cls = "" if meas else "projected-value"
            sub = "mesuré" if meas else f"± {std:.1f} pts"
            with mcols[i]:
                st.markdown(f"""
                <div class="metric-card {cls}">
                    <div class="metric-label">{period}</div>
                    <div class="metric-value {val_cls}">{score:.1f}%</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        # ---- Graphique de trajectoire (matplotlib) ----
        x = list(range(len(periods)))
        upper = [min(100, s + sd) for s, sd in zip(scores, stds)]
        lower = [max(0, s - sd) for s, sd in zip(scores, stds)]

        fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=160)

        # Bandes de fond par niveau de maturité (signature visuelle)
        ax.axhspan(0, 40, color="#C0392B", alpha=0.05, zorder=0)
        ax.axhspan(40, 80, color=AMBER, alpha=0.05, zorder=0)
        ax.axhspan(80, 100, color=SAGE, alpha=0.05, zorder=0)

        # Bande d'incertitude
        ax.fill_between(x, lower, upper, color=AMBER, alpha=0.14, zorder=1, linewidth=0)

        # Segments mesuré / projeté
        ax.plot(x[:3], scores[:3], marker="o", color=NAVY, linewidth=2.6, markersize=8, label="Mesuré (T0–T2)", zorder=3)
        ax.plot(x[2:], scores[2:], marker="o", color=AMBER, linewidth=2.6, markersize=8, linestyle="--", label="Projeté (N+1, N+2)", zorder=3)

        ax.axvline(2, color="#9AA5B1", linestyle=":", linewidth=1.1, zorder=2)
        ax.text(2.06, 6, "au-delà : projection", fontsize=8.5, color=SLATE, style="italic")

        ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=9.5, color=NAVY)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score IMG (%)", fontsize=9.5, color=SLATE)
        ax.tick_params(axis="y", labelsize=8.5, colors=SLATE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#E2E5EA")
        ax.legend(loc="lower right", fontsize=8.8, frameon=False)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Markov ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probabilité de niveau de maturité</div><div class="section-caption">Chaîne de Markov empirique appliquée à partir de T2</div>', unsafe_allow_html=True)

        current_state = score_to_state(t2_pred)
        dist_n1 = markov_forecast_distribution(P, current_state, 1)
        dist_n2 = markov_forecast_distribution(P, current_state, 2)

        badge_class = {0: "badge-risque", 1: "badge-maitrise", 2: "badge-optimise"}
        st.markdown(f'Niveau actuel (T2) : <span class="badge {badge_class[current_state]}">{STATES[current_state]}</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

        bar_colors = ["#C0392B", "#B8860B", "#178A4C"]
        for i, state in enumerate(STATES):
            pct_n1 = dist_n1[i] * 100
            pct_n2 = dist_n2[i] * 100
            st.markdown(f"""
            <div style="margin-bottom:0.85rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#1E3A5F; margin-bottom:0.25rem;">
                    <span style="font-weight:600;">{state}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;">N+1 : {pct_n1:.0f}%  ·  N+2 : {pct_n2:.0f}%</span>
                </div>
                <div style="background:#EDEFF2; border-radius:6px; height:10px; overflow:hidden; position:relative;">
                    <div style="background:{bar_colors[i]}; opacity:0.35; width:{pct_n1:.1f}%; height:100%; position:absolute; border-radius:6px;"></div>
                    <div style="background:{bar_colors[i]}; width:{pct_n2:.1f}%; height:100%; position:absolute; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Barre pleine = probabilité à N+2 · barre claire = probabilité à N+1")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Détail LOOCV du modèle sélectionné ----
        with st.expander("Voir le détail de la validation (Leave-One-Out) du Gaussian Process"):
            feat_df, feature_cols = build_features(df)
            feat_df["T0"] = df["T0"]; feat_df["T1"] = df["T1"]
            from sklearn.model_selection import LeaveOneOut
            from sklearn.preprocessing import StandardScaler
            from model_core import make_gp_pipeline
            X = feat_df.values; y = df["T2"].values; orgs = df["org_id"].values
            loo = LeaveOneOut()
            rows = []
            for train_idx, test_idx in loo.split(X):
                scaler = StandardScaler().fit(X[train_idx])
                gp_fold = make_gp_pipeline(n_features=X.shape[1])
                gp_fold.fit(scaler.transform(X[train_idx]), y[train_idx])
                pred = gp_fold.predict(scaler.transform(X[test_idx]))[0]
                rows.append({"Organisation testée": orgs[test_idx][0], "T2 réel": y[test_idx][0],
                             "T2 prédit": round(pred, 1), "Erreur absolue": round(abs(pred - y[test_idx][0]), 2)})
            detail_df = pd.DataFrame(rows)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            st.caption(f"MAE = moyenne des erreurs ci-dessus = **{detail_df['Erreur absolue'].mean():.2f} points**")

st.markdown("""
<div class="footnote">
Modèle : Gaussian Process Regression (scikit-learn), sélectionné parmi 7 algorithmes en validation
Leave-One-Organization-Out · Chaîne de Markov empirique pour la lecture par niveau de maturité ·
Les projections N+1/N+2 sont présentées avec une incertitude croissante — aucune organisation de
l'échantillon n'ayant été observée au-delà de T2.
</div>
""", unsafe_allow_html=True)
