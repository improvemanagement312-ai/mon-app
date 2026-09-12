"""
Feuille de style de l'application — identité visuelle « instrument d'audit ».
Palette : encre / marine / bleu signal / sauge / ambre / papier / ardoise.
Typographie : Source Serif 4 (titres), Inter (corps), IBM Plex Mono (chiffres).
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0F2138;
    --navy: #1E3A5F;
    --signal: #2F6FED;
    --sage: #178A4C;
    --amber: #B8860B;
    --paper: #F7F8FA;
    --slate: #5B6472;
    --line: #E2E5EA;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Fond général */
.stApp { background: var(--paper); }

/* Cache les éléments Streamlit par défaut superflus */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 1180px; }

/* ---------- Bandeau d'en-tête ---------- */
.hero {
    background: linear-gradient(120deg, var(--ink) 0%, var(--navy) 65%, #274873 100%);
    border-radius: 14px;
    padding: 2.1rem 2.4rem 1.7rem 2.4rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute; top: 0; right: 0; bottom: 0; width: 45%;
    background: repeating-linear-gradient(135deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 2px, transparent 2px, transparent 14px);
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    color: #9FB6D9;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Source Serif 4', serif;
    color: #FFFFFF;
    font-size: 2.05rem;
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    color: #C7D4E8;
    font-size: 0.98rem;
    max-width: 640px;
    line-height: 1.5;
}
.ladder {
    display: flex; gap: 4px; margin-top: 1.2rem; max-width: 420px; position: relative; z-index: 2;
}
.ladder-seg { flex: 1; height: 7px; border-radius: 4px; opacity: 0.9; }
.ladder-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #9FB6D9; margin-top: 0.4rem; display: flex; justify-content: space-between; max-width: 420px; }

/* ---------- Cartes de section ---------- */
.section-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
}
.section-title {
    font-family: 'Source Serif 4', serif;
    color: var(--ink);
    font-size: 1.28rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.section-caption {
    color: var(--slate);
    font-size: 0.88rem;
    margin-bottom: 0.9rem;
}

/* ---------- Cartes métriques ---------- */
.metric-card {
    border-radius: 10px;
    padding: 0.85rem 0.95rem 0.75rem 0.95rem;
    border: 1px solid var(--line);
    background: #FFFFFF;
    text-align: left;
    height: 100%;
}
.metric-card.measured { border-top: 3px solid var(--navy); }
.metric-card.projected { border-top: 3px solid var(--amber); border-style: dashed; border-top-style: solid; background: #FFFBF3; }
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1;
}
.metric-value.projected-value { color: #8A6300; }
.metric-sub { font-size: 0.72rem; color: var(--slate); margin-top: 0.3rem; }

/* ---------- Badges de niveau de maturité ---------- */
.badge {
    display: inline-block; padding: 0.18rem 0.6rem; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-risque { background: #FBEAEA; color: #B3261E; }
.badge-maitrise { background: #FFF4E0; color: #8A6300; }
.badge-optimise { background: #E7F5EC; color: var(--sage); }

/* ---------- Bandeau d'avertissement données ---------- */
.data-flag {
    background: #FFF4E0; border: 1px solid #F0D9A8; border-radius: 10px;
    padding: 0.7rem 1rem; font-size: 0.85rem; color: #6B4E00; margin-bottom: 1rem;
}

/* ---------- Widgets Streamlit ---------- */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--navy); font-weight: 600;
}
.stButton > button {
    background: var(--signal); color: white; border-radius: 8px; border: none;
    font-family: 'Inter', sans-serif; font-weight: 600; padding: 0.55rem 1.4rem;
}
.stButton > button:hover { background: var(--navy); color: white; }

section[data-testid="stSidebar"] {
    background: #FFFFFF; border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* Dataframes */
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }

/* Footer note */
.footnote {
    color: var(--slate); font-size: 0.78rem; text-align: center; margin-top: 1.4rem;
    padding-top: 1rem; border-top: 1px solid var(--line); line-height: 1.6;
}
</style>
"""

MATURITY_COLORS = {"Inexistante/Risquée": "#C0392B", "Maîtrisée": "#B8860B", "Optimisée": "#178A4C"}
