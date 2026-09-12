"""
Modèle prédictif de l'Indice de Maturité de la Gouvernance (IMG).

Architecture hybride :
  1. Modèle structurel (chaîne de Markov à 3 états) — capture la dynamique
     qualitative de maturation observée dans le cadre théorique (COBIT/ISO/NIST).
  2. Couche Machine Learning (Gaussian Process Regression) — prédit le score
     IMG continu à partir du score initial et des covariables organisationnelles,
     avec quantification d'incertitude native (essentiel avec n=10).

Validation : Leave-One-Organization-Out Cross-Validation (LOO-CV), seule
méthode défendable statistiquement avec un aussi petit échantillon.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

STATES = ["Inexistante/Risquée", "Maîtrisée", "Optimisée"]


def score_to_state(score):
    if score < 40:
        return 0
    elif score < 80:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# 1. Préparation des features
# ---------------------------------------------------------------------------

def build_features(df):
    """Encode les covariables organisationnelles en matrice numérique."""
    X = df.copy()
    X["secteur_privé"] = (X["secteur"] == "privé").astype(int)
    feature_cols = ["engagement_dg", "budget_secu", "suivi_formel", "secteur_privé"]
    return X[feature_cols], feature_cols


# ---------------------------------------------------------------------------
# 2. Gaussian Process Regression (couche ML)
# ---------------------------------------------------------------------------

def make_gp_pipeline(n_features=5):
    """
    Noyau RBF isotrope (une seule longueur de portée partagée par toutes les
    variables standardisées) + bruit blanc. Choisi après comparaison empirique :
    plus précis en LOOCV qu'une version ARD (longueur par variable) avec
    seulement 9 organisations en entraînement — l'ARD, plus flexible, a
    besoin de plus de données pour ne pas sur-ajuster ses 6 hyperparamètres.
    Conséquence : le modèle peut sous-représenter l'effet individuel d'une
    variable fortement corrélée à une autre (voir diagnostic de colinéarité
    dans l'application) — c'est un compromis assumé, pas un oubli.
    """
    kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=42)


def fit_stage_model(df, target_col, extra_features=None):
    """
    Entraîne un GP prédisant `target_col` à partir de T0 + covariables
    (+ éventuellement une variable intermédiaire déjà prédite, ex. T1 pour prédire T2).
    Retourne le modèle entraîné + le scaler.
    """
    feat_df, feature_cols = build_features(df)
    feat_df = feat_df.copy()
    feat_df["T0"] = df["T0"]
    if extra_features:
        for col in extra_features:
            feat_df[col] = df[col]

    X = feat_df.values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gp = make_gp_pipeline(n_features=X.shape[1])
    gp.fit(X_scaled, y)

    return gp, scaler, list(feat_df.columns)


def predict_stage(gp, scaler, feature_row_dict, columns):
    X = np.array([[feature_row_dict.get(c, 0.0) for c in columns]])  # fallback 0 si clé manquante
    X_scaled = scaler.transform(X)
    mean, std = gp.predict(X_scaled, return_std=True)
    return float(mean[0]), float(std[0])


# ---------------------------------------------------------------------------
# 3. Chaîne de Markov (modèle structurel)
# ---------------------------------------------------------------------------

def build_markov_transition_matrix(df):
    """Estime la matrice de transition empirique à partir des 20 transitions
    observées (T0->T1 et T1->T2 sur les 10 organisations)."""
    transitions = np.zeros((3, 3))
    for _, row in df.iterrows():
        s0, s1, s2 = score_to_state(row.T0), score_to_state(row.T1), score_to_state(row.T2)
        transitions[s0, s1] += 1
        transitions[s1, s2] += 1

    # Normalisation ligne par ligne (avec lissage de Laplace pour éviter les zéros absolus)
    P = (transitions + 0.5) / (transitions.sum(axis=1, keepdims=True) + 1.5)
    return P


def markov_forecast_distribution(P, current_state, n_steps):
    """Distribution de probabilité sur les états après n_steps transitions."""
    dist = np.zeros(3)
    dist[current_state] = 1.0
    for _ in range(n_steps):
        dist = dist @ P
    return dist


# ---------------------------------------------------------------------------
# 4. Validation croisée LOO — comparaison de 7 modèles de Machine Learning
# ---------------------------------------------------------------------------

def get_candidate_models(n_features=5):
    """
    Retourne le dictionnaire des modèles ML candidats pour prédire l'IMG.
    Tous sont de vrais algorithmes de Machine Learning supervisé (pas de
    modèle purement statistique/heuristique dans cette liste).
    """
    return {
        # Méthodes à noyau (kernel methods) — bien adaptées à n petit
        "Gaussian Process": make_gp_pipeline(n_features=n_features),
        "SVR (noyau RBF)": SVR(kernel="rbf", C=10.0, epsilon=1.0, gamma="scale"),

        # Régression linéaire régularisée / bayésienne
        "Ridge": Ridge(alpha=1.0),
        "Bayesian Ridge": BayesianRidge(),

        # Méthode à base d'instances (instance-based learning)
        "k-NN (k=3)": KNeighborsRegressor(n_neighbors=3),

        # Méthodes d'ensemble (nécessitent plus de données — inclus pour comparaison)
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
    }


def loocv_compare(df, target_col="T2"):
    feat_df, feature_cols = build_features(df)
    feat_df["T0"] = df["T0"]
    if target_col == "T2":
        feat_df["T1"] = df["T1"]
    X = feat_df.values
    y = df[target_col].values

    loo = LeaveOneOut()
    model_names = list(get_candidate_models(n_features=X.shape[1]).keys())
    results = {name: [] for name in model_names}

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        models = get_candidate_models(n_features=X.shape[1])  # ré-instancier à chaque pli (évite les fuites d'état)
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)[0]
            results[name].append(abs(pred - y_test[0]))

    return {model: {"MAE": float(np.mean(errs)), "std": float(np.std(errs))} for model, errs in results.items()}


def extrapolate_future(t0, t1, t2, std_t2, n_periods=2, period_len=1.0):
    """Projection N+1/N+2 au-dela de T2, avec incertitude croissante."""
    t_t2 = 2.0
    r = (t2 - t0) / np.log(1 + t_t2)

    forecasts = []
    current_t = t_t2
    current_std = std_t2
    for step in range(1, n_periods + 1):
        current_t += period_len
        proj = t0 + r * np.log(1 + current_t)
        proj = max(0.0, min(99.5, proj))
        current_std = current_std * 1.4
        forecasts.append({"period": "N+" + str(step), "score": round(float(proj), 1),
                           "std": round(float(current_std), 1)})
    return forecasts


# ---------------------------------------------------------------------------
# 5. Diagnostic de colinéarité entre covariables
# ---------------------------------------------------------------------------

def collinearity_diagnostic(df, threshold=0.7):
    """
    Calcule la matrice de corrélation entre les covariables organisationnelles
    et repère les paires fortement corrélées (|r| > threshold). Une variable
    fortement corrélée à une autre verra son effet individuel sous-estimé par
    n'importe quel modèle de régression (ML ou statistique) : le modèle ne
    peut pas distinguer laquelle des deux cause réellement la variation
    observée. Utile pour expliquer pourquoi faire varier une covariable dans
    l'application peut avoir peu d'effet visible sur la prédiction.
    """
    feat_df, _ = build_features(df)
    corr = feat_df.corr()
    flagged = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                flagged.append((cols[i], cols[j], round(float(r), 2)))
    return corr, flagged


if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    print("=== Validation croisée LOO (prédiction de T2) ===")
    comparison = loocv_compare(df, target_col="T2")
    for model, metrics in comparison.items():
        print(f"  {model:15s} MAE = {metrics['MAE']:.2f} pts  (std={metrics['std']:.2f})")

    print("\n=== Matrice de transition de Markov (empirique) ===")
    P = build_markov_transition_matrix(df)
    print(pd.DataFrame(P, index=STATES, columns=STATES).round(3))

    print("\n=== Exemple de prédiction pour une organisation ===")
    gp_t1, scaler_t1, cols_t1 = fit_stage_model(df, "T1")
    gp_t2, scaler_t2, cols_t2 = fit_stage_model(df, "T2", extra_features=["T1"])

    example = {"engagement_dg": 60, "budget_secu": 2.0, "suivi_formel": 1, "secteur_privé": 1, "T0": 35}
    t1_pred, t1_std = predict_stage(gp_t1, scaler_t1, example, cols_t1)
    print(f"  T1 prédit : {t1_pred:.1f}% (+/- {t1_std:.1f})")

    example["T1"] = t1_pred
    t2_pred, t2_std = predict_stage(gp_t2, scaler_t2, example, cols_t2)
    print(f"  T2 prédit : {t2_pred:.1f}% (+/- {t2_std:.1f})")
