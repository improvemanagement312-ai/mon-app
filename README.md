# Instrument prédictif IMG — Guide de démarrage

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre dans le navigateur (http://localhost:8501).

## Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit (mise en page, cartes, graphiques). |
| `style.py` | Feuille de style CSS (couleurs, typographie, composants visuels). |
| `model_core.py` | Modèles ML (7 algorithmes comparés), Markov, LOOCV, extrapolation. |
| `data.csv` | Données des 10 organisations (synthétique — à remplacer). |
| `requirements.txt` | Dépendances Python. |

## Remplacer par les données réelles

Éditez `data.csv` en conservant les colonnes : `org_id, secteur, engagement_dg, budget_secu, suivi_formel, T0, T1, T2`.
Aucune autre modification n'est nécessaire — modèles et graphiques se recalculent automatiquement.

## Notes de design

- **Palette** : encre `#0F2138` / marine `#1E3A5F` / bleu signal `#2F6FED` / sauge `#178A4C` (optimisée) / ambre `#B8860B` (incertitude/projection).
- **Typographie** : Source Serif 4 (titres), Inter (corps), IBM Plex Mono (valeurs chiffrées) — chargées via Google Fonts dans `style.py`. Sans connexion internet lors de l'exécution, le navigateur retombe sur les polices système ; l'app reste lisible.
- **Signature visuelle** : l'échelle de maturité à 3 paliers (Inexistante/Risquée → Maîtrisée → Optimisée) revient dans le bandeau d'en-tête, en fond du graphique de trajectoire, et dans les badges de niveau — pour ancrer visuellement le cadre théorique du mémoire.
- **Cartes métriques** : bordure pleine marine pour les valeurs mesurées (T0–T2), bordure ambre en tirets pour les valeurs projetées (N+1, N+2) — distinction visuelle immédiate entre mesure et projection.

## Tests effectués (aucune erreur constatée)

- Syntaxe des 3 fichiers Python validée (`py_compile`).
- Pipeline complet (chargement → entraînement → prédiction → Markov) exécuté sans erreur sur les 10 organisations de l'échantillon.
- Cas limites testés : T0 = 0 %, T0 = 100 %, engagement/budget à 0, profil incomplet (variable manquante), et cas de régression (T2 < T0) — toutes les prédictions restent bornées entre 0 et 100 %.
- Balises HTML/CSS de l'interface vérifiées équilibrées (aucune balise `<div>` ou `<span>` mal fermée).
- Diagnostic de colinéarité entre covariables intégré et testé (explique pourquoi certaines variables peuvent sembler avoir peu d'effet — voir l'expander dans l'app).
- Comparaison des 7 modèles ML (Leave-One-Out) reproduite à l'identique : Gaussian Process en tête avec une MAE de 1,84 point.

## Checklist avant la soutenance

- [ ] Remplacer `data.csv` par les données réelles des 10 organisations.
- [ ] Relancer l'app en local (`streamlit run app.py`) et vérifier que rien n'affiche d'erreur.
- [ ] Déployer sur Streamlit Community Cloud (voir ci-dessous) et tester le lien public.
- [ ] Ouvrir le lien 10-15 minutes avant la soutenance (l'app peut être en veille après 12h d'inactivité).
- [ ] Avoir une capture d'écran ou un export PDF de secours en cas de souci de connexion.

## Déploiement en ligne (soutenance)

1. Poussez le dossier sur un dépôt GitHub.
2. Sur [share.streamlit.io](https://share.streamlit.io), connectez le dépôt, pointez sur `app.py`.
3. Lien public généré en quelques minutes — testez-le la veille de la soutenance.

## Pistes d'évolution (hors périmètre actuel)

Trois prolongements possibles sont documentés dans le rapport joint (`rapport_final_travail.pdf`, section 6) :
classification supervisée du niveau de maturité, clustering des organisations (CAH + ACP),
et NLP sur les recommandations d'audit. Aucun n'est implémenté dans cette application —
ce sont des directions pour un travail futur, pas des fonctionnalités actuelles.
