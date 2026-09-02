# Conversion Rate Challenge 



## Contexte

[www.datascienceweekly.org](https://www.datascienceweekly.org) est une newsletter hebdomadaire sur la data science. L'équipe souhaite comprendre le comportement des visiteurs de son site et construire un **modèle prédictif de l'abonnement à la newsletter** (conversion), afin d'identifier des leviers d'action pour améliorer le taux de conversion.

La métrique d'évaluation du concours est le **F1-score**, adaptée au fort déséquilibre de la variable cible (~3 % de conversions).

## Structure du projet

```
.
├── conversionrate.ipynb          # Notebook principal (EDA, préprocessing, modélisation, prédictions)
├── data/
│   ├── conversion_data_train.csv # Données étiquetées (entraînement)
│   └── conversion_data_test.csv  # Données sans étiquette (prédictions à soumettre)
├── conversion_data_test_hh.csv   # Fichier de prédictions soumis au leaderboard
├── requirements.txt
└── README.md
```

## Données

| Colonne               | Type        | Description                                    |
|-----------------------|-------------|------------------------------------------------|
| `country`             | catégorielle| Pays de l'utilisateur (US, China, UK, Germany) |
| `age`                 | numérique   | Âge de l'utilisateur                           |
| `new_user`            | binaire     | 1 si nouvel utilisateur, 0 sinon               |
| `source`              | catégorielle| Canal d'acquisition (Seo, Ads, Direct)         |
| `total_pages_visited` | numérique   | Nombre de pages visitées pendant la session    |
| `converted`           | binaire     | **Cible** — 1 si abonnement à la newsletter    |

## Démarche

1. **EDA** — Analyse exploratoire avec **Plotly** : distributions (âge, pages visitées), taux de conversion par pays / source / type d'utilisateur, matrice de corrélation, détection des valeurs aberrantes.
2. **Préprocessing** — Sélection des features, standardisation des variables numériques (`StandardScaler`) et encodage one-hot des catégorielles (`OneHotEncoder`, `drop='first'`) via un `ColumnTransformer`.
3. **Modélisation** — Régression logistique régularisée (`LogisticRegressionCV`) avec pondération des classes, évaluée au F1-score (train/test stratifié) et matrices de confusion.
4. **Prédictions** — Réentraînement du meilleur modèle sur l'ensemble des données étiquetées, puis génération du fichier de prédictions `.csv` pour le leaderboard.
5. **Analyse & recommandations** — Interprétation des coefficients du modèle pour identifier les leviers d'amélioration du taux de conversion.

## Installation

```bash
# Créer et activer un environnement virtuel (exemple avec conda)
conda create -n conversion-rate python=3.11
conda activate conversion-rate

# Installer les dépendances
pip install -r requirements.txt
```

## Résultats

- Modèle de référence : régression logistique régularisée — F1-score évalué sur un test set stratifié (33 %).
- La variable la plus prédictive est `total_pages_visited`, suivie de `new_user` (effet négatif).
- Les prédictions sont exportées dans `conversion_data_test_hh.csv` pour évaluation indépendante sur le leaderboard.

## Auteur

**Henintsoa HASINAVALONA**
