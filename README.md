# 🍷 Wine Quality Prediction


> Une application intelligente de prédiction de qualité du vin basée sur le Machine Learning, avec une interface graphique moderne et des visualisations interactives.



## Aperçu

Cette application utilise un modèle **Random Forest** optimisé pour prédire la qualité des vins rouges et blancs sur une échelle de **0 à 10**, à partir de 12 caractéristiques physico-chimiques.

### Variables d'entrée

| Variable | Description | Plage typique | Valeur par défaut |
|----------|-------------|---------------|-------------------|
| `fixed acidity` | Acidité fixe | 4.0 - 10.0 | 6.8 |
| `volatile acidity` | Acidité volatile | 0.1 - 1.6 | 0.35 |
| `citric acid` | Acide citrique | 0.0 - 1.7 | 0.3 |
| `residual sugar` | Sucre résiduel | 0.5 - 15.5 | 3.0 |
| `chlorides` | Chlorures | 0.01 - 0.6 | 0.08 |
| `free sulfur dioxide` | SO₂ libre | 2 - 72 | 35 |
| `total sulfur dioxide` | SO₂ total | 9 - 440 | 138 |
| `density` | Densité | 0.990 - 1.039 | 0.996 |
| `pH` | pH | 2.7 - 4.0 | 3.3 |
| `sulphates` | Sulfates | 0.3 - 2.0 | 0.65 |
| `alcohol` | Alcool (%) | 8.0 - 15.0 | 10.5 |
| `type` | Type de vin | 0=Blanc, 1=Rouge | 0 |

##  Fonctionnalités

-  **Prédiction en temps réel** - Obtenez une qualité prédite instantanément avec intervalle de confiance
-  **Visualisations interactives** - Explorez l'importance des variables, les corrélations, les résidus et plus
-  **Interface moderne** - Design sombre professionnel avec CustomTkinter
-  **Métriques détaillées** - R², RMSE, MAE, MAPE et intervalle de confiance (95%)
-  **Exploration des données** - Visualisez et analysez le dataset intégré (4898 échantillons)
-  **Analyse de surapprentissage** - Indicateur visuel de la performance du modèle
- **Validation croisée** - Évaluation robuste avec 5-Fold cross-validation
-  **Interprétation automatique** - Code couleur et message explicatif selon la qualité prédite


