# 🍷 Wine Quality Prediction

> Prédiction de la qualité du vin avec Machine Learning — Application desktop interactive construite avec Random Forest et CustomTkinter.


## Description

Ce projet prédit la **qualité d'un vin** (score de 0 à 10) à partir de ses caractéristiques physico-chimiques, en utilisant un modèle **Random Forest Regressor** optimisé contre le surapprentissage.

L'application inclut une interface graphique complète permettant de :
- saisir les paramètres d'un vin et obtenir une prédiction instantanée,
- visualiser les performances du modèle,
- explorer interactivement le dataset.



##  Aperçu

| Formulaire de prédiction | Visualisations |
|:---:|:---:|
| Saisie des 12 paramètres physico-chimiques | Importance des variables, corrélations, résidus |


##  Structure du projet


projet_wine_quality/
│
├── entrainement.py              # Script d'entraînement du modèle
├── app.py                       # Application desktop (CustomTkinter)
├── wine-quality-white-and-red.csv  # Dataset (vins blancs + rouges)
│
├── model.pkl                    # Modèle entraîné (généré)
├── scaler.pkl                   # StandardScaler (généré)
├── feature_names.pkl            # Noms des features (généré)
├── model_metrics.json           # Métriques du modèle (généré)
├── feature_importance.csv       # Importance des variables (généré)
└── model_analysis.png           # Graphiques d'analyse (généré)


##  Dataset

- **Source :** [UCI ML Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Taille :** 6 497 lignes × 13 colonnes
- **Types :** Vins rouges + vins blancs fusionnés
- **Variable cible :** `quality` (score entier de 3 à 9)

| Variable | Description | Exemple |
|---|---|---|
| `fixed acidity` | Acidité fixe | 7.0 |
| `volatile acidity` | Acidité volatile | 0.35 |
| `citric acid` | Acide citrique | 0.30 |
| `residual sugar` | Sucre résiduel | 3.0 |
| `chlorides` | Chlorures | 0.08 |
| `free sulfur dioxide` | SO₂ libre | 35 |
| `total sulfur dioxide` | SO₂ total | 138 |
| `density` | Densité | 0.996 |
| `pH` | pH | 3.3 |
| `sulphates` | Sulfates | 0.65 |
| `alcohol` | Taux d'alcool (%) | 10.5 |
| `type` | Type de vin (0=Blanc, 1=Rouge) | 0 |



##  Modèle

### Algorithme : Random Forest Regressor

Hyperparamètres optimisés pour réduire le surapprentissage :

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,          # Limite la profondeur des arbres
    min_samples_split=10,  # Empêche la fragmentation excessive
    min_samples_leaf=4,    # Feuilles avec minimum 4 échantillons
    max_features='sqrt',   # Sélection aléatoire des features
    random_state=42
)

### Résultats

| Métrique | Train | Test |
|---|---|---|
| R² | 0.6004 | 0.4570 |
| RMSE | 0.5563 | 0.6230 |
| MAE | 0.4302 | 0.4916 |
| MAPE | 7.71% | 8.69% |

**Validation croisée 5-Fold :**
- R² moyen : `0.4133 ± 0.0230`
- RMSE moyen : `0.6739 ± 0.0168`

**Ratio de surapprentissage :** `0.1200` →  **Bonne généralisation**

### Top variables les plus impactantes

| Variable | Importance |
|---|---|
| alcohol | 0.2374 |
| volatile acidity | 0.1253 |
| density | 0.1158 |
| chlorides | 0.0810 |
| free sulfur dioxide | 0.0742 |



##  Installation & Lancement

### Prérequis

```bash
Python 3.8+
```

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn customtkinter
```

### 3. Entraîner le modèle

```bash
python entrainement.py
```

> Génère automatiquement : `model.pkl`, `scaler.pkl`, `feature_names.pkl`, `model_metrics.json`, `feature_importance.csv`, `model_analysis.png`

### 4. Lancer l'application

```bash
python app.py


## Fonctionnalités de l'application

| Fonctionnalité | Description |
|---|---|
|  **Prédiction** | Saisie des 12 paramètres → score prédit avec intervalle de confiance |
|  **Visualisations** | Importance des variables, corrélations, performance Train/Test, résidus, distribution |
|  **Explorateur de données** | Affichage interactif du dataset (début, fin, échantillon aléatoire) |
| ↻ **Reset** | Réinitialisation rapide des champs avec les valeurs par défaut |

### Interprétation des scores

| Score | Statut |
|---|---|
| ≥ 7 |  **EXCELLENT** |
| ≥ 6 |  **BON** |
| ≥ 5 |  **ACCEPTABLE** |
| < 5 |  **FAIBLE** |



## Technologies utilisées

- **Python** — langage principal
- **scikit-learn** — Random Forest, StandardScaler, métriques, validation croisée
- **Pandas / NumPy** — manipulation et préparation des données
- **Matplotlib** — visualisations intégrées dans l'interface
- **CustomTkinter** — interface graphique moderne (thème dark)


##  Auteur

**Bosco** — Étudiant en L3 IA & Big Data  
Grand École de l'Innovation Technologique (GE-IT) — Antananarivo, Madagascar



.
