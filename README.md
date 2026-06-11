# 🍷 Wine Quality Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2%2B-purple.svg)](https://customtkinter.tomschimansky.com/)

> Application de prédiction de qualité du vin basée sur le Machine Learning avec interface graphique moderne.

---

## 📋 Table des matières

1. [Aperçu](#-aperçu)
2. [Fonctionnalités](#-fonctionnalités)
3. [Installation](#-installation)
4. [Utilisation](#-utilisation)
5. [Performance du modèle](#-performance-du-modèle)
6. [Variables impactantes](#-variables-impactantes)
7. [Structure du projet](#-structure-du-projet)
8. [Technologies](#-technologies)
9. [Auteur](#-auteur)

---

## 🎯 Aperçu

Cette application utilise un modèle **Random Forest** optimisé pour prédire la qualité des vins (rouges et blancs) sur une échelle de **0 à 10**, à partir de 12 caractéristiques physico-chimiques.

### Variables d'entrée

| Variable | Description | Plage | Défaut |
|----------|-------------|-------|--------|
| fixed acidity | Acidité fixe | 4.0 - 10.0 | 6.8 |
| volatile acidity | Acidité volatile | 0.1 - 1.6 | 0.35 |
| citric acid | Acide citrique | 0.0 - 1.7 | 0.3 |
| residual sugar | Sucre résiduel | 0.5 - 15.5 | 3.0 |
| chlorides | Chlorures | 0.01 - 0.6 | 0.08 |
| free sulfur dioxide | SO₂ libre | 2 - 72 | 35 |
| total sulfur dioxide | SO₂ total | 9 - 440 | 138 |
| density | Densité | 0.990 - 1.039 | 0.996 |
| pH | pH | 2.7 - 4.0 | 3.3 |
| sulphates | Sulfates | 0.3 - 2.0 | 0.65 |
| alcohol | Alcool (%) | 8.0 - 15.0 | 10.5 |
| type | Type de vin | 0=Blanc, 1=Rouge | 0 |

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 🔮 **Prédiction temps réel** | Qualité instantanée avec intervalle de confiance 95% |
| 📊 **5 visualisations** | Importance, corrélations, performance, résidus, distribution |
| 🎨 **Interface moderne** | Design sombre professionnel |
| 🔍 **Exploration des données** | Visualisation du dataset intégré |
| 📈 **Métriques détaillées** | R², RMSE, MAE, MAPE |
| 📉 **Anti-surapprentissage** | Validation croisée 5-Fold |

### Code couleur des résultats

| Qualité | Couleur | Interprétation |
|---------|---------|----------------|
| ≥ 7 | 🟢 Vert | 🌟 EXCELLENT |
| ≥ 6 | 🔵 Bleu | ✓ BON |
| ≥ 5 | 🟡 Jaune | ◐ ACCEPTABLE |
| < 5 | 🔴 Rouge | ✗ FAIBLE |

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/SoloniainaBosco/wine-quality-prediction.git
cd wine-quality-prediction


##  Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation rapide

sur bash
# 1. Cloner le dépôt
git clone https://github.com/SoloniainaBosco/wine-quality-prediction.git
cd wine-quality-prediction

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python entrainement.py
python app.py

### Interface utilisateur
  Remplir le formulaire : Entrez les 12 caractéristiques du vin
  Cliquer sur "PRÉDIRE" : Obtenez la qualité estimée
  Explorer les visualisations : Utilisez les onglets pour analyser les données

### Validation croisée (5-Fold)
| Métrique|	Moyenne|Écart-type
-------------------------------
| R² |	0.4133|	0.0230 |
| RMSE|	0.6739|	0.0168 |
| MAE	| 0.5203|	0.0082  |



