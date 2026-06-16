# Wine Quality Prediction

Prediction of wine quality using Machine Learning — interactive desktop application built with Random Forest and CustomTkinter.


---

## Description

This project predicts the **quality of a wine** (score from 0 to 10) based on its physicochemical characteristics, using a **Random Forest Regressor** model optimized to reduce overfitting.

The application provides a complete graphical interface to:
- enter wine parameters and get an instant prediction,
- visualize model performance,
- interactively explore the dataset.

---

## Project Structure

```
wine-quality-prediction/
│
├── entrainement.py                 # Model training script
├── app.py                          # Desktop application (CustomTkinter)
├── wine-quality-white-and-red.csv  # Dataset (red + white wines)
│
├── model.pkl                       # Trained model (generated)
├── scaler.pkl                      # StandardScaler (generated)
├── feature_names.pkl               # Feature names (generated)
├── model_metrics.json              # Model metrics (generated)
├── feature_importance.csv          # Feature importances (generated)
└── model_analysis.png              # Analysis charts (generated)
```

---

## Dataset

- **Source:** [UCI ML Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Size:** 6,497 rows x 13 columns
- **Content:** Red and white wines combined
- **Target variable:** `quality` (integer score from 3 to 9)

| Variable | Description | Example |
|---|---|---|
| `fixed acidity` | Fixed acidity | 7.0 |
| `volatile acidity` | Volatile acidity | 0.35 |
| `citric acid` | Citric acid | 0.30 |
| `residual sugar` | Residual sugar | 3.0 |
| `chlorides` | Chlorides | 0.08 |
| `free sulfur dioxide` | Free SO2 | 35 |
| `total sulfur dioxide` | Total SO2 | 138 |
| `density` | Density | 0.996 |
| `pH` | pH | 3.3 |
| `sulphates` | Sulphates | 0.65 |
| `alcohol` | Alcohol (%) | 10.5 |
| `type` | Wine type (0=White, 1=Red) | 0 |

---

## Model

### Algorithm: Random Forest Regressor

Hyperparameters tuned to reduce overfitting:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,          # Limits tree depth
    min_samples_split=10,  # Prevents excessive splitting
    min_samples_leaf=4,    # Minimum 4 samples per leaf
    max_features='sqrt',   # Random feature selection
    random_state=42
)
```

### Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.6004 | 0.4570 |
| RMSE | 0.5563 | 0.6230 |
| MAE | 0.4302 | 0.4916 |
| MAPE | 7.71% | 8.69% |

**5-Fold Cross-Validation:**
- Mean R²: `0.4133 ± 0.0230`
- Mean RMSE: `0.6739 ± 0.0168`

**Overfitting ratio:** `0.1200` — Good generalization

### Most Influential Features

| Feature | Importance |
|---|---|
| alcohol | 0.2374 |
| volatile acidity | 0.1253 |
| density | 0.1158 |
| chlorides | 0.0810 |
| free sulfur dioxide | 0.0742 |

---

## Installation

### Requirements

Python 3.8 or higher.

### Clone the repository

```bash
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn customtkinter
```

### Train the model

```bash
python entrainement.py
```

This generates: `model.pkl`, `scaler.pkl`, `feature_names.pkl`, `model_metrics.json`, `feature_importance.csv`, `model_analysis.png`.

### Launch the application

```bash
python app.py
```

---

## Application Features

| Feature | Description |
|---|---|
| Prediction | Enter the 12 parameters and get a predicted score with confidence interval |
| Visualizations | Feature importance, correlations, Train/Test performance, residuals, distribution |
| Data Explorer | Interactive display of the dataset (first rows, last rows, random sample) |
| Reset | Reset all fields to default values |

### Score Interpretation

| Score | Label |
|---|---|
| >= 7 | Excellent |
| >= 6 | Good |
| >= 5 | Acceptable |
| < 5 | Poor |

---

## Stack

- **Python** — core language
- **scikit-learn** — Random Forest, StandardScaler, metrics, cross-validation
- **Pandas / NumPy** — data manipulation and preprocessing
- **Matplotlib** — charts embedded in the interface
- **CustomTkinter** — modern dark-theme GUI

---

## Author

**Bosco** — L3 AI & Big Data student  
Grand Ecole de l'Innovation Technologique (GE-IT) — Antananarivo, Madagascar

---

## License

This project is licensed under the [MIT License](LICENSE).
