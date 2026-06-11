import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error, explained_variance_score)
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" ENTRAÎNEMENT DU MODÈLE - WINE QUALITY")
print("="*80)

# CHARGEMENT DES DONNÉES
try:
    df = pd.read_csv('wine-quality-white-and-red.csv')
    print(f" Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f" Variables: {list(df.columns)}\n")
except FileNotFoundError:
    print(" Erreur: Le fichier 'wine-quality-white-and-red.csv' n'existe pas!")
    exit()

# ANALYSE EXPLORATOIRE
print(" Statistiques descriptives:")
print(df.describe())

print("\n Distribution de la variable cible (qualité):")
print(df['quality'].value_counts().sort_index())

# CORRÉLATIONS 
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['quality'].sort_values(ascending=False)
print("\n CORRÉLATIONS AVEC LA QUALITÉ:")
print(correlations)

# PRÉPARATION DES DONNÉES
df_encoded = df.copy()
if 'type' in df_encoded.columns:
    df_encoded['type'] = (df_encoded['type'] == 'Red').astype(int)

X = df_encoded.drop('quality', axis=1)
y = df_encoded['quality']

print(f"\n✓ X shape: {X.shape} | y shape: {y.shape}")

# ÉTAPE 5: NORMALISATION 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Données normalisées - Train: {X_train_scaled.shape[0]} | Test: {X_test_scaled.shape[0]}")

print("\n" + "="*80)
print(" OPTIMISATION DU MODÈLE - RÉDUCTION DU SURAPPRENTISSAGE")
print("="*80)

#ÉTAPE 6: ENTRAÎNEMENT AVEC HYPERPARAMÈTRES OPTIMISÉS 
"""
Pour réduire le surapprentissage, on utilise:
- max_depth plus petit (évite d'apprendre les détails du train)
- min_samples_split plus grand (force le modèle à être générique)
- min_samples_leaf plus grand (empêche les feuilles avec 1 échantillon)
- max_features='sqrt' (sélection aléatoire des features)
"""

print("\n Entraînement du modèle avec hyperparamètres optimisés...")
print("   Paramètres anti-surapprentissage:")
print("   - max_depth: 10 (peu profond)")
print("   - min_samples_split: 10 (décourage la fragmentation)")
print("   - min_samples_leaf: 4 (feuilles avec au minimum 4 samples)")
print("   - max_features: 'sqrt' (sélection aléatoire)")
print("   - n_estimators: 100 (nombre d'arbres)")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,              #  Réduit (évite surapprentissage)
    min_samples_split=10,      #  Augmenté
    min_samples_leaf=4,        #  Augmenté
    max_features='sqrt',       #Sélection aléatoire des features
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Modèle entraîné")

# VALIDATION CROISÉE
print("\n📊 Validation croisée 5-Fold...")
scoring_metrics = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=scoring_metrics)

cv_r2_scores = cv_results['test_r2']
cv_rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
cv_mae_scores = -cv_results['test_neg_mae']

print("\n✓ Résultats de la validation croisée:")
print(f"  R² CV:   {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"  RMSE CV: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
print(f"  MAE CV:  {cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}")

# ÉTAPE 8: PRÉDICTIONS ET MÉTRIQUES
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Métriques train
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
ev_train = explained_variance_score(y_train, y_pred_train)

# Métriques test
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
ev_test = explained_variance_score(y_test, y_pred_test)

# ÉTAPE 9: ANALYSE DU SURAPPRENTISSAGE
"""
Formule correcte du ratio de surapprentissage:
Ratio = (RMSE_test - RMSE_train) / RMSE_train

Plus ce ratio est petit, meilleure est la généralisation
"""

overfitting_ratio = (rmse_test - rmse_train) / rmse_train
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test
residual_std = np.std(residuals_test)

print("\n" + "="*80)
print("RÉSULTATS FINAUX")
print("="*80)

print(f"\n{'Métrique':<25} {'Train':<15} {'Test':<15} {'Écart':<12}")
print("-"*80)
print(f"{'R² (Determination)':<25} {r2_train:.4f}         {r2_test:.4f}         {abs(r2_train-r2_test):.4f}")
print(f"{'EV (Explained Variance)':<25} {ev_train:.4f}         {ev_test:.4f}         {abs(ev_train-ev_test):.4f}")
print(f"{'RMSE':<25} {rmse_train:.4f}         {rmse_test:.4f}         {abs(rmse_train-rmse_test):.4f}")
print(f"{'MAE':<25} {mae_train:.4f}         {mae_test:.4f}         {abs(mae_train-mae_test):.4f}")
print(f"{'MAPE (%)':<25} {mape_train*100:.2f}%          {mape_test*100:.2f}%          {abs(mape_train-mape_test)*100:.2f}%")
print("-"*80)

# Analyse du surapprentissage
print(f"\n ANALYSE DU SURAPPRENTISSAGE:")
print(f"   Formule: Ratio = (RMSE_test - RMSE_train) / RMSE_train")
print(f"   Ratio: {overfitting_ratio:.4f} ({overfitting_ratio*100:.2f}%)")

if overfitting_ratio < 0.1:
    print(f"   EXCELLENT - Très bonne généralisation!")
elif overfitting_ratio < 0.2:
    print(f"    BON - Bonne généralisation")
elif overfitting_ratio < 0.5:
    print(f"    ACCEPTABLE - Léger surapprentissage")
else:
    print(f"    PROBLÉMATIQUE - Surapprentissage détecté")

#  ÉTAPE 10: IMPORTANCE DES VARIABLES 
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n⭐ TOP 10 VARIABLES LES PLUS IMPACTANTES:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<20} {row['importance']:.4f}")

# ÉTAPE 11: VISUALISATION DES RÉSULTATS
print("\n📊 Génération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Train vs Test
ax = axes[0, 0]
metrics_names = ['R²', 'RMSE', 'MAE']
train_vals = [r2_train, rmse_train, mae_train]
test_vals = [r2_test, rmse_test, mae_test]
x = np.arange(len(metrics_names))
width = 0.35
ax.bar(x - width/2, train_vals, width, label='Train', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, test_vals, width, label='Test', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylabel('Valeur')
ax.set_title('Train vs Test')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Importance des variables
ax = axes[0, 1]
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'].values, color='#e74c3c')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Variables Impactantes')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Prédictions vs Réalité
ax = axes[1, 0]
ax.scatter(y_test, y_pred_test, alpha=0.5, color='#3498db', s=30)
lim = [y_test.min(), y_test.max()]
ax.plot(lim, lim, 'r--', lw=2, label='Parfait')
ax.set_xlabel('Qualité Réelle')
ax.set_ylabel('Qualité Prédite')
ax.set_title(f'Prédictions vs Réalité (R²: {r2_test:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Distribution des résidus
ax = axes[1, 1]
ax.hist(residuals_test, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zéro')
ax.set_xlabel('Résidus (Erreurs)')
ax.set_ylabel('Fréquence')
ax.set_title(f'Distribution des Résidus (Std: {residual_std:.4f})')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Visualisations sauvegardées: model_analysis.png")
plt.close()

print("\n" + "="*80)
print("💾 SAUVEGARDE DU MODÈLE")
print("="*80)

# Sauvegarde du modèle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ model.pkl")

# Sauvegarde du scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ scaler.pkl")

# Sauvegarde des noms de features
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✓ feature_names.pkl")

# Sauvegarde des métriques
metrics_data = {
    'r2_test': float(r2_test),
    'r2_train': float(r2_train),
    'rmse_test': float(rmse_test),
    'rmse_train': float(rmse_train),
    'mae_test': float(mae_test),
    'mae_train': float(mae_train),
    'mape_test': float(mape_test),
    'mape_train': float(mape_train),
    'ev_test': float(ev_test),
    'ev_train': float(ev_train),
    'overfitting_ratio': float(overfitting_ratio),
    'residual_std': float(residual_std),
    'cv_r2_mean': float(cv_r2_scores.mean()),
    'cv_r2_std': float(cv_r2_scores.std()),
    'cv_rmse_mean': float(cv_rmse_scores.mean()),
    'cv_rmse_std': float(cv_rmse_scores.std()),
    'cv_mae_mean': float(cv_mae_scores.mean()),
    'cv_mae_std': float(cv_mae_scores.std()),
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)
print("   ✓ model_metrics.json")

# Sauvegarde de l'importance des features
feature_importance.to_csv('feature_importance.csv', index=False)
print("   ✓ feature_importance.csv")

print("\n" + "="*80)
print("✅ ENTRAÎNEMENT TERMINÉ - MODÈLE OPTIMISÉ ET PRÊT")
print("="*80)
print("\nFichiers générés:")
print("  1. model.pkl")
print("  2. scaler.pkl")
print("  3. feature_names.pkl")
print("  4. model_metrics.json")
print("  5. feature_importance.csv")
print("  6. model_analysis.png")
print("\nLance maintenant: python app.py")
print("="*80 + "\n")