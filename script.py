import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error, explained_variance_score)
from sklearn.model_selection import cross_validate
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# ==================== CHARGEMENT ET PRÉPARATION DES DONNÉES ====================
print("📊 Chargement des données Wine Quality...")

try:
    df = pd.read_csv('wine-quality-white-and-red.csv')
    print("✓ Données chargées depuis wine-quality-white-and-red.csv")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'wine-quality-white-and-red.csv' n'a pas été trouvé")
    exit()

print(f"✓ {len(df)} samples chargés")
print(f"✓ Variables: {list(df.columns)}\n")

if 'type' in df.columns:
    print(f"Répartition: {df['type'].value_counts().to_dict()}\n")

# ==================== ANALYSE EXPLORATOIRE ====================
print("📈 ANALYSE EXPLORATOIRE DES DONNÉES")
print(f"Distribution qualité:\n{df['quality'].value_counts().sort_index()}\n")
print("Statistiques descriptives:")
print(df.describe())

# Corrélations
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['quality'].sort_values(ascending=False)
print("\n🔗 Corrélations avec la qualité:")
print(correlations)

# ==================== PRÉPARATION DU MODÈLE ====================
df_encoded = df.copy()
if 'type' in df_encoded.columns:
    df_encoded['type'] = (df_encoded['type'] == 'Red').astype(int)

X = df_encoded.drop('quality', axis=1)
y = df_encoded['quality']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== OPTIMISATION DU MODÈLE ====================
print("\n" + "="*70)
print("🤖 OPTIMISATION ET ENTRAÎNEMENT DU MODÈLE")
print("="*70)

# Grid Search pour optimiser Random Forest
print("\n⏳ Optimisation Random Forest avec GridSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train_scaled, y_train)

print(f"✓ Meilleurs paramètres: {grid_search.best_params_}")
print(f"✓ Meilleur score CV: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

# ==================== VALIDATION CROISÉE ====================
print("\n📊 VALIDATION CROISÉE (5-Fold)")
scoring_metrics = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=scoring_metrics)

cv_r2_scores = cv_results['test_r2']
cv_rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
cv_mae_scores = -cv_results['test_neg_mae']

print(f"R² CV: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std():.4f})")
print(f"RMSE CV: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")
print(f"MAE CV: {cv_mae_scores.mean():.4f} (+/- {cv_mae_scores.std():.4f})")

# ==================== PRÉDICTIONS ET MÉTRIQUES ====================
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Métriques d'entraînement
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
ev_train = explained_variance_score(y_train, y_pred_train)

# Métriques de test
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
ev_test = explained_variance_score(y_test, y_pred_test)

# Détection du surapprentissage
overfitting_ratio = (rmse_train - rmse_test) / rmse_test
print("\n" + "="*70)
print("📈 MÉTRIQUES DE PERFORMANCE (RÉGRESSION)")
print("="*70)
print(f"\n{'Métrique':<25} {'Train':<15} {'Test':<15} {'Écart':<10}")
print("-"*70)
print(f"{'R² (Coeff. Détermin.)':<25} {r2_train:.4f}         {r2_test:.4f}         {abs(r2_train-r2_test):.4f}")
print(f"{'EV (Explained Variance)':<25} {ev_train:.4f}         {ev_test:.4f}         {abs(ev_train-ev_test):.4f}")
print(f"{'RMSE (Root Mean Sq.)':<25} {rmse_train:.4f}         {rmse_test:.4f}         {abs(rmse_train-rmse_test):.4f}")
print(f"{'MAE (Mean Abs. Error)':<25} {mae_train:.4f}         {mae_test:.4f}         {abs(mae_train-mae_test):.4f}")
print(f"{'MAPE (%)':<25} {mape_train*100:.2f}%          {mape_test*100:.2f}%          {abs(mape_train-mape_test)*100:.2f}%")
print("-"*70)
print(f"⚠️  Ratio surapprentissage: {overfitting_ratio:.4f}")
if overfitting_ratio < 0.2:
    print("✓ Bon généralisation - Pas de surapprentissage significatif")
else:
    print("⚠️  Surapprentissage détecté")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n⭐ TOP 10 VARIABLES IMPACTANTES:")
print(feature_importance.head(10).to_string(index=False))

# Résidus
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test
residual_std = np.std(residuals_test)

print(f"\n📊 ANALYSE DES RÉSIDUS:")
print(f"Moyenne des résidus test: {np.mean(residuals_test):.4f}")
print(f"Écart-type des résidus: {residual_std:.4f}")

# ==================== INTERFACE GRAPHIQUE ====================
class WineQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Wine Quality Prediction System - Advanced")
        self.geometry("1600x950")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.input_vars = {}
        
        # Frame principal avec tabs
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== SECTION GAUCHE: PRÉDICTIONS =====
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        title_label = ctk.CTkLabel(left_frame, text="🍷 PRÉDICTION DE QUALITÉ", 
                                    font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Grille d'entrée
        input_frame = ctk.CTkScrollableFrame(left_frame, height=400)
        input_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        row = 0
        for i, feature in enumerate(X.columns):
            col = i % 2
            if col == 0:
                row += 1
            
            ctk.CTkLabel(input_frame, text=f"{feature}:", font=("Arial", 10)).grid(
                row=row, column=col*2, sticky="w", padx=5, pady=5
            )
            
            entry = ctk.CTkEntry(input_frame, width=100)
            entry.grid(row=row, column=col*2+1, sticky="w", padx=5, pady=5)
            self.input_vars[feature] = entry
            entry.insert(0, str(round(X[feature].mean(), 2)))
        
        # Boutons d'action
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        predict_btn = ctk.CTkButton(button_frame, text="🔮 PRÉDIRE", 
                                     command=self.predict, 
                                     font=("Arial", 12, "bold"),
                                     fg_color="#2ecc71", text_color="black")
        predict_btn.pack(side="left", padx=5)
        
        reset_btn = ctk.CTkButton(button_frame, text="🔄 RÉINITIALISER",
                                  command=self.reset_inputs,
                                  font=("Arial", 11))
        reset_btn.pack(side="left", padx=5)
        
        # Résultat
        self.result_label = ctk.CTkLabel(
            left_frame, text="Qualité prédite: --",
            font=("Arial", 16, "bold"), text_color="#2ecc71"
        )
        self.result_label.pack(pady=10)
        
        # Métriques du modèle
        metrics_frame = ctk.CTkFrame(left_frame, fg_color="#2a2a2a", border_width=2, border_color="#3498db")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        metrics_title = ctk.CTkLabel(metrics_frame, text="📊 MÉTRIQUES GLOBALES", 
                                     font=("Arial", 11, "bold"), text_color="#3498db")
        metrics_title.pack(pady=(8, 5))
        
        metrics_text = f"""R² Test: {r2_test:.4f} | Train: {r2_train:.4f}
EV Test: {ev_test:.4f} | Train: {ev_train:.4f}
RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}
MAPE: {mape_test*100:.2f}%

CV R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}
Surapp.: {overfitting_ratio:.4f} {'✓ OK' if overfitting_ratio < 0.2 else '⚠️ Détecté'}"""
        
        metrics_label = ctk.CTkLabel(metrics_frame, text=metrics_text, 
                                     font=("Arial", 9), justify="left", text_color="#ecf0f1")
        metrics_label.pack(padx=10, pady=10)
        
        # ===== SECTION DROITE: VISUALISATIONS =====
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        # Sélection du graphique
        viz_frame = ctk.CTkFrame(right_frame)
        viz_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(viz_frame, text="Visualisations:", font=("Arial", 12, "bold")).pack(side="left")
        
        self.viz_var = ctk.StringVar(value="correlations")
        options = [
            ("Corrélations", "correlations"),
            ("Importance", "importance"),
            ("Résidus", "residuals"),
            ("Prédictions", "predictions"),
            ("CV Scores", "cv_scores"),
            ("Distribution", "distribution")
        ]
        
        for text, val in options:
            rb = ctk.CTkRadioButton(viz_frame, text=text, variable=self.viz_var, 
                                    value=val, command=self.update_visualization)
            rb.pack(side="left", padx=10)
        
        # Canvas pour matplotlib
        self.canvas_frame = ctk.CTkFrame(right_frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.update_visualization()
    
    def predict(self):
        try:
            input_data = []
            for feature in X.columns:
                val = float(self.input_vars[feature].get())
                input_data.append(val)
            
            input_array = np.array([input_data])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            
            # Intervalle de confiance
            confiance = 0.95
            z = 1.96  # pour 95%
            intervalle = z * residual_std
            
            self.result_label.configure(
                text=f"Qualité: {prediction:.2f}/10 ± {intervalle:.2f}",
                text_color="#2ecc71" if prediction >= 6 else "#e74c3c"
            )
        except ValueError:
            self.result_label.configure(text="❌ Entrées invalides!", text_color="#e74c3c")
    
    def reset_inputs(self):
        for feature in X.columns:
            self.input_vars[feature].delete(0, "end")
            self.input_vars[feature].insert(0, str(round(X[feature].mean(), 2)))
        self.result_label.configure(text="Qualité prédite: --", text_color="#2ecc71")
    
    def update_visualization(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(8, 6.5), dpi=100, facecolor='#212121')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        
        viz_type = self.viz_var.get()
        
        if viz_type == "correlations":
            top_corr = correlations.drop('quality').head(10)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr.values]
            y_pos = np.arange(len(top_corr))
            ax.barh(y_pos, top_corr.values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_corr.index, color='white')
            ax.set_xlabel('Corrélation', color='white')
            ax.set_title('Top 10 Corrélations', color='white', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='white', linestyle='-', linewidth=0.5)
        
        elif viz_type == "importance":
            top_features = feature_importance.head(10)
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'].values, color='#3498db')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'].values, color='white')
            ax.set_xlabel('Importance', color='white')
            ax.set_title('Top 10 Variables Impactantes', color='white', fontsize=12, fontweight='bold')
        
        elif viz_type == "residuals":
            ax.scatter(y_pred_test, residuals_test, alpha=0.6, color='#3498db', s=20)
            ax.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
            ax.fill_between(sorted(y_pred_test), -2*residual_std, 2*residual_std, 
                           alpha=0.2, color='#3498db', label='±2σ')
            ax.set_xlabel('Prédictions', color='white')
            ax.set_ylabel('Résidus', color='white')
            ax.set_title(f'Analyse des Résidus (RMSE: {rmse_test:.4f})', 
                        color='white', fontsize=12, fontweight='bold')
            ax.legend(facecolor='#1f1f1f', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2)
        
        elif viz_type == "predictions":
            ax.scatter(y_test, y_pred_test, alpha=0.5, color='#3498db', s=20)
            lim = [y_test.min(), y_test.max()]
            ax.plot(lim, lim, 'r--', lw=2, label='Parfait')
            ax.set_xlabel('Qualité Réelle', color='white')
            ax.set_ylabel('Qualité Prédite', color='white')
            ax.set_title(f'Prédictions vs Réalité (R²: {r2_test:.4f})', 
                        color='white', fontsize=12, fontweight='bold')
            ax.legend(facecolor='#1f1f1f', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2)
        
        elif viz_type == "cv_scores":
            cv_metrics = ['R²', 'RMSE', 'MAE']
            means = [cv_r2_scores.mean(), cv_rmse_scores.mean(), cv_mae_scores.mean()]
            stds = [cv_r2_scores.std(), cv_rmse_scores.std(), cv_mae_scores.std()]
            
            x_pos = np.arange(len(cv_metrics))
            ax.bar(x_pos, means, yerr=stds, capsize=10, color='#9b59b6', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cv_metrics, color='white')
            ax.set_ylabel('Score', color='white')
            ax.set_title('Validation Croisée (5-Fold)', color='white', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='y')
        
        elif viz_type == "distribution":
            ax.hist(y, bins=20, color='#9b59b6', alpha=0.7, edgecolor='white', label='Données')
            ax.axvline(y.mean(), color='#2ecc71', linestyle='--', linewidth=2, label='Moyenne')
            ax.set_xlabel('Qualité', color='white')
            ax.set_ylabel('Fréquence', color='white')
            ax.set_title('Distribution de la Qualité', color='white', fontsize=12, fontweight='bold')
            ax.legend(facecolor='#1f1f1f', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, axis='y')
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    app = WineQualityApp()
    app.mainloop()