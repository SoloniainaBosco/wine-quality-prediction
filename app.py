import pickle
import json
import numpy as np
import pandas as pd
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Chargement du modèle sauvegardé...")


# Charger les fichiers sauvegardés
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    feature_importance = pd.read_csv('feature_importance.csv')
    print("✓ Tous les fichiers chargés\n")
except FileNotFoundError as e:
    print(f"❌ Erreur: {e}")
    exit()


# Hints pour chaque variable
hints = {
    'fixed acidity': ('Acidité Fixe', '4.0 - 10.0', '6.8'),
    'volatile acidity': ('Acidité Volatile', '0.1 - 1.6', '0.35'),
    'citric acid': ('Acide Citrique', '0.0 - 1.7', '0.3'),
    'residual sugar': ('Sucre Résiduel', '0.5 - 15.5', '3.0'),
    'chlorides': ('Chlorures', '0.01 - 0.6', '0.08'),
    'free sulfur dioxide': ('SO₂ Libre (dioxyde de soufre libre)', '2 - 72', '35'),
    'total sulfur dioxide': ('SO₂ (dioxyde de soufre) Total', '9 - 440', '138'),
    'density': ('Densité', '0.990 - 1.039', '0.996'),
    'pH': ('pH', '2.7 - 4.0', '3.3'),
    'sulphates': ('Sulfates', '0.3 - 2.0', '0.65'),
    'alcohol': ('Alcool %', '8.0 - 15.0', '10.5'),
    'type': ('Type de Vin', '0=Blanc, 1=Rouge', '0')
}

# INTERFACE GRAPHIQUE
class WineQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("🍷 Prédiction Qualité Vin ")
        self.geometry("1600x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configuration de la fenêtre
        self.resizable(True, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.input_vars = {}
        self.data = None  # Pour stocker les données
        self.load_data()  # Charger les données
        
        # Frame principal
        main_container = ctk.CTkFrame(self, fg_color="#0f1419")
        main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # ===== SECTION GAUCHE: FORMULAIRE MODERNE =====
        left_section = self.create_left_section(main_container)
        left_section.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        # ===== SECTION DROITE: VISUALISATIONS =====
        right_section = self.create_right_section(main_container)
        right_section.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
    def load_data(self):
        """Charger le dataset"""
        try:
            self.data = pd.read_csv('wine-quality-white-and-red.csv')
            print("✓ Dataset chargé")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du dataset: {e}")
            self.data = None
    
    def create_left_section(self, parent):
        """Crée la section gauche avec formulaire"""
        left_frame = ctk.CTkFrame(parent, fg_color="transparent")
        left_frame.grid_rowconfigure(2, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        

    # Header avec gradient
        header = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        title = ctk.CTkLabel(header, text="🍷 Prédiction Qualité", 
                            font=("Segoe UI", 24, "bold"), text_color="#00d4ff")
        title.pack(pady=15, padx=20)
        
        subtitle = ctk.CTkLabel(header, text="Analysez vos vins avec l'IA", 
                               font=("Segoe UI", 11), text_color="#888")
        subtitle.pack(pady=(0, 15), padx=20)
        

    # Formulaire scrollable
        form_frame = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        form_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        form_frame.grid_rowconfigure(0, weight=1)
        form_frame.grid_columnconfigure(0, weight=1)
        
        scroll_frame = ctk.CTkScrollableFrame(form_frame, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        row = 0
        for i, feature in enumerate(feature_names):
            col = i % 2
            if col == 0:
                row += 1
            
            label_text, range_text, default_val = hints.get(feature, (feature, '', ''))
            

        # Conteneur pour chaque input
            input_container = ctk.CTkFrame(scroll_frame, fg_color="#242d3d", corner_radius=10)
            input_container.grid(row=row, column=col, sticky="ew", padx=8, pady=10)
            input_container.grid_columnconfigure(0, weight=1)
            

        # Label
            label = ctk.CTkLabel(input_container, text=label_text, 
                               font=("Segoe UI", 10, "bold"), text_color="#00d4ff")
            label.pack(anchor="w", padx=12, pady=(10, 3))
            

        # Range hint
            hint_label = ctk.CTkLabel(input_container, text=f"Range: {range_text}", 
                                    font=("Segoe UI", 8), text_color="#666")
            hint_label.pack(anchor="w", padx=12, pady=(0, 5))
            

        # Entry
            entry = ctk.CTkEntry(input_container, placeholder_text=default_val,
                               fg_color="#1a1f2e", text_color="#00d4ff",
                               border_color="#00d4ff", border_width=1,
                               font=("Segoe UI", 10))
            entry.pack(fill="x", padx=12, pady=(0, 10))
            entry.insert(0, default_val)
            self.input_vars[feature] = entry
        

    # Section des boutons
        button_container = ctk.CTkFrame(left_frame, fg_color="transparent")
        button_container.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        button_container.grid_columnconfigure((0, 1, 2), weight=1)  # Modifié pour 3 colonnes
        
        # Bouton pour explorer les données
        explore_btn = ctk.CTkButton(
            button_container, text="📊 EXPLORER", command=self.show_data_explorer,
            font=("Segoe UI", 12, "bold"), height=45,
            fg_color="#9b59b6", text_color="#ffffff",  # Couleur violette pour différencier
            hover_color="#8e44ad", corner_radius=10
        )
        explore_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        
        predict_btn = ctk.CTkButton(
            button_container, text="🔮 PRÉDIRE", command=self.predict,
            font=("Segoe UI", 13, "bold"), height=45,
            fg_color="#00d4ff", text_color="#0f1419", 
            hover_color="#00a8cc", corner_radius=10
        )
        predict_btn.grid(row=0, column=1, sticky="ew", padx=8)
        
        reset_btn = ctk.CTkButton(
            button_container, text="↻ RESET", command=self.reset_inputs,
            font=("Segoe UI", 12), height=45,
            fg_color="#1a1f2e", text_color="#888",
            border_color="#00d4ff", border_width=1,
            hover_color="#242d3d", corner_radius=10
        )
        reset_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        

    # Résultat
        result_container = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        result_container.grid(row=4, column=0, sticky="ew", pady=(0, 20))
        
        self.result_label = ctk.CTkLabel(
            result_container, text="--",
            font=("Segoe UI", 32, "bold"), text_color="#00d4ff"
        )
        self.result_label.pack(pady=15)
        
        self.interpretation_label = ctk.CTkLabel(
            result_container, text="Entrez les valeurs et cliquez PRÉDIRE",
            font=("Segoe UI", 10), text_color="#888"
        )
        self.interpretation_label.pack(pady=(0, 15))
        

    # Métriques compactes
        metrics_container = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        metrics_container.grid(row=5, column=0, sticky="ew")
        metrics_container.grid_columnconfigure((0, 1, 2), weight=1)
        
        metrics_data = [
            ("R² Score", f"{metrics['r2_test']:.3f}", "#00d4ff"),
            ("RMSE", f"{metrics['rmse_test']:.3f}", "#00d4ff"),
            ("Généralisation", "✓ Excellente", "#00ff00" if metrics['overfitting_ratio'] < 0.2 else "#ffaa00")
        ]
        
        for i, (label, value, color) in enumerate(metrics_data):
            metric_box = ctk.CTkFrame(metrics_container, fg_color="#0f1419", corner_radius=10)
            metric_box.grid(row=0, column=i, sticky="ew", padx=8, pady=15)
            
            ctk.CTkLabel(metric_box, text=label, font=("Segoe UI", 9), 
                        text_color="#888").pack(pady=(8, 2))
            ctk.CTkLabel(metric_box, text=value, font=("Segoe UI", 13, "bold"), 
                        text_color=color).pack(pady=(2, 8))
        
        return left_frame
    
    def create_right_section(self, parent):
        """Crée la section droite avec visualisations"""
        right_frame = ctk.CTkFrame(parent, fg_color="transparent")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        

    # Header visualisations
        viz_header = ctk.CTkFrame(right_frame, fg_color="#1a1f2e", corner_radius=15)
        viz_header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        viz_header.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(viz_header, text="📊 Visualisations", 
                            font=("Segoe UI", 16, "bold"), text_color="#00d4ff")
        title.pack(anchor="w", padx=20, pady=(12, 8))
        

    # Boutons de sélection
        buttons_frame = ctk.CTkFrame(viz_header, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 12))
        
        self.viz_var = ctk.StringVar(value="importance")
        
        options = [
            ("Importance", "importance"),
            ("Corrélations", "correlations"),
            ("Performance", "performance"),
            ("Résidus", "residuals"),
            ("Distribution", "distribution"),
        ]
        
        values = [text for text, _ in options]
        
        seg_btn = ctk.CTkSegmentedButton(
            buttons_frame, values=values, variable=self.viz_var,
            command=self.update_visualization,
            font=("Segoe UI", 9)
        )
        seg_btn.set("Importance")
        seg_btn.pack(side="left", fill="x", expand=True, padx=4)
        
    
     # Canvas pour matplotlib
        canvas_frame = ctk.CTkFrame(right_frame, fg_color="#1a1f2e", corner_radius=15)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas_frame = canvas_frame
        self.update_visualization()
        
        return right_frame
    
    def show_data_explorer(self):
        """Affiche une fenêtre d'exploration des données"""
        if self.data is None:
            self.show_error("Impossible de charger les données. Vérifiez que le fichier 'wine-quality-white-and-red.csv' existe.")
            return
        
        # Créer une nouvelle fenêtre
        explorer_window = ctk.CTkToplevel(self)
        explorer_window.title("🔍 Exploration des Données")
        explorer_window.geometry("1200x700")
        explorer_window.resizable(True, True)
        
        # Configurer la grille
        explorer_window.grid_rowconfigure(0, weight=1)
        explorer_window.grid_columnconfigure(0, weight=1)
        
        # Frame principal
        main_frame = ctk.CTkFrame(explorer_window, fg_color="#0f1419")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        title = ctk.CTkLabel(header, text="📊 Exploration du Dataset", 
                            font=("Segoe UI", 20, "bold"), text_color="#9b59b6")
        title.pack(pady=15, padx=20)
        
        # Informations générales
        info_frame = ctk.CTkFrame(header, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        col1, col2, col3, col4 = ctk.CTkFrame(info_frame, fg_color="transparent"), \
                                ctk.CTkFrame(info_frame, fg_color="transparent"), \
                                ctk.CTkFrame(info_frame, fg_color="transparent"), \
                                ctk.CTkFrame(info_frame, fg_color="transparent")
        
        col1.pack(side="left", expand=True, padx=5)
        col2.pack(side="left", expand=True, padx=5)
        col3.pack(side="left", expand=True, padx=5)
        col4.pack(side="left", expand=True, padx=5)
        
        ctk.CTkLabel(col1, text=f"Lignes: {self.data.shape[0]}", 
                    font=("Segoe UI", 11), text_color="#888").pack()
        ctk.CTkLabel(col2, text=f"Colonnes: {self.data.shape[1]}", 
                    font=("Segoe UI", 11), text_color="#888").pack()
        ctk.CTkLabel(col3, text=f"Valeurs: {self.data.size:,}", 
                    font=("Segoe UI", 11), text_color="#888").pack()
        ctk.CTkLabel(col4, text=f"Valeurs manquantes: {self.data.isnull().sum().sum()}", 
                    font=("Segoe UI", 11), text_color="#888").pack()
        
        # Contrôles d'affichage
        controls_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Options d'affichage
        options_frame = ctk.CTkFrame(controls_frame, fg_color="#1a1f2e", corner_radius=10)
        options_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(options_frame, text="Options d'affichage:", 
                     font=("Segoe UI", 11, "bold"), text_color="#9b59b6").pack(anchor="w", padx=15, pady=10)
        
        # Type d'affichage
        display_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        display_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.display_type = ctk.StringVar(value="head")
        ctk.CTkRadioButton(display_frame, text="Début du dataset", variable=self.display_type, 
                          value="head", font=("Segoe UI", 10), text_color="#888").pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(display_frame, text="Fin du dataset", variable=self.display_type, 
                          value="tail", font=("Segoe UI", 10), text_color="#888").pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(display_frame, text="Échantillon aléatoire", variable=self.display_type, 
                          value="sample", font=("Segoe UI", 10), text_color="#888").pack(side="left")
        
        # Nombre de lignes
        rows_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        rows_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkLabel(rows_frame, text="Nombre de lignes:", 
                     font=("Segoe UI", 10), text_color="#888").pack(side="left")
        
        self.n_rows = ctk.CTkSlider(rows_frame, from_=5, to=100, number_of_steps=19, 
                                   command=self.update_data_display)
        self.n_rows.set(20)
        self.n_rows.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        self.n_rows_label = ctk.CTkLabel(rows_frame, text="20", 
                                        font=("Segoe UI", 10, "bold"), text_color="#9b59b6")
        self.n_rows_label.pack(side="left", padx=(10, 0))
        
        # Bouton de rafraîchissement
        refresh_btn = ctk.CTkButton(controls_frame, text="🔄 Rafraîchir", 
                                   command=self.update_data_display,
                                   font=("Segoe UI", 11), height=40,
                                   fg_color="#9b59b6", text_color="#ffffff",
                                   hover_color="#8e44ad", corner_radius=10)
        refresh_btn.pack(side="right", fill="y")
        
        # Zone d'affichage des données
        data_display_frame = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        data_display_frame.grid(row=2, column=0, sticky="nsew")
        data_display_frame.grid_rowconfigure(0, weight=1)
        data_display_frame.grid_columnconfigure(0, weight=1)
        
        # Scrollable frame pour les données
        scroll_frame = ctk.CTkScrollableFrame(data_display_frame, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Frame pour afficher les données sous forme de tableau
        self.data_table_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        self.data_table_frame.pack(fill="both", expand=True)
        
        # Initialiser l'affichage
        self.update_data_display()
        
        # Statistiques rapides
        stats_frame = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(15, 0))
        
        ctk.CTkLabel(stats_frame, text="📈 Statistiques rapides:", 
                    font=("Segoe UI", 12, "bold"), text_color="#9b59b6").pack(anchor="w", padx=20, pady=10)
        
        # Quelques statistiques
        stats_grid = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_grid.pack(fill="x", padx=20, pady=(0, 15))
        
        # Colonnes numériques
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sample_col = numeric_cols[0] if len(numeric_cols) > 0 else self.data.columns[0]
            
            col1, col2, col3, col4 = ctk.CTkFrame(stats_grid, fg_color="transparent"), \
                                    ctk.CTkFrame(stats_grid, fg_color="transparent"), \
                                    ctk.CTkFrame(stats_grid, fg_color="transparent"), \
                                    ctk.CTkFrame(stats_grid, fg_color="transparent")
            
            col1.pack(side="left", expand=True)
            col2.pack(side="left", expand=True)
            col3.pack(side="left", expand=True)
            col4.pack(side="left", expand=True)
            
            # Exemple de statistiques pour une colonne
            if sample_col in self.data.columns:
                ctk.CTkLabel(col1, text=f"Colonne: {sample_col}", 
                           font=("Segoe UI", 10), text_color="#888").pack()
                ctk.CTkLabel(col2, text=f"Moyenne: {self.data[sample_col].mean():.2f}", 
                           font=("Segoe UI", 10), text_color="#888").pack()
                ctk.CTkLabel(col3, text=f"Min: {self.data[sample_col].min():.2f}", 
                           font=("Segoe UI", 10), text_color="#888").pack()
                ctk.CTkLabel(col4, text=f"Max: {self.data[sample_col].max():.2f}", 
                           font=("Segoe UI", 10), text_color="#888").pack()
        
        # Bouton de fermeture
        close_btn = ctk.CTkButton(main_frame, text="Fermer", 
                                 command=explorer_window.destroy,
                                 font=("Segoe UI", 12), height=40,
                                 fg_color="#1a1f2e", text_color="#888",
                                 border_color="#9b59b6", border_width=1,
                                 hover_color="#242d3d", corner_radius=10)
        close_btn.grid(row=4, column=0, sticky="e", pady=(15, 0))
    
    def update_data_display(self, *args):
        """Met à jour l'affichage des données"""
        if not hasattr(self, 'data_table_frame'):
            return
        
        # Mettre à jour le label du slider
        n_rows = int(self.n_rows.get())
        self.n_rows_label.configure(text=str(n_rows))
        
        # Nettoyer le frame existant
        for widget in self.data_table_frame.winfo_children():
            widget.destroy()
        
        # Récupérer les données selon l'option sélectionnée
        display_type = self.display_type.get()
        
        if display_type == "head":
            data_to_display = self.data.head(n_rows)
            title = f"{n_rows} premières lignes"
        elif display_type == "tail":
            data_to_display = self.data.tail(n_rows)
            title = f"{n_rows} dernières lignes"
        else:  # sample
            data_to_display = self.data.sample(n=min(n_rows, len(self.data)))
            title = f"{n_rows} lignes aléatoires"
        
        # Afficher le titre
        ctk.CTkLabel(self.data_table_frame, text=title,
                    font=("Segoe UI", 12, "bold"), text_color="#00d4ff").pack(anchor="w", pady=(0, 10))
        
        # Créer un tableau simple
        # En-têtes
        headers_frame = ctk.CTkFrame(self.data_table_frame, fg_color="#242d3d", corner_radius=5)
        headers_frame.pack(fill="x", pady=(0, 5))
        
        # Afficher les noms de colonnes
        for i, col in enumerate(data_to_display.columns):
            header_label = ctk.CTkLabel(headers_frame, text=str(col), 
                                       font=("Segoe UI", 10, "bold"), 
                                       text_color="#9b59b6")
            header_label.grid(row=0, column=i, padx=8, pady=8, sticky="ew")
            headers_frame.grid_columnconfigure(i, weight=1)
        
        # Afficher les données
        for row_idx, (_, row) in enumerate(data_to_display.iterrows()):
            row_frame = ctk.CTkFrame(self.data_table_frame, fg_color="#1a1f2e" if row_idx % 2 == 0 else "#242d3d")
            row_frame.pack(fill="x", pady=1)
            
            for col_idx, (col_name, value) in enumerate(row.items()):
                # Formater la valeur
                if isinstance(value, float):
                    display_value = f"{value:.4f}" if abs(value) < 0.001 else f"{value:.2f}"
                else:
                    display_value = str(value)
                
                # Créer le label
                cell_label = ctk.CTkLabel(row_frame, text=display_value,
                                         font=("Segoe UI", 9),
                                         text_color="#888" if row_idx % 2 == 0 else "#aaa")
                cell_label.grid(row=0, column=col_idx, padx=8, pady=4, sticky="w")
                row_frame.grid_columnconfigure(col_idx, weight=1)
    
    def show_error(self, message):
        """Affiche un message d'erreur"""
        error_window = ctk.CTkToplevel(self)
        error_window.title("Erreur")
        error_window.geometry("400x150")
        
        ctk.CTkLabel(error_window, text="❌ Erreur", 
                    font=("Segoe UI", 16, "bold"), text_color="#ff4444").pack(pady=20)
        
        ctk.CTkLabel(error_window, text=message, 
                    font=("Segoe UI", 11), text_color="#888", wraplength=350).pack(padx=20)
        
        ctk.CTkButton(error_window, text="OK", command=error_window.destroy,
                     fg_color="#1a1f2e", text_color="#888").pack(pady=20)
    
    def predict(self):
        """Effectue une prédiction"""
        try:
            input_data = []
            for feature in feature_names:
                val = float(self.input_vars[feature].get())
                input_data.append(val)
            
            input_array = np.array([input_data])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            
            residual_std = metrics['residual_std']
            z = 1.96
            intervalle = z * residual_std
            
            if prediction >= 7:
                couleur = "#00ff00"
                status = "🌟 EXCELLENT"
            elif prediction >= 6:
                couleur = "#00d4ff"
                status = "✓ BON"
            elif prediction >= 5:
                couleur = "#ffaa00"
                status = "◐ ACCEPTABLE"
            else:
                couleur = "#ff4444"
                status = "✗ FAIBLE"
            
            self.result_label.configure(
                text=f"{prediction:.2f}/10",
                text_color=couleur
            )
            
            self.interpretation_label.configure(
                text=f"{status} | Intervalle de confiance: [{prediction - intervalle:.2f}, {prediction + intervalle:.2f}]",
                text_color=couleur
            )
        
        except ValueError:
            self.result_label.configure(text="Erreur", text_color="#ff4444")
            self.interpretation_label.configure(
                text="Vérifiez les valeurs entrées",
                text_color="#ff4444"
            )
    
    def reset_inputs(self):
        """Réinitialise les champs"""
        for feature in feature_names:
            self.input_vars[feature].delete(0, "end")
            _, _, default_val = hints.get(feature, ('', '', ''))
            self.input_vars[feature].insert(0, default_val)
        self.result_label.configure(text="--", text_color="#00d4ff")
        self.interpretation_label.configure(text="Entrez les valeurs et cliquez PRÉDIRE")
    
    def update_visualization(self, value=None):
        """Met à jour la visualisation"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        viz_type = self.viz_var.get()
        

    # Container pour graphique + explication
        viz_container = ctk.CTkFrame(self.canvas_frame, fg_color="transparent")
        viz_container.pack(fill="both", expand=True)
        viz_container.grid_rowconfigure(0, weight=1)
        viz_container.grid_columnconfigure(0, weight=1)
        

    # Canvas matplotlib
        canvas_frame = ctk.CTkFrame(viz_container, fg_color="#1a1f2e", corner_radius=15)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        fig = Figure(figsize=(9, 5.5), dpi=90, facecolor='#1a1f2e')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0f1419')
        ax.tick_params(colors='#888', labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('#2a3f5f')
            spine.set_linewidth(0.5)
        
        if viz_type == "Importance":
            self.plot_importance(ax, fig)
            explanation = "📊 Montre les 10 variables les plus influentes sur la qualité du vin. Plus la barre est longue, plus la variable affecte la prédiction."
        elif viz_type == "Corrélations":
            self.plot_correlations(ax, fig)
            explanation = "🔗 Les barres vertes = corrélation positive (+ valeur → + qualité). Les rouges = corrélation négative (+ valeur → - qualité)."
        elif viz_type == "Performance":
            fig2 = Figure(figsize=(9, 5.5), dpi=90, facecolor='#1a1f2e')
            self.plot_performance(fig2)
            explanation = "📈 À gauche: Comparaison Train vs Test. À droite: Ratio de surapprentissage (< 0.2 = bon). Les écarts petits = généralisation correcte."
            fig = fig2
            ax = None
        elif viz_type == "Résidus":
            self.plot_residuals(ax, fig)
            explanation = "📍 Montre les erreurs du modèle. Les points proches de 0 = prédictions exactes. Les bandes grises = zone d'erreur normale."
        elif viz_type == "Distribution":
            self.plot_distribution(ax, fig)
            explanation = "📉 Histogramme montrant la distribution des qualités dans les données. La plupart des vins ont une qualité 5-6 (ordinaires)."
        
        if ax:
            ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    
    # Explication en bas
        explanation_frame = ctk.CTkFrame(viz_container, fg_color="#1a1f2e", corner_radius=15)
        explanation_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        explanation_frame.grid_columnconfigure(0, weight=1)
        
        explanation_label = ctk.CTkLabel(
            explanation_frame, text=explanation,
            font=("Segoe UI", 10), text_color="#888", wraplength=800, justify="left"
        )
        explanation_label.pack(padx=15, pady=12)
    
    def plot_importance(self, ax, fig):
        """Importance des variables"""
        top = feature_importance.head(10)
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top['importance'].values, color='#00d4ff', edgecolor='#00a8cc', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top['feature'].values, fontsize=9, color='#888')
        ax.set_xlabel('Importance', color='#888', fontsize=10)
        ax.set_title('Variables les Plus Impactantes', color='#00d4ff', fontsize=12, fontweight='bold', pad=15)
        ax.invert_yaxis()
    
    def plot_correlations(self, ax, fig):
        """Corrélations"""
        df = pd.read_csv('wine-quality-white-and-red.csv')
        corr = df.select_dtypes(include=[np.number]).corr()['quality'].sort_values(ascending=False)
        top_corr = corr.drop('quality').head(10)
        
        colors = ['#00ff00' if x > 0 else '#ff4444' for x in top_corr.values]
        y_pos = np.arange(len(top_corr))
        ax.barh(y_pos, top_corr.values, color=colors, edgecolor='white', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_corr.index, fontsize=9, color='#888')
        ax.set_xlabel('Corrélation', color='#888', fontsize=10)
        ax.set_title('Corrélations avec la Qualité', color='#00d4ff', fontsize=12, fontweight='bold', pad=15)
        ax.axvline(x=0, color='#2a3f5f', linestyle='-', linewidth=1)
        ax.invert_yaxis()
    
    def plot_performance(self, fig):
        """Performance Train vs Test + Surapprentissage"""
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#0f1419')
            ax.tick_params(colors='#888', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#2a3f5f')
        
    
    # Métriques Train vs Test
        metrics_names = ['R²', 'RMSE', 'MAE']
        train = [metrics['r2_train'], metrics['rmse_train'], metrics['mae_train']]
        test = [metrics['r2_test'], metrics['rmse_test'], metrics['mae_test']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        ax1.bar(x - width/2, train, width, label='Train', color='#00ff00', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.bar(x + width/2, test, width, label='Test', color='#00d4ff', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, color='#888', fontsize=9)
        ax1.set_ylabel('Valeur', color='#888', fontsize=9)
        ax1.set_title('Train vs Test', color='#00d4ff', fontsize=11, fontweight='bold')
        ax1.legend(facecolor='#1a1f2e', edgecolor='#2a3f5f', labelcolor='#888', fontsize=8)
        ax1.grid(True, alpha=0.1, axis='y')
        


    # Surapprentissage - Gauge
        ratio = metrics['overfitting_ratio']
        
    
    # Créer une jauge
        colors_gauge = ['#00ff00', '#f39c12', '#ff4444']
        thresholds = [0.2, 0.5, 1.0]
    

    # Barres de référence
        for i, (thresh, color) in enumerate(zip(thresholds, colors_gauge)):
            ax2.axvline(x=thresh, color=color, linestyle='--', linewidth=2, alpha=0.6)
        

    # Barre du ratio actuel
        couleur_ratio = '#00ff00' if ratio < 0.2 else '#f39c12' if ratio < 0.5 else '#ff4444'
        ax2.barh([0], [ratio], height=0.3, color=couleur_ratio, edgecolor='white', linewidth=2, alpha=0.9)
        
        ax2.set_xlim(0, 0.8)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Ratio de Surapprentissage', color='#888', fontsize=9)
        ax2.set_title('Indicateur Surapprentissage', color='#00d4ff', fontsize=11, fontweight='bold')
        ax2.set_yticks([])
        

    # Texte du ratio
        status_text = f'{ratio:.3f} - '
        if ratio < 0.2:
            status_text += '✓ Excellent'
        elif ratio < 0.5:
            status_text += '⚠ Acceptable'
        else:
            status_text += '✗ Problématique'
        
        ax2.text(ratio + 0.02, 0.15, status_text, color='white', fontweight='bold', fontsize=9, va='center')
        ax2.grid(True, alpha=0.1, axis='x')
    
    def plot_residuals(self, ax, fig):
        """Résidus"""
        df = pd.read_csv('wine-quality-white-and-red.csv')
        df_enc = df.copy()
        if 'type' in df_enc.columns:
            df_enc['type'] = (df_enc['type'] == 'Red').astype(int)
        X = df_enc.drop('quality', axis=1)
        y = df_enc['quality']
        X_s = scaler.transform(X)
        y_pred = model.predict(X_s)
        res = y - y_pred
        
        ax.scatter(y_pred, res, alpha=0.4, color='#00d4ff', s=20, edgecolor='none')
        ax.axhline(y=0, color='#ff4444', linestyle='--', linewidth=2)
        ax.set_xlabel('Prédictions', color='#888', fontsize=10)
        ax.set_ylabel('Résidus', color='#888', fontsize=10)
        ax.set_title('Analyse des Résidus', color='#00d4ff', fontsize=12, fontweight='bold', pad=15)
    
    def plot_distribution(self, ax, fig):
        """Distribution"""
        df = pd.read_csv('wine-quality-white-and-red.csv')
        ax.hist(df['quality'], bins=20, color='#00d4ff', edgecolor='#00a8cc', alpha=0.8)
        ax.axvline(df['quality'].mean(), color='#00ff00', linestyle='--', linewidth=2, label=f'Moy: {df["quality"].mean():.2f}')
        ax.set_xlabel('Qualité', color='#888', fontsize=10)
        ax.set_ylabel('Fréquence', color='#888', fontsize=10)
        ax.set_title('Distribution de la Qualité', color='#00d4ff', fontsize=12, fontweight='bold', pad=15)
        ax.legend(facecolor='#1a1f2e', edgecolor='#2a3f5f', labelcolor='#888')


# LANCEMENT
if __name__ == "__main__":
    print("Application lancée!\n")
    app = WineQualityApp()
    app.mainloop()