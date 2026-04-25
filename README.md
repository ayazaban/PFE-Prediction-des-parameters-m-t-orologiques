# PFE-Prediction-des-parameters-m-t-orologiques
# Prédiction des Paramètres Météorologiques & Agronomiques  
## À l’aide d’images satellitaires et d’APIs open-source  
**Projet de Fin d’Études – 2025**  
**Réalisé avec Google Colab + Python + ML avancé**

![Python](https://img.shields.io/badge/python-3.10-blue)
![Pandas](https://img.shields.io/badge/pandas-2.2-green)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-optimized-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/license-MIT-blue)

---

### Résultat final (démo live)

 Application Streamlit interactive (déployable en 1 clic)  
 Prédiction en temps réel pour n’importe quelle coordonnée GPS au Maroc  
https://meteo-agro-maroc.streamlit.app

---

### Objectif du projet

Développer un système complet de **prédiction haute précision** de :
- Température (°C)
- Précipitations (mm)
- Humidité relative (%)
- Rayonnement solaire
- Vitesse et direction du vent
- VPD, ETo, etc.

**Pour des régions jamais vues par le modèle** → généralisation spatiale réelle (clé pour l’agriculture de précision)

---

### Données utilisées

| Source                | Résolution | Période     | Variables principales                     | Taille |
|-----------------------|------------|-------------|-------------------------------------------|--------|
| Données terrain (24 stations) | Journalière | 2012–2025 | Temp, Précip, HR, Vent, Rayonnement, Batterie | 197 fichiers → **1 376 858 lignes** |
| ERA5 (Copernicus)     | Mensuelle  | 2012–2025 | Temp, Précip, Rayonnement, Vent           | CSV    |
| Open-Meteo API        | Mensuelle  | 2012–2025 | Temp min/max, Précip                      | CSV    |
| NASA POWER            | Mensuelle  | 2012–2025 | Rayonnement solaire, Temp                 | CSV    |

---

### Pipeline complet (17 cellules Colab)

| Cellule | Description |
|--------|-----------|
| 0–1    | Setup + montage Drive |
| 2      | Chargement robuste des 197 fichiers .xls XML (SpreadsheetML) → 100% succès |
| 3      | Chargement ERA5 + Open-Meteo + NASA POWER |
| 4      | Harmonisation journalier → mensuel |
| 5      | Fusion multi-sources (3 approches testées – approche B retenue) |
| 6      | Feature Engineering avancé (50+ variables) : sin/cos, VPD, lags régionaux, anomalies |
| 7      | Détection automatique de leakage + exclusion métadonnées |
| 8      | Split temporel 60/20/20 + LOLO CV (Leave-One-Location-Out) |
| 9      | Entraînement RF + XGBoost avec Optuna (50 trials) |
| 10     | Tableau comparatif + barplot |
| 11     | SHAP feature importance + beeswarm |
| 12     | Validation spatiale LOLO (24 régions) → généralisation prouvée |
| 13     | Calibration complète (courbe, résidus, Q-Q plot) |
| 14     | Séries temporelles réel vs prédit |
| 15     | Fonction `predict_new_region(lat, lon, année, mois)` |
| 16     | Génération automatique de `app.py` Streamlit |
| 17     | Rapport final + phrase de soutenance prête à l'emploi |

---

### Performances obtenues (Test set)

| Modèle     | Température (RMSE) | Précipitations (RMSE) | R² Temp | R² Précip |
|------------|---------------------|-------------------------|---------|-----------|
| Random Forest | 2.41°C             | 18.7 mm                | 0.94    | 0.71      |
| **XGBoost (Optuna)** | **2.18°C**        | **16.3 mm**            | **0.96**| **0.76**  |

Validation LOLO (régions jamais vues) : RMSE moyen 2.6°C → **généralisation spatiale excellente**

---

### Fonctionnalités de l’application finale (Streamlit)

- Saisie latitude/longitude (ou clic sur carte)
- Sélection année/mois
- Prédiction instantanée Température + Précipitations ± incertitude
- Prévisions 6 mois glissants
- Carte interactive Folium
- Dashboard Plotly
- 100% déployable sur Streamlit Cloud

---

### Structure du dépôt
