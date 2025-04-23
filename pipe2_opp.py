import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import time

start = time.time()

# --------------------------
# 1. Configuration initiale
# Création du dossier opportunity
os.makedirs('opportunity', exist_ok=True)

# Paramètres fixes
SEED = 42
N_ITER = 50
CV = 5

# --------------------------
# 2. Chargement et préparation des données
df = pd.read_csv("./processed_data.csv")
X = df.drop("ScoreOpportunity", axis=1)
y = df["ScoreOpportunity"]

# Sélection des mêmes variables que pour Risk (à adapter si nécessaire)
selected_features = [
    'outils_utilises_chatgpt',
    'taches_ia_traduction',
    'taches_ia_planification',
    'impact_ia_enseignement',
    'age',
    'bac',
    'utilisation_ia',
    'impact_ia_emploi',
    'domaine_etudes',
    'reaction_nouveautes_ia',
    'pret_payer_ia_si_efficace',
    'ScoreRisk'
]

X = X[selected_features]

# --------------------------
# 3. Prétraitement des données
# Encodage des catégorielles
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

# Séparation des types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = categorical_cols

# Pipeline de prétraitement
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=15))  # Sélection des 15 meilleures features
    ]), numeric_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
])

# --------------------------
# 4. Pipeline et optimisation
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(
        n_estimators=200,
        random_state=SEED,
        eval_metric='rmse'
    ))
])

# Espace de recherche hyperparamètres
param_dist = {
    'xgb__max_depth': randint(3, 10),
    'xgb__learning_rate': uniform(0.01, 0.3),
    'xgb__subsample': uniform(0.6, 0.4),
    'xgb__colsample_bytree': uniform(0.6, 0.4),
    'xgb__gamma': uniform(0, 5),
    'xgb__reg_alpha': uniform(0, 3),
    'xgb__reg_lambda': uniform(0, 3)
}

# Split des données
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Recherche aléatoire
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=N_ITER,
    cv=CV,
    scoring='neg_root_mean_squared_error',
    random_state=SEED,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

# --------------------------
# 5. Sauvegarde des résultats
# Meilleur modèle
best_pipeline = search.best_estimator_

# Réentraînement avec early stopping
X_train_trans = best_pipeline.named_steps['preprocessor'].transform(X_train)
X_val_trans = best_pipeline.named_steps['preprocessor'].transform(X_val)

best_pipeline.named_steps['xgb'].fit(
    X_train_trans, y_train,
    eval_set=[(X_val_trans, y_val)],
    verbose=10
)

# Évaluation finale
val_pred = best_pipeline.predict(X_val)
final_rmse = np.sqrt(np.mean((val_pred - y_val)**2))

# --------------------------
# 6. Feature Importance
def imp_plot(importances, features, filename, color='red'):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances, y=features, color=color)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Récupération des features sélectionnées
num_mask = best_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['selector'].get_support()
selected_num_features = numeric_features[num_mask]
all_features = list(selected_num_features) + list(categorical_features)

importances = best_pipeline.named_steps['xgb'].feature_importances_
imp_plot(importances, all_features, 'opportunity/feature_importances.png', 'red')

# --------------------------
# 7. Sauvegarde complète
# Modèle complet
joblib.dump(best_pipeline, 'opportunity/best_model.pkl')

# Métriques et paramètres
with open('opportunity/model_metrics.txt', 'w') as f:
    f.write(f"Meilleurs paramètres: {search.best_params_}\n")
    f.write(f"Meilleur RMSE (CV): {-search.best_score_:.4f}\n")
    f.write(f"RMSE final (validation): {final_rmse:.4f}\n")
    f.write(f"Features sélectionnées:\n")
    for feat in all_features:
        f.write(f"- {feat}\n")

# --------------------------
# 8. Rapport final
print("\n" + "="*50)
print("RAPPORT FINAL DU MODÈLE OPPORTUNITY")
print("="*50)
print(f"\n• Meilleurs paramètres trouvés:\n{search.best_params_}")
print(f"\n• Performance du modèle:")
print(f"  - RMSE (cross-validation): {-search.best_score_:.4f}")
print(f"  - RMSE (validation set): {final_rmse:.4f}")
print("\n• Features sélectionnées:")
for i, feat in enumerate(all_features[:15]):  # Affiche les 15 premières
    print(f"  {i+1}. {feat}")
print("\n• Fichiers générés dans 'opportunity/':")
print("  - best_model.pkl : Modèle complet")
print("  - feature_importances.png : Importance des variables")
print("  - model_metrics.txt : Détails des performances")
print("\n" + "="*50)

end = time.time()
print(f"\nDurée d'exécution: {end - start:.2f} secondes") # 9.83s