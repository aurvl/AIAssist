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
import os
import time

start = time.time()
os.makedirs('risk', exist_ok=True)
SEED = 42
N_ITER = 50
CV = 5

# 1. Chargement et sélection des données
df = pd.read_csv("./processed_data.csv")
selected_features = [
    'outils_utilises_chatgpt',
    'taches_ia_traduction',
    'taches_ia_planification',
    'impact_ia_enseignement',
    'age',
    'bac',
    'statut_etudiant',
    'utilisation_ia',
    'impact_ia_emploi',
    'domaine_etudes',
    'reaction_nouveautes_ia',
    'pret_payer_ia_si_efficace',
    'ScoreRisk'
]
df = df[selected_features]

# 2. Préparation des données
X = df.drop("ScoreRisk", axis=1)
y = df["ScoreRisk"]

# 3. Encodage des catégorielles
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

# 4. Définition du préprocesseur
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = categorical_cols

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k='all'))
    ]), numeric_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
])

# 5. Pipeline complet avec XGBoost
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(
        n_estimators=100,
        random_state=SEED,
        eval_metric='rmse'
    ))
])

# 6. Espace de recherche des hyperparamètres
param_dist = {
    'xgb__max_depth': randint(3, 10),
    'xgb__learning_rate': uniform(0.01, 0.3),
    'xgb__subsample': uniform(0.6, 0.4),
    'xgb__colsample_bytree': uniform(0.6, 0.4),
    'xgb__gamma': uniform(0, 5),
    'xgb__reg_alpha': uniform(0, 3),
    'xgb__reg_lambda': uniform(0, 3)
}

# 7. Randomized Search CV
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=N_ITER,
    cv=CV,
    scoring='neg_root_mean_squared_error',
    random_state=SEED,
    verbose=2
)

# 8. Split des données
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Recherche des meilleurs hyperparamètres
search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres: {search.best_params_}")
print(f"Meilleur score (RMSE): {-search.best_score_:.4f}")

# 10. Récupération du meilleur modèle
best_pipeline = search.best_estimator_

# 11. Réentraînement avec early stopping
X_train_trans = best_pipeline.named_steps['preprocessor'].transform(X_train)
X_val_trans = best_pipeline.named_steps['preprocessor'].transform(X_val)

best_pipeline.named_steps['xgb'].fit(
    X_train_trans, y_train,
    eval_set=[(X_val_trans, y_val)],
    verbose=10
)

# 12. Évaluation finale
val_pred = best_pipeline.predict(X_val)
final_rmse = np.sqrt(np.mean((val_pred - y_val)**2))
print(f"\nRMSE final sur validation: {final_rmse:.4f}")

# 13. Feature Importance
feature_names = list(numeric_features) + list(categorical_features)
importances = best_pipeline.named_steps['xgb'].feature_importances_

plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=feature_names, palette='rocket')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('risk/feature_importances.png')
plt.close()

# Courbes d'apprentissage
results = best_pipeline.named_steps['xgb'].evals_result()
plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['rmse'], label='Validation RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend()
plt.savefig('risk/learning_curve.png')
plt.close()

with open('risk/model_metrics.txt', 'w') as f:
    f.write(f"Meilleurs paramètres: {search.best_params_}\n")
    f.write(f"Meilleur RMSE (CV): {-search.best_score_:.4f}\n")
    f.write(f"RMSE final (validation): {final_rmse:.4f}\n")
    f.write(f"Features sélectionnées:\n")
    for feat in feature_names:
        f.write(f"- {feat}\n")
    
# 14. Sauvegarde du modèle complet
joblib.dump(best_pipeline, 'risk/risk_model_pipeline.pkl')
print("\nModèle final sauvegardé sous 'risk_model_pipeline.pkl'")

# 15. Rapport final
print("\n" + "="*50)
print("RAPPORT FINAL DU MODÈLE RISK")
print("="*50)
print(f"\n• Meilleurs paramètres trouvés:\n{search.best_params_}")
print(f"\n• Performance du modèle:")
print(f"  - RMSE (cross-validation): {-search.best_score_:.4f}")
print(f"  - RMSE (validation set): {final_rmse:.4f}")
print("\n• Fichiers générés dans le dossier 'risk/':")
print("  - best_model.pkl : Modèle entraîné complet")
print("  - feature_importances.png : Importance des variables")
print("  - learning_curve.png : Courbe d'apprentissage")
print("  - model_metrics.txt : Métriques et paramètres")
print("\n" + "="*50)

end = time.time()
print(f"\nTemps d'exécution total: {end - start:.2f} secondes") # 16.50s