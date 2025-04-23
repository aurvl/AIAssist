import os
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint

def main():
    # --------------------------
    # 1. Configuration initiale
    start_time = time.time()
    os.makedirs('recommendation', exist_ok=True)
    SEED = 42
    N_ITER = 30
    CV = 5

    # --------------------------
    # 2. Chargement et préparation des données
    try:
        dt = pd.read_csv("./recom_data.csv")
        
        # Sélection des colonnes pertinentes
        selected_cols = [
            'outils_utilises_chatgpt', 'taches_ia_traduction', 'taches_ia_planification',
            'impact_ia_enseignement', 'age', 'bac', 'utilisation_ia', 'impact_ia_emploi',
            'domaine_etudes', 'reaction_nouveautes_ia', 'pret_payer_ia_si_efficace',
            'ScoreOpportunity', 'ScoreRisk', 'statut_etudiant', 'profil_recom'
        ]
        dt = dt[selected_cols].copy()

        # Vérification des données manquantes
        if dt.isnull().sum().sum() > 0:
            print("Attention : données manquantes détectées")
            dt = dt.dropna()

        # Conversion des profils en catégories numériques
        y_map = {'Minimaliste': 0, 'Big Tech Lover': 1, 'Power User': 2, 'Curieux': 3}
        dt['profil_recom'] = dt['profil_recom'].map(y_map).astype('category')

        # Séparation features/target
        X = dt.drop(columns=['profil_recom'])
        y = dt['profil_recom']

    except Exception as e:
        print(f"Erreur lors du chargement des données : {str(e)}")
        return

    # --------------------------
    # 3. Pipeline de traitement
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=10))  # Sélection des 10 meilleures features
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # --------------------------
    # 4. Définition des modèles à tester
    models_config = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=SEED),
            'params': {
                'clf__n_estimators': randint(50, 200),
                'clf__max_depth': randint(3, 10),
                'clf__min_samples_split': randint(2, 10)
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(multi_class='multinomial', 
                                     solver='lbfgs', 
                                     max_iter=1000,
                                     random_state=SEED),
            'params': {
                'clf__C': uniform(0.1, 10),
                'clf__penalty': ['l2', None]
            }
        }
    }

    # --------------------------
    # 5. Entraînement et optimisation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=y
    )

    best_models = {}
    for model_name, config in models_config.items():
        try:
            print(f"\n>>> Optimisation du modèle {model_name} <<<")
            
            # Création du pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', config['model'])
            ])

            # Recherche d'hyperparamètres
            search = RandomizedSearchCV(
                pipeline,
                config['params'],
                n_iter=N_ITER,
                cv=CV,
                scoring='accuracy',
                n_jobs=-1,
                random_state=SEED,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            best_models[model_name] = search

            # Évaluation
            val_pred = search.best_estimator_.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            
            print(f"\nMeilleurs paramètres ({model_name}):")
            for k, v in search.best_params_.items():
                print(f"- {k}: {v}")
                
            print(f"\nAccuracy (validation): {val_acc:.4f}")
            print("\nRapport de classification:")
            print(classification_report(y_val, val_pred))

        except Exception as e:
            print(f"Erreur avec le modèle {model_name}: {str(e)}")
            continue

    # --------------------------
    # 6. Sauvegarde des résultats
    if not best_models:
        print("\nAucun modèle n'a pu être entraîné avec succès.")
        return

    # Sélection du meilleur modèle
    best_model_name = max(best_models, key=lambda x: best_models[x].best_score_)
    best_model = best_models[best_model_name].best_estimator_

    # Feature Importance (si disponible)
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        try:
            # Récupération des noms de features
            num_features = numeric_features[best_model.named_steps['preprocessor']
                                    .named_transformers_['num']
                                    .named_steps['selector']
                                    .get_support()]
            
            cat_features = best_model.named_steps['preprocessor'] \
                                    .named_transformers_['cat'] \
                                    .get_feature_names_out(categorical_features)
            
            all_features = list(num_features) + list(cat_features)
            importances = best_model.named_steps['clf'].feature_importances_
            
            # Tri par importance
            idx_sorted = np.argsort(importances)[::-1]
            importances_sorted = importances[idx_sorted]
            features_sorted = [all_features[i] for i in idx_sorted]
            
            # Visualisation
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances_sorted[:15], y=features_sorted[:15], palette='viridis')
            plt.title('Top 15 des features les plus importantes')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig('recommendation/feature_importances.png')
            plt.close()
        except Exception as e:
            print(f"Erreur lors de la création du graphique d'importance: {str(e)}")

    # Sauvegarde du modèle
    joblib.dump(best_model, 'recommendation/best_model.pkl')

    # Rapport de classification
    val_pred = best_model.predict(X_val)
    report = classification_report(y_val, val_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('recommendation/classification_report.csv')

    # --------------------------
    # 7. Rapport final
    print("\n" + "="*50)
    print("RÉSULTATS FINAUX".center(50))
    print("="*50)
    print(f"\nModèle retenu: {best_model_name}")
    print(f"Accuracy sur validation: {accuracy_score(y_val, val_pred):.4f}")
    print("\nFichiers générés:")
    print("- recommendation/best_model.pkl : Modèle entraîné")
    print("- recommendation/feature_importances.png : Graphique d'importance")
    print("- recommendation/classification_report.csv : Métriques détaillées")
    
    exec_time = time.time() - start_time
    print(f"\nTemps d'exécution total: {exec_time:.2f} secondes") # 15.43s
    print("="*50)

if __name__ == "__main__":    
    main()