import pandas as pd
import numpy as np
import re
import joblib
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from textblob_fr import PatternTagger, PatternAnalyzer
from textblob import Blobber
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Téléchargement des ressources NLTK
nltk.download('stopwords')

# 1. Chargement et nettoyage initial des données
def load_and_clean_data(filepath):
    """Charge et nettoie les données initiales"""
    df = pd.read_csv(filepath)
    df = df[df['complete'] >= 2]  # Filtre sur les réponses complètes
    
    df.drop(columns=['impact_ia_performance'], errors='ignore', inplace=True)
    
    # Conversion des colonnes Y/N en binaire
    for col in df.columns:
        if df[col].dtype == object and df[col].isin(["Y", "N"]).any():
            df[col] = df[col].map({"Y": 1, "N": 0})
    
    return df

# 2. Calcul du score de risque
def calculate_risk_score(row) -> int:
    """Calcule le score de risque basé sur l'utilisation de l'IA"""
    score = 0
    
    # Dépendance
    if row['dependance_ia'] == 2: 
        score += 3
    elif row['dependance_ia'] == 1: 
        score += 1
        
    # Confiance
    if row['confiance_ia'] >= 8: 
        score += 2
        
    # Vérification des infos
    if row['verification_infos_ia'] in [0, 1]: 
        score += 3
    elif row['verification_infos_ia'] == 2: 
        score += 1
    
    # Partage d'infos confidentielles
    if row['partage_infos_confidentielles_ia'] == 'Oui, sans problème': 
        score += 4
    elif row['partage_infos_confidentielles_ia'] == 'Oui, avec méfiance': 
        score += 2
        
    # Prise de décision
    if row['accord_ia_decision'] in [2, 3]: 
        score += 2
        
    # Réaction aux nouveautés
    if row['reaction_nouveautes_ia'] == "J'aime être parmi les premiers à essayer ces innovations, même s'il y a des risques potentiels.":
        score += 2
    elif row['reaction_nouveautes_ia'] == "Je suis curieux(se) et prêt(e) à l'essayer rapidement, mais avec prudence.":
        score += 1
    
    return score

# 3. Reclassification des tâches IA
def reclassify_ai_tasks(row):
    """Reclasse les tâches IA 'autres' dans des catégories existantes"""
    if pd.isna(row['taches_ia_autre']):
        return row

    tache = row['taches_ia_autre'].lower()
    if 'math' in tache or 'exercice' in tache or 'flash card' in tache:
        row['taches_ia_revision'] = 1
    elif 'correction de fautes' in tache or 'grammaire' in tache:
        row['taches_ia_redaction'] = 1
    elif 'planification' in tache and 'vacances' in tache:
        row['taches_ia_planification'] = 1
    
    return row

# 4. Analyse de texte
def clean_text(text):
    """Nettoie le texte pour analyse"""
    if pd.isna(text): 
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return " ".join([word for word in text.split() if word not in stopwords.words('french')])

def generate_word_cloud(text_column, save_path='wordcloud.png'):
    """Génère un nuage de mots"""
    text = text_column.apply(clean_text).str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    return Counter(text.split()).most_common(20)

# 5. Pipeline de traitement principal
def process_data(df):
    """Pipeline complet de traitement des données"""
    # Calcul des scores
    df['ScoreRisk'] = df.apply(calculate_risk_score, axis=1)
    
    # Reclassification des tâches
    df = df.apply(reclassify_ai_tasks, axis=1)
    df.drop(columns=['taches_ia_autre'], inplace=True)
    
    # Traitement des outils utilisés
    df['outils_utilises_autre'] = df['outils_utilises_autre'].fillna('').astype(str)
    df['outils_utilises_perplexity'] = df['outils_utilises_autre'].str.lower().str.contains('perplexity').astype(int)
    df['outils_utilises_autre'] = (~df['outils_utilises_autre'].str.lower().str.contains('perplexity')).astype(int)
    
    # Nettoyage des domaines d'études
    df['domaine_etudes'] = df.apply(lambda x: 
        FIELD_OF_STUDY_MAPPING.get(x['domaine_etudes_autre'], "Autre") 
        if x['domaine_etudes'] == 'Autre' and pd.notna(x['domaine_etudes_autre'])
        else x['domaine_etudes'], axis=1)
    
    df['domaine_etudes'] = df['domaine_etudes'].fillna("Autre")
    
    # Conversion des niveaux d'éducation
    df['bac'] = df['diplome_en_cours'].map(BAC_LEVEL_MAP)
    df['bac'] = df.apply(lambda x: 
        BAC_LEVEL_MAP.get(x['diplome_plus_eleve']) 
        if pd.isna(x['bac']) and pd.notna(x['diplome_plus_eleve'])
        else x['bac'], axis=1)
    
    # Conversion du genre
    df['genre'] = df['genre'].map(GENDER_MAP)
    
    # Nettoyage final
    df.drop(columns=['domaine_etudes_autre', 'diplome_en_cours', 'diplome_plus_eleve'], inplace=True)
    df['pret_payer_15e_ia'] = df['pret_payer_15e_ia'].fillna(0)
    
    # Standardisation du texte
    df['attentes_futures_ia'] = df['attentes_futures_ia'].replace('.', np.nan)
    df['domaine_etudes'] = df['domaine_etudes'].replace({
        'Sciences et ingénierie.': 'Sciences et ingénierie',
        'Arts, lettres et sciences sociales': 'Arts, lettres et sciences humaines et sociales',
        'Arts, lettres et sciences humaines.': 'Arts, lettres et sciences humaines et sociales',
        'Sciences sociales': 'Arts, lettres et sciences humaines et sociales',
        'Économie et gestion.': 'Économie et gestion'
    })
    
    return df

# 6. Imputation des valeurs manquantes
def impute_missing_values(df):
    """Impute les valeurs manquantes"""
    columns_to_impute = ['dependance_ia', 'ia_menace_travail', 'impact_ia_emploi', 'bac', 
                        'impact_ia_enseignement', 'age', 'genre', 'statut_etudiant']
    
    existing_cols = [col for col in columns_to_impute if col in df.columns]
    
    numerical_cols = df[existing_cols].select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df[existing_cols].select_dtypes(include=['object']).columns
    
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    joblib.dump((num_imputer, cat_imputer), 'imputers.pkl')
    return df

# 7. Modélisation des attentes futures
def predict_future_expectations(df):
    """Prédit les attentes futures manquantes avec gestion des variables catégorielles"""
    # Préparation des données textuelles
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    df['attentes_futures_ia'] = df['attentes_futures_ia'].apply(clean_text)
    df['ia_futures_sentiment'] = df['attentes_futures_ia'].apply(
        lambda x: tb(x).sentiment[0] if x.strip() else np.nan)
    
    # Identification des colonnes
    exclude_cols = ['id', 'complete', 'attentes_futures_ia', 'ia_futures_sentiment',
                   'ia_favorite_autre', 'temps_total'] + [col for col in df.columns if col.startswith('duree_')]
    
    # Séparation des types de variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = [col for col in df.columns if col not in exclude_cols + categorical_cols]
    
    # Séparation train/test
    train = df[df['ia_futures_sentiment'].notna()]
    test = df[df['ia_futures_sentiment'].isna()]
    
    if len(train) > 0 and len(test) > 0:
        # Pipeline avec preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        param_dist = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2']
        }
        
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1,
            error_score='raise'  # Pour voir les erreurs détaillées
        )
        
        try:
            # Entraînement
            search.fit(train[numerical_cols + categorical_cols], train['ia_futures_sentiment'])
            
            # Prédiction
            preds = search.best_estimator_.predict(test[numerical_cols + categorical_cols])
            df.loc[test.index, 'ia_futures_sentiment'] = preds
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
            # Fallback: imputation par la médiane
            median_val = train['ia_futures_sentiment'].median()
            df.loc[test.index, 'ia_futures_sentiment'] = median_val
    
    return df

# 8. Calcul du score d'opportunité
def compute_opportunity_score(row):
    """Calcule un score d'opportunité composite"""
    score = 0
    
    # 1. Variété des tâches IA
    tache_cols = [col for col in row.index if col.startswith('taches_ia_')]
    score += sum(row[tache] for tache in tache_cols if tache in row)
    
    # 2. Disposition à payer
    if row.get('pret_payer_30e_ia', 0): 
        score += 1
    if row.get('pret_payer_15e_ia', 0): 
        score += 0.5
    
    # 3. Confiance et vérification
    score += row.get('confiance_ia', 0) / 10
    if row.get('verification_infos_ia', 0) < 2: 
        score += 1
    
    # 4. Nombre d'outils utilisés
    outils_cols = [col for col in row.index if col.startswith('outils_utilises_')]
    score += sum(row[outil] for outil in outils_cols if outil in row) * 0.3
    
    # 5. Ajustement par le score de risque
    score -= row.get('ScoreRisk', 0) * 0.2
    
    return round(score, 2)

# Dictionnaires de mapping
FIELD_OF_STUDY_MAPPING = {
    'Commerce': 'Économie et gestion',
    'Banque': 'Économie et gestion',
    'Finance': 'Économie et gestion',
    'Marketing / communication': 'Économie et gestion',
    'Master MBFA': 'Économie et gestion',
    'Ressources humaines et communication': 'Sciences sociales',
    'Gestion des ressources humaines': 'Sciences sociales',
    'Informatique': 'Sciences et ingénierie',
    'informatique': 'Sciences et ingénierie',
    'Mathématique et informatique': 'Sciences et ingénierie',
    'MASS': 'Sciences et ingénierie',
    'Architecture': 'Sciences et ingénierie',
    'Bâtiment Travaux Public': 'Sciences et ingénierie',
    'Génie civil': 'Sciences et ingénierie',
    'Journalisme plurimedia': 'Arts, lettres et sciences sociales',
    'Psychologie': 'Arts, lettres et sciences sociales',
    'Éducation': 'Arts, lettres et sciences sociales',
    'Coiffure': 'Arts, lettres et sciences sociales',
}

BAC_LEVEL_MAP = {0: 'bac', 1: 'bac+2', 2: 'bac+3', 3: 'bac+5', 4: 'bac+8'}
GENDER_MAP = {'Masculin': 0, 'Féminin': 1, 'Autre': 0}

# Exécution principale
if __name__ == "__main__":
    # Chargement et traitement
    df = load_and_clean_data("./dataset.csv")
    df = process_data(df)
    
    # Analyse textuelle
    top_words = generate_word_cloud(df['attentes_futures_ia'])
    print("Mots les plus fréquents:", top_words)
    
    # Imputation
    df = impute_missing_values(df)
    
    # Prédiction des attentes
    df = predict_future_expectations(df)
    
    # Calcul des scores
    df['ScoreOpportunity'] = df.apply(compute_opportunity_score, axis=1)
    
    # Nettoyage final
    cols_to_drop = ['id', 'complete', 'ia_favorite_autre', 'pret_payer_100e_ia', 'pret_payer_40e_ia', 
                    'pret_payer_20e_ia', 'pret_payer_5e_ia', 'pret_payer_50e_ia', 'attentes_futures_ia',
                    'temps_total', *[col for col in df.columns if col.startswith('duree_')]]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Sauvegarde
    print(df.columns)
    df.to_csv('processed_data.csv', index=False)
    print("Traitement terminé. Données sauvegardées.")