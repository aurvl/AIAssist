import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import joblib
import time

start = time.time()

# --------------------------
# 1. Configuration initiale
os.makedirs('clustering', exist_ok=True)

# --------------------------
# 2. Chargement et préparation des données
dtviz = pd.read_csv("./recom_data.csv")

# Conversion des listes d'outils
dtviz["outils_recom"] = dtviz["outils_recom"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Feature engineering
dtviz['nb_outils'] = dtviz['outils_recom'].apply(len)
prof = dtviz['profil_recom']

# --------------------------
# 3. Visualisations de base
# Nombre d'outils par utilisateur
plt.figure(figsize=(10, 6))
dtviz['nb_outils'].value_counts().sort_index().plot(kind='bar')
plt.title("Nombre d'outils IA utilisés")
plt.xlabel("Nombre d'outils")
plt.ylabel("Nombre d'utilisateurs")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('clustering/nombre_outils.png')
plt.close()

# Répartition des profils
plt.figure(figsize=(10, 6))
dtviz['profil_recom'].value_counts().plot(kind='bar')
plt.title("Répartition des profils recommandés")
plt.xlabel("Profil")
plt.ylabel("Nombre d'utilisateurs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('clustering/repartition_profils.png')
plt.close()

# --------------------------
# 4. Analyse des tâches par profil
taches = [
    'taches_ia_redaction', 'taches_ia_recherche', 'taches_ia_traduction',
    'taches_ia_programmation', 'taches_ia_presentation', 'taches_ia_analyse_donnees',
    'taches_ia_brainstorming', 'taches_ia_planification', 'taches_ia_revision',
    'taches_ia_cv_lettre', 'taches_ia_entretien', 'taches_ia_creation_media'
]

# Analyse globale
task_usage = dtviz.groupby("profil_recom")[taches].sum().T
task_usage.to_csv('clustering/usage_taches_par_profil.csv')

# Analyse normalisée
task_usage_norm = dtviz.groupby("profil_recom")[taches].mean().T
for col in task_usage_norm.columns:
    task_usage_norm[col] = task_usage_norm[col] / task_usage_norm[col].sum()

task_usage_norm = task_usage_norm.sort_values(by='Curieux', ascending=False)

# Visualisation comparée
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.barplot(data=task_usage_norm.reset_index(), x='Curieux', y='index', ax=ax[0])
ax[0].set_title("Utilisation relative des IA par tâche (Curieux)")
ax[0].set_xlabel("Proportion d'utilisation")
ax[0].set_ylabel("Tâches")
ax[0].set_xlim(0, 0.3)

sns.barplot(data=task_usage_norm.reset_index(), x='Power User', y='index', ax=ax[1])
ax[1].set_title("Utilisation relative des IA par tâche (Power User)")
ax[1].set_xlabel("Proportion d'utilisation")
ax[1].set_xlim(0, 0.3)

plt.tight_layout()
plt.savefig('clustering/comparaison_taches_profils.png')
plt.close()

# --------------------------
# 5. Clustering des utilisateurs
# Préparation des données
dt_cluster = dtviz.drop(columns=['outils_recom', 'profil_recom'])
num_vars = dt_cluster.select_dtypes(include=['int64', 'float64']).columns
cat_vars = dt_cluster.select_dtypes(include=['object']).columns

# Pipeline de prétraitement
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_vars),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars)
])

X_processed = preprocessor.fit_transform(dt_cluster)

# Réduction de dimension
pca = PCA(n_components=0.95)  # Conserve 95% de la variance
X_pca = pca.fit_transform(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)
X_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Sauvegarde du modèle
joblib.dump({
    'preprocessor': preprocessor,
    'pca': pca,
    'kmeans': kmeans
}, 'clustering/clustering_model.pkl')

# --------------------------
# 6. Visualisation des clusters
Kdf = X_pca.copy()
Kdf['cluster'] = clusters
Kdf['profil'] = prof.values

# Scatter plot avec profils
plt.figure(figsize=(12, 8))
sns.scatterplot(data=Kdf, x='PC1', y='PC2', hue='profil', palette='Set1')
plt.title('Répartition des profils dans l\'espace PCA')
plt.grid()
plt.tight_layout()
plt.savefig('clustering/profils_pca.png')
plt.close()

# Scatter plot avec clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=Kdf, x='PC1', y='PC2', hue='cluster', palette='Set1')
plt.title('Clusters dans l\'espace PCA')
plt.grid()
plt.tight_layout()
plt.savefig('clustering/clusters_pca.png')
plt.close()

# Ellipses des clusters
palette = sns.color_palette('Set1', n_colors=Kdf['cluster'].nunique())
cluster_colors = {k: palette[v] for k, v in zip(Kdf['cluster'].unique(), range(Kdf['cluster'].nunique()))}

plt.figure(figsize=(10, 8))
ax = plt.gca()

for cluster_id in Kdf['cluster'].unique():
    cluster_data = Kdf[Kdf['cluster'] == cluster_id]
    
    # Points
    plt.scatter(
        cluster_data['PC1'], 
        cluster_data['PC2'],
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.6
    )
    
    # Ellipse
    mean_x, mean_y = cluster_data[['PC1', 'PC2']].mean()
    std_x, std_y = cluster_data[['PC1', 'PC2']].std()
    
    ellipse = Ellipse(
        (mean_x, mean_y),
        width=std_x * 2,
        height=std_y * 2,
        edgecolor=cluster_colors[cluster_id],
        facecolor='none',
        linestyle='--',
        linewidth=2
    )
    ax.add_patch(ellipse)
    plt.text(mean_x, mean_y, str(cluster_id), 
            fontsize=12, ha='center', va='center')

plt.title('Clusters des utilisateurs avec ellipses de distribution')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('clustering/clusters_ellipses.png')
plt.close()

# --------------------------
# 7. Analyse des clusters
# Matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(dt_cluster[num_vars].corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation des variables numériques')
plt.tight_layout()
plt.savefig('clustering/matrice_correlation.png')
plt.close()

# Profils par cluster
cluster_profile = pd.crosstab(Kdf['cluster'], Kdf['profil'], normalize='index')
cluster_profile.to_csv('clustering/profils_par_cluster.csv')

# Caractéristiques des clusters
cluster_means = Kdf.groupby('cluster')[X_pca.columns].mean()
cluster_means.to_csv('clustering/caracteristiques_clusters.csv')

# --------------------------
# 8. Rapport final
print("\n" + "="*50)
print("RAPPORT DE VISUALISATION TERMINÉ")
print("="*50)
print("\nFichiers générés dans le dossier 'clustering/':")
print("- nombre_outils.png : Répartition du nombre d'outils utilisés")
print("- repartition_profils.png : Distribution des profils utilisateurs")
print("- comparaison_taches_profils.png : Usage des IA par tâche")
print("- clusters_*.png : Visualisations des clusters")
print("- clustering_model.pkl : Modèles de clustering sauvegardés")
print("- *.csv : Données d'analyse exportées")
print("\n" + "="*50)

end = time.time()
print(f"Temps d'exécution : {end - start:.2f} secondes") # 4.72s