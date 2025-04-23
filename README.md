# Assistant IA – Profil et Prédictions Personnalisées

Cette application Streamlit permet de :

-   Prédire un score de **risque perçu** lié à l'adoption de l'IA.

-   Prédire un score d'**opportunité perçue** de l'IA.

-   Identifier un **profil d'utilisateur IA** à partir des usages et perceptions.

## 🛠️ Installation

Crée un environnement virtuel et installe les dépendances :

``` bash
python -m venv venv
source venv/bin/activate  # ou venv\\Scripts\\activate sur Windows
pip install -r requirements.txt
```

## 🚀 Lancer l'application

``` bash
streamlit run app.py
```

## 📁 Structure des fichiers

``` text
.
├── app.py                      # Code principal de l'application Streamlit
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
├── risk/
│   └── risk_model_pipeline.pkl
├── opportunity/
│   └── best_model.pkl
├── recommendation/
│   └── best_model.pkl
```

## 🧠 À propos des prédictions

Les prédictions s'enchaînent ainsi :

1.  Prédiction du **ScoreRisk** (modèle `risk/`)
2.  Prédiction du **ScoreOpportunity** (modèle `opportunity/`, sans `statut_etudiant`)
3.  Prédiction du **Profil IA** (modèle `recommendation/`, avec toutes les variables + scores)

## ✉️ Contact

Pour toute question : [Me Contacter](https://aurvl.github.io/portfolio/index.html#contacts)

------------------------------------------------------------------------

*© 2025 Aurel – Tous droits réservés*