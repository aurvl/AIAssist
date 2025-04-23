import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load models
risk_model = joblib.load("risk/risk_model_pipeline.pkl")
opportunity_model = joblib.load("opportunity/best_model.pkl")
recommendation_model = joblib.load("recommendation/best_model.pkl")

# Profil mapping
y_map_inv = {
    0: (
        "Minimaliste",
        "Vous utilisez peu d'outils, souvent un seul comme ChatGPT. Cela montre une approche ciblée ou une prudence dans l'exploration des outils IA.",
        "Conseil : Explorez davantage avec OpenAI ChatGPT pour approfondir vos usages."
    ),
    1: (
        "Big Tech Lover",
        "Votre profil reflète une confiance forte envers les grands acteurs technologiques (Microsoft, Google, OpenAI, Claude).",
        "Conseil : Continuez à tirer profit des écosystèmes intégrés, comme Microsoft Copilot ou Claude pour plus d'efficacité."
    ),
    2: (
        "Power User",
        "Vous êtes un utilisateur expérimenté et polyvalent, explorant toutes les IA disponibles. Vous cherchez à tirer le meilleur de chacune.",
        "Conseil : Continuez ce mix complet incluant Gemini, Grok ou d'autres outils spécialisés pour des usages variés et avancés."
    ),
    3: (
        "Curieux",
        "Vous êtes dans une phase d'exploration active avec 3 à 4 outils. Cela montre une envie de découverte tout en gardant une approche raisonnée.",
        "Conseil : Essayez des IA moins connues comme Mistral ou Perplexity AI pour enrichir votre expérience."
    )
}

st.title("Assistant IA - Profil et Prédictions Personnalisées")

st.header("1. Vos habitudes avec l'IA")
outils_utilises_chatgpt = st.selectbox("Utilisez-vous ChatGPT ?", [1, 0])
tache = st.selectbox("Quelle tâche courante utilisez-vous avec l'IA ?", ["Aucune", "Traduction", "Planification", "Les deux"])
impact_ia_enseignement = st.selectbox("Quel impact pensez-vous que l'IA a sur l'enseignement ?", ['Largement', 'Radicalement', 'Légèrement', 'Pas du tout'])
age = st.selectbox("Votre tranche d'âge", ['40-60 ans', '30-40 ans.', '25-30 ans.', '18-25 ans.', 'Moins de 18 ans.'])
bac = st.selectbox("Votre niveau d'études", ['bac+5', 'bac+3', 'bac+2', 'bac+8', 'bac'])
statut_etudiant = st.selectbox("Êtes-vous actuellement étudiant(e) ?", [1.0, 0.0])
utilisation_ia = st.selectbox("Fréquence d'utilisation de l'IA", [2.0, 3.0, 1.0, 0.0])
impact_ia_emploi = st.selectbox("Quel impact pensez-vous que l'IA aura sur l'emploi ?", [
    'Elle ouvrira de nouvelles opportunités professionnelles.',
    'Elle aidera les travailleurs plutôt que de les remplacer.',
    'Elle remplacera de nombreux emplois.',
    'Elle supprimera certains postes, mais seulement dans certains secteurs.',
    'Je ne sais pas encore.'
])
domaine_etudes = st.selectbox("Votre domaine d'études", [
    'Sciences et ingénierie', 'Médecine et santé.', 'Économie et gestion', 'Arts, lettres et sciences humaines et sociales',
    'Droit et science politique', 'Autre'])
reaction_nouveautes_ia = st.selectbox("Quelle est votre réaction face aux nouveautés IA ?", [
    "Je suis curieux(se) et prêt(e) à l'essayer rapidement, mais avec prudence.",
    "Je préfère attendre que d'autres testent avant d'utiliser.",
    "J’aime être parmi les premiers à essayer ces innovations, même s'il y a des risques potentiels."
])
pret_payer_ia_si_efficace = st.selectbox("Seriez-vous prêt à payer si une IA est vraiment efficace ?", [0, 1])

# Transform to DataFrame
input_dict = {
    "outils_utilises_chatgpt": [outils_utilises_chatgpt],
    "taches_ia_traduction": [1 if tache in ["Traduction", "Les deux"] else 0],
    "taches_ia_planification": [1 if tache in ["Planification", "Les deux"] else 0],
    "impact_ia_enseignement": [impact_ia_enseignement],
    "age": [age],
    "bac": [bac],
    "statut_etudiant": [statut_etudiant],
    "utilisation_ia": [utilisation_ia],
    "impact_ia_emploi": [impact_ia_emploi],
    "domaine_etudes": [domaine_etudes],
    "reaction_nouveautes_ia": [reaction_nouveautes_ia],
    "pret_payer_ia_si_efficace": [pret_payer_ia_si_efficace]
}

df = pd.DataFrame(input_dict)

# Predict ScoreRisk
score_risk = risk_model.predict(df)[0]

# Predict ScoreOpportunity (drop statut_etudiant)
df_opportunity = df.drop(columns=["statut_etudiant"])
df_opportunity["ScoreRisk"] = score_risk
score_opportunity = opportunity_model.predict(df_opportunity)[0]

# Predict profil IA (full df with risk and opportunity)
df["ScoreRisk"] = score_risk
df["ScoreOpportunity"] = score_opportunity
profil_pred = recommendation_model.predict(df)[0]

score_risk = (score_risk - 0) / (14 - 0) * 9 + 1
score_opportunity = (score_opportunity - 0.4) / (14.5 - 0.4) * 9 + 1

# Display
st.subheader("2. Vos scores personnalisés")
col1, col2 = st.columns(2)
with col1:
    st.metric("Score de Risque", f"{score_risk:.2f}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_risk,
        gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "red"}},
        title={'text': "Niveau de Risque"}
    ))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.metric("Score d'Opportunité", f"{score_opportunity:.2f}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_opportunity,
        gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "green"}},
        title={'text': "Potentiel d'Opportunité"}
    ))
    st.plotly_chart(fig, use_container_width=True)

# Analyse complémentaire des scores
st.markdown("""
#### 📊 Interprétation des scores

Les scores sont **standardisés sur une échelle de 0 à 10** :
- `0` représente l’absence totale de risque ou d’opportunité perçue,
- `10` représente un niveau maximal.
""")

risk_text = ""
if score_risk <= 3:
    risk_text = "🟢 **Risque très faible** : Vous vous sentez à l’aise avec l’IA, sans crainte marquée."
elif score_risk <= 7:
    risk_text = "🟡 **Risque modéré** : Vous percevez des risques mais êtes potentiellement ouvert(e) à l’expérimentation."
else:
    risk_text = "🔴 **Risque élevé** : L’IA génère chez vous une forte inquiétude ou défiance."

opp_text = ""
if score_opportunity <= 3:
    opp_text = "🔴 **Opportunité perçue comme faible** : Vous voyez peu de bénéfices dans l’usage de l’IA."
elif score_opportunity <= 7:
    opp_text = "🟡 **Opportunité modérée** : Vous percevez certains bénéfices, sans être pleinement convaincu."
else:
    opp_text = "🟢 **Opportunité élevée** : Vous voyez dans l’IA un fort potentiel personnel ou professionnel."

st.info(risk_text)
st.success(opp_text)

# Profil
st.subheader("3. Votre profil IA")
label, desc, conseil = y_map_inv[profil_pred]
st.markdown(f"**Profil identifié :** `{label}`")
st.info(desc)
st.success(conseil)

st.caption("*App créée par Aurel pour explorer les comportements face à l'IA | [Me Contacter](https://aurvl.github.io/portfolio/index.html#contacts) © 2025*")