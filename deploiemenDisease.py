# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire nope
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# Charger les données pour les analyses
file_path = 'HeartDiseaseUCI.csv'
diamonds_df = pd.read_csv(file_path)


# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

loaded_model = load_model()
best_model = loaded_model["model"]
best_threshold = loaded_model["threshold"]

# Titre et description
st.title("Prédiction des maladies cardiaques")
st.write("Ce modèle prédit si une personne est **saine** ou **malade** en fonction des caractéristiques suivantes :")

# Sidebar pour saisir les caractéristiques
st.sidebar.header("Entrez les informations du patient")

# Options des types de douleur thoracique
chest_pain_options = {
    "Angine typique": 1,
    "Angine atypique": 2,
    "Douleur non angineuse": 3,
    "Asymptomatique": 4
}

# Entrées utilisateur
age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=50)
sex = st.sidebar.selectbox("Sexe", [('Femme', 0), ('Homme', 1)], index=1)
cp = st.sidebar.selectbox("Type de douleur thoracique", list(chest_pain_options.keys()), index=0)
trestbps = st.sidebar.number_input("Tension artérielle au repos (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Cholestérol sérique (mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.sidebar.selectbox("Glycémie à jeun > 120 mg/dl", [("Non", 0), ("Oui", 1)], index=0)
restecg = st.sidebar.selectbox("Résultat électrocardiographique au repos", ["Normal", "Anomalies de l'onde ST-T", "Hypertrophie ventriculaire gauche"], index=0)
thalach = st.sidebar.number_input("Fréquence cardiaque maximale atteinte", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Angine induite par l'exercice", [("Non", 0), ("Oui", 1)], index=0)
oldpeak = st.sidebar.number_input("Dépression du segment ST", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Pente du segment ST", ["Ascendante", "Plate", "Descendante"], index=0)
ca = st.sidebar.selectbox("Nombre de vaisseaux principaux", [0, 1, 2, 3], index=0)
thal = st.sidebar.selectbox("Thalassémie", [("Normal", 3), ("Défaut fixe", 6), ("Défaut réversible", 7)], index=0)

# Convertir les données utilisateur en tableau NumPy
input_data = np.array([
    [
        age, sex[1], chest_pain_options[cp], trestbps, chol, fbs[1],
        ["Normal", "Anomalies de l'onde ST-T", "Hypertrophie ventriculaire gauche"].index(restecg),
        thalach, exang[1], oldpeak, ["Ascendante", "Plate", "Descendante"].index(slope) + 1,
        ca, thal[1]
    ]
])

# Affichage des données sous forme de tableau et radar chart
st.subheader("Données saisies")
data = pd.DataFrame(input_data, columns=[
    "Âge", "Sexe", "Type douleur thoracique", "Tension artérielle au repos",
    "Cholestérol", "Glycémie à jeun", "Résultat ECG", "Fréquence cardiaque maximale",
    "Angine induite par exercice", "Dépression ST", "Pente segment ST", "Nb vaisseaux principaux", "Thalassémie"
])

st.write(data)

# Créer un radar chart pour visualiser les données
categories = [
    "Âge", "Tension artérielle au repos", "Cholestérol", "Fréquence cardiaque maximale",
    "Dépression ST", "Pente segment ST", "Nb vaisseaux principaux"
]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=input_data[0, [0, 3, 4, 7, 9, 10, 11]],
    theta=categories,
    fill='toself',
    name='Patient'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(input_data[0, [0, 3, 4, 7, 9, 10, 11]].max(), 220)]
        )
    ),
    showlegend=False
)

st.subheader("Visualisation des caractéristiques (Radar Chart)")
st.plotly_chart(fig)

# Prédiction
if st.button("Prédire"):
    y_scores = (
        best_model.predict_proba(input_data)[:, 1]
        if hasattr(best_model, "predict_proba")
        else best_model.decision_function(input_data)
    )
    y_pred = (y_scores >= best_threshold).astype(int)
    result = "Malade" if y_pred[0] == 1 else "Saine"
    st.write(f"**Résultat de la prédiction** : La personne est {result}.")
