import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="👨‍⚕️",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #14532d;
        color: #86efac;
        padding: 18px 24px;
        border-radius: 8px;
        border: 1px solid #22c55e;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 25px;
    }
    .disclaimer-box {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

DISEASE_MAP = {
    1: ("Psoriasis", "A chronic autoimmune skin condition causing rapid skin cell buildup, creating scaling, inflammation, and red plaques."),
    2: ("Seborrheic Dermatitis", "A common inflammatory skin condition causing flaky, yellowish scales primarily on oily areas like the scalp and face."),
    3: ("Lichen Planus", "An inflammatory disease characterized by itchy, purplish, flat-topped bumps typically with polygonal shapes and oral involvement."),
    4: ("Pityriasis Rosea", "A mild, self-limiting skin rash often presenting with an initial herald patch followed by smaller oval reddish spots."),
    5: ("Chronic Dermatitis", "Persistent eczema/dermatitis causing thickened skin (acanthosis), recurring itching, and skin barrier compromise."),
    6: ("Pityriasis Rubra Pilaris", "A rare group of chronic disorders characterized by reddish-orange scaling plaques and follicular keratotic papules.")
}

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_trained_model()

# Header
st.markdown('<div class="main-title">👨‍⚕️ Skin Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Please provide the following information about the symptoms:</div>', unsafe_allow_html=True)

# Define dropdown options
SEVERITY_OPTIONS = {"Absent": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
BORDER_OPTIONS = {"Absent": 0, "Slightly": 1, "Moderate": 2, "Definite": 3}
BINARY_OPTIONS = {"No": 0, "Yes": 1}
BINARY_OR_SEV_OPTIONS = {"No": 0, "Yes / Mild": 1, "Moderate": 2, "Severe": 3}

# Form inputs
features = {}

# Clinical Symptoms
features["erythema"] = SEVERITY_OPTIONS[st.selectbox("Erythema (Redness of the skin)", list(SEVERITY_OPTIONS.keys()), index=1)]
features["scaling"] = SEVERITY_OPTIONS[st.selectbox("Scaling (Flaky/peeling skin)", list(SEVERITY_OPTIONS.keys()), index=2)]
features["definite_borders"] = BORDER_OPTIONS[st.selectbox("Definite Borders (Clear edge of rash)", list(BORDER_OPTIONS.keys()), index=1)]
features["itching"] = SEVERITY_OPTIONS[st.selectbox("Itching (Itchy skin)", list(SEVERITY_OPTIONS.keys()), index=2)]
features["koebner_phenomenon"] = SEVERITY_OPTIONS[st.selectbox("Koebner Phenomenon (Rash after skin trauma)", list(SEVERITY_OPTIONS.keys()), index=3)]
features["polygonal_papules"] = SEVERITY_OPTIONS[st.selectbox("Polygonal Papules (Small raised lesions)", list(SEVERITY_OPTIONS.keys()), index=2)]
features["follicular_papules"] = SEVERITY_OPTIONS[st.selectbox("Follicular Papules (Bumps near hair follicles)", list(SEVERITY_OPTIONS.keys()), index=3)]
features["oral_mucosal_involvement"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Oral Mucosal Involvement (Inside mouth affected)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["knee_elbow_involvement"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Knee/Elbow Involvement", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["scalp_involvement"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Scalp Involvement", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["family_history"] = BINARY_OPTIONS[st.selectbox("Family History", list(BINARY_OPTIONS.keys()), index=0)]

# Histopathological / Clinical Infiltration
features["melanin_incontinence"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Melanin Incontinence (Loss of pigment)", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["eosinophils_infiltrate"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Eosinophils in Infiltrate", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["pnl_infiltrate"] = BINARY_OR_SEV_OPTIONS[st.selectbox("PNL Infiltrate (Type of white cells)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["fibrosis_papillary_dermis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Fibrosis of Papillary Dermis", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["exocytosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Exocytosis (White cells in top skin layer)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["acanthosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Acanthosis (Thickening of skin)", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["hyperkeratosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Hyperkeratosis (Thick outer skin)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["parakeratosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Parakeratosis (Nuclei in skin cells)", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["clubbing_rete_ridges"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Clubbing of Rete Ridges", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["elongation_rete_ridges"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Elongation of Rete Ridges", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["thinning_suprapapillary_epidermis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Thinning of Suprapapillary Epidermis", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["spongiform_pustule"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Spongiform Pustule (White cell clusters)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["munro_microabcess"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Munro Microabscess (Pus in skin)", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["focal_hypergranulosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Focal Hypergranulosis (Thick granular layer)", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["disappearance_granular_layer"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Disappearance of Granular Layer", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["basal_layer_damage"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Basal Layer Damage", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["spongiosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Spongiosis (Fluid between skin cells)", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["saw_tooth_appearance_retes"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Saw-Tooth Appearance of Rete", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["follicular_horn_plug"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Follicular Horn Plug", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["perifollicular_parakeratosis"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Perifollicular Parakeratosis", list(BINARY_OR_SEV_OPTIONS.keys()), index=1)]
features["inflammatory_mononuclear_infiltrate"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Inflammatory Mononuclear Infiltrate", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]
features["band_like_infiltrate"] = BINARY_OR_SEV_OPTIONS[st.selectbox("Band-like Infiltrate", list(BINARY_OR_SEV_OPTIONS.keys()), index=0)]

# Demographic
features["age"] = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)

# Predict button
if st.button("Predict Skin Disease", type="primary"):
    input_values = np.array(list(features.values())).reshape(1, -1)
    
    if model is not None:
        prediction = model.predict(input_values)[0]
        disease_name, disease_desc = DISEASE_MAP.get(prediction, ("Unknown Condition", "Consult a physician."))
        
        st.markdown(
            f'<div class="result-box">🩺 <b>Predicted Disease:</b> {disease_name}</div>',
            unsafe_allow_html=True
        )
        st.info(f"**Clinical Description:** {disease_desc}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_values)[0]
            st.markdown("#### Prediction Confidence Breakdown:")
            prob_df = pd.DataFrame({
                "Disease": [DISEASE_MAP[i][0] for i in model.classes_],
                "Confidence": [f"{p*100:.1f}%" for p in probs]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
    else:
        st.warning("Model file 'model.pkl' not found. Please run 'python train.py' to train and save the model first.")

# Medical Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Medical Disclaimer:</b> This application is an educational and supportive machine learning diagnostic aid built using clinical and histopathological classification models. It is not intended to replace professional dermatological diagnosis, clinical consultations, or biopsy evaluations.
</div>
""", unsafe_allow_html=True)
