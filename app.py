import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Generate & Train (cached so it runs only once) ──
@st.cache_resource
def train_model():
    np.random.seed(42)
    n, n1 = 303, 163

    def make(n, d):
        r = np.random
        return {
            'age':      r.normal(56 if d else 52, 9, n).clip(29,77).astype(int),
            'sex':      r.choice([0,1], n, p=[0.25,0.75] if d else [0.45,0.55]),
            'cp':       r.choice([0,1,2,3], n, p=[0.08,0.18,0.24,0.50] if d else [0.22,0.28,0.28,0.22]),
            'trestbps': r.normal(134 if d else 129, 18, n).clip(90,200).astype(int),
            'chol':     r.normal(251 if d else 242, 52, n).clip(126,564).astype(int),
            'fbs':      r.choice([0,1], n, p=[0.84,0.16]),
            'restecg':  r.choice([0,1,2], n, p=[0.48,0.46,0.06]),
            'thalach':  r.normal(139 if d else 158, 22, n).clip(71,202).astype(int),
            'exang':    r.choice([0,1], n, p=[0.41,0.59] if d else [0.83,0.17]),
            'oldpeak':  np.round(r.exponential(1.6 if d else 0.6, n).clip(0,6.2), 1),
            'slope':    r.choice([0,1,2], n, p=[0.08,0.58,0.34] if d else [0.18,0.38,0.44]),
            'ca':       r.choice([0,1,2,3], n, p=[0.28,0.32,0.24,0.16] if d else [0.70,0.18,0.08,0.04]),
            'thal':     r.choice([1,2,3], n, p=[0.06,0.22,0.72] if d else [0.10,0.14,0.76]),
            'target':   [int(d)]*n
        }

    df = pd.concat([pd.DataFrame(make(n1,True)), pd.DataFrame(make(n-n1,False))],
                   ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])

    return model, scaler, X.columns.tolist(), acc, auc, df

model, scaler, features, acc, auc, df = train_model()

# ── Header ────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("*Random Forest ML Model — Cleveland Heart Disease Dataset*")
st.markdown("---")

# ── Model Stats Banner ────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Model Accuracy", f"{acc*100:.1f}%")
c2.metric("ROC-AUC Score", f"{auc*100:.1f}%")
c3.metric("Training Patients", "303")

st.markdown("---")

# ── Sidebar Inputs ────────────────────────────
st.sidebar.title("🩺 Enter Patient Details")
st.sidebar.markdown("Adjust the sliders and dropdowns below:")

age      = st.sidebar.slider("Age (years)", 29, 77, 54)
sex      = st.sidebar.radio("Sex", ["Female", "Male"])
cp_map   = {"Typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3}
cp       = st.sidebar.selectbox("Chest Pain Type", list(cp_map.keys()))
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dl)", 126, 564, 250)
fbs      = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
ecg_map  = {"Normal":0, "ST-T Abnormality":1, "LV Hypertrophy":2}
restecg  = st.sidebar.selectbox("Resting ECG Result", list(ecg_map.keys()))
thalach  = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
exang    = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
oldpeak  = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
slope_map= {"Upsloping":0, "Flat":1, "Downsloping":2}
slope    = st.sidebar.selectbox("ST Slope", list(slope_map.keys()))
ca       = st.sidebar.selectbox("Major Vessels Blocked (0–3)", [0,1,2,3])
thal_map = {"Normal":1, "Fixed Defect":2, "Reversible Defect":3}
thal     = st.sidebar.selectbox("Thalassemia Type", list(thal_map.keys()))

# ── Predict Button ────────────────────────────
predict = st.sidebar.button("🔍 Predict Now", use_container_width=True)

# ── Prediction Result ─────────────────────────
if predict:
    patient = np.array([[
        age, 1 if sex=="Male" else 0, cp_map[cp],
        trestbps, chol, 1 if fbs=="Yes" else 0,
        ecg_map[restecg], thalach, 1 if exang=="Yes" else 0,
        oldpeak, slope_map[slope], ca, thal_map[thal]
    ]])
    patient_s = scaler.transform(patient)
    pred  = model.predict(patient_s)[0]
    prob  = model.predict_proba(patient_s)[0][1]

    st.subheader("🎯 Prediction Result")

    if pred == 1:
        st.error(f"## ⚠️ HIGH RISK — Heart Disease Detected")
        st.error(f"**Probability of heart disease: {prob*100:.1f}%**")
    else:
        st.success(f"## ✅ LOW RISK — No Heart Disease Detected")
        st.success(f"**Probability of heart disease: {prob*100:.1f}%**")

    # Progress bar
    st.markdown("**Risk Level:**")
    st.progress(float(prob))
    st.caption(f"{prob*100:.1f}% chance of heart disease")

    st.markdown("---")

    # Feature importance chart
    st.subheader("📊 What influenced this prediction?")
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e24b4a' if f in ['ca','thal','cp','oldpeak'] else '#185FA5' for f in fi.index]
    fi.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Red = High Risk Indicators)")
    st.pyplot(fig)
    plt.close()

else:
    # Show instructions when app first loads
    st.subheader("👈 How to use this app")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Step 1:** Fill in the patient details in the left sidebar

        **Step 2:** Click the **Predict Now** button

        **Step 3:** See the result and risk probability instantly
        """)
    with col2:
        st.warning("""
        ⚕️ **Medical Disclaimer**

        This app is for **educational purposes only**.
        It is NOT a substitute for professional medical diagnosis.
        Always consult a qualified doctor.
        """)

    # Dataset overview
    st.markdown("---")
    st.subheader("📈 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        df['target'].value_counts().plot(kind='bar', ax=ax, color=['#3B6D11','#A32D2D'])
        ax.set_xticklabels(['No Disease','Disease'], rotation=0)
        ax.set_title('Patient Distribution')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        df[df['target']==0]['age'].hist(ax=ax2, alpha=0.6, color='#3B6D11', label='No Disease', bins=20)
        df[df['target']==1]['age'].hist(ax=ax2, alpha=0.6, color='#A32D2D', label='Disease', bins=20)
        ax2.set_title('Age Distribution by Target')
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn | Heart Disease Prediction ML Model")
