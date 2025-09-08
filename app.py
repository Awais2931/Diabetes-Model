import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ===========================
# LOAD MODEL
# ===========================
MODEL_FILENAME = "diabetes_model.pkl"
try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ===========================
# TITLE
# ===========================
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🩺 Diabetes Prediction App</h1>
    <p style='text-align: center; color: #555;'>Analyze your health values, predict diabetes risk, and get personalized insights.</p>
    """,
    unsafe_allow_html=True
)

# ===========================
# USER INPUTS
# ===========================
st.sidebar.header("📝 Enter Patient Details")

def user_input():
    glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300,
                                      step=1, value=None, placeholder="e.g., 110")
    blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200,
                                             step=1, value=None, placeholder="e.g., 80")
    insulin = st.sidebar.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900,
                                      step=1, value=None, placeholder="e.g., 85")
    bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0,
                                  step=0.1, value=None, placeholder="e.g., 25.5")
    age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120,
                                  step=1, value=None, placeholder="e.g., 35")
    location = st.sidebar.text_input("📍 Location (City, Country)", placeholder="e.g., Narowal, Pakistan")
    return glucose, blood_pressure, insulin, bmi, age, location

glucose, blood_pressure, insulin, bmi, age, location = user_input()

# ===========================
# HEALTHY RANGES
# ===========================
healthy_ranges = {
    "Glucose": (70, 140),
    "Blood Pressure": (60, 120),
    "Insulin": (16, 166),
    "BMI": (18.5, 24.9),
    "Age": (18, 50)  # Reference only
}
healthy_targets = {k: (low + high) / 2 for k, (low, high) in healthy_ranges.items()}

# ===========================
# PREDICTION LOGIC
# ===========================
if st.sidebar.button("🔍 Predict Diabetes"):
    if None in [glucose, blood_pressure, insulin, bmi, age] or location == "":
        st.warning("⚠️ Please enter all patient details (including location) before predicting.")
    else:
        features = np.array([[glucose, blood_pressure, insulin, bmi, age]])
        prediction = model.predict(features)

        try:
            proba = model.predict_proba(features)[0][1] * 100  # Probability of diabetes
        except:
            proba = 50  # fallback if model doesn't support predict_proba

        health_score = int(100 - proba)  # Higher score = healthier

        # ===========================
        # RESULTS IN TABS
        # ===========================
        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Charts", "💡 Recommendations"])

        with tab1:
            st.subheader("🔎 Prediction Result")
            if prediction[0] == 1:
                st.error(f"⚠️ The model predicts this person **may have diabetes**.")
            else:
                st.success(f"✅ The model predicts this person is **unlikely to have diabetes**.")

            st.metric("Diabetes Risk Probability", f"{proba:.1f}%")
            st.metric("Health Score", f"{health_score}/100")
            st.markdown(f"**📍 Location:** {location}")

        with tab2:
            # Categories
            categories = list(healthy_ranges.keys())
            patient_values = [glucose, blood_pressure, insulin, bmi, age]
            target_values = [healthy_targets[cat] for cat in categories]

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, 4))
            bar_width = 0.35
            x = np.arange(len(categories))
            colors = []
            attention_needed = []

            for cat, val in zip(categories, patient_values):
                low, high = healthy_ranges[cat]
                if val < low or val > high:
                    colors.append("#E74C3C")  # Red
                    attention_needed.append(f"- **{cat}** is outside the healthy range ({low}–{high}).")
                else:
                    colors.append("#27AE60")  # Green

            ax.bar(x - bar_width/2, patient_values, bar_width, label="Your Value", color=colors, alpha=0.85)
            ax.bar(x + bar_width/2, target_values, bar_width, label="Healthy Target", color="#3498DB", alpha=0.6)

            ax.set_ylabel("Value")
            ax.set_title("Your Health vs. Healthy Targets", fontsize=14, color="#2E4053")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.legend()
            st.pyplot(fig)

        with tab3:
            st.subheader("⚠️ Health Areas Needing Attention")
            if attention_needed:
                for msg in attention_needed:
                    st.markdown(msg)
            else:
                st.success("✅ All entered values are within the healthy range.")

            # Personalized tips
            tips = {
                "Glucose": "Consider reducing sugar intake and increasing physical activity.",
                "Blood Pressure": "Try reducing salt intake and managing stress.",
                "Insulin": "Consult with a doctor about insulin resistance and diet control.",
                "BMI": "A balanced diet and regular exercise may help improve BMI.",
                "Age": "Regular health checkups are recommended."
            }
            if attention_needed:
                st.markdown("### 💡 Suggested Next Steps")
                for cat, val in zip(categories, patient_values):
                    low, high = healthy_ranges[cat]
                    if val < low or val > high:
                        st.markdown(f"- {tips[cat]}")

        # ===========================
        # PDF EXPORT (via matplotlib)
        # ===========================
        def create_pdf():
            buf = BytesIO()
            fig, ax = plt.subplots(figsize=(8.5, 6))

            ax.axis("off")
            report_text = f"""
            Diabetes Prediction Report
            --------------------------
            Prediction: {'Diabetes' if prediction[0]==1 else 'No Diabetes'}
            Risk Probability: {proba:.1f}%
            Health Score: {health_score}/100

            Patient Values:
            - Glucose: {glucose}
            - Blood Pressure: {blood_pressure}
            - Insulin: {insulin}
            - BMI: {bmi}
            - Age: {age}
            - Location: {location}
            """
            ax.text(0, 1, report_text, va="top", fontsize=12, family="monospace")

            plt.tight_layout()
            plt.savefig(buf, format="pdf")
            buf.seek(0)
            return buf

        pdf_buffer = create_pdf()
        st.download_button("📥 Download Report (PDF)", data=pdf_buffer,
                           file_name="diabetes_report.pdf", mime="application/pdf")

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        👨‍💻 <b>Developed by Awais Ali</b> <br>
        🎓 University of Narowal <br>
        📧 <a href="mailto:awaisshahid6890@gmail.com">awaisshahid6890@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)


