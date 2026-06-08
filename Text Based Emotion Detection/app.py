import streamlit as st
import joblib
import pandas as pd
import base64
from datetime import datetime

# Load trained model
model = joblib.load('forensic_model.pkl')

# Page setup
st.set_page_config(
    page_title="Cybercrime Forensics",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ Text-Based Emotion & Threat Detection")
st.markdown("**Forensic Decision Support System for Cybercrime Investigation**")

col1, col2 = st.columns([2, 1])

#LEFT SIDE 
with col1:
    st.subheader("📝 Evidence Analysis")

    user_input = st.text_area(
        "Paste Chat / Email / Social Media Evidence:",
        height=200
    )

    if st.button("Analyze Text"):
        if user_input.strip() != "":
            prediction = model.predict([user_input])[0]
            probs = model.predict_proba([user_input])[0]
            classes = model.classes_

            st.divider()
            st.markdown(f"### 🧠 Classification: **{prediction.upper()}**")

            # Alerts
            if prediction == "threat":
                st.error("🚨 THREAT DETECTED: Immediate investigation required.")
            elif prediction == "anger":
                st.warning("⚠️ AGGRESSION DETECTED: Potential hostile intent.")
            elif prediction == "joy":
                st.success("✅ LOW RISK: Positive sentiment.")
            else:
                st.info(f"ℹ️ Status: {prediction.capitalize()}")

            # Legal Report
            report = f"""
FORENSIC ANALYSIS REPORT
-----------------------
Date: {datetime.now()}

Evidence Text:
"{user_input}"

Detected Emotion: {prediction.upper()}
Confidence Score: {max(probs)*100:.2f}%

Status:
{'FLAGGED FOR INVESTIGATION' if prediction in ['threat','anger'] else 'CLEARED'}
"""

            b64 = base64.b64encode(report.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="forensic_report.txt">📄 Download Legal Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.warning("Please enter evidence text.")

# RIGHT SIDE
with col2:
    st.subheader("📊 Confidence Metrics")

    if user_input.strip() != "":
        df = pd.DataFrame({
            "Emotion": classes,
            "Confidence": probs
        })
        st.bar_chart(df.set_index("Emotion"))

        with st.expander("ℹ️ Model Information"):
            st.write(
                "This system uses an ensemble of Logistic Regression, "
                "Random Forest, and SVM trained on forensic text data."
            )
