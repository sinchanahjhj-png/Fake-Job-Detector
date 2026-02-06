import streamlit as st
import pickle

# -----------------------------
# Load trained model + vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🚨")

st.title("🚨 Fake Job Posting Detector")
st.write("Check whether a job posting is Genuine or Fraudulent.")

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("✍ Enter Job Description")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Check Job"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a job description.")
    else:
        # Convert input text to vector
        input_vector = vectorizer.transform([user_input])

        # Model prediction
        prediction = model.predict(input_vector)[0]

        # Prediction probability
        probability = model.predict_proba(input_vector)[0]

        # -----------------------------
        # Manual Suspicious Rule Check
        # -----------------------------
        suspicious_keywords = [
            "whatsapp",
            "telegram",
            "send money",
            "registration fee",
            "processing fee",
            "pay fee",
            "contact on",
            "call this number",
            "urgent hiring",
            "limited seats",
            "earn money fast"
        ]

        suspicious_flag = any(
            word in user_input.lower() for word in suspicious_keywords
        )

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        # -----------------------------
        # Final Decision Logic
        # -----------------------------
        if prediction == 1 or suspicious_flag:
            st.error("❌ Fraudulent / Suspicious Job Detected")

            # Show model confidence if model flagged fraud
            if prediction == 1:
                st.write(
                    f"Model Confidence: {round(probability[1] * 100, 2)}%"
                )

            # Show rule-based warning
            if suspicious_flag:
                st.warning(
                    "⚠ Suspicious keywords detected (WhatsApp / Fee / Urgency pattern)"
                )

        else:
            st.success(
                f"✅ Genuine Job ({round(probability[0] * 100, 2)}% confidence)"
            )

