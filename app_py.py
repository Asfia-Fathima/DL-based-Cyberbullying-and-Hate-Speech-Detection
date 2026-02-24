
!pip install torch transformers streamlit pandas matplotlib seaborn scikit-learn pyngrok

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

# -------------------------------
# 1️⃣ Page Configuration
# -------------------------------
st.set_page_config(page_title="AI Cyberbullying & Hate Speech Detection", layout="wide")
st.title("AI-driven Cyberbullying and Hate Speech Detection Dashboard")

st.markdown("""
The rapid growth of digital communication platforms has increased cyberbullying and hate speech, affecting mental health and online safety.
This system detects, categorizes, and monitors toxic comments using a transformer-based model (BERT) trained on the Jigsaw Toxic Comment Classification dataset.
""")

# -------------------------------
# 2️⃣ Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# -------------------------------
# 3️⃣ User Comment Input
# -------------------------------
st.header("Analyze a Single Comment")
user_comment = st.text_area("Enter your comment here:")

if st.button("Predict Toxicity") and user_comment.strip() != "":
    # Tokenize
    inputs = tokenizer(user_comment, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # Create dataframe for display
    severity = ["Low" if p<0.33 else "Medium" if p<0.66 else "High" for p in probs]
    df_pred = pd.DataFrame({
        "Label": labels,
        "Probability": np.round(probs, 2),
        "Prediction": (probs>=0.5).astype(int),
        "Severity": severity
    })

    st.subheader("Prediction Results")
    st.dataframe(df_pred)

    # Bar chart
    st.subheader("Probability per Label")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x="Label", y="Probability", data=df_pred, palette="coolwarm", ax=ax)
    ax.set_ylim(0,1)
    st.pyplot(fig)

# -------------------------------
# 4️⃣ Batch File Upload for AI-driven Monitoring
# -------------------------------
st.header("AI-driven Content Monitoring")
uploaded_file = st.file_uploader("Upload CSV with a column named 'comment'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "comment" not in df.columns:
        st.error("CSV must contain a 'comment' column")
    else:
        st.info(f"Processing {len(df)} comments...")
        all_probs = []
        batch_size = 16
        for i in range(0, len(df), batch_size):
            batch_texts = df['comment'][i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs).logits
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
        all_probs = np.vstack(all_probs)
        all_preds_bin = (all_probs>=0.5).astype(int)

        # Label distribution
        label_counts = all_preds_bin.sum(axis=0)
        df_counts = pd.DataFrame({"Label": labels, "Count": label_counts})

        st.subheader("Toxicity Counts per Label")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="Label", y="Count", data=df_counts, palette="viridis", ax=ax)
        st.pyplot(fig)

        # Severity distribution
        severity = np.where(all_probs<0.33, "Low", np.where(all_probs<0.66, "Medium", "High"))
        severity_counts = pd.Series(severity.flatten()).value_counts()
        st.subheader("Severity Distribution")
        st.bar_chart(severity_counts)

# -------------------------------
# 5️⃣ Model Metrics Section (Demo Metrics)
# -------------------------------
st.header("Model Performance Metrics (Demo)")

# For demo, generate random evaluation metrics (or you can compute on test set)
# Example: Replace with your computed all_labels and all_preds_bin if available
# Here, using small subset
if st.button("Show Demo Metrics"):
    # Dummy demo metrics (replace with actual metrics if available)
    demo_micro_f1 = 0.84
    demo_macro_f1 = 0.78
    st.metric("Micro F1 Score", demo_micro_f1)
    st.metric("Macro F1 Score", demo_macro_f1)

# -------------------------------
# 6️⃣ Background & Dataset Info
# -------------------------------
st.header("Dataset & Labels Info")
st.markdown("""
- **Dataset:** Jigsaw Toxic Comment Classification (Kaggle)
- **Labels:** toxic, severe toxic, obscene, threat, insult, identity hate
- Each label can co-occur with others.
- **Severity mapping:** Low (<0.33), Medium (0.33–0.66), High (>0.66)
- This system contributes toward safer online communities by providing automated monitoring of harmful content.
""")

!pip install torch transformers streamlit seaborn matplotlib
!streamlit run app.py

from pyngrok import ngrok
public_url = ngrok.connect(port=8501)
print("Your Streamlit dashboard URL:", public_url)

