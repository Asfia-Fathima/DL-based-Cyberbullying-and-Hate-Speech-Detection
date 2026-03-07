import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- 1. Page Config & GUARDIAN AI Theme ---
st.set_page_config(page_title="Guardian AI", layout="wide")

# This block hides the CSS logic so it doesn't appear as text on your page
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .legal-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #1e40af;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Backend: Model Loading ---
@st.cache_resource
def load_model():
    # Using the toxic-bert model from your project requirements
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return probs

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🛡️ Guardian AI</h1>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Main Checker", "Dataset Insights"],
        icons=["shield-check", "database"],
        default_index=0,
        styles={
            "container": {"background-color": "#111827"},
            "nav-link": {"color": "white", "text-align": "left"},
            "nav-link-selected": {"background-color": "#374151"},
        }
    )

# --- 4. Main Page Content ---
if selected == "Main Checker":
    st.title("🚫 Cyberbullying & Hate Speech Detection with Legal Awareness")
    
    # --- SEGMENT 1: Input Content ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Input Content")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("Example: Neutral"): st.session_state.user_text = "The weather is quite nice today for a walk."
    if col_s2.button("Example: Harsh"): st.session_state.user_text = "You are a pathetic loser, I hate everything about you!"
    if col_s3.button("Example: Threatening"): st.session_state.user_text = "I am going to find where you live and hurt you."

    user_input = st.text_area(
        "Enter text to analyze:", 
        value=st.session_state.get("user_text", ""), 
        height=120
    )
    analyze_btn = st.button("Run Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SEGMENT 2: Analysis Results ---
    if analyze_btn and user_input:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Analysis Breakdown")
        
        probs = predict(user_input)
        
        # Displaying the bar graph as seen in your reference image
        fig = go.Figure()
        for label, prob in zip(labels, probs):
            fig.add_trace(go.Bar(
                y=[label], x=[prob],
                orientation='h',
                marker=dict(color="#b91c1c" if prob > 0.5 else "#10b981"),
            ))
        fig.update_layout(xaxis=dict(range=[0, 1]), height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SEGMENT 3: Legal Framework ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Indian Legal Framework & Punishments ⚖️")
    
    # Defining legal sections for display
    laws = [
        ("Section 66E (IT Act)", "Violation of privacy.", "Up to 3 years imprisonment or ₹2 lakh fine."),
        ("Section 67 (IT Act)", "Transmitting obscene material.", "Up to 3 years imprisonment + fine."),
        ("Section 507 (IPC)", "Criminal intimidation.", "Up to 2 years imprisonment."),
        ("Section 509 (IPC)", "Insulting the modesty of a woman.", "Up to 3 years imprisonment + fine.")
    ]
    
    for title, desc, punishment in laws:
        st.markdown(f"""
            <div class="legal-card">
                <h4 style='margin:0;'>{title}</h4>
                <p style='margin:5px 0;'><b>Offense:</b> {desc}</p>
                <p style='margin:0; color:#b91c1c;'><b>Punishment:</b> {punishment}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Dataset Insights":
    st.title("📈 Dataset Insights")
    st.info("Information about the Jigsaw dataset used to train the BERT model goes here.")
