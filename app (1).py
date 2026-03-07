import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- 1. Page Config & project Theme ---
st.set_page_config(page_title="🚫 Cyberbullying & Hate Speech Detection", layout="wide")

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

# --- 2. Backend: Model Loading & logic ---
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
    st.markdown("<h1 style='text-align: center;'>🛡️🚫 Cyberbullying & Hate Speech Detection</h1>", unsafe_allow_html=True)
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
    st.subheader("Sample comments")
    st.write('Enter Text or Select any suggestion and analyze')
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
    is_toxic = any(p > 0.5 for p in probs)
        
    if is_toxic:
        st.error(" **High Toxicity Detected:** This content may violate safety guidelines.")
    else:
        st.success("**Safe Content:** No significant toxicity detected.")

    custom_colors = ["#b91c1c", "#8b0000", "#15803d", "#65a30d", "#1d4ed8", "#4338ca"]
    
    fig = go.Figure()
  
    for label, prob, color in zip(labels, probs, custom_colors):
        fig.add_trace(go.Bar(
            y=[label], 
            x=[prob],
            orientation='v',
            # Using the custom color for each specific bar
            marker=dict(color=color),
            # Adding a text label to show the percentage on the bar
            text=f"{prob:.1%}",
            textposition='auto'
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Probability Score"), 
        height=350, 
        margin=dict(l=0, r=0, t=10, b=10), 
        showlegend=False,
        # Making the background transparent to match your white card
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
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

# --- . Dataset Insights Page ---
elif selected == "Dataset Insights":
    st.title("Model Performance & Dataset Analytics")
    st.markdown("The model is trained on the Jigsaw Toxic Comment Classification dataset.")

    # --- 1. Model Performance (Metrics + Accuracy) ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Model Evaluation Metrics")
    
    metrics_data = {
        "Category": labels,
        "Precision": [1.0, 0.0, 0.92, 1.0, 0.61, 1.0],
        "Recall": [0.4, 0.0, 0.22, 0.51, 0.32, 0.15],
        "F1 Score": [0.58, 0.0, 0.35, 0.68, 0.42, 0.25],
        "Accuracy": [0.94, 0.99, 0.95, 0.98, 0.92, 0.99]
    }
    
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=labels, y=metrics_data["Precision"], name="Precision", marker_color='#60A5FA'))
    fig_metrics.add_trace(go.Bar(x=labels, y=metrics_data["Recall"], name="Recall", marker_color='#FBBF24'))
    fig_metrics.add_trace(go.Bar(x=labels, y=metrics_data["F1 Score"], name="F1 Score", marker_color='#F472B6'))
    
    fig_metrics.add_trace(go.Scatter(
        x=labels, y=metrics_data["Accuracy"], 
        name="Accuracy", mode='lines+markers',
        line=dict(color='#10B981', width=3)
    ))
    
    fig_metrics.update_layout(barmode='group', height=450, yaxis_range=[0, 1.1], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. Dataset Preview ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    sample_df = pd.DataFrame([
        {"id": "0000997932d777bf", "toxic": 0, "obscene": 0, "compound_pol": 0.5574, "comment_len": 46},
        {"id": "0002bcb3da6cb337", "toxic": 1, "obscene": 1, "compound_pol": -0.7783, "comment_len": 8},
        {"id": "0005c987bdfc9d4b", "toxic": 1, "obscene": 0, "compound_pol": -0.4588, "comment_len": 54}
    ])
    st.dataframe(sample_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. BERT Explanation ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Architecture: BERT")
    st.write("This model uses Bidirectional Encoder Representations from Transformers to analyze word context rather than just isolated keywords.")
    st.markdown('</div>', unsafe_allow_html=True)
