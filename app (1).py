import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta

# --- 1. Page Config ---
st.set_page_config(page_title="Model | Monitoring Dashboard", layout="wide")

# --- 2. Professional CSS Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f1f5f9; }
    .main-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: left;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e293b;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
    }
    .legal-card {
        background-color: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Backend Logic ---
@st.cache_resource
def load_model():
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

# --- 4. Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #f1f5f9;'>Model</h2>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Main Checker", "Dataset Insights"],
        icons=["shield-check", "graph-up"],
        default_index=0,
        styles={
            "container": {"background-color": "#0f172a"},
            "nav-link": {"color": "#94a3b8", "text-align": "left"},
            "nav-link-selected": {"background-color": "#1e293b", "color": "#f1f5f9"},
        }
    )

# --- 5. Main Checker Page ---
if selected == "Main Checker":
    st.markdown("<h2 style='color: #1e293b;'>Cyberbullying & Hate Speech Detection</h2>", unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Analyze Text")
        
        # Suggestion buttons
        c1, c2, c3 = st.columns(3)
        if c1.button("Neutral"): st.session_state.user_text = "The weather is quite nice today for a walk."
        if c2.button("Harsh"): st.session_state.user_text = "You are a pathetic loser, I hate everything about you!"
        if c3.button("Threat"): st.session_state.user_text = "I am going to find where you live and hurt you."

        user_input = st.text_area("Input Content", value=st.session_state.get("user_text", ""), height=150)
        analyze_btn = st.button("Run Analysis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Analysis Breakdown")
        if analyze_btn and user_input:
            probs = predict(user_input)
            
            # Show Warning Box based on image
            if any(p > 0.5 for p in probs):
                st.error("DANGER: High toxicity or hate speech detected!")
            else:
                st.success("CLEAN: No significant toxicity detected.")

            # Bar Plot
            fig = go.Figure()
            colors = ["#ef4444", "#991b1b", "#10b981", "#059669", "#3b82f6", "#1d4ed8"]
            for label, prob, color in zip(labels, probs, colors):
                fig.add_trace(go.Bar(y=[label], x=[prob], orientation='h', marker=dict(color=color),
                                     text=f"{prob:.1%}", textposition='auto'))
            
            fig.update_layout(xaxis=dict(range=[0, 1]), height=300, margin=dict(l=0, r=0, t=10, b=10),
                              showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Results will appear here after analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Legal Section
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Indian Legal Framework")
    l1, l2 = st.columns(2)
    laws = [
        ("Section 66E (IT Act)", "Violation of privacy.", "3 years jail / ₹2L fine"),
        ("Section 67 (IT Act)", "Obscene material.", "3 years jail + fine"),
        ("Section 507 (IPC)", "Criminal intimidation.", "2 years imprisonment"),
        ("Section 509 (IPC)", "Insulting modesty.", "3 years jail + fine")
    ]
    for i, (title, desc, punishment) in enumerate(laws):
        with (l1 if i < 2 else l2):
            st.markdown(f'<div class="legal-card"><b>{title}</b>: {desc}<br><span style="color:#ef4444;">{punishment}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. Dataset Insights Page (Dashboard Style) ---
elif selected == "Dataset Insights":
    st.markdown("<h2 style='color: #1e293b;'>AI-driven Content Monitoring Dashboard</h2>", unsafe_allow_html=True)
    
    # Top Row: Pie and Trends
    row1_c1, row1_c2, row1_c3 = st.columns([1, 1.5, 1])
    
    with row1_c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("<b>Severity Distribution</b>", unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(labels=['Low Severity', 'High Severity'], values=[85, 15], hole=.6, marker_colors=['#3b82f6', '#94a3b8'])])
        fig_pie.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_c2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("<b>Toxicity Trend Over Time</b>", unsafe_allow_html=True)
        dates = [(datetime.now() - timedelta(days=x)).strftime('%d %b') for x in range(10, 0, -1)]
        trends = [1, 1.5, 1.2, 2, 2.5, 4, 3.8, 5, 4.5, 6]
        fig_trend = go.Figure(go.Scatter(x=dates, y=trends, fill='tozeroy', line_color='#3b82f6'))
        fig_trend.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_c3:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("<b>Category Frequency</b>", unsafe_allow_html=True)
        # Based on Training counts image
        fig_freq = go.Figure(go.Bar(x=labels, y=[38, 2, 11, 1, 23, 1], marker_color='#3b82f6'))
        fig_freq.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_freq, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Middle Row: Metrics and detailed performance
    row2_c1, row2_c2 = st.columns([1, 2.5])
    
    with row2_c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("<b>Model Performance Metrics</b>", unsafe_allow_html=True)
        # Metrics from reference image
        perf = {"Accuracy": "98.2%", "Precision": "97.5%", "Recall": "96.8%", "F1-Score": "97.1%"}
        for k, v in perf.items():
            st.markdown(f'<div style="margin-bottom:10px;"><span class="metric-label">{k}:</span> <span class="metric-value">{v}</span></div>', unsafe_allow_html=True)
        st.caption("Trained on Jigsaw Dataset")
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_c2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("<b>Per-label Evaluation (Precision, Recall, F1)</b>", unsafe_allow_html=True)
        # Data from classification report image
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Bar(x=labels, y=[1.0, 0, 0.92, 1.0, 0.61, 1.0], name="Precision", marker_color='#60a5fa'))
        fig_eval.add_trace(go.Bar(x=labels, y=[0.4, 0, 0.22, 0.51, 0.32, 0.15], name="Recall", marker_color='#fbbf24'))
        fig_eval.add_trace(go.Bar(x=labels, y=[0.58, 0, 0.35, 0.68, 0.42, 0.25], name="F1 Score", marker_color='#f472b6'))
        fig_eval.update_layout(barmode='group', height=300, margin=dict(l=0, r=0, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_eval, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bottom Row: Flagged Comments
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.write("<b>Recent Flagged Comments</b>", unsafe_allow_html=True)
    logs = pd.DataFrame({
        "Comment Text": ["This user is a complete idiot.", "I will find you and hurt you.", "The weather is okay."],
        "Detected Category": ["Insult", "Threat", "Neutral"],
        "Predicted Severity": ["Medium", "High", "Low"],
        "Action": ["Review", "Flagged", "Approved"]
    })
    st.table(logs)
    st.markdown('</div>', unsafe_allow_html=True)
