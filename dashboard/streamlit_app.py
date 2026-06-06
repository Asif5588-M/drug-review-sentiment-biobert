import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(
    page_title="Drug Review Sentiment Analyzer",
    page_icon="💊",
    layout="centered"
)

# ── Model Load ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL = "asif-nawaz-ml/biobert_drug_sentiment"  # HuggingFace path
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    return tokenizer, model

st.title("💊 Drug Review Sentiment Analyzer")
st.caption("BioBERT fine-tuned on 215,000+ patient drug reviews | By Asif Nawaz")
st.divider()

# ── Load ───────────────────────────────────────────────────
with st.spinner("Loading BioBERT model..."):
    tokenizer, model = load_model()

# ── Input ──────────────────────────────────────────────────
st.subheader("Enter a patient drug review:")
review = st.text_area(
    label="Review",
    placeholder="e.g. This medication worked great for my anxiety with minimal side effects...",
    height=150,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ── Predict ────────────────────────────────────────────────
if analyze and review.strip():
    with st.spinner("Analyzing..."):
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1)[0]

        labels = ["Negative", "Neutral", "Positive"]
        scores = {l: float(p) for l, p in zip(labels, probs)}
        pred   = max(scores, key=scores.get)

    st.divider()

    # Result
    color = {"Positive":"#3B6D11","Neutral":"#BA7517","Negative":"#E8593C"}[pred]
    emoji = {"Positive":"😊","Neutral":"😐","Negative":"😞"}[pred]

    st.markdown(f"""
    <div style='text-align:center; padding:20px; background:{color}22;
                border-radius:12px; border:1px solid {color}44;'>
        <h2 style='color:{color}; margin:0;'>{emoji} {pred}</h2>
        <p style='color:#666; margin:4px 0 0;'>Predicted Sentiment</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Confidence bars
    st.subheader("Confidence Scores:")
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        col_a, col_b = st.columns([3,1])
        col_a.progress(score, text=label)
        col_b.write(f"**{score*100:.1f}%**")

elif analyze and not review.strip():
    st.warning("Please enter a review first!")

st.divider()

# ── Examples ───────────────────────────────────────────────
st.subheader("Try these examples:")
examples = [
    "This medication completely changed my life. No side effects and works perfectly!",
    "It helped a little but the side effects were terrible. Would not recommend.",
    "Seems okay so far, been taking it for 2 weeks. Not sure yet if it's working."
]

for ex in examples:
    if st.button(ex[:60]+"...", use_container_width=True):
        st.session_state['example'] = ex

st.divider()
st.caption("Model: BioBERT | Dataset: UCI Drug Reviews (215k) | Accuracy: 83.2% | F1: 0.848")