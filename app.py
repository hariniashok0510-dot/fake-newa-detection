
import os
import time
import joblib
import streamlit as st
from datetime import datetime
from preprocess import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f0f4f8; color: #1a202c; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3a5f 0%, #2d6a9f 100%) !important; }
section[data-testid="stSidebar"] * { color: #e8f4fd !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px; }
.hero { background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 60%, #1a8cff 100%); border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(30,58,95,0.25); position: relative; overflow: hidden; }
.hero::after { content: '🔍'; position: absolute; right: 2.5rem; font-size: 5rem; opacity: 0.12; }
.hero h1 { color: #ffffff; font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero p { color: rgba(255,255,255,0.75); margin: 0.4rem 0 0; font-size: 1rem; }
.hero-badge { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 0.3rem 0.9rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600; display: inline-block; margin-top: 0.8rem; }
.card { background: #ffffff; border-radius: 16px; padding: 1.8rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; margin-bottom: 1.2rem; }
.card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 1rem; }
textarea { background: #f8fafc !important; border: 2px solid #e2e8f0 !important; border-radius: 12px !important; color: #1a202c !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.88rem !important; }
textarea:focus { border-color: #2d6a9f !important; box-shadow: 0 0 0 3px rgba(45,106,159,0.1) !important; }
.stButton > button { background: linear-gradient(135deg, #1e3a5f, #2d6a9f) !important; color: white !important; font-weight: 700 !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; width: 100% !important; box-shadow: 0 4px 15px rgba(30,58,95,0.3) !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(30,58,95,0.4) !important; }
.result-real { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #16a34a; border-radius: 16px; padding: 1.8rem 2rem; text-align: center; animation: slideUp 0.4s ease; }
.result-real .emoji { font-size: 2.5rem; }
.result-real .label { color: #15803d; font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
.result-real .sub { color: #166534; font-size: 0.9rem; }
.result-fake { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 2px solid #dc2626; border-radius: 16px; padding: 1.8rem 2rem; text-align: center; animation: slideUp 0.4s ease; }
.result-fake .emoji { font-size: 2.5rem; }
.result-fake .label { color: #dc2626; font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
.result-fake .sub { color: #991b1b; font-size: 0.9rem; }
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.2rem; }
.stat-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; text-align: center; }
.stat-box .num { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #2d6a9f; }
.stat-box .lbl { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
.history-item { display: flex; align-items: center; justify-content: space-between; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
.history-text { font-size: 0.82rem; color: #374151; flex: 1; margin-right: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.badge-real { background: #dcfce7; color: #15803d; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.badge-fake { background: #ffe4e6; color: #dc2626; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.history-time { font-size: 0.7rem; color: #94a3b8; margin-left: 0.5rem; }
.tip-box { background: #eff6ff; border-left: 4px solid #2d6a9f; border-radius: 0 10px 10px 0; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.85rem; color: #1e40af; }
@keyframes slideUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
.stProgress > div > div { background: linear-gradient(90deg, #1e3a5f, #2d6a9f) !important; border-radius: 10px !important; }
.stProgress > div { background: #e2e8f0 !important; border-radius: 10px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_real' not in st.session_state:
    st.session_state.total_real = 0
if 'total_fake' not in st.session_state:
    st.session_state.total_fake = 0

@st.cache_resource(show_spinner=False)
def load_model():
    mp = os.path.join('model', 'model.pkl')
    vp = os.path.join('model', 'vectorizer.pkl')
    if not os.path.exists(mp) or not os.path.exists(vp):
        return None, None
    return joblib.load(mp), joblib.load(vp)

model, vectorizer = load_model()

with st.sidebar:
    st.markdown("### 🔍 Fake News Detector")
    st.markdown("**📊 Session Stats**")
    col1, col2 = st.columns(2)
    col1.metric("✅ Real", st.session_state.total_real)
    col2.metric("🚨 Fake", st.session_state.total_fake)
    total_s = st.session_state.total_real + st.session_state.total_fake
    if total_s > 0:
        st.markdown(f"**Fake Rate:** {int(st.session_state.total_fake/total_s*100)}%")
        st.progress(st.session_state.total_fake / total_s)
    st.markdown("---")
    st.markdown("**ℹ️ How it works**")
    st.markdown("1. Text cleaned & stemmed\n2. TF-IDF vectorization\n3. ML model predicts\n4. Confidence shown")
    st.markdown("---")
    st.markdown(f"**Model:** {'✅ Loaded' if model else '❌ Not Found'}")
    st.markdown("**Algorithm:** Passive Aggressive")
    st.markdown("**Accuracy:** ~95%")
    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.total_real = 0
            st.session_state.total_fake = 0
            st.rerun()

st.markdown("""
<div class="hero">
    <div>
        <h1>Fake News Detector</h1>
        <p>Paste any news article — get an instant AI-powered verdict</p>
        <span class="hero-badge">🤖 NLP + Machine Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found! Run `python train_model.py` first.")
    st.stop()

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<div class="card"><div class="card-title">📰 News Article Input</div>', unsafe_allow_html=True)
    news_input = st.text_area(label="", height=240, placeholder="Paste the full news article here...", label_visibility="collapsed", key="news_input")
    wc = len(news_input.split()) if news_input.strip() else 0
    if wc > 0:
        color = "#15803d" if wc >= 20 else "#d97706"
        st.markdown(f"<p style='font-size:0.82rem;color:{color};margin:0'>📝 {wc} words {'✓ Good length' if wc >= 20 else '— add more for better accuracy'}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    analyse_btn = st.button("🔍  Analyse Article", use_container_width=True)

    if analyse_btn:
        if not news_input.strip():
            st.warning("⚠️ Please paste some news text first.")
        elif wc < 5:
            st.warning("⚠️ Please enter at least 5 words.")
        else:
            with st.spinner("🔄 Analysing article..."):
                time.sleep(0.6)
                cleaned = clean_text(news_input)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                score = model.decision_function(vectorized)[0]
                confidence = min(round(abs(score) * 18, 1), 99.9)
            st.session_state.history.insert(0, {'text': news_input[:75] + "..." if len(news_input) > 75 else news_input, 'prediction': prediction, 'confidence': confidence, 'time': datetime.now().strftime("%H:%M"), 'words': wc})
            if prediction == "REAL":
                st.session_state.total_real += 1
            else:
                st.session_state.total_fake += 1
            if prediction == "REAL":
                st.markdown('<div class="result-real"><div class="emoji">✅</div><div class="label">REAL NEWS</div><div class="sub">This article appears credible and legitimate</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fake"><div class="emoji">🚨</div><div class="label">FAKE NEWS</div><div class="sub">This article shows patterns of misinformation</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stats-grid"><div class="stat-box"><div class="num">{confidence}%</div><div class="lbl">Confidence</div></div><div class="stat-box"><div class="num">{wc}</div><div class="lbl">Words</div></div><div class="stat-box"><div class="num">{len(cleaned.split())}</div><div class="lbl">Features</div></div></div>', unsafe_allow_html=True)
            st.markdown("<br>**Confidence Level**", unsafe_allow_html=True)
            st.progress(min(confidence / 100, 1.0))
            st.markdown('<div class="tip-box">💡 <strong>Tip:</strong> Always cross-verify with trusted sources like Reuters, BBC, or AP News.</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card"><div class="card-title">📊 Session Analysis</div>', unsafe_allow_html=True)
    total = st.session_state.total_real + st.session_state.total_fake
    if total == 0:
        st.markdown("<p style='color:#94a3b8;font-size:0.85rem;text-align:center;padding:1rem 0'>Analyse articles to see chart</p>", unsafe_allow_html=True)
    else:
        import pandas as pd
        chart_data = pd.DataFrame({'Category': ['✅ Real', '🚨 Fake'], 'Count': [st.session_state.total_real, st.session_state.total_fake]}).set_index('Category')
        st.bar_chart(chart_data, height=200)
        real_pct = int(st.session_state.total_real / total * 100)
        st.markdown(f'<div style="display:flex;gap:0.8rem;margin-top:0.5rem;"><div style="flex:1;background:#dcfce7;border-radius:8px;padding:0.6rem;text-align:center;"><span style="color:#15803d;font-weight:700;font-size:1.1rem">{real_pct}%</span><div style="color:#166534;font-size:0.75rem">Real</div></div><div style="flex:1;background:#ffe4e6;border-radius:8px;padding:0.6rem;text-align:center;"><span style="color:#dc2626;font-weight:700;font-size:1.1rem">{100-real_pct}%</span><div style="color:#991b1b;font-size:0.75rem">Fake</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🕒 Analysis History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color:#94a3b8;font-size:0.85rem;text-align:center;padding:1rem 0'>No history yet</p>", unsafe_allow_html=True)
    else:
        for item in st.session_state.history[:8]:
            badge = '<span class="badge-real">✅ REAL</span>' if item['prediction'] == 'REAL' else '<span class="badge-fake">🚨 FAKE</span>'
            st.markdown(f'<div class="history-item"><span class="history-text">{item["text"]}</span>{badge}<span class="history-time">{item["time"]}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #e2e8f0;">Built with ❤️ using Streamlit + Scikit-learn | Fake News Detector</div>', unsafe_allow_html=True)

