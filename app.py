"""
MCI Detection - Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, time
from PIL import Image
from mri_validator import validate_brain_mri

st.set_page_config(
    page_title="MCI Detection | Alzheimer's Early Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.main-header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  padding:2.5rem 2rem;border-radius:16px;margin-bottom:2rem;text-align:center;
  box-shadow:0 8px 32px rgba(0,0,0,.3);}
.main-header h1{color:#fff;font-size:2.2rem;font-weight:700;margin:0 0 .4rem;}
.main-header p{color:#a0aec0;font-size:1rem;margin:0;}
.badge{display:inline-block;background:rgba(99,179,237,.15);color:#63b3ed;
  border:1px solid rgba(99,179,237,.3);padding:2px 12px;border-radius:20px;
  font-size:.75rem;margin-top:8px;font-weight:600;}

.result-cn{background:linear-gradient(135deg,#1a365d,#2a4a7f);border:2px solid #4299e1;
  border-radius:16px;padding:1.8rem;text-align:center;box-shadow:0 4px 20px rgba(66,153,225,.25);}
.result-mci{background:linear-gradient(135deg,#742a2a,#9b2335);border:2px solid #fc8181;
  border-radius:16px;padding:1.8rem;text-align:center;box-shadow:0 4px 20px rgba(252,129,129,.25);}

.metric-box{background:#1e2a3a;border:1px solid #2d3748;border-radius:12px;
  padding:1rem;text-align:center;margin-bottom:.5rem;}
.metric-val{font-size:1.8rem;font-weight:700;color:#63b3ed;}
.metric-lbl{font-size:.72rem;color:#718096;text-transform:uppercase;
  letter-spacing:.8px;margin-top:2px;}
.info-card{background:#1e2a3a;border-left:4px solid #4299e1;border-radius:0 10px 10px 0;
  padding:1rem 1.2rem;margin-bottom:1rem;color:#cbd5e0;font-size:.88rem;line-height:1.6;}
.warn-card{background:#2d1f1f;border-left:4px solid #fc8181;border-radius:0 10px 10px 0;
  padding:1rem 1.2rem;margin-bottom:1rem;color:#fed7d7;font-size:.88rem;}
.valid-badge{background:rgba(72,187,120,.12);border:1px solid rgba(72,187,120,.35);
  border-radius:8px;padding:8px 14px;margin-bottom:16px;}
.stButton>button{background:linear-gradient(135deg,#4299e1,#3182ce)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-weight:600!important;font-size:.95rem!important;width:100%!important;}
.footer{text-align:center;color:#4a5568;font-size:.78rem;
  padding:2rem 0 1rem;border-top:1px solid #2d3748;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    for path in ["saved_models/best_model_finetuned.keras",
                 "saved_models/best_model.keras"]:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    return None, None

def preprocess_pil(img):
    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def prob_chart(cn, mci):
    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    bars = ax.barh(["CN","MCI"],[cn,mci],
                   color=["#4299e1","#fc8181"],height=.4,edgecolor="none")
    for bar,v in zip(bars,[cn,mci]):
        ax.text(v+.8,bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%",va="center",color="white",fontsize=11,fontweight="bold")
    ax.set_xlim(0,115); ax.spines[:].set_visible(False)
    ax.tick_params(colors="#a0aec0")
    for lbl in ax.get_yticklabels():
        lbl.set_color("white"); lbl.set_fontsize(11); lbl.set_fontweight("bold")
    ax.set_xlabel("Probability (%)",color="#a0aec0",fontsize=9)
    ax.grid(axis="x",color="#2d3748",linewidth=.5)
    fig.tight_layout(pad=.4)
    return fig


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About This Tool")
    st.markdown("""<div class='info-card'>
    Combines <b>ResNet50</b> and <b>DenseNet121</b> to classify brain MRI scans as:<br><br>
    • <b style='color:#63b3ed'>CN</b> — Cognitively Normal<br>
    • <b style='color:#fc8181'>MCI</b> — Mild Cognitive Impairment<br><br>
    Non-brain images are <b>automatically rejected</b>.
    </div>""", unsafe_allow_html=True)
    st.markdown("### 📋 Accepted Images")
    st.markdown("""<div class='info-card'>
    ✅ Brain MRI scans (JPG / PNG)<br>
    ✅ Axial, coronal, sagittal slices<br>
    ❌ Photos, selfies, screenshots<br>
    ❌ CT scans, X-rays, coloured images
    </div>""", unsafe_allow_html=True)
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""<div class='warn-card'>
    Research prototype only — not a certified medical device.
    Always consult a qualified neurologist.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style='color:#4a5568;font-size:.78rem;text-align:center;'>
    St. Thomas College of Engineering<br>and Technology, Mattannur<br><br>
    Adith C &nbsp;|&nbsp; Harinand P<br>
    Sruthin M &nbsp;|&nbsp; Navaneeth G<br><br>
    Guide: Ms Neethu T Reji
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🧠 MCI Detection System</h1>
  <p>Early Detection of Mild Cognitive Impairment from Brain MRI</p>
  <span class='badge'>ResNet50 + DenseNet121 · Stacked Feature Fusion · MRI Validation</span>
</div>
""", unsafe_allow_html=True)

model, model_path = load_model()
if model is None:
    st.error("⚠️ No trained model found. Run `2_model_training.py` first.")
    st.stop()
else:
    st.success(f"✅ Model loaded: `{model_path}`")

# ── Layout ────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📤 Upload MRI Image")
    uploaded = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"],
        help="Only grayscale brain MRI scans are accepted."
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption=f"📷 {uploaded.name}", use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{pil_img.size[0]}</div>
                <div class='metric-lbl'>Width (px)</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{pil_img.size[1]}</div>
                <div class='metric-lbl'>Height (px)</div></div>""",
                unsafe_allow_html=True)
        st.write("")
        analyse = st.button("🔍  Analyse This MRI Scan")
    else:
        st.info("👆 Upload a brain MRI image (JPG or PNG) to get started.")
        analyse = False


with right:
    st.markdown("### 📊 Analysis Result")

    if uploaded and analyse:
        pil_fresh = Image.open(uploaded)

        # ══════════════════════════════════════
        # STEP 1 — VALIDATE IMAGE
        # ══════════════════════════════════════
        with st.spinner("🔎 Checking image…"):
            is_valid, reason = validate_brain_mri(pil_fresh)

        if not is_valid:
            # ── BIG RED ERROR — cannot be missed ─────────
            st.error("🚫  **Invalid Image — Not a Brain MRI Scan**")

            st.markdown("""
            <div style="
                background: #1a0505;
                border: 2px solid #e53e3e;
                border-radius: 14px;
                padding: 1.8rem 2rem;
                margin-top: 0.5rem;
            ">
                <div style="
                    font-size: 3rem;
                    text-align: center;
                    margin-bottom: 1rem;
                ">🚫</div>

                <div style="
                    font-size: 1.25rem;
                    font-weight: 700;
                    color: #fc8181;
                    text-align: center;
                    margin-bottom: 0.5rem;
                ">INVALID IMAGE</div>

                <div style="
                    font-size: 0.9rem;
                    color: #fed7d7;
                    text-align: center;
                    margin-bottom: 1.2rem;
                ">This is <b>not</b> a brain MRI scan.<br>
                No analysis has been performed.</div>

                <div style="
                    background: rgba(229,62,62,0.1);
                    border: 1px solid rgba(229,62,62,0.35);
                    border-radius: 10px;
                    padding: 1rem 1.2rem;
                ">
                    <div style="
                        color: #fc8181;
                        font-size: 0.78rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.8px;
                        margin-bottom: 8px;
                    ">📋 Why was this rejected?</div>
                    <div style="
                        color: #fed7d7;
                        font-size: 0.84rem;
                        line-height: 1.9;
                        white-space: pre-line;
                    ">""" + reason + """</div>
                </div>

                <div style="
                    margin-top: 1.2rem;
                    background: rgba(255,255,255,0.04);
                    border-radius: 8px;
                    padding: 0.8rem 1rem;
                    font-size: 0.82rem;
                    color: #a0aec0;
                    line-height: 1.7;
                ">
                    <b style="color:#e2e8f0;">Please upload:</b><br>
                    ✔ A grayscale 2D brain MRI slice<br>
                    ✔ Axial, coronal, or sagittal view<br>
                    ✔ From an MRI scanner (not CT / X-ray)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Extra hint in an expander
            st.write("")
            with st.expander("💡 What does a valid brain MRI look like?"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Accepted**")
                    st.markdown("""
- Grayscale brain MRI
- Dark/black scanner background
- Bright brain tissue in centre
- Axial / coronal / sagittal slice
""")
                with col_b:
                    st.markdown("**❌ Rejected**")
                    st.markdown("""
- Colour photos (flowers, people)
- Selfies or portraits
- CT scans or X-ray images
- Screenshots / documents
- Any non-medical image
""")

        else:
            # ══════════════════════════════════════
            # STEP 2 — RUN MODEL
            # ══════════════════════════════════════
            with st.spinner("🧠 Analysing brain MRI…"):
                img_batch = preprocess_pil(pil_fresh)
                bar = st.progress(0)
                for i in range(1, 101):
                    time.sleep(0.005)
                    bar.progress(i)
                probs = model.predict(
                    {"resnet_input": img_batch, "densenet_input": img_batch},
                    verbose=0
                )[0]

            pred  = int(np.argmax(probs))
            cn_p  = float(probs[0]) * 100
            mci_p = float(probs[1]) * 100
            conf  = max(cn_p, mci_p)

            st.write("")
            st.markdown("""<div class='valid-badge'>
                <span style='color:#9ae6b4;font-size:0.83rem;'>
                ✅ &nbsp;Brain MRI validated successfully — analysis complete</span>
            </div>""", unsafe_allow_html=True)

            if pred == 0:
                st.markdown(f"""<div class='result-cn'>
                    <div style='font-size:2.8rem;'>✅</div>
                    <div style='font-size:1.6rem;font-weight:700;color:#63b3ed;margin:6px 0;'>
                        Cognitively Normal</div>
                    <div style='color:#e2e8f0;'>Confidence: <b>{conf:.1f}%</b></div>
                    <div style='color:#a0aec0;font-size:.82rem;margin-top:8px;'>
                        No significant signs of MCI detected.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-mci'>
                    <div style='font-size:2.8rem;'>⚠️</div>
                    <div style='font-size:1.6rem;font-weight:700;color:#fc8181;margin:6px 0;'>
                        Mild Cognitive Impairment</div>
                    <div style='color:#e2e8f0;'>Confidence: <b>{conf:.1f}%</b></div>
                    <div style='color:#fed7d7;font-size:.82rem;margin-top:8px;'>
                        Signs consistent with MCI detected. Please consult a neurologist.</div>
                </div>""", unsafe_allow_html=True)

            st.write("")
            st.markdown("**Probability Breakdown**")
            fig = prob_chart(cn_p, mci_p)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            m1, m2, m3 = st.columns(3)
            for col, label, value, color in [
                (m1, "CN Score",   cn_p,  "#63b3ed"),
                (m2, "MCI Score",  mci_p, "#fc8181"),
                (m3, "Confidence", conf,  "#68d391"),
            ]:
                with col:
                    st.markdown(f"""<div class='metric-box'>
                        <div class='metric-val' style='color:{color};'>{value:.1f}%</div>
                        <div class='metric-lbl'>{label}</div></div>""",
                        unsafe_allow_html=True)

            st.write("")
            st.markdown("""<div class='warn-card'>
                ⚕️ <b>Medical Disclaimer:</b> For research purposes only.
                Not a substitute for professional medical diagnosis.
            </div>""", unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""<div style='background:#1e2a3a;border-radius:12px;
            padding:3rem 2rem;text-align:center;margin-top:1rem;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🧠</div>
            <div style='color:#718096;'>Upload a brain MRI scan on the left<br>
            and click <b>Analyse</b> to see results.</div></div>""",
            unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:#1e2a3a;border-radius:12px;
            padding:3rem 2rem;text-align:center;margin-top:1rem;'>
            <div style='font-size:3rem;margin-bottom:12px;'>👈</div>
            <div style='color:#718096;'>Click <b>Analyse This MRI Scan</b>
            to run the prediction.</div></div>""",
            unsafe_allow_html=True)

st.markdown("""<div class='footer'>
    Early Detection of MCI using Stacked Feature Fusion of Pretrained CNNs &nbsp;|&nbsp;
    St. Thomas College of Engineering and Technology, Mattannur &nbsp;|&nbsp; 2026
</div>""", unsafe_allow_html=True)
