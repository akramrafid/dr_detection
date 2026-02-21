import streamlit as st
import requests
import base64
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "DR Detection System",
    page_icon  = "🔬",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A3557, #2E6DA4);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid;
        text-align: center;
        margin: 10px 0;
    }
    .advice-box {
        background: #EBF5FB;
        border-left: 5px solid #2E6DA4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #DEE2E6;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Diabetic Retinopathy Detection System</h1>
    <p>AI-powered retinal fundus image analysis using EfficientNet-B5 + ViT-B/16 Ensemble</p>
    <p><small>Python Based Project Development | Group 09 | 10th Semester</small></p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses deep learning to automatically grade 
    diabetic retinopathy severity from retinal fundus images.
    
    **DR Grades:**
    - 🟢 Grade 0 — No DR
    - 🟡 Grade 1 — Mild
    - 🟠 Grade 2 — Moderate  
    - 🔴 Grade 3 — Severe
    - 🟣 Grade 4 — Proliferative
    
    **Model Performance:**
    - EfficientNet-B5 QWK: 0.9098
    - ViT-B/16 QWK: 0.9090
    - Ensemble QWK: **0.9030**
    
    **Dataset:** APTOS 2019
    3,662 retinal fundus images
    """)

    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")

    # ── Check API status ──────────────────────────────────────────
    if st.button("🔌 Check API Status"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ API Running\n\nDevice: {data['device']}")
            else:
                st.error("❌ API not responding")
        except:
            st.error("❌ Cannot connect to API")

# ── Main content ──────────────────────────────────────────────────
st.markdown("### 📤 Upload Retinal Fundus Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type   = ["png", "jpg", "jpeg"],
    help   = "Upload a retinal fundus image for DR grading"
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"File: {uploaded_file.name}")

    with col2:
        if st.button("🔍 Analyse Image", type="primary", use_container_width=True):
            with st.spinner("🧠 Analysing retinal image..."):
                try:
                    # ── Call API ──────────────────────────────────
                    files    = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        f"{api_url}/predict",
                        files   = {"file": (uploaded_file.name,
                                            uploaded_file.getvalue(),
                                            "image/png")},
                        timeout = 30
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            # ── Grade result ──────────────────────
                            grade = result["grade"]
                            color = result["grade_color"]
                            label = result["grade_label"]

                            st.markdown(f"""
                            <div class="result-box" style="border-color: {color}; background: {color}22;">
                                <h2 style="color: {color};">Grade {grade}: {label}</h2>
                                <p style="font-size: 18px;">Raw Score: {result['raw_score']:.3f} / 4.0</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # ── Metrics ───────────────────────────
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("DR Grade", f"{grade} / 4")
                            with m2:
                                st.metric("Severity", label)
                            with m3:
                                st.metric("Raw Score", f"{result['raw_score']:.3f}")

                            # ── Clinical advice ───────────────────
                            st.markdown(f"""
                            <div class="advice-box">
                                <b>🏥 Clinical Recommendation:</b><br>
                                {result['advice']}
                            </div>
                            """, unsafe_allow_html=True)

                            # ── Images side by side ───────────────
                            st.markdown("### 🔥 Grad-CAM++ Explainability")
                            st.caption("Red regions show where the model focused most")

                            img_col1, img_col2 = st.columns(2)

                            with img_col1:
                                st.markdown("**Original (Preprocessed)**")
                                orig_bytes = base64.b64decode(result["original_b64"])
                                st.image(Image.open(io.BytesIO(orig_bytes)),
                                        use_container_width=True)

                            with img_col2:
                                st.markdown("**Grad-CAM++ Heatmap**")
                                heat_bytes = base64.b64decode(result["heatmap_b64"])
                                st.image(Image.open(io.BytesIO(heat_bytes)),
                                        use_container_width=True)

                    else:
                        st.error(f"❌ API Error: {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure FastAPI is running!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    Diabetic Retinopathy Detection System | Python Based Project Development<br>
    Group 09 | Ovick Hassan | Shah Md. Imtiaz Chowdhury | Akram Rafid | Muntasir Adnan Eram
</div>
""", unsafe_allow_html=True)