"""
Streamlit Frontend — Credit Card Fraud Detection
Sends transaction features to the FastAPI backend and displays the prediction.
"""

import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
    .fraud-box {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        border-radius: 12px; padding: 24px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 4px 20px rgba(255,68,68,0.4);
    }
    .safe-box {
        background: linear-gradient(135deg, #00c853 0%, #007c2f 100%);
        border-radius: 12px; padding: 24px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 4px 20px rgba(0,200,83,0.4);
    }
    .metric-card {
        background: #1e2130; border-radius: 10px; padding: 16px;
        text-align: center; border: 1px solid #2d3250;
    }
    .stSlider > div > div > div > div { background-color: #3d5af1; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Credit Card Fraud Detector")
st.markdown("Enter transaction details below. The model will predict whether it is **fraudulent or legitimate**.")
st.divider()

# ── API Health Check ──────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

api_ok = check_api()
if not api_ok:
    st.warning("⚠️ **FastAPI server is not running.** Start it with: `uvicorn api.main:app --reload`", icon="⚠️")

# ── Input Form ────────────────────────────────────────────────────────────────
tab_manual, tab_json = st.tabs(["📝 Manual Input", "📋 Paste JSON"])

with tab_manual:
    st.markdown("### Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=406.0, format="%.2f")
        amount_val = st.number_input("Amount ($)", value=1.00, min_value=0.0, format="%.2f")

    st.markdown("### Anonymized Features (V1 – V28)")
    st.caption("These are PCA-transformed features. Use values from your dataset or leave at 0.0 for a neutral transaction.")

    v_vals = {}
    cols = st.columns(7)
    for i in range(1, 29):
        with cols[(i - 1) % 7]:
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

    st.markdown("")
    predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True, disabled=not api_ok)

    if predict_btn:
        payload = {"Time": time_val, "Amount": amount_val, **v_vals}
        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.divider()
                st.markdown("## 🔎 Prediction Result")

                if result["prediction"] == 1:
                    st.markdown(f'<div class="fraud-box">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-box">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)

                st.markdown("")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#aaa;margin:0">Fraud Probability</p>
                        <p style="font-size:2rem;font-weight:700;margin:0;color:{'#ff4444' if result['prediction']==1 else '#00c853'}">
                            {result['fraud_probability']*100:.2f}%
                        </p>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#aaa;margin:0">Decision Threshold</p>
                        <p style="font-size:2rem;font-weight:700;margin:0;color:#e0e0e0">
                            {result['threshold']:.2f}
                        </p>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#aaa;margin:0">Model Label</p>
                        <p style="font-size:2rem;font-weight:700;margin:0;color:#e0e0e0">
                            {result['label']}
                        </p>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📄 Raw API Response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure it is running at `http://127.0.0.1:8000`.")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_json:
    st.markdown("### Paste a JSON transaction")
    st.caption("Paste a full transaction object (Time, V1–V28, Amount).")

    sample = {
        "Time": 406.0,
        "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998,
        "V5": -0.522, "V6": -1.427, "V7": -2.537, "V8": 1.392,
        "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.900,
        "V13": -0.595, "V14": -4.289, "V15": 0.390, "V16": -1.141,
        "V17": -2.831, "V18": -0.017, "V19": 0.416, "V20": 0.126,
        "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
        "V25": 0.045, "V26": 0.177, "V27": 0.261, "V28": -0.143,
        "Amount": 1.0,
    }

    json_input = st.text_area(
        "JSON Input",
        value=json.dumps(sample, indent=2),
        height=300,
    )

    json_btn = st.button("🔍 Predict from JSON", type="primary", disabled=not api_ok)
    if json_btn:
        try:
            payload = json.loads(json_input)
            with st.spinner("Analyzing..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                if result["prediction"] == 1:
                    st.markdown(f'<div class="fraud-box">🚨 FRAUD DETECTED — {result["fraud_probability"]*100:.2f}% probability</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-box">✅ LEGITIMATE — {result["fraud_probability"]*100:.2f}% fraud probability</div>', unsafe_allow_html=True)
                st.json(result)
        except json.JSONDecodeError:
            st.error("Invalid JSON. Please check your input.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem'>Credit Card Fraud Detection · XGBoost + FastAPI + Streamlit</p>",
    unsafe_allow_html=True
)
