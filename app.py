from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import plotly.express as px  # noqa: F401 — kept for future chart expansions
import plotly.graph_objects as go
import streamlit as st
from streamlit import cache_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PAGE CONFIG + STATIC CSS (rendered once)
# =============================================================================

st.set_page_config(
    page_title="🏥 AI Medical Prescription Verification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
.main-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 2rem;}
.metric-card {background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #667eea; margin: 1rem 0;}
.alert-high {background-color: #fee; border: 1px solid #fcc; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;}
.alert-moderate {background-color: #fff4e6; border: 1px solid #ffd700; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;}
.alert-low {background-color: #f0f8ff; border: 1px solid #87ceeb; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;}
.success-box {background-color: #f0fff0; border: 1px solid #90ee90; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;}
.ai-insight {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
.stSelectbox > div > div > select {background-color: #f8f9fa;}
.drug-card {background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 0.5rem 0;}
.interaction-high {border-left-color: #dc3545 !important;}
.interaction-moderate {border-left-color: #ffc107 !important;}
.interaction-low {border-left-color: #17a2b8 !important;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
# CONFIGURATION + HTTP TRANSPORT LAYER
# =============================================================================

API_URL = "https://ai-medical-prescription.onrender.com"
_API_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _build_sync_http_client() -> httpx.Client:
    """Build pooled httpx.Client; degrade to HTTP/1.1 if h2 extra not installed."""
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=30.0)
    try:
        return httpx.Client(base_url=API_BASE_URL, timeout=_API_TIMEOUT, limits=limits, http2=True)
    except ImportError:
        logger.warning("h2 package not installed — Streamlit client using HTTP/1.1")
        return httpx.Client(base_url=API_BASE_URL, timeout=_API_TIMEOUT, limits=limits, http2=False)


@st.cache_resource(show_spinner=False)
def _get_http_client() -> httpx.Client:
    """One TCP-connection-pooled httpx.Client per Streamlit session cache entry.
    Keeps connections alive across API calls (vs. `requests` which rebuilds them)."""
    return _build_sync_http_client()


@cache_data(ttl=30, show_spinner=False)
def cached_get(endpoint: str) -> Dict[str, Any]:
    """Cached GET for read-only, infrequently changing endpoints (root, health)."""
    try:
        resp = _get_http_client().get(endpoint)
        resp.raise_for_status()
        return {"success": True, "data": resp.json()}
    except httpx.ConnectError:
        return {"success": False,
                "error": "❌ Cannot connect to AI backend server. Please ensure the FastAPI server is running on localhost:8000"}
    except httpx.TimeoutException:
        return {"success": False, "error": "⏱️ Request timed out. The AI analysis is taking longer than expected."}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"🚫 Server error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"🔥 Unexpected error: {str(e)}"}


def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Uncached API call for mutation / personalized endpoints."""
    client = _get_http_client()
    try:
        if method == "POST":
            resp = client.post(endpoint, json=data)
        else:
            resp = client.get(endpoint)
        resp.raise_for_status()
        return {"success": True, "data": resp.json()}
    except httpx.ConnectError:
        return {"success": False,
                "error": "❌ Cannot connect to AI backend server. Please ensure the FastAPI server is running on localhost:8000"}
    except httpx.TimeoutException:
        return {"success": False, "error": "⏱️ Request timed out. The AI analysis is taking longer than expected."}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"🚫 Server error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"🔥 Unexpected error: {str(e)}"}


# =============================================================================
# SESSION STATE  (initialized once per session, guarded by sentinel)
# =============================================================================

def _init_session() -> None:
    sentinel = "__init_done__"
    if st.session_state.get(sentinel):
        return
    defaults = {
        "analysis_results": None,
        "patient_drugs": [],
        "analysis_history": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state[sentinel] = True


_init_session()


# =============================================================================
# SHARED VIEW HELPERS (pure, side-effect free)
# =============================================================================

_SEVERITY_COLOR: Dict[str, str] = {
    "High": "#dc3545",
    "Moderate": "#ffc107",
    "Low": "#17a2b8",
}


@lru_cache(maxsize=32)
def get_severity_color(severity: str) -> str:
    return _SEVERITY_COLOR.get(severity, "#6c757d")


def create_safety_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Safety Score"},
        delta={"reference": 80},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightgray"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_interaction_chart(interactions: List[Dict[str, Any]]) -> Optional[go.Figure]:
    if not interactions:
        return None

    # Vectorized counting via pandas on normalized list avoids repeated dict gets.
    severities = [i.get("severity", "Unknown") for i in interactions]
    counts = pd.Series(severities).value_counts()

    color_map = {"High": "#dc3545", "Moderate": "#ffc107", "Low": "#17a2b8"}
    bar_colors = [color_map.get(s, "#6c757d") for s in counts.index]

    fig = go.Figure(data=[go.Bar(x=list(counts.index), y=list(counts.values), marker_color=bar_colors)])
    fig.update_layout(
        title="Drug Interactions by Severity",
        xaxis_title="Severity Level",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_risk_analysis_chart(results: Dict[str, Any], patient_age: int, drug_count: int) -> go.Figure:
    interaction_risk = len(results.get("interactions", [])) * 10
    allergy_risk = len(results.get("allergy_warnings", [])) * 25
    age_risk = 20 if patient_age > 75 else 5
    poly_risk = max(0, (drug_count - 3) * 5)
    overall_risk = 100 - results.get("safety_score", 100)

    categories = ["Interactions", "Allergies", "Age Factors", "Polypharmacy", "Overall Safety"]
    risk_scores = [interaction_risk, allergy_risk, age_risk, poly_risk, overall_risk]

    fig = go.Figure(data=go.Scatterpolar(
        r=risk_scores,
        theta=categories,
        fill="toself",
        name="Risk Assessment",
        line_color="rgb(255, 99, 71)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Comprehensive Risk Analysis",
        height=500,
    )
    return fig


# =============================================================================
# MAIN APP  (navigation orchestrator)
# =============================================================================

def main() -> None:
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Medical Prescription Verification System</h1>
        <p>Advanced AI-Powered Drug Interaction & Safety Analysis with HuggingFace & Google Gemini</p>
        <p>🤖 Intelligent Clinical Decision Support • 🔍 Advanced NLP Analysis • ⚡ Real-time Verification</p>
    </div>
    """, unsafe_allow_html=True)

    # Cached connection check — avoids hammering backend on every widget interaction.
    api_status = cached_get("/")
    if not api_status["success"]:
        st.error(api_status["error"])
        st.info("💡 **Quick Fix**: Start the FastAPI backend server with: `python main.py` or `uvicorn main:app`")
        return

    with st.expander("🔌 AI System Status", expanded=False):
        c1, c2 = st.columns(2)
        api_data = api_status["data"]
        with c1:
            st.success(f"**Backend Status**: {api_data.get('status', 'Unknown')}")
            st.info(f"**Version**: {api_data.get('version', 'Unknown')}")
        with c2:
            for service, info in api_data.get("ai_services", {}).items():
                st.write(f"**{service.title()}**: {info.get('status', 'Unknown')}")

    st.sidebar.markdown("## 🎛️ Control Panel")
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Home", "📊 Analysis Dashboard", "🤖 AI Insights", "📋 History", "⚙️ Settings"],
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Analysis Dashboard":
        show_analysis_dashboard()
    elif page == "🤖 AI Insights":
        show_ai_insights()
    elif page == "📋 History":
        show_history()
    elif page == "⚙️ Settings":
        show_settings()


# =============================================================================
# HOME PAGE
# =============================================================================

def show_home_page() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Patient Information")

    patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
    patient_weight = st.sidebar.number_input(
        "Weight (kg)", min_value=0.0, max_value=500.0, value=70.0, step=0.1, help="Patient's weight in kilograms",
    )

    conditions = st.sidebar.text_area(
        "Medical Conditions",
        placeholder="Enter conditions separated by commas (e.g., diabetes, hypertension, COPD)",
        help="List patient's current medical conditions",
    )
    conditions_list = [c.strip() for c in conditions.split(",") if c.strip()]

    allergies = st.sidebar.text_area(
        "Known Allergies",
        placeholder="Enter allergies separated by commas (e.g., penicillin, aspirin, sulfa)",
        help="List patient's known drug allergies",
    )
    allergies_list = [a.strip() for a in allergies.split(",") if a.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Analysis Options")
    use_ai_analysis = st.sidebar.checkbox(
        "🤖 Enable Advanced AI Analysis", value=True,
        help="Use HuggingFace and Google Gemini AI for enhanced analysis",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 💊 Prescription Entry")
        entry_method = st.radio(
            "Choose prescription entry method:",
            ["📝 Manual Entry", "📄 Text Analysis (NLP)", "💊 Quick Drug Search"],
        )
        if entry_method == "📝 Manual Entry":
            show_manual_entry()
        elif entry_method == "📄 Text Analysis (NLP)":
            show_text_analysis()
        else:
            show_quick_search()

    with col2:
        st.markdown("## 📋 Current Prescription")
        _render_patient_drugs()
        if st.session_state.patient_drugs:
            st.markdown("---")
            if st.button("🔍 **ANALYZE PRESCRIPTION**", type="primary", use_container_width=True):
                perform_analysis(patient_age, patient_weight, conditions_list, allergies_list, use_ai_analysis)
        else:
            st.info("📝 No drugs added yet. Add medications using the form on the left.")

    if st.session_state.analysis_results:
        st.markdown("---")
        display_analysis_results()


def _render_patient_drugs() -> None:
    """Render current prescription list with indexed remove buttons."""
    drugs = st.session_state.patient_drugs
    if not drugs:
        return

    to_remove: List[int] = []
    for i, drug in enumerate(drugs):
        st.markdown(f"""
        <div class="drug-card">
            <strong>{drug['name']}</strong><br>
            📊 Dosage: {drug.get('dosage', 'Not specified')}<br>
            ⏰ Frequency: {drug.get('frequency', 'Not specified')}
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"❌ Remove {drug['name']}", key=f"remove_{i}"):
            to_remove.append(i)

    if to_remove:
        # Remove in descending order so indices stay valid; single mutation + one rerun.
        for i in reversed(to_remove):
            st.session_state.patient_drugs.pop(i)
        st.rerun()


# --- Entry methods -----------------------------------------------------------

def show_manual_entry() -> None:
    with st.form("drug_entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            drug_name = st.text_input("💊 Drug Name", placeholder="e.g., Aspirin")
            dosage = st.text_input("📊 Dosage", placeholder="e.g., 100mg")
        with c2:
            frequency = st.text_input("⏰ Frequency", placeholder="e.g., Once daily")

        if st.form_submit_button("➕ Add Drug", use_container_width=True) and drug_name:
            st.session_state.patient_drugs.append({
                "name": drug_name.strip(),
                "dosage": dosage.strip() or None,
                "frequency": frequency.strip() or None,
            })
            st.success(f"✅ Added {drug_name} to prescription")
            st.rerun()


def show_text_analysis() -> None:
    st.markdown("### 📄 Prescription Text Analysis")
    st.info("🤖 **AI-Powered**: Our advanced NLP system will extract drug information from medical text using HuggingFace models.")

    medical_text = st.text_area(
        "Enter prescription or medical text:",
        placeholder="""Example:
Patient prescribed:
- Aspirin 100mg once daily for cardioprotection
- Metformin 500mg twice daily with meals for diabetes
- Lisinopril 10mg once daily for hypertension
- Warfarin 5mg daily (target INR 2.0-3.0)""",
        height=200,
    )

    if st.button("🔍 Extract Drugs with AI", type="primary"):
        if medical_text.strip():
            extract_drugs_from_text(medical_text.strip())
        else:
            st.warning("Please enter some medical text first.")


def show_quick_search() -> None:
    st.markdown("### 🔍 Quick Drug Search")
    common_drugs = [
        "Aspirin", "Paracetamol", "Metformin", "Warfarin", "Digoxin",
        "Furosemide", "Lisinopril", "Spironolactone", "Simvastatin",
        "Clopidogrel", "Rivaroxaban", "Apixaban", "Atorvastatin",
        "Amlodipine", "Losartan", "Omeprazole", "Ibuprofen",
    ]
    selected = st.selectbox("Select from common drugs:", [""] + common_drugs)

    c1, c2 = st.columns(2)
    with c1:
        custom_drug = st.text_input("Or enter custom drug name:")
    drug_name = selected or custom_drug

    if drug_name:
        c1a, c2a = st.columns(2)
        with c1a:
            dosage = st.text_input("Dosage:", key="quick_dosage")
        with c2a:
            frequency = st.text_input("Frequency:", key="quick_frequency")
        if st.button("➕ Add to Prescription", type="primary"):
            st.session_state.patient_drugs.append({
                "name": drug_name,
                "dosage": dosage or None,
                "frequency": frequency or None,
            })
            st.success(f"✅ Added {drug_name}")
            st.rerun()


# --- NLP extraction ----------------------------------------------------------

def extract_drugs_from_text(medical_text: str) -> None:
    with st.spinner("🤖 AI is analyzing the medical text..."):
        request_data = {
            "drugs": [],
            "patient": {"age": 45, "conditions": [], "allergies": []},
            "medical_text": medical_text,
            "use_ai_analysis": True,
        }
        result = make_api_request("/analyze", method="POST", data=request_data)

    if not result["success"]:
        st.error(result["error"])
        return

    extracted = result["data"].get("extracted_drugs", [])
    if not extracted:
        st.warning("🤔 No drugs were extracted from the text. Try adding them manually.")
        return

    st.success(f"🎉 AI extracted {len(extracted)} drugs from the text!")
    st.markdown("### 📋 Review Extracted Drugs:")

    # Use a form-like pattern: collect actions via session-state callback keys.
    for i, drug in enumerate(extracted):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.write(f"**{drug['name']}**")
            conf = drug.get("ai_confidence")
            if conf:
                pct = conf * 100
                st.progress(pct / 100)
                st.caption(f"AI Confidence: {pct:.1f}%")
        with c2:
            st.write(f"📊 {drug.get('dosage', 'N/A')}")
        with c3:
            if st.button(f"➕ Add", key=f"add_extracted_{i}"):
                st.session_state.patient_drugs.append(drug)
                st.success(f"Added {drug['name']}")
                st.rerun()


# --- Analysis orchestration --------------------------------------------------

def perform_analysis(
    age: int, weight: float, conditions: List[str], allergies: List[str], use_ai: bool,
) -> None:
    if not st.session_state.patient_drugs:
        st.warning("Please add at least one drug before analyzing.")
        return

    request_data = {
        "drugs": st.session_state.patient_drugs,
        "patient": {
            "age": age, "weight": weight,
            "conditions": conditions, "allergies": allergies,
        },
        "use_ai_analysis": use_ai,
    }

    # Real progress: stages are tied to what the backend actually does.
    # NO artificial sleeps — progress advances immediately based on phases.
    progress = st.progress(0)
    status = st.empty()

    with st.spinner("🔬 Performing comprehensive AI analysis..."):
        status.text("📤 Preparing request for AI systems...")
        progress.progress(15)

        status.text("🌐 Submitting to backend (HuggingFace + Gemini pipeline)...")
        progress.progress(35)

        result = make_api_request("/analyze", method="POST", data=request_data)

        status.text("📥 Processing results...")
        progress.progress(85)

    progress.progress(100)
    status.text("✅ Analysis complete!")
    # Clean up widgets without forcing extra rerender
    progress.empty()
    status.empty()

    if result["success"]:
        st.session_state.analysis_results = result["data"]
        st.session_state.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "patient_age": age,
            "drugs_count": len(st.session_state.patient_drugs),
            "safety_score": result["data"].get("safety_score", 0),
            "ai_enabled": use_ai,
        })
        st.success("🎉 Analysis completed successfully!")
    else:
        st.error(result["error"])


# =============================================================================
# RESULT DISPLAY
# =============================================================================

def display_analysis_results() -> None:
    results = st.session_state.analysis_results
    if not results:
        return

    st.markdown("# 📊 Analysis Results")

    safety_score = results.get("safety_score", 0)
    interactions = results.get("interactions", [])
    allergy_warnings = results.get("allergy_warnings", [])

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.plotly_chart(create_safety_gauge(safety_score), use_container_width=True)
    with c2:
        st.metric("Safety Score", f"{safety_score:.1f}/100")
        if safety_score >= 80:
            st.success("🟢 Low Risk")
        elif safety_score >= 60:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")
    with c3:
        st.metric("Interactions Found", len(interactions))
        st.metric("Allergy Warnings", len(allergy_warnings))

    tabs = st.tabs(["🚨 Interactions", "💊 Dosage", "⚠️ Allergies", "🔄 Alternatives", "🤖 AI Insights"])
    with tabs[0]:
        display_interactions(interactions)
    with tabs[1]:
        display_dosage_recommendations(results.get("dosage_recommendations", []))
    with tabs[2]:
        display_allergy_warnings(allergy_warnings)
    with tabs[3]:
        display_alternatives(results.get("alternatives", []))
    with tabs[4]:
        display_ai_insights(results)


def display_interactions(interactions: List[Dict[str, Any]]) -> None:
    if not interactions:
        st.success("✅ No drug interactions detected!")
        return
    st.warning(f"⚠️ Found {len(interactions)} drug interactions")
    fig = create_interaction_chart(interactions)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    for it in interactions:
        sev = it.get("severity", "Unknown")
        css_class = f"interaction-{sev.lower()}"
        st.markdown(f"""
        <div class="drug-card {css_class}">
            <h4>⚠️ {it.get('drug1')} ↔️ {it.get('drug2')}</h4>
            <p><strong>Severity:</strong> <span style="color: {get_severity_color(sev)}">{sev}</span></p>
            <p><strong>Description:</strong> {it.get('description', 'N/A')}</p>
            <p><strong>Recommendation:</strong> {it.get('recommendation', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_dosage_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    if not recommendations:
        st.success("✅ No dosage adjustments needed based on current information")
        return
    st.info(f"📊 {len(recommendations)} dosage recommendations found")
    for rec in recommendations:
        st.markdown(f"""
        <div class="drug-card">
            <h4>💊 {rec.get('drug_name')}</h4>
            <p><strong>Current Dosage:</strong> {rec.get('current_dosage')}</p>
            <p><strong>Recommended Dosage:</strong> {rec.get('recommended_dosage')}</p>
            <p><strong>Notes:</strong> {rec.get('notes', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_allergy_warnings(warnings: List[Dict[str, Any]]) -> None:
    if not warnings:
        st.success("✅ No allergy contraindications detected")
        return
    st.error(f"🚨 CRITICAL: {len(warnings)} allergy warnings!")
    for w in warnings:
        st.markdown(f"""
        <div class="alert-high">
            <h4>🚨 {w.get('drug_name')} - CONTRAINDICATED</h4>
            <p><strong>Allergen:</strong> {w.get('allergen')}</p>
            <p><strong>Severity:</strong> {w.get('severity')}</p>
            <p><strong>Recommendation:</strong> {w.get('recommendation')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_alternatives(alternatives: List[Dict[str, Any]]) -> None:
    if not alternatives:
        st.success("✅ No alternative medications needed")
        return
    st.info(f"🔄 {len(alternatives)} alternative medications suggested")
    for alt in alternatives:
        st.markdown(f"""
        <div class="success-box">
            <h4>🔄 Alternative for {alt.get('original_drug')}</h4>
            <p><strong>Suggested Alternative:</strong> {alt.get('alternative')}</p>
            <p><strong>Dosage:</strong> {alt.get('dosage')}</p>
            <p><strong>Reason:</strong> {alt.get('reason')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_ai_insights(results: Dict[str, Any]) -> None:
    ai_plain = results.get("ai_insights", [])
    if ai_plain:
        st.markdown("### 🧠 Traditional AI Insights")
        for insight in ai_plain:
            st.markdown(f'<div class="ai-insight">{insight}</div>', unsafe_allow_html=True)

    enhanced = results.get("enhanced_ai_insights", [])
    if enhanced:
        st.markdown("### 🚀 Enhanced AI Insights")
        for i in enhanced:
            pct = i.get("confidence", 0) * 100
            color = "#28a745" if pct >= 80 else ("#ffc107" if pct >= 60 else "#dc3545")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); border-left: 4px solid {color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: {color};">🤖 {i.get('source', 'AI System')}</h4>
                <p><strong>Insight:</strong> {i.get('insight', 'No insight available')}</p>
                <p><strong>Category:</strong> {i.get('type', 'general').replace('_', ' ').title()}</p>
                <div style="background-color: #f8f9fa; border-radius: 4px; padding: 0.5rem;">
                    <strong>Confidence Score:</strong> {pct:.1f}%
                    <div style="background-color: {color}; height: 10px; width: {pct}%; border-radius: 5px; margin-top: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    hf = results.get("huggingface_analysis", {})
    if hf:
        st.markdown("### 🤗 HuggingFace AI Analysis")
        with st.expander("View HuggingFace Results", expanded=False):
            st.json(hf)

    gemini = results.get("gemini_analysis", {})
    if gemini:
        st.markdown("### ✨ Google Gemini Analysis")
        with st.expander("View Gemini Results", expanded=False):
            st.json(gemini)

    for w in results.get("warnings", []):
        st.warning(w)


# =============================================================================
# ANALYSIS DASHBOARD
# =============================================================================

def show_analysis_dashboard() -> None:
    st.markdown("# 📊 Analysis Dashboard")
    results = st.session_state.analysis_results
    if not results:
        st.info("🔍 No analysis results available. Please run an analysis first.")
        return

    safety_score = results.get("safety_score", 0)
    interactions_data = results.get("interactions", [])
    allergy_warnings = results.get("allergy_warnings", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Safety Score", f"{safety_score:.1f}")
    c2.metric("Interactions", len(interactions_data))
    c3.metric("Allergy Warnings", len(allergy_warnings))
    c4.metric("Alternatives", len(results.get("alternatives", [])))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_safety_gauge(safety_score), use_container_width=True)
    with col2:
        fig = create_interaction_chart(interactions_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interactions to display")

    st.markdown("### 💊 Drug Analysis Summary")
    drugs = st.session_state.patient_drugs
    if not drugs:
        return

    # Precompute lookup indices once → O(drugs + interactions + allergies) instead of nested loops.
    drug_lower_list = [d["name"].lower() for d in drugs]
    interaction_count_map: Dict[str, int] = {name: 0 for name in drug_lower_list}
    for it in interactions_data:
        a, b = it.get("drug1", "").lower(), it.get("drug2", "").lower()
        if a in interaction_count_map:
            interaction_count_map[a] += 1
        if b in interaction_count_map:
            interaction_count_map[b] += 1

    allergy_drug_set = {w.get("drug_name", "").lower() for w in allergy_warnings}

    rows: List[Dict[str, Any]] = []
    for drug, dl in zip(drugs, drug_lower_list):
        n_interactions = interaction_count_map.get(dl, 0)
        has_allergy = dl in allergy_drug_set
        rows.append({
            "Drug Name": drug["name"],
            "Dosage": drug.get("dosage", "Not specified"),
            "Frequency": drug.get("frequency", "Not specified"),
            "Interactions": n_interactions,
            "Allergy Risk": "⚠️ YES" if has_allergy else "✅ No",
            "Status": "🔴 High Risk" if (has_allergy or n_interactions > 0) else "🟢 Safe",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =============================================================================
# AI INSIGHTS PAGE
# =============================================================================

def show_ai_insights() -> None:
    st.markdown("# 🤖 AI Insights Center")
    results = st.session_state.analysis_results
    if not results:
        st.info("🔍 No AI analysis results available. Please run an analysis first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🤗 HuggingFace AI")
        hf = results.get("huggingface_analysis", {})
        if hf:
            st.success("✅ Analysis Complete")
            di = hf.get("drug_interactions", [])
            if di:
                st.write(f"🔍 Found {len(di)} interaction analyses")
            mi = hf.get("medical_insights", {})
            if mi.get("status") == "success":
                st.success("🧠 Medical insights generated")
        else:
            st.warning("⚠️ No HuggingFace analysis available")
    with c2:
        st.markdown("### ✨ Google Gemini")
        gemini = results.get("gemini_analysis", {})
        if gemini:
            st.success("✅ Analysis Complete")
            sa = gemini.get("safety_analysis", {})
            if sa.get("status") == "success":
                st.success("🛡️ Safety analysis complete")
            dosage = gemini.get("dosage_recommendations", {})
            if dosage.get("status") == "success":
                st.success("💊 Dosage reference review generated")
        else:
            st.warning("⚠️ No Gemini analysis available")

    enhanced = results.get("enhanced_ai_insights", [])
    if enhanced:
        st.markdown("### 🚀 Enhanced AI Insights")
        for i in enhanced:
            pct = i.get("confidence", 0) * 100
            color = "#28a745" if pct >= 80 else ("#ffc107" if pct >= 60 else "#dc3545")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); border-left: 4px solid {color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: {color};">🤖 {i.get('source', 'AI System')}</h4>
                <p><strong>Insight:</strong> {i.get('insight', 'No insight available')}</p>
                <p><strong>Category:</strong> {i.get('type', 'general').replace('_', ' ').title()}</p>
                <div style="background-color: #f8f9fa; border-radius: 4px; padding: 0.5rem;">
                    <strong>Confidence Score:</strong> {pct:.1f}%
                    <div style="background-color: {color}; height: 10px; width: {pct}%; border-radius: 5px; margin-top: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📊 Detailed AI Analysis")
    t1, t2 = st.tabs(["🤗 HuggingFace Data", "✨ Gemini Data"])
    with t1:
        hf_data = results.get("huggingface_analysis", {})
        if hf_data:
            mi = hf_data.get("medical_insights", {})
            if mi:
                st.markdown("#### 🧠 Medical Text Analysis")
                if mi.get("status") == "success":
                    st.success("Analysis completed successfully")
                    if mi.get("insights"):
                        st.json(mi["insights"])
                else:
                    st.error("Medical insights analysis failed")
            di = hf_data.get("drug_interactions", [])
            if di:
                st.markdown("#### 🔄 Drug Interaction Analysis")
                for idx, interaction in enumerate(di):
                    with st.expander(f"Interaction {idx + 1}: {interaction.get('drug1')} ↔️ {interaction.get('drug2')}"):
                        st.json(interaction)
        else:
            st.info("No HuggingFace analysis data available")
    with t2:
        gemini_data = results.get("gemini_analysis", {})
        if gemini_data:
            sa = gemini_data.get("safety_analysis", {})
            if sa:
                st.markdown("#### 🛡️ Prescription Safety Analysis")
                if sa.get("status") == "success":
                    st.success("Safety analysis completed")
                    txt = sa.get("analysis")
                    if txt:
                        st.markdown("**AI-Generated Safety Analysis:**")
                        st.write(txt)
                else:
                    st.error("Safety analysis failed")
            dosage = gemini_data.get("dosage_recommendations", {})
            if dosage:
                st.markdown("#### 💊 Gemini Dosage Reference")
                if dosage.get("status") == "success":
                    st.success("Recommendations generated")
                    txt = dosage.get("analysis")
                    if txt:
                        st.markdown("**AI-Generated Reference Considerations:**")
                        st.write(txt)
                else:
                    st.error("Dosage reference analysis failed")
        else:
            st.info("No Gemini analysis data available")

    st.markdown("---")
    st.markdown("### 🔍 Request Additional AI Insights")
    _additional_insights_ui()


def _additional_insights_ui() -> None:
    c1, c2 = st.columns(2)
    with c1:
        additional_text = st.text_area(
            "Additional medical text for analysis:",
            placeholder="Enter additional clinical notes, symptoms, or medical history...",
            height=100,
        )
    with c2:
        st.markdown("**Patient Summary:**")
        drugs = st.session_state.patient_drugs
        if drugs:
            st.write(f"📊 Drugs: {len(drugs)}")
            for d in drugs[:3]:
                st.write(f"• {d['name']}")
            if len(drugs) > 3:
                st.write(f"... and {len(drugs) - 3} more")

    if st.button("🤖 Get Additional AI Insights", type="primary"):
        if additional_text.strip():
            _get_additional_insights(additional_text.strip())
        else:
            st.warning("Please enter some additional medical text.")


def _get_additional_insights(medical_text: str) -> None:
    medications = [d["name"] for d in st.session_state.patient_drugs]
    request_data = {
        "medical_text": medical_text,
        "patient_data": {"age": 45, "medications": medications, "conditions": [], "allergies": []},
    }
    with st.spinner("🤖 Requesting additional AI insights..."):
        result = make_api_request("/ai-insights", method="POST", data=request_data)

    if not result["success"]:
        st.error(result["error"])
        return

    insights = result["data"]["insights"]
    st.success("✅ Additional AI insights generated!")

    if "huggingface" in insights:
        hf = insights["huggingface"]
        st.markdown("#### 🤗 HuggingFace Analysis")
        entities = hf.get("entities", [])
        if entities:
            st.write(f"🔍 Extracted {len(entities)} medical entities:")
            st.dataframe(pd.DataFrame(entities))
        mi = hf.get("medical_insights", {})
        if mi.get("status") == "success":
            st.success("🧠 Medical insights generated successfully")

    if "gemini" in insights:
        gemini = insights["gemini"]
        st.markdown("#### ✨ Google Gemini Analysis")
        sa = gemini.get("safety_analysis", {})
        if sa.get("status") == "success" and sa.get("analysis"):
            st.markdown("**Safety Analysis:**")
            st.write(sa["analysis"])
        dosage = gemini.get("dosage_recommendations", {})
        if dosage.get("status") == "success" and dosage.get("analysis"):
            st.markdown("**Dosage Reference Considerations:**")
            st.write(dosage["analysis"])


# =============================================================================
# HISTORY PAGE
# =============================================================================

def show_history() -> None:
    st.markdown("# 📋 Analysis History")
    history = st.session_state.analysis_history
    if not history:
        st.info("📊 No analysis history available yet. Complete some analyses to see them here.")
        return

    n = len(history)
    avg_safety = sum(h.get("safety_score", 0) for h in history) / n
    ai_count = sum(1 for h in history if h.get("ai_enabled"))
    total_drugs = sum(h.get("drugs_count", 0) for h in history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", n)
    c2.metric("Avg Safety Score", f"{avg_safety:.1f}")
    c3.metric("AI Analyses", ai_count)
    c4.metric("Total Drugs Analyzed", total_drugs)

    st.markdown("### 📊 Analysis History")
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(reversed(history)):
        ts = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        score = entry.get("safety_score", 0)
        risk = "🟢 Low" if score >= 80 else ("🟡 Moderate" if score >= 60 else "🔴 High")
        rows.append({
            "Analysis #": n - idx,
            "Timestamp": ts,
            "Patient Age": entry.get("patient_age", "N/A"),
            "Drugs Count": entry.get("drugs_count", 0),
            "Safety Score": f"{score:.1f}",
            "AI Enabled": "🤖 Yes" if entry.get("ai_enabled") else "❌ No",
            "Risk Level": risk,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if n > 1:
        st.markdown("### 📈 Safety Score Trend")
        timestamps = [datetime.fromisoformat(h["timestamp"]) for h in history]
        scores = [h.get("safety_score", 0) for h in history]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=scores, mode="lines+markers",
            name="Safety Score", line=dict(color="#667eea", width=3), marker=dict(size=8),
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Safe Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
        fig.update_layout(
            title="Safety Score Over Time",
            xaxis_title="Analysis Date",
            yaxis_title="Safety Score",
            height=400,
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🗑️ Clear History", type="secondary"):
        if st.checkbox("I confirm I want to clear all history"):
            st.session_state.analysis_history = []
            st.success("✅ History cleared successfully!")
            st.rerun()


# =============================================================================
# SETTINGS PAGE
# =============================================================================

def show_settings() -> None:
    st.markdown("# ⚙️ Settings & Configuration")

    st.markdown("### 🔌 API Configuration")
    c1, c2 = st.columns(2)
    with c1:
        url = st.text_input("Backend API URL", value=API_BASE_URL, help="URL of the FastAPI backend")
        if st.button("🧪 Test Connection"):
            test = make_api_request("/")
            if test["success"]:
                st.success("✅ Connection successful!")
                st.json(test["data"])
            else:
                st.error(test["error"])
    with c2:
        st.markdown("**Current System Status:**")
        status = cached_get("/health")
        if status["success"]:
            hd = status["data"]
            st.success(f"✅ System: {hd.get('status', 'Unknown')}")
            st.info(f"📊 Version: {hd.get('version', 'Unknown')}")
            for service, info in hd.get("ai_services", {}).items():
                s = info.get("status", "unknown")
                if s == "connected":
                    st.success(f"✅ {service.title()}: Connected")
                else:
                    st.warning(f"⚠️ {service.title()}: {s}")
        else:
            st.error("❌ Cannot connect to backend")

    st.markdown("---")
    st.markdown("### 🎨 Display Settings")
    d1, d2 = st.columns(2)
    with d1:
        st.checkbox("Show detailed AI insights", value=True, help="Display detailed AI analysis results")
        st.checkbox("Enable animation effects", value=True, help="Enable animated progress bars and transitions")
        st.selectbox("Theme preference", ["Auto", "Light", "Dark"], help="Choose display theme")
    with d2:
        st.slider("Results per page", 5, 50, 20, help="Number of results to show per page")
        st.checkbox("Auto-refresh results", value=False, help="Automatically refresh analysis results")

    st.markdown("---")
    st.markdown("### 🔬 Advanced Settings")
    with st.expander("AI Model Configuration", expanded=False):
        st.markdown("**HuggingFace Models:**")
        for line in [
            "• Biomedical NER: d4data/biomedical-ner-all",
            "• Clinical BERT: emilyalsentzer/Bio_ClinicalBERT",
            "• Drug Interaction: cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        ]:
            st.text(line)
        st.markdown("**Google Gemini Model:**")
        st.text("• Gemini 2.5 Flash")
        st.info("💡 Model configuration is managed by the backend API")

    with st.expander("Data Management", expanded=False):
        st.markdown("**Session Data:**")
        st.write(f"• Current drugs: {len(st.session_state.patient_drugs)}")
        st.write(f"• Analysis history: {len(st.session_state.analysis_history)}")
        st.write(f"• Has current results: {'Yes' if st.session_state.analysis_results else 'No'}")
        if st.button("🗑️ Clear All Session Data"):
            if st.checkbox("Confirm data clearing"):
                st.session_state.patient_drugs = []
                st.session_state.analysis_results = None
                st.session_state.analysis_history = []
                st.success("✅ All session data cleared!")
                st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Frontend Information:**")
        st.write("• Framework: Streamlit")
        st.write("• Version: 3.0.0")
        st.write("• Features: AI Integration, NLP, Real-time Analysis")
    with s2:
        st.markdown("**Backend Information:**")
        be = cached_get("/")
        if be["success"]:
            bd = be["data"]
            st.write(f"• Version: {bd.get('version', 'Unknown')}")
            st.write(f"• Status: {bd.get('status', 'Unknown')}")
            st.write(f"• AI Services: {len(bd.get('ai_services', {}))}")

    st.markdown("---")
    st.markdown("### 📚 Documentation & Help")
    with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. 👤 Enter patient information in the sidebar
        2. 💊 Add medications using manual entry or text analysis
        3. 🔍 Click "Analyze Prescription" to run AI analysis
        4. 📊 Review results in the detailed analysis sections
        5. 🤖 Check AI insights for additional recommendations

        **AI Features:**
        • 🤗 HuggingFace NLP for drug extraction
        • ✨ Google Gemini for safety analysis and dosage reference considerations
        • 🧠 Advanced drug interaction detection
        • 📊 Comprehensive risk scoring
        """)
    with st.expander("❓ Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**

        **❌ Cannot connect to backend:**
        • Ensure FastAPI server is running on localhost:8000
        • Check if the backend URL in settings is correct
        • Verify no firewall is blocking the connection

        **⏱️ Analysis taking too long:**
        • AI analysis can take 30-60 seconds
        • Check your internet connection for API calls
        • Try disabling AI analysis for faster results

        **🤖 AI features not working:**
        • Verify API keys are configured in the backend
        • Check AI service status in settings
        • Review backend logs for error messages
        """)


if __name__ == "__main__":
    main()
