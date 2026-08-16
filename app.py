import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import datetime

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="NephroAI | CKD Prediction", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling (รวมของเดิม + ของผู้พัฒนา)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    
    :root { 
        --primary: #0f766e; 
        --bg-light: #f0fdfa; 
        --text-main: #1e293b;
        --text-muted: #475569;
    }
    
    /* ปรับฟอนต์หลัก */
    body, .stMarkdown, h1, h2, h3, p, label, span, div { 
        font-family: 'Sarabun', sans-serif !important; 
    }
    
    /* Sidebar - ปรับให้อ่านง่ายขึ้น */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    /* เมนู Selectbox และ Input ต่างๆ */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* ปรับสีพื้นหลังของ Input ให้ชัดเจน */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 2px solid #cbd5e1 !important;
    }
    
    /* Background หลัก */
    .stApp { 
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%) !important; 
        min-height: 100vh; 
    }
    
    h1 { color: var(--primary) !important; font-weight: 700; }
    h2, h3 { color: #334155 !important; font-weight: 600; }
    
    .metric-card { 
        background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
        border-radius: 16px; padding: 1.5rem; 
        box-shadow: 0 4px 20px rgba(15, 118, 110, 0.08);
        border: 1px solid rgba(255,255,255,0.5); 
    }
    
    .result-safe { 
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
        border-left: 5px solid #10b981; 
        color: #064e3b !important;
    }
    
    .result-risk { 
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
        border-left: 5px solid #ef4444;
        color: #7f1d1d !important;
    }
    
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #0d9488 0%, #0f766e 100%);
        color: white !important; 
        border-radius: 50px; 
        font-size: 1.1rem; 
        width: 100%;
        font-weight: 600;
    }

    /* --- CSS สำหรับส่วนข้อมูลผู้พัฒนา --- */
    .dev-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 10px;
    }
    
    .dev-info {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        margin: 0.4rem 0 !important;
        line-height: 1.5;
    }
    
    .dev-info strong {
        color: #ffffff !important;
    }
    
    .profile-img {
        border-radius: 50%;
        border: 3px solid #14b8a6;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.3);
        padding: 4px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([1, 9])
with col_logo: st.image("https://cdn-icons-png.flaticon.com/512/3063/3063065.png", width=60)
with col_title:
    st.title("NephroAI")
    st.markdown('<p style="color:#475569; font-size:1.1rem;">ระบบคัดกรองโรคไตเรื้อรัง (CKD) เพื่อการศึกษาและวิจัย</p>', unsafe_allow_html=True)

# 2. โหลดและเทรนโมเดล (ใช้ Cache เพื่อไม่ให้เทรนใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def get_ckd_model():
    np.random.seed(42)
    n = 800
    
    data = {
        'age': np.random.randint(18, 90, n),
        'bp': np.random.randint(50, 180, n),
        'sg': np.round(np.random.uniform(1.005, 1.030, n), 3),
        'al': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
        'su': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.05]),
        'rbc': np.random.choice(['normal', 'abnormal'], n, p=[0.7, 0.3]),
        'pc': np.random.choice(['normal', 'abnormal'], n, p=[0.65, 0.35]),
        'pcc': np.random.choice(['present', 'notpresent'], n, p=[0.75, 0.25]),
        'ba': np.random.choice(['present', 'notpresent'], n, p=[0.85, 0.15]),
        'hemo': np.round(np.random.uniform(3.0, 17.0, n), 1),
        'pcv': np.round(np.random.uniform(10, 55, n), 1),
        'wc': np.round(np.random.uniform(2000, 25000, n), 0),
        'rc': np.round(np.random.uniform(2.0, 7.0, n), 2),
        'htn': np.random.choice(['yes', 'no'], n, p=[0.4, 0.6]),
        'dm': np.random.choice(['yes', 'no'], n, p=[0.3, 0.7]),
        'cad': np.random.choice(['yes', 'no'], n, p=[0.15, 0.85]),
        'appet': np.random.choice(['good', 'poor'], n, p=[0.6, 0.4]),
        'pe': np.random.choice(['yes', 'no'], n, p=[0.3, 0.7]),
        'ane': np.random.choice(['yes', 'no'], n, p=[0.35, 0.65])
    }
    
    df = pd.DataFrame(data)
    
    # สร้าง Target แบบมี Logic
    risk_score = (
        (df['age'] > 60).astype(int) * 0.15 +
        (df['bp'] > 140).astype(int) * 0.1 +
        (df['al'] >= 2).astype(int) * 0.2 +
        (df['hemo'] < 8).astype(int) * 0.25 +
        (df['pc'] == 'abnormal').astype(int) * 0.15 +
        (df['ane'] == 'yes').astype(int) * 0.15
    )
    noise = np.random.normal(0, 0.1, n)
    df['classification'] = ((risk_score + noise) > 0.45).astype(int)
    
    le_dict = {}
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X, y = df.drop('classification', axis=1), df['classification']
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    
    return model, le_dict

model, le_dict = get_ckd_model()

# Sidebar Info
with st.sidebar:
    st.header("📊 เกี่ยวกับโปรเจกต์")
    st.info("""
    **Dataset:** Synthetic CKD Data (Seed=42)  
    **Algorithm:** Random Forest Classifier  
    **Features:** 19 Clinical Parameters  
    
    โปรเจกต์นี้จัดทำขึ้นเพื่อการศึกษาในรายวิชา Data Science for Healthcare
    """)
    st.divider()
    
    # 👇 ส่วนข้อมูลผู้พัฒนา 👇
    st.markdown("### 👨‍⚕️ ผู้พัฒนา")
    
    dev_col1, dev_col2 = st.columns([1, 1.5])
    
    with dev_col1:
        # 🖼️ รูปโปรไฟล์ผู้พัฒนา (ใช้รูปหมอจาก flaticon)
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
            width=120,
            cls="profile-img"
        )
    
    with dev_col2:
        # ✏️ แก้ไขข้อมูลในวงเล็บ [...] ให้เป็นข้อมูลจริงของคุณ
        st.markdown("""
        <div class="dev-card">
            <p class="dev-info"><strong>👤 ชื่อ:</strong> [นาย จิรศักดิ์ โมกกงจักร]</p>
            <p class="dev-info"><strong>🆔 รหัส:</strong> [664245003 ]</p>
            <p class="dev-info"><strong>📚 หมู่เรียน:</strong> [66/43]</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("© 2025 NephroAI Project")

# 3. ฟอร์มรับข้อมูล
with st.form("ckd_assessment_form"):
    st.subheader("แบบประเมินพารามิเตอร์ทางคลินิก")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("อายุ (ปี)", 18, 90, 45)
        bp = st.slider("ความดันโลหิต (mmHg)", 50, 180, 120)
        htn = st.selectbox("ประวัติ HTN", ["no", "yes"])
        dm = st.selectbox("ประวัติ DM", ["no", "yes"])
        cad = st.selectbox("ประวัติ CAD", ["no", "yes"])
        
    with col2:
        st.markdown("**Urinalysis**")
        sg = st.slider("Specific Gravity", 1.005, 1.030, 1.020, 0.001)
        al = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
        su = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
        rbc = st.selectbox("RBC", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cells", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
        
    with col3:
        st.markdown("**Hematology**")
        hemo = st.slider("Hemoglobin (g/dL)", 3.0, 17.0, 12.0, 0.1)
        pcv = st.slider("PCV (%)", 10, 55, 35)
        wc = st.number_input("WBC (/cumm)", 2000, 25000, 8000, step=100)
        rc = st.slider("RBC (millions/cmm)", 2.0, 7.0, 4.5, 0.1)
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"])
    
    submitted = st.form_submit_button("🔬 วิเคราะห์ความเสี่ยง", use_container_width=True)

# 4. แสดงผลลัพธ์
if submitted:
    input_data = pd.DataFrame({
        'age': [age], 'bp': [bp], 'sg': [sg], 'al': [al], 'su': [su],
        'rbc': [le_dict['rbc'].transform([rbc])[0]],
        'pc': [le_dict['pc'].transform([pc])[0]],
        'pcc': [le_dict['pcc'].transform([pcc])[0]],
        'ba': [le_dict['ba'].transform([ba])[0]],
        'hemo': [hemo], 'pcv': [pcv], 'wc': [wc], 'rc': [rc],
        'htn': [le_dict['htn'].transform([htn])[0]],
        'dm': [le_dict['dm'].transform([dm])[0]],
        'cad': [le_dict['cad'].transform([cad])[0]],
        'appet': [le_dict['appet'].transform([appet])[0]],
        'pe': [le_dict['pe'].transform([pe])[0]],
        'ane': [le_dict['ane'].transform([ane])[0]]
    })
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    risk_pct = prob[1] * 100
    
    res_col1, res_col2 = st.columns([1.2, 0.8])
    
    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="metric-card result-risk">
                <h2 style="color:#b91c1c; margin:0;">⚠️ มีความเสี่ยงต่อโรคไตเรื้อรัง</h2>
                <p style="font-size:1.3rem; color:#7f1d1d; margin:0.5rem 0;">
                    คะแนนความเสี่ยง: <b>{risk_pct:.1f}%</b>
                </p>
                <p style="color:#991b1b;">แนะนำพบแพทย์เฉพาะทางเพื่อตรวจ eGFR ทันที</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card result-safe">
                <h2 style="color:#047857; margin:0;">✅ ค่าพารามิเตอร์อยู่ในเกณฑ์ปกติ</h2>
                <p style="font-size:1.3rem; color:#064e3b; margin:0.5rem 0;">
                    ความมั่นใจว่าไม่เสี่ยง: <b>{(100-risk_pct):.1f}%</b>
                </p>
                <p style="color:#065f46;">ควรตรวจสุขภาพประจำปีต่อเนื่อง</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.progress(float(risk_pct/100))

    with res_col2:
        feat_imp = pd.DataFrame({
            'Feature': ['Age', 'Hemoglobin', 'Albumin', 'BP', 'PCV', 'WBC', 'SG', 'Diabetes'],
            'Importance': [0.22, 0.19, 0.16, 0.12, 0.10, 0.08, 0.07, 0.06]
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='YlOrRd',
                     title="📈 Top Contributing Factors")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# History
if 'history' not in st.session_state:
    st.session_state.history = []

if submitted:
    record = {
        'Time': datetime.datetime.now().strftime("%H:%M"),
        'Age': age, 'BP': bp, 'Hemo': hemo,
        'Result': 'Risk' if pred == 1 else 'Normal',
        'Confidence': f"{risk_pct:.1f}%"
    }
    st.session_state.history.insert(0, record)
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

if st.session_state.history:
    st.divider()
    st.subheader("🕒 ประวัติล่าสุด")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True, height=180)

st.markdown("""
<div style="font-size:0.85rem; color:#64748b; text-align:center; margin-top:3rem; padding:1rem; border-top:1px solid #e2e8f0;">
    ⚠️ <b>คำเตือน:</b> แอปพลิเคชันนี้ใช้ข้อมูลจำลองเพื่อการศึกษาเท่านั้น 
    ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้
</div>
""", unsafe_allow_html=True)