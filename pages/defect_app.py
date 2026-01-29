import streamlit as st
import os
import sys
import subprocess
import time

# --- 0. NUCLEAR INSTALLATION (สูตรล้างเครื่อง) ---
# ส่วนนี้จะทำงานก่อนทุกอย่าง เพื่อบังคับลงไลบรารีใหม่ล่าสุดให้ได้
try:
    import google.generativeai as genai
    current_ver = genai.__version__
    
    # ถ้าเวอร์ชันเก่ากว่า 0.8.3 สั่งลบและลงใหม่ทันที
    if current_ver < "0.8.3":
        st.toast(f"Found old library v{current_ver}. Upgrading...", icon="🔄")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai>=0.8.3"])
        st.rerun() # รีสตาร์ทแอปทันที
except:
    # ถ้ายังไม่มี ก็สั่งลงเลย
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.8.3"])
    st.rerun()

# --- เริ่มต้น Import ปกติ ---
import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd

# --- 1. Config & Setup ---
st.set_page_config(page_title="NSSUS Universal QA", page_icon="🏭", layout="wide")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    # 🧠 SYSTEM: MODEL DISCOVERY (ถาม Server ว่ามีโมเดลอะไรบ้าง)
    # เราจะไม่เดาชื่อแล้ว เราจะให้ Server บอกมาเลย
    valid_model = None
    try:
        # ดึงรายชื่อโมเดลทั้งหมดที่ API Key นี้ใช้ได้
        model_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_list.append(m.name)
        
        # แสดงเวอร์ชันและรายชื่อโมเดลใน Sidebar (เพื่อ Debug)
        st.sidebar.header("🔧 System Status")
        st.sidebar.text(f"Lib Version: {genai.__version__}")
        
        # Logic เลือกโมเดลอัตโนมัติ (Flash > Pro > Vision)
        target_keywords = ['flash', 'gemini-1.5', 'vision']
        
        for keyword in target_keywords:
            for m_name in model_list:
                if keyword in m_name:
                    valid_model = genai.GenerativeModel(m_name)
                    st.sidebar.success(f"Active Model: {m_name}")
                    break
            if valid_model: break
        
        # ถ้าหาไม่เจอจริงๆ ให้ User เลือกเองจาก Dropdown ที่ Sidebar
        if not valid_model:
            st.sidebar.error("Auto-detect failed. Please select:")
            selected = st.sidebar.selectbox("Manual Select:", model_list)
            if selected:
                valid_model = genai.GenerativeModel(selected)

    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()
        
    if valid_model:
        model = valid_model
    else:
        st.error("❌ CRITICAL: ไม่พบโมเดลที่รองรับ (กรุณาเช็ก API Key หรือ Library)")
        st.stop()

else:
    st.error("❌ ไม่พบ API Key กรุณาตรวจสอบใน Secrets")
    st.stop()

# --- 2. KNOWLEDGE BASE (ข้อมูลไลน์ผลิต) ---
LINE_CONFIG = {
    "CDCM (Cold Rolling Mill)": {
        "Product": "CR (Cold Rolled Steel)",
        "Param1": {"name": "Rolling Force", "unit": "MN", "default": 1500, "min": 0, "max": 3000},
        "Param2": {"name": "Strip Tension", "unit": "kN", "default": 50, "min": 0, "max": 200},
        "Param3": {"name": "Rolling Speed", "unit": "mpm", "default": 1200, "min": 0, "max": 2000},
        "Defect_Focus": "Scale, Chatter marks, Edge cracks, Shape defects (Buckle)"
    },
    "CGL (Continuous Galvanizing Line)": {
        "Product": "GA/GI (Galvanized Steel)",
        "Param1": {"name": "Zinc Pot Temp", "unit": "°C", "default": 460, "min": 400, "max": 500},
        "Param2": {"name": "Air Knife Pressure", "unit": "kPa", "default": 40, "min": 0, "max": 100},
        "Param3": {"name": "Line Speed", "unit": "mpm", "default": 180, "min": 0, "max": 300},
        "Defect_Focus": "Dross, Spangle defects, Uncoated spots, Zinc adhesion issues"
    },
    "EPL (Electrolytic Plating Line)": {
        "Product": "TP/TFS (Tinplate/Tin Free)",
        "Param1": {"name": "Current Density", "unit": "A/dm²", "default": 20, "min": 0, "max": 100},
        "Param2": {"name": "Plating Solution Temp", "unit": "°C", "default": 50, "min": 20, "max": 80},
        "Param3": {"name": "Line Speed", "unit": "mpm", "default": 400, "min": 0, "max": 800},
        "Defect_Focus": "Pinholes, Plating burns, Rust, Scratch (from Anode)"
    }
}

# --- 3. Save Function ---
def save_log(timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level):
    file_name = 'production_logs_v2.csv'
    # Check if file exists to determine if header is needed
    header_needed = not os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if header_needed:
            writer.writerow(['Timestamp', 'Line', 'Lot No.', 'Param 1', 'Param 2', 'Param 3', 'Status', 'Defect', 'Risk'])
        writer.writerow([timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level])

# --- 4. UI Layout ---
st.title("🏭 NSSUS Universal Process QA")
st.markdown("---")

# Select Line
st.subheader("📍 Select Production Line")
selected_line_name = st.selectbox("Choose Process Unit:", list(LINE_CONFIG.keys()))
current_config = LINE_CONFIG[selected_line_name]

st.markdown(f"**active Module:** `{current_config['Product']}`")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📦 Lot Number**")
        lot_number = st.text_input("Lot No.", value="LOT-2026-X001", label_visibility="collapsed")
    with c2:
        p1_cfg = current_config['Param1']
        st.markdown(f"**⚙️ {p1_cfg['name']}**")
        p1_val = st.number_input("P1", value=p1_cfg['default'], label_visibility="collapsed")
    with c3:
        p2_cfg = current_config['Param2']
        st.markdown(f"**⚙️ {p2_cfg['name']}**")
        p2_val = st.number_input("P2", value=p2_cfg['default'], label_visibility="collapsed")
    with c4:
        p3_cfg = current_config['Param3']
        st.markdown(f"**⏩ {p3_cfg['name']}**")
        p3_val = st.number_input("P3", value=p3_cfg['default'], label_visibility="collapsed")

st.markdown("---")

# Analysis Section
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1️⃣ Visual Inspection")
    uploaded_file = st.file_uploader(f"Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Inspection Point: {selected_line_name}", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Expert Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("2️⃣ AI Expert Findings")
    
    if uploaded_file and run_btn:
        with st.spinner(f"Consulting {selected_line_name} Expert Module..."):
            try:
                # Prompt setup
                prompt = f"""
                Role: Senior Process Engineer at NSSUS. Line: {selected_line_name}.
                Analyze image for defects: {current_config['Defect_Focus']}.
                Machine Data: {p1_val}, {p2_val}, {p3_val}.
                Task: Detect defects and decide Pass/Fail.
                Response:
                [STATUS]: (PASS / FAIL)
                [DEFECT_DETECTED]: (Name)
                [ANALYSIS]: (Explanation)
                """
                
                # เรียก AI
                response = model.generate_content([prompt, image])
                result_text = response.text
                
                if "[STATUS]: FAIL" in result_text or "Critical" in result_text:
                    st.error(f"🚨 FAIL: Defect Detected")
                    status = "FAIL"
                else:
                    st.success(f"✅ PASS: Quality Approved")
                    status = "PASS"
                
                with st.container(border=True):
                    st.markdown(result_text)
                    
                # Save Log
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_log(current_time, selected_line_name, lot_number, p1_val, p2_val, p3_val, status, "AI Check", "Low")
                
            except Exception as e:
                st.error(f"Processing Error: {e}")

# History
st.divider()
st.subheader("📜 History Log")
if os.path.isfile('production_logs_v2.csv'):
    df = pd.read_csv('production_logs_v2.csv')
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
