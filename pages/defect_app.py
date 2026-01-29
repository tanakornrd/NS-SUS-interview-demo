import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. Config & Setup ---
st.set_page_config(page_title="NSSUS Universal QA", page_icon="🏭", layout="wide")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ ไม่พบ API Key กรุณาตั้งค่าใน Streamlit Secrets ก่อนครับ")
    st.stop()

# --- 🧠 KNOWLEDGE BASE (สมองของระบบ) ---
# นี่คือส่วนที่ทำให้ระบบฉลาดและรองรับหลายไลน์ผลิตตามรูปภาพ Process Map
LINE_CONFIG = {
    "CDCM (Cold Rolling Mill)": {
        "Product": "CR (Cold Rolled Steel)",
        "Param1": {"name": "Rolling Force", "unit": "MN", "default": 1500, "min": 0, "max": 3000},
        "Param2": {"name": "Strip Tension", "unit": "kN", "default": 50, "min": 0, "max": 200},
        "Param3": {"name": "Rolling Speed", "unit": "mpm", "default": 1200, "min": 0, "max": 2000},
        "Defect_Focus": "Scale, Chatter marks, Edge cracks, Shape defects (Buckle)",
        "Process_Desc": "Rolling Hot Rolled Coil to get required thickness and shape."
    },
    "CGL (Continuous Galvanizing Line)": {
        "Product": "GA/GI (Galvanized Steel)",
        "Param1": {"name": "Zinc Pot Temp", "unit": "°C", "default": 460, "min": 400, "max": 500},
        "Param2": {"name": "Air Knife Pressure", "unit": "kPa", "default": 40, "min": 0, "max": 100},
        "Param3": {"name": "Line Speed", "unit": "mpm", "default": 180, "min": 0, "max": 300},
        "Defect_Focus": "Dross, Spangle defects, Uncoated spots, Zinc adhesion issues",
        "Process_Desc": "Coating Zinc to prevent rust. Critical points are Pot Temp and Air Knife."
    },
    "EPL (Electrolytic Plating Line)": {
        "Product": "TP/TFS (Tinplate/Tin Free)",
        "Param1": {"name": "Current Density", "unit": "A/dm²", "default": 20, "min": 0, "max": 100},
        "Param2": {"name": "Plating Solution Temp", "unit": "°C", "default": 50, "min": 20, "max": 80},
        "Param3": {"name": "Line Speed", "unit": "mpm", "default": 400, "min": 0, "max": 800},
        "Defect_Focus": "Pinholes, Plating burns, Rust, Scratch (from Anode)",
        "Process_Desc": "Electrolytic process for Tin/Chrome coating. Watch out for electrical issues."
    },
    "CAL (Continuous Annealing Line)": {
        "Product": "Annealed CR",
        "Param1": {"name": "Soaking Temp", "unit": "°C", "default": 800, "min": 600, "max": 900},
        "Param2": {"name": "Furnace Pressure", "unit": "mmAq", "default": 20, "min": 0, "max": 50},
        "Param3": {"name": "Cooling Rate", "unit": "°C/s", "default": 50, "min": 0, "max": 100},
        "Defect_Focus": "Heat buckle, Oxidation (Color), Pick-up marks",
        "Process_Desc": "Heat treatment to improve mechanical properties."
    }
}

# --- ฟังก์ชันบันทึก ---
def save_log(timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level):
    file_name = 'production_logs_v2.csv' # เปลี่ยนชื่อไฟล์เพราะโครงสร้างเปลี่ยน
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Line', 'Lot No.', 'Param 1', 'Param 2', 'Param 3', 'Status', 'Defect', 'Risk'])
        writer.writerow([timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level])

# --- UI Setup ---
st.title("🏭 NSSUS Universal Process QA")
st.markdown("---")

# 🟢 1. Line Selection (เลือกไลน์ผลิตก่อน)
st.subheader("📍 Select Production Line")
selected_line_name = st.selectbox("Choose Process Unit:", list(LINE_CONFIG.keys()))

# ดึง Config ของไลน์ที่เลือกมา
current_config = LINE_CONFIG[selected_line_name]

# 🟢 2. Dynamic Control Panel (เปลี่ยนปุ่มตามไลน์ที่เลือก)
st.markdown(f"**active Module:** `{current_config['Product']}` | **Process:** *{current_config['Process_Desc']}*")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("**📦 Lot Number**")
        lot_number = st.text_input("Lot No.", value="LOT-2026-X001", label_visibility="collapsed")
    
    with c2:
        # ดึงชื่อ Parameter 1 มาแสดง
        p1_cfg = current_config['Param1']
        st.markdown(f"**⚙️ {p1_cfg['name']} ({p1_cfg['unit']})**")
        p1_val = st.number_input("P1", value=p1_cfg['default'], min_value=p1_cfg['min'], max_value=p1_cfg['max'], label_visibility="collapsed")

    with c3:
        # ดึงชื่อ Parameter 2 มาแสดง
        p2_cfg = current_config['Param2']
        st.markdown(f"**⚙️ {p2_cfg['name']} ({p2_cfg['unit']})**")
        p2_val = st.number_input("P2", value=p2_cfg['default'], min_value=p2_cfg['min'], max_value=p2_cfg['max'], label_visibility="collapsed")

    with c4:
        # ดึงชื่อ Parameter 3 มาแสดง
        p3_cfg = current_config['Param3']
        st.markdown(f"**⏩ {p3_cfg['name']} ({p3_cfg['unit']})**")
        p3_val = st.number_input("P3", value=p3_cfg['default'], min_value=p3_cfg['min'], max_value=p3_cfg['max'], label_visibility="collapsed")

st.markdown("---")

# 🟢 3. Analysis Section
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1️⃣ Visual Inspection")
    uploaded_file = st.file_uploader(f"Upload Image for {current_config['Product']}", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Inspection Point: {selected_line_name}", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Expert Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("2️⃣ AI Expert Findings")
    
    if uploaded_file and run_btn:
        if not lot_number:
            st.warning("⚠️ Please enter Lot Number")
        else:
            with st.spinner(f"Consulting {selected_line_name} Expert Module..."):
                try:
                    # --- 🔥 DYNAMIC PROMPT (หัวใจสำคัญของการเพิ่มความแม่นยำ) ---
                    # เราจะส่ง "บริบทเฉพาะทาง" ของแต่ละไลน์ไปให้ AI
                    prompt = f"""
                    Role: You are a Top-Tier Process Engineer at NSSUS specializing in {selected_line_name}.
                    Your expertise covers defects specifically for: {current_config['Product']}.
                    
                    Target Lot: {lot_number}
                    
                    Machine Telemetry:
                    - {current_config['Param1']['name']}: {p1_val} {current_config['Param1']['unit']}
                    - {current_config['Param2']['name']}: {p2_val} {current_config['Param2']['unit']}
                    - {current_config['Param3']['name']}: {p3_val} {current_config['Param3']['unit']}
                    
                    Likely Defects in this process: {current_config['Defect_Focus']}
                    
                    Task:
                    1. VISUAL: Analyze the image for specific defects related to {current_config['Product']}. 
                       (e.g., if Galvanizing, look for Spangle/Dross. If Tinplate, look for Pinholes).
                    2. CORRELATION: Correlate the visual finding with the machine telemetry provided.
                       (e.g., Low Zinc Pot Temp -> Dross Risk).
                    3. DECISION: Pass or Fail based on high standards.
                    
                    Response Format (Markdown):
                    [STATUS]: (PASS / FAIL)
                    [DEFECT_DETECTED]: (Specific Name or "None")
                    [CONFIDENCE_SCORE]: (0-100%)
                    [ROOT_CAUSE_ANALYSIS]:
                    - Observation: (What you see)
                    - Technical Link: (How parameters might have caused this)
                    [RECOMMENDED_ACTION]: (Specific adjustment for the operator)
                    ตอบเป็นภาษาไทย
                    """
                    
                    response = model.generate_content([prompt, image])
                    result_text = response.text
                    
                    # --- Display Logic ---
                    if "[STATUS]: FAIL" in result_text:
                        st.error(f"🚨 FAIL: Defect Detected in {lot_number}")
                        risk = "High"
                        status = "FAIL"
                    elif "Critical" in result_text:
                        st.error("🚨 CRITICAL STOP")
                        risk = "Critical"
                        status = "FAIL"
                    else:
                        st.success(f"✅ PASS: Quality Approved")
                        risk = "Low"
                        status = "PASS"
                    
                    with st.container(border=True):
                        st.markdown("### 📝 Engineering Report")
                        st.markdown(result_text)
                    
                    # Extract Defect Name simple logic
                    defect_name = "Normal"
                    if "DEFECT_DETECTED]:" in result_text:
                        try:
                            defect_name = result_text.split("[DEFECT_DETECTED]:")[1].split("\n")[0].strip()
                        except: pass
                        
                    # Save to new CSV structure
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_log(current_time, selected_line_name, lot_number, p1_val, p2_val, p3_val, status, defect_name, risk)
                    
                except Exception as e:
                    st.error(f"System Error: {e}")

# 🟢 History Log
st.divider()
st.subheader("📜 Multi-Line Production Log")
log_file = 'production_logs_v2.csv'
if os.path.isfile(log_file):
    df = pd.read_csv(log_file)
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
