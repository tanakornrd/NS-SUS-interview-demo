import streamlit as st
import os
import sys
import subprocess
import time
import random

# --- 0. FORCE UPDATE SYSTEM ---
try:
    import google.generativeai as genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.8.3"])
    st.rerun()

import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd
import datetime

# --- 1. Config & Setup ---
st.set_page_config(page_title="NS-SUS Defect Inspection", layout="wide")

# ตั้งค่า API Key (ถ้ามี)
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- 2. SIDEBAR CONFIG (แผงควบคุมลับสำหรับคน Demo) ---
st.sidebar.title("🔧 Developer Settings")
use_simulation = st.sidebar.toggle("Simulation Mode (For Demo)", value=True, help="เปิดโหมดนี้เพื่อจำลองผลลัพธ์โดยไม่ใช้ Quota Google")

if use_simulation:
    st.sidebar.success("✅ SIMULATION ACTIVE: ระบบจะจำลองคำตอบเสมือนจริง (ไม่กิน Quota)")
    force_fail = st.sidebar.checkbox("⚠️ Force Defect (สั่งให้เจอของเสีย)", value=False)
else:
    st.sidebar.warning("LIVE AI MODE: ระบบจะเรียกใช้ Google Gemini จริง (ระวัง Quota)")
    
st.sidebar.divider()
st.sidebar.markdown("### 🗑️ Database Management")
if st.sidebar.button("ล้างประวัติการตรวจ (Reset Logs)", type="primary", use_container_width=True):
    log_file = 'production_logs_v2.csv'
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            st.toast("🧹 History Log Cleared!", icon="✅") # แจ้งเตือนแบบ Toast สวยๆ
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.info("Log file is already empty.")

# --- 3. LOGIC & DATA (Updated based on NSSUS.pdf) ---
LINE_CONFIG = {
    "CDCM (Continuous Descaling & Cold Rolling)": { 
        # อ้างอิงจาก  รวมกระบวนการ Descaling และ Rolling ไว้ด้วยกัน
        "Product": "CR (Cold Rolled Steel)",
        "Param1": {"name": "Pickling Acid Temp (°C)", "unit": "°C", "default": 85, "min": 60, "max": 100}, # เพิ่มส่วน Descaling 
        "Param2": {"name": "Rolling Force (MN)", "unit": "MN", "default": 1500, "min": 0, "max": 3000},
        "Param3": {"name": "Rolling Speed (mpm)", "unit": "mpm", "default": 1200, "min": 0, "max": 2000},
        "Defect_Focus": "Residual Scale, Pickling stain, Chatter marks, Edge cracks" # เพิ่ม Defect จากการกัดกรด
    },
    "CGL (Continuous Galvanizing Line)": {
        "Product": "GA/GI (Galvanized Steel)",
        "Param1": {"name": "Annealing Furnace Temp (°C)", "unit": "°C", "default": 800, "min": 700, "max": 900}, # เพิ่มส่วน Annealing 
        "Param2": {"name": "Zinc Pot Temp (°C)", "unit": "°C", "default": 460, "min": 440, "max": 480},
        "Param3": {"name": "Air Knife Pressure (kPa)", "unit": "kPa", "default": 40, "min": 0, "max": 100}, # ควบคุม Coating Weight [cite: 80]
        "Defect_Focus": "Dross, Uncoated spots, Zinc adhesion (Peeling), Fluting"
    },
    "EPL (Electrolytic Plating Line)": {
        "Product": "TP/TFS (Tinplate/Tin Free)",
        "Param1": {"name": "Plating Current Density (A/dm²)", "unit": "A/dm²", "default": 20, "min": 0, "max": 100},
        "Param2": {"name": "Reflow Temperature (°C)", "unit": "°C", "default": 250, "min": 230, "max": 300}, # เพิ่ม Reflow Process 
        "Param3": {"name": "Coating Weight (g/m²)", "unit": "g/m²", "default": 2.8, "min": 1.0, "max": 11.0}, # สำคัญสำหรับ TP/TFS 
        "Defect_Focus": "Pinholes, Plating burns (White/Black), Reflow stain, Woodgrain"
    }
}

def save_log(timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level):
    file_name = 'production_logs_v2.csv'
    header_needed = not os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if header_needed:
            writer.writerow(['Timestamp', 'Line', 'Lot No.', 'Param 1', 'Param 2', 'Param 3', 'Status', 'Defect', 'Risk'])
        writer.writerow([timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level])

# --- 4. UI Layout ---
# --- 4. UI Layout (ปรับปรุงใหม่: จัดระเบียบ UI) ---
st.title("NS-SUS Defect Inspection")
st.markdown("---")

st.subheader("Select Production Line")
selected_line_name = st.selectbox("Choose Process Unit:", list(LINE_CONFIG.keys()))
current_config = LINE_CONFIG[selected_line_name]

st.info(f"📍 **Active Module:** `{current_config['Product']}`")

# === ZONE 1: PARAMETERS (ปรับจูน UI ใหม่: แยกแถวชื่อกับแถวช่องกรอก) ===
with st.container(border=True):
    # -------------------------------------------------------------
    # 📝 แถวที่ 1: ชื่อตัวแปร (Labels) -> บังคับติดขอบบนเสมอ
    # -------------------------------------------------------------
    l1, l2, l3, l4 = st.columns(4)
    
    # ใช้ style height เพื่อดันให้พื้นที่ข้อความเท่ากัน (เผื่อชื่อยาว 2 บรรทัด)
    with l1: st.markdown(f"**Lot Number**")
    with l2: st.markdown(f"**{current_config['Param1']['name']}**")
    with l3: st.markdown(f"**{current_config['Param2']['name']}**")
    with l4: st.markdown(f"**{current_config['Param3']['name']}**")

    # -------------------------------------------------------------
    # ⌨️ แถวที่ 2: ช่องกรอกข้อมูล (Inputs) -> บังคับเรียงกันข้างล่าง
    # -------------------------------------------------------------
    i1, i2, i3, i4 = st.columns(4)
    
    with i1: 
        # label_visibility="collapsed" คือซ่อนชื่อในตัว Input (เพราะเราเขียนไว้ข้างบนแล้ว)
        lot_number = st.text_input("Lot", value="LOT-2026-X001", label_visibility="collapsed")
    with i2: 
        p1_val = st.number_input("P1", value=current_config['Param1']['default'], label_visibility="collapsed")
    with i3: 
        p2_val = st.number_input("P2", value=current_config['Param2']['default'], label_visibility="collapsed")
    with i4: 
        p3_val = st.number_input("P3", value=current_config['Param3']['default'], label_visibility="collapsed")

# === ZONE 2: INSPECTION & UPLOAD (แบ่งซ้ายขวา) ===
# col_visual (ซ้าย 70%) = เอารูปไว้ตรงนี้ให้ใหญ่ๆ
# col_control (ขวา 30%) = เอาปุ่ม Upload ไว้ข้างๆ
col_visual, col_control = st.columns([2, 1]) 

with col_control:
    st.subheader("Controls")
    uploaded_file = st.file_uploader(f"Upload Image (CCTV)", type=["jpg", "png", "jpeg"])
    
    run_btn = False
    if uploaded_file:
        st.success("Image Loaded!")
        st.markdown("Ready to analyze...")
        # ปุ่มกดรัน ย้ายมาอยู่ตรงนี้ กดง่ายๆ
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

with col_visual:
    st.subheader("Visual Inspection Monitor (แทนภาพจากกล้องวงจรปิดที่ตรวจสอบสินค้าในไลน์ผลิต)")
    if uploaded_file:
        image = Image.open(uploaded_file)
        # แสดงรูปเต็มความกว้างคอลัมน์
        st.image(image, caption=f"Live Feed: {selected_line_name}", use_container_width=True)
    else:
        # แสดงกรอบว่างๆ ให้รู้ว่ารอรูป
        st.info("Waiting for image upload...")
        st.markdown(
            """
            <div style="border: 2px dashed #ccc; padding: 50px; text-align: center; color: #ccc;">
                NO SIGNAL INPUT
            </div>
            """, unsafe_allow_html=True
        )

# === ZONE 3: AI RESULT (ย้ายมาไว้ข้างล่าง เต็มจอ) ===
if uploaded_file and run_btn:
    st.divider()
    st.subheader("Analysis Result")
    
    with st.spinner(f"Consulting {selected_line_name} Expert Module..."):
        
        result_text = ""
        status = "PASS"
        
        # === LOGIC การทำงาน (เหมือนเดิม) ===
        if use_simulation:
            time.sleep(2.0) # ลดเวลาลงนิดนึงจะได้ทันใจ
            
            if force_fail:
                defects = current_config['Defect_Focus'].split(', ')
                chosen_defect = defects[0] if defects else "Surface Crack"
                result_text = f"""
                ### 🚨 [STATUS]: FAIL
                **Defect Detected:** {chosen_defect}
                **Confidence Score:** 94.5%
                
                ---
                **🔬 Engineering Analysis:**
                * **Observation:** Detected significant {chosen_defect} on the material surface.
                * **Root Cause:** Abnormal parameter settings (P1: {p1_val}) correlated with surface stress.
                
                **🛠️ Recommended Action:**
                * Immediate stop recommended. 
                * Check roller conditions and adjust P1 parameter.
                """
                status = "FAIL"
            else:
                result_text = f"""
                ### ✅ [STATUS]: PASS
                **Defect Detected:** None
                **Confidence Score:** 98.2%
                
                ---
                **🔬 Engineering Analysis:**
                * **Observation:** Surface texture appears consistent and free of defects.
                * **Compliance:** Meets strict quality standards for {current_config['Product']}.
                
                **🛠️ Recommended Action:**
                * Continue production. Parameters are stable.
                """
                status = "PASS"
        
        else:
            # 📡 LIVE MODE
            try:
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                except:
                    model = genai.GenerativeModel('gemini-pro')

                prompt = f"""
                Role: Senior Process Engineer at NS-Siam United Steel, เป็นนักคำนวณที่คุ้มค่าที่สุดในโลก และเป็นผู้วางแผนการบริหารการจัดการ production line ที่คุ้มค่าที่สุดในโลก. Line: {selected_line_name}.
                Analyze image for defects: {current_config['Defect_Focus']}.
                Response format: [STATUS]: ตอบแค่คำว่า "PASS" หรือ "FAIL"
                * [DEFECT_DETECTED]: ...(ตรวจพบอะไร กระชับคำตอบใน 1-2 ประโยค)
                * [ANALYSIS]: ...(วิเคราะห์ว่า defect เป็นผลมาจากอะไร parameter ไหนมีผลต่อ defect กระชับคำตอบใน 1-2 ประโยค)
                * [NEXT STEP]: ...(ต้องทำอะไรต่อ ต้องปรับค่า parameter ไหน หรือทำยังไงเพื่อแก้ไขโดยส่งผลกระทบต่อกระบวนการน้อยที่สุด ลดต้นทุนทางเศรษฐศาสตร์ คำนวณความคุ้มค่าเชิงการดำเนินงาน, เศรษฐ์ศาสตร์, ทางแก้ทางวิศวกรรม กระชับคำตอบใน 1-3 ประโยค)
                Respond in Thai.
                """
                response = model.generate_content([prompt, image])
                result_text = response.text
                
                if "FAIL" in result_text.upper():
                    status = "FAIL"
                else:
                    status = "PASS"
                    
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                status = "ERROR"

        # === DISPLAY RESULT (แสดงผลแบบเต็มจอ) ===
        if status != "ERROR":
            # ใช้สีพื้นหลังแบ่งแยกชัดเจน
            if status == "FAIL":
                st.error("🚨 DEFECT DETECTED")
                box_color = "#FFEBEB" # สีแดงอ่อน
            else:
                st.success("✅ QUALITY APPROVED")
                box_color = "#E8FDF5" # สีเขียวอ่อน
            
            # สร้างกล่องผลลัพธ์สวยๆ
            with st.container(border=True):
                st.markdown(result_text)
            
            # Save Log
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode_label = "Simulated" if use_simulation else "AI Check"
            save_log(current_time, selected_line_name, lot_number, p1_val, p2_val, p3_val, status, mode_label, "Low")

st.divider()
st.subheader("History Log")
if os.path.isfile('production_logs_v2.csv'):
    df = pd.read_csv('production_logs_v2.csv')
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
