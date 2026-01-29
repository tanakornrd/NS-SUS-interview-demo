import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. Config & Setup ---
st.set_page_config(page_title="NSSUS Predictive QA", page_icon="🏭", layout="wide")

# ตรวจสอบ API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ ไม่พบ API Key กรุณาตั้งค่าใน Streamlit Secrets ก่อนครับ")
    st.stop()

# --- ฟังก์ชันบันทึก ---
def save_log(timestamp, lot_id, machine_temp, pressure, speed, status, prediction, risk_level):
    file_name = 'production_logs.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Lot No.', 'Temp(C)', 'Pressure(Bar)', 'Speed(m/min)', 'Status', 'AI Prediction', 'Risk Level'])
        writer.writerow([timestamp, lot_id, machine_temp, pressure, speed, status, prediction, risk_level])

# --- 2. UI Setup: Dashboard Header ---
st.title("🏭 NSSUS Predictive Quality Assurance")
st.markdown("---")

# 🟢 ส่วนที่ 1: Control Panel (ปรับ UI ให้เรียงสวยงาม)
st.subheader("⚙️ Production Parameters")

c1, c2, c3, c4 = st.columns(4)

# ใช้ CSS Hack เล็กน้อยเพื่อให้บรรทัดเท่ากัน (ใส่ <br> และตัวอักษรสีจางๆ)
with c1:
    st.markdown("**📦 Product Lot Number**<br><span style='color:gray; font-size:0.8em'>Lot No. for tracking</span>", unsafe_allow_html=True)
    lot_number = st.text_input("Lot Number", value="LOT-2026-A001", label_visibility="collapsed")

with c2:
    # ย้าย Norm ลงมาบรรทัดล่าง ตามสั่งเจ้านาย
    st.markdown("**🌡️ Temp (°C)**<br><span style='color:gray; font-size:0.8em'>(Norm: 800-900)</span>", unsafe_allow_html=True)
    machine_temp = st.number_input("Temp", min_value=0, max_value=1500, value=850, step=10, label_visibility="collapsed")

with c3:
    st.markdown("**⬇️ Pressure (Bar)**<br><span style='color:gray; font-size:0.8em'>(Standard: 200)</span>", unsafe_allow_html=True)
    pressure = st.number_input("Pressure", min_value=0, max_value=1000, value=200, step=5, label_visibility="collapsed")

with c4:
    st.markdown("**⏩ Speed (m/min)**<br><span style='color:gray; font-size:0.8em'>(Target: 1200)</span>", unsafe_allow_html=True)
    line_speed = st.number_input("Speed", min_value=0, max_value=3000, value=1200, step=50, label_visibility="collapsed")

st.markdown("---")

# 🟢 ส่วนที่ 2: Inspection Area
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1️⃣ CCTV / Image Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Monitoring: {lot_number}", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("🚀 Run Predictive Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("2️⃣ AI Analysis Result")
    
    if uploaded_file is not None and run_button:
        if not lot_number:
            st.warning("⚠️ กรุณาระบุเลข Lot Number ก่อนวิเคราะห์ครับ")
        else:
            with st.spinner(f"Analyzing Lot {lot_number}..."):
                try:
                    # --- 🔥 PROMPT ENGINEERING (สั่งให้ตอบเป็น Bullet) ---
                    prompt = f"""
                    Role: You are a Senior QA Engineer at a Steel Factory (NS-SUS). 
                    
                    Target Product Lot No: {lot_number}
                    Current Machine Conditions:
                    - Temp: {machine_temp} °C (Norm: 800-900)
                    - Pressure: {pressure} Bar
                    - Speed: {line_speed} m/min
                    
                    Criteria:
                    1. PASS: Minor texture, water stains, light scratches.
                    2. FAIL: Cracks, heavy scale, holes.
                    
                    Task:
                    Analyze the image and machine data.
                    
                    Response Format (Use Markdown for readability):
                    [STATUS]: (PASS / FAIL)
                    [DEFECT_TYPE]: (Defect Name OR "None")
                    [ANALYSIS]:
                    - (Point 1: Describe visual findings clearly)
                    - (Point 2: Explain if it meets acceptance criteria)
                    - (Point 3: Relate to machine parameters if relevant)
                    
                    [RISK_PREDICTION]: (One sentence prediction)
                    
                    [HOW TO PREVENT] : 
                    - บอกวิธีแก้ไข next step ต้องปรับค่าอะไร หรือต้องจัดการกับปัญหายังไงโดยคำนึงถึงผลทางเศรษฐศาสตร์และความต่อเนื่องของกระบวนการผลิตเป็นหลัก กระชับใน 2 ประโยค
                    
                    ตอบทั้งหมดเป็นภาษาไทย
                    """
                    
                    response = model.generate_content([prompt, image])
                    result_text = response.text
                    
                    # --- แสดงผล ---
                    status = "PASS"
                    risk_level = "Low"
                    
                    if "[STATUS]: FAIL" in result_text:
                        status = "FAIL"
                        risk_level = "High"
                        st.error(f"🚨 DETECTED: พบความผิดปกติใน Lot {lot_number}")
                    elif "Critical" in result_text:
                        status = "FAIL"
                        risk_level = "Critical"
                        st.error("🚨 CRITICAL WARNING!")
                    else:
                        st.success(f"✅ Lot {lot_number} : ผ่านเกณฑ์ (PASS)")
                    
                    # ✅ ใช้ st.markdown แสดงผล จะทำให้อ่านง่าย มีตัวหนา มี bullet
                    st.markdown("### 📝 Detailed Report")
                    with st.container(border=True): # ใส่กรอบให้นิดนึงเพื่อความสวยงาม
                        st.markdown(result_text)
                    
                    # บันทึก Log
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    prediction = "Normal"
                    if "DEFECT_TYPE]:" in result_text:
                        try:
                            prediction = result_text.split("[DEFECT_TYPE]:")[1].split("\n")[0].strip()
                        except:
                            prediction = "See Details"
                    
                    save_log(current_time, lot_number, machine_temp, pressure, line_speed, status, prediction, risk_level)
                    st.toast(f"บันทึกข้อมูลเรียบร้อย!", icon="💾")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif uploaded_file is None:
        st.info("👈 กรุณาอัปโหลดรูปภาพทางฝั่งซ้ายเพื่อเริ่มการทำงาน")

# 🟢 ส่วนที่ 3: History Log
st.divider()
st.subheader("📜 Production History Log")

log_file = 'production_logs.csv'
if os.path.isfile(log_file):
    try:
        df = pd.read_csv(log_file)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    except Exception:
        os.remove(log_file)
        st.warning("⚠️ รีเซ็ตฐานข้อมูลใหม่ (File Reset)")
else:
    st.info("ยังไม่มีข้อมูลในระบบ")
