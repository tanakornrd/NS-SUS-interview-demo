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

# --- ฟังก์ชันบันทึก (เพิ่ม lot_id เข้ามา) ---
def save_log(timestamp, lot_id, machine_temp, pressure, speed, prediction, risk_level):
    file_name = 'defect_history.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # ถ้าสร้างไฟล์ใหม่ ให้เพิ่มหัวตาราง Lot No. ด้วย
        if not file_exists:
            writer.writerow(['Timestamp', 'Lot No.', 'Temp(C)', 'Pressure(Bar)', 'Speed(m/min)', 'AI Prediction', 'Risk Level'])
        
        # บันทึกข้อมูลครบทุกช่อง
        writer.writerow([timestamp, lot_id, machine_temp, pressure, speed, prediction, risk_level])

# --- 2. UI Setup ---
st.title("🏭 NSSUS Predictive Quality Assurance")
st.caption("ระบบทำนายโอกาสเกิด Defect จากสภาพเครื่องจักรและภาพหน้างาน (CCTV)")

col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("⚙️ Control Panel")
    st.info("ระบุข้อมูลการผลิตปัจจุบัน")
    
    # ✅ ส่วนที่เพิ่มมา: ช่องกรอก Lot Number
    st.markdown("### 📦 Product Identification")
    lot_number = st.text_input("ระบุเลข Lot Number", value="LOT-2026-A001", placeholder="เช่น LOT-XXXX-XXXX")
    
    st.markdown("---")
    st.markdown("### ⚙️ Machine Parameters")
    
    st.write("🌡️ Temperature (°C)")
    machine_temp = st.number_input("อุณหภูมิ", min_value=0, max_value=1500, value=850, step=10, label_visibility="collapsed")
    
    st.write("⬇️ Rolling Pressure (Bar)")
    pressure = st.number_input("แรงกด", min_value=0, max_value=1000, value=200, step=5, label_visibility="collapsed")
    
    st.write("⏩ Line Speed (m/min)")
    line_speed = st.number_input("ความเร็วไลน์ผลิต", min_value=0, max_value=3000, value=1200, step=50, label_visibility="collapsed")
    
    st.markdown("---")
    st.header("📹 CCTV Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

with col_display:
    st.header("📊 Real-time Monitor")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Monitoring Lot: {lot_number}", width=500)
# --- 🎯 PROMPT ฉบับแม่นยำ (Calibrated Prompt) ---
                        prompt = f"""
                        Role: You are a Senior QA Engineer at NS-SUS Factory. 
                        Your job is to prevent FALSE ALARMS. You only flag defects that are clearly visible and affect product quality.
                        
                        Target Product Lot No: {lot_number}
                        
                        Current Machine Conditions:
                        - Temperature: {machine_temp} °C (Normal Range: 800-900)
                        - Pressure: {pressure} Bar
                        - Speed: {line_speed} m/min
                        
                        Standard Acceptance Criteria:
                        1. ACCEPTABLE (Pass): Minor surface texture, water stains, or very faint scratches (light reflection) are NORMAL. Do not flag these.
                        2. REJECT (Fail): Deep cracks, heavy scale, severe scratches, holes, or distinct discoloration.
                        
                        Task:
                        1. Analyzes the image strictly based on the criteria above.
                        2. If the image looks mostly clean or ambiguous -> Result is "PASS".
                        3. If there is a CLEAR defect -> Result is "FAIL".
                        4. Combine visual finding with machine parameters to predict future risk.
                        
                        Response Format (Strictly follow this):
                        [STATUS]: (PASS / FAIL)
                        [CONFIDENCE_SCORE]: (0-100%)
                        [DEFECT_TYPE]: (Name of defect OR "None")
                        [ANALYSIS]: (Brief explanation why you chose Pass/Fail)
                        [RISK_PREDICTION]: (Based on machine params)
                        ตอบเป็นภาษาไทย
                        """
                        
                        # ส่งข้อมูลให้ AI
                        response = model.generate_content([prompt, image])
                        result_text = response.text
                        
                        # --- Logic การแสดงผลแบบใหม่ (ฉลาดขึ้น) ---
                        # ถ้า AI บอกว่า FAIL และมั่นใจเกิน 70% ถึงจะเตือน
                        if "[STATUS]: FAIL" in result_text:
                            st.error(f"🚨 DETECTED: พบความผิดปกติใน Lot {lot_number}")
                            risk_level = "High"
                        elif "Critical" in result_text:
                            st.error("🚨 CRITICAL WARNING!")
                            risk_level = "Critical"
                        else:
                            # เคสปกติ ให้ขึ้นสีเขียวสบายตา
                            st.success(f"✅ Lot {lot_number} : ผ่านเกณฑ์ (PASS)")
                            risk_level = "Low"
                            
                        st.markdown("### 🧠 AI Analysis Details")
                        st.code(result_text, language='yaml') # แสดงผลแบบ Code Block จะดู Professional ขึ้น
                        Response Format:
                        [RISK_LEVEL]: (Low / Medium / High / Critical)
                        [PREDICTION]: (Defect Name)
                        [ADVICE]: (Action for operator)
                        """
                        
                        response = model.generate_content([prompt, image])
                        result_text = response.text
                        
                        # Logic แสดงผล
                        if "High" in result_text or "Critical" in result_text:
                            st.error(f"🚨 WARNING: Lot {lot_number} มีความเสี่ยงสูง!")
                        elif "Medium" in result_text:
                            st.warning("⚠️ Caution: Abnormal Condition Warning")
                        else:
                            st.success(f"✅ Lot {lot_number} ปกติ: Conditions เหมาะสม")
                            
                        st.markdown("### 🧠 AI Assessment")
                        st.write(result_text)
                        
                        # บันทึก Log
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        risk_level = "Low"
                        if "Critical" in result_text: risk_level = "Critical"
                        elif "High" in result_text: risk_level = "High"
                        elif "Medium" in result_text: risk_level = "Medium"
                        
                        save_log(current_time, lot_number, machine_temp, pressure, line_speed, result_text, risk_level)
                        st.toast(f"บันทึกข้อมูล Lot {lot_number} เรียบร้อย!", icon="💾")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Waiting for CCTV Input...")

st.divider()
st.subheader("📜 Production History Log")
if os.path.isfile('defect_history.csv'):
    df = pd.read_csv('defect_history.csv')
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
