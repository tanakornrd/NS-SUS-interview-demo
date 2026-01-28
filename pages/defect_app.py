import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. ตั้งค่า API Key ---
GOOGLE_API_KEY = "AIzaSyBCPSibe8SD3TnEJe0IXw3RDvWi9nTshOo" 

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- ฟังก์ชันสำหรับบันทึกข้อมูลลงไฟล์ CSV ---
def save_to_csv(defect_type, analysis_text):
    file_name = 'defect_history.csv'
    # ตรวจสอบว่ามีไฟล์อยู่แล้วหรือยัง
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # ถ้ายังไม่มีไฟล์ ให้สร้างหัวตารางก่อน
        if not file_exists:
            writer.writerow(['Date & Time', 'Defect Type (AI Prediction)', 'Full Analysis'])
        
        # บันทึกข้อมูลใหม่ต่อท้าย
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_time, defect_type, analysis_text])

# --- 2. หน้าตาแอป (UI) ---
st.set_page_config(page_title="NSSUS Tech Service V2", page_icon="📝")

st.title("📝 NSSUS Smart Defect Log")
st.write("ระบบวิเคราะห์และบันทึกประวัติ Defect อัตโนมัติ")

# สร้างแท็บ 2 หน้า: หน้าวิเคราะห์ กับ หน้าดูประวัติ
tab1, tab2 = st.tabs(["🔍 วิเคราะห์ Defect", "📂 ประวัติการเคลม (History)"])

with tab1:
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ Defect", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='รูปภาพสินค้าที่มีปัญหา', width=400)
        
        if st.button('🔍 วิเคราะห์และบันทึกผล'):
            with st.spinner('AI กำลังวิเคราะห์และลงบันทึก...'):
                try:
                    # สั่ง AI
                    prompt = """
                    คุณคือผู้เชี่ยวชาญ NSSUS จงวิเคราะห์รูปนี้:
                    1. ระบุชื่อ Defect สั้นๆ (เช่น Pitting, Scratch) เอาไว้เป็นหัวข้อ
                    2. วิเคราะห์สาเหตุและวิธีแก้ไขโดยละเอียด
                    
                    ตอบกลับมาในรูปแบบ:
                    [ชื่อ Defect]
                    [รายละเอียดการวิเคราะห์]
                    """
                    response = model.generate_content([prompt, image])
                    text_result = response.text
                    
                    # แยกชื่อ Defect ออกมาจากบรรทัดแรก (เพื่อเก็บลงตารางสวยๆ)
                    lines = text_result.split('\n')
                    defect_name = lines[0].replace('*', '').strip() # เอาชื่อบรรทัดแรกมา
                    
                    # แสดงผล
                    st.success(f"ผลการวิเคราะห์: {defect_name}")
                    st.write(text_result)
                    
                    # บันทึกลงไฟล์
                    save_to_csv(defect_name, text_result)
                    st.toast("✅ บันทึกข้อมูลลง History เรียบร้อยแล้ว!", icon="💾")
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")

with tab2:
    st.header("📂 ประวัติการวิเคราะห์ทั้งหมด")
    st.write("ข้อมูลจะถูกบันทึกอยู่ในไฟล์ `defect_history.csv` ในเครื่องนี้")
    
    # โหลดไฟล์ CSV มาแสดงเป็นตาราง
    if os.path.isfile('defect_history.csv'):
        df = pd.read_csv('defect_history.csv')
        st.dataframe(df, use_container_width=True)
    else:
        st.info("ยังไม่มีข้อมูลประวัติ (ลองไปที่หน้าวิเคราะห์แล้วกดบันทึกดูครับ)")