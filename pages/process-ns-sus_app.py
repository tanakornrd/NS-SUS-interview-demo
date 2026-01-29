import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import io

# ==========================================
# 1. ระบบจัดการฐานข้อมูล (จำลองด้วย CSV)
# ==========================================
DB_FILE = 'tracking_db.csv'

def init_db():
    expected_columns = ['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days', 'Current_Handler', 'Action_History', 'Final_Decision', 'Resolution_Note']
    
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(DB_FILE, index=False)
    else:
        # 🛠️ AUTO-MIGRATION SYSTEM (ฉลาดขึ้น)
        df = pd.read_csv(DB_FILE)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        
        # ถ้ามีคอลัมน์ไม่ครบ หรือ Current_Handler เป็น "System" ให้ซ่อม
        if missing_cols or 'Current_Handler' in df.columns:
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = "" # สร้างคอลัมน์ว่างไว้ก่อน
            
            # 🚑 FIX DATA: ถ้า Handler เป็น System หรือว่างเปล่า ให้ก๊อปจาก Department มาใส่เลย
            mask = (df['Current_Handler'] == "System") | (df['Current_Handler'].isna()) | (df['Current_Handler'] == "")
            if 'Department' in df.columns:
                df.loc[mask, 'Current_Handler'] = df.loc[mask, 'Department']
            
            df.to_csv(DB_FILE, index=False)

def save_to_db(lot_id, complaint, dept, status, days):
    df = pd.read_csv(DB_FILE)
    new_data = pd.DataFrame({
        'Lot_ID': [lot_id],
        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M")],
        'Complaint': [complaint],
        'Department': [dept],
        'Status': [status],
        'Estimated_Days': [days],
        'Current_Handler': [dept],
        'Action_History': [f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Case Created -> Assigned to {dept}"],
        'Final_Decision': [""],
        'Resolution_Note': [""]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_status(lot_id, new_status, action_note, next_handler=None, final_decision=None, resolution_note=None):
    df = pd.read_csv(DB_FILE)
    idx = df[df['Lot_ID'].astype(str) == str(lot_id)].index
    if not idx.empty:
        df.loc[idx, 'Status'] = new_status
        history = df.loc[idx, 'Action_History'].values[0]
        new_history = f"{history} || [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {action_note}"
        df.loc[idx, 'Action_History'] = new_history
        
        if next_handler:
            df.loc[idx, 'Current_Handler'] = next_handler
            
        if final_decision:
             df.loc[idx, 'Final_Decision'] = final_decision
             
        if resolution_note:
             df.loc[idx, 'Resolution_Note'] = resolution_note
            
        df.to_csv(DB_FILE, index=False)
        return True
    return False

def get_all_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    return pd.read_csv(DB_FILE)

init_db()

# ==========================================
# 2. ส่วนสมอง AI
# ==========================================
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('complaints_data.csv'):
            data = {
                'text': ['สนิมขึ้น', 'ขนาดไม่ได้', 'ส่งช้า', 'สินค้าบุบ', 'สีเพี้ยน', 'ความแข็งไม่ได้มาตรฐาน', 'ขนส่งทำของเสียหาย'],
                'department': ['QC', 'QC', 'Logistics', 'Logistics', 'QC', 'R&D', 'Logistics']
            }
            pd.DataFrame(data).to_csv('complaints_data.csv', index=False)
            
        df = pd.read_csv('complaints_data.csv')
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(df['text'], df['department'])
        return model
    except Exception as e:
        return None

global_model = load_model()

# ==========================================
# 3. Helper Functions (Report)
# ==========================================
def generate_final_report(case_data):
    content = f"""
    ========================================
    OFFICIAL RESOLUTION LETTER
    NIPPON STEEL & SUMIKIN (NSSUS)
    ========================================
    Date: {datetime.now().strftime("%Y-%m-%d")}
    Ref Lot ID: {case_data['Lot_ID']}
    
    To: Valued Customer
    
    Subject: Result of Claim Investigation
    
    Regarding your complaint about "{case_data['Complaint']}", 
    our Quality Assurance team has completed the investigation.
    
    ----------------------------------------
    FINAL DECISION: {case_data['Final_Decision']}
    ----------------------------------------
    
    DETAIL & RESOLUTION:
    {case_data['Resolution_Note']}
    
    We apologize for any inconvenience caused and appreciate your partnership.
    
    Sincerely,
    Customer Service Department
    NSSUS
    ========================================
    """
    return content

# ==========================================
# 4. User Interface
# ==========================================
st.set_page_config(page_title="Smart Claim Tracking", page_icon="📦", layout="wide")

st.title("📦 NSSUS Smart Claim & Tracking Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Summary", "📝 Submit New Case", "✅ Workflow Approval", "🔍 Customer Tracking"])

df = get_all_data()

# --- TAB 1: Dashboard ---
with tab1:
    st.header("📈 ภาพรวมการจัดการข้อร้องเรียน")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        total_cases = len(df)
        completed_cases = len(df[df['Status'] == 'Case Closed'])
        pending_cases = total_cases - completed_cases
        
        col1.metric("Total Cases", total_cases)
        col2.metric("Closed/Resolved", completed_cases)
        col3.metric("Pending", pending_cases)
        col4.metric("Avg. Resolution Time", "2.5 Days")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("สถานะงาน (Status)")
            if 'Status' in df.columns:
                st.bar_chart(df['Status'].value_counts())
        with c2:
            st.subheader("ผลการตัดสิน (Outcome)")
            if 'Final_Decision' in df.columns:
                outcomes = df[df['Final_Decision'] != ""]['Final_Decision'].value_counts()
                if not outcomes.empty:
                    st.bar_chart(outcomes)
    else:
        st.info("ยังไม่มีข้อมูลเคสในระบบ")

# --- TAB 2: Submit ---
with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Submit New Claim")
        lot_input = st.text_input("ระบุเลข Lot สินค้า (Lot No.):", placeholder="เช่น LOT-2026-001")
        complaint_input = st.text_area("อาการเสีย (Complaint):", height=100)
        
        if st.button("🚀 Process & Save", type="primary"):
            if lot_input and complaint_input and global_model:
                with st.spinner("AI Processing..."):
                    time.sleep(1)
                    predicted_dept = global_model.predict([complaint_input])[0]
                    status = f"Assigned to {predicted_dept}"
                    days = 3
                    if predicted_dept == "R&D": days = 7
                    elif predicted_dept == "Logistics": days = 2
                    
                    save_to_db(lot_input, complaint_input, predicted_dept, status, days)
                st.success(f"✅ บันทึกข้อมูลสำเร็จ! ส่งต่อให้แผนก **{predicted_dept}**")
            else:
                st.warning("กรุณากรอกข้อมูลให้ครบถ้วน")
    with col2:
        st.info("💡 **AI Auto-Routing**\nระบบจะวิเคราะห์ข้อความและส่งงานไปยังแผนกที่เกี่ยวข้องอัตโนมัติ")

# --- TAB 3: Workflow (ปรับปรุงใหม่: มี Tab แยก History) ---
with tab3:
    st.header("✅ Workflow & Action Center")
    user_dept = st.selectbox("เลือกฝ่ายของคุณ (Simulate User Role):", ["QC", "R&D", "Logistics", "Customer Service", "System Admin"])
    
    # สร้าง Sub-Tabs เพื่อแยกงานค้าง กับ งานเสร็จ
    subtab_active, subtab_history = st.tabs(["⚡ งานรอดำเนินการ (Pending)", "📜 ประวัติงานที่เสร็จแล้ว (History)"])

    if not df.empty:
        # === LOGIC การกรองงาน (FILTERING) ===
        # 1. กรองงานที่เสร็จแล้ว (Case Closed)
        completed_tasks = df[df['Status'] == 'Case Closed']
        
        # 2. กรองงานที่ยังไม่เสร็จ (Active)
        active_tasks_all = df[df['Status'] != 'Case Closed']
        
        # 3. กรองตาม User Role (สำหรับงาน Active)
        my_active_tasks = pd.DataFrame()
        if user_dept == "System Admin":
            my_active_tasks = active_tasks_all # Admin เห็นงานค้างทั้งหมด
        else:
            if 'Current_Handler' in df.columns:
                my_active_tasks = active_tasks_all[active_tasks_all['Current_Handler'] == user_dept]

        # === SHOW ACTIVE TASKS ===
        with subtab_active:
            if user_dept == "System Admin":
                st.info(f"👀 System Admin Mode: กำลังดูงานค้างทั้งหมดในระบบ ({len(my_active_tasks)} เคส)")
            
            if not my_active_tasks.empty:
                for index, row in my_active_tasks.iterrows():
                    # การ์ดแสดงงาน
                    with st.container():
                        st.markdown(f"### 📌 {row['Lot_ID']}")
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.info(f"**อาการ:** {row['Complaint']}")
                            st.markdown(f"**สถานะปัจจุบัน:** `{row['Status']}`")
                            st.markdown(f"**ผู้รับผิดชอบ:** `{row['Current_Handler']}`")
                            with st.expander("ดูประวัติ (History)"):
                                if pd.notna(row['Action_History']):
                                    for h in str(row['Action_History']).split(' || '):
                                        st.caption(f"• {h}")
                        
                        with c2:
                            st.write("### 🛠️ Action Zone")
                            # ถ้าเป็น Admin ให้เตือนว่ากำลังแก้ในนามใคร
                            if user_dept == "System Admin":
                                st.caption(f"⚠️ คุณกำลังจัดการงานแทนฝ่าย: **{row['Current_Handler']}**")

                            # Logic ปุ่มกด (แยกตามแผนกที่ถือเรื่อง)
                            current_handler = row['Current_Handler']
                            
                            # ถ้าเป็นตาของ CS (Customer Service)
                            if current_handler == "Customer Service":
                                st.markdown("#### ⚖️ Final Decision")
                                decision = st.selectbox("ผลการพิจารณา:", 
                                    ["✅ อนุมัติเคลม (Approve)", "🤝 ประนีประนอม (Compromise)", "❌ ปฏิเสธ (Reject)"],
                                    key=f"dec_{row['Lot_ID']}")
                                resolution_msg = st.text_area("ข้อความถึงลูกค้า:", key=f"res_{row['Lot_ID']}")
                                
                                if st.button("🏁 Close Case", type="primary", key=f"close_{row['Lot_ID']}"):
                                    update_status(row['Lot_ID'], "Case Closed", f"CS Decision: {decision}", 
                                                  next_handler="Completed", final_decision=decision, resolution_note=resolution_msg)
                                    st.success("ปิดเคสเรียบร้อย! ย้ายไปที่ tab History แล้ว")
                                    st.rerun()
                            
                            # ถ้าเป็นตาของแผนกอื่น (QC, R&D, Logistics)
                            else:
                                action_note = st.text_input("ผลการตรวจสอบ:", key=f"note_{row['Lot_ID']}")
                                if st.button("ส่งต่อให้ Customer Service", key=f"fwd_{row['Lot_ID']}"):
                                    update_status(row['Lot_ID'], "Investigation Complete", 
                                                  f"{current_handler}: {action_note}", next_handler="Customer Service")
                                    st.success("ส่งงานต่อเรียบร้อย!")
                                    st.rerun()
                        st.divider()
            else:
                st.success(f"🎉 ไม่มีงานค้างสำหรับฝ่าย {user_dept} ครับ")

        # === SHOW HISTORY ===
        with subtab_history:
            st.markdown(f"### 🗂️ รายการที่ปิดเคสแล้วทั้งหมด ({len(completed_tasks)} เคส)")
            if not completed_tasks.empty:
                st.dataframe(completed_tasks[['Lot_ID', 'Date', 'Complaint', 'Final_Decision', 'Status']])
            else:
                st.info("ยังไม่มีเคสที่ปิดงานเสร็จสิ้น")
    else:
        st.info("ไม่มีข้อมูลในระบบ")

# --- TAB 4: Customer Tracking ---
with tab4:
    st.subheader("🔍 Track Your Claim Status")
    track_id = st.text_input("กรอกเลข Lot ที่ต้องการค้นหา:", placeholder="Enter Lot No...", key="track_input")
    
    if st.button("🔎 Search", key="track_btn"):
        df_latest = get_all_data()
        if not df_latest.empty:
            result = df_latest[df_latest['Lot_ID'].astype(str) == str(track_id)]
            if not result.empty:
                res = result.iloc[-1]
                st.success("✅ พบข้อมูลสินค้า")
                
                status_val = 30
                status_str = str(res['Status'])
                if "Investigation" in status_str: status_val = 60
                if "Case Closed" in status_str: status_val = 100
                st.progress(status_val)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Lot ID:** {res['Lot_ID']}")
                    st.markdown(f"**Status:** `{res['Status']}`")
                with c2:
                    st.markdown(f"**Dept:** {res['Department']}")
                    st.markdown(f"**Handler:** {res['Current_Handler']}")
                
                st.divider()
                
                if res['Status'] == 'Case Closed':
                    st.markdown("### 📢 ผลการพิจารณา")
                    decision_text = str(res['Final_Decision'])
                    if "Approve" in decision_text: st.success(f"🎉 {decision_text}")
                    elif "Reject" in decision_text: st.error(f"⚠️ {decision_text}")
                    else: st.warning(f"🤝 {decision_text}")
                    st.info(f"**รายละเอียด:**\n{res['Resolution_Note']}")
                    
                    report_content = generate_final_report(res)
                    st.download_button(label="📄 ดาวน์โหลดจดหมาย (Official Letter)", data=report_content, file_name=f"Resolution_{res['Lot_ID']}.txt")
                else:
                    st.info("🕒 กำลังตรวจสอบครับ")
                    
                with st.expander("Timeline"):
                    if pd.notna(res['Action_History']):
                        for h in str(res['Action_History']).split(' || '):
                            st.caption(f"• {h}")
            else:
                st.error("ไม่พบข้อมูล Lot ID นี้")
