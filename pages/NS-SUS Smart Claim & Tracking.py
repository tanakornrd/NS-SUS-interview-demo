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
# 1. ระบบจัดการฐานข้อมูล
# ==========================================
DB_FILE = 'tracking_db_v3_mcs.csv' 

def init_db():
    expected_columns = ['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days', 'Current_Handler', 'Action_History', 'Final_Decision', 'Resolution_Note']
    
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols: df[col] = ""
        
        if 'Current_Handler' in df.columns and 'Department' in df.columns:
            mask = (df['Current_Handler'] == "System") | (df['Current_Handler'].isnull())
            if mask.any():
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
        'Action_History': [f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Case Created -> AI Assigned to {dept}"],
        'Final_Decision': [""],
        'Resolution_Note': [""]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_status(lot_id, new_status, action_note, next_handler=None, final_decision=None, resolution_note=None, force_handler=None):
    df = pd.read_csv(DB_FILE)
    idx = df[df['Lot_ID'].astype(str) == str(lot_id)].index
    if not idx.empty:
        if force_handler:
            df.loc[idx, 'Current_Handler'] = force_handler
            df.loc[idx, 'Department'] = force_handler 
            new_status = f"Re-assigned to {force_handler}"
            action_note += f" (⚠️ MCS Manual Override)"

        df.loc[idx, 'Status'] = new_status
        history = df.loc[idx, 'Action_History'].values[0]
        new_history = f"{history} || [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {action_note}"
        df.loc[idx, 'Action_History'] = new_history
        
        if next_handler: df.loc[idx, 'Current_Handler'] = next_handler
        if final_decision: df.loc[idx, 'Final_Decision'] = final_decision
        if resolution_note: df.loc[idx, 'Resolution_Note'] = resolution_note
            
        df.to_csv(DB_FILE, index=False)
        return True
    return False

def get_all_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    return pd.read_csv(DB_FILE)

# === HELPER: สร้างรายงาน Text File ===
def generate_customer_report(case_data):
    report = f"""
    =======================================================
    OFFICIAL CLAIM RESOLUTION REPORT
    NS-SUS SMART CLAIM TRACKING SYSTEM
    =======================================================
    Date: {datetime.now().strftime("%Y-%m-%d")}
    Reference Lot ID: {case_data['Lot_ID']}
    
    -------------------------------------------------------
    CASE INFORMATION
    -------------------------------------------------------
    Issue Reported: {case_data['Complaint']}
    Date Reported: {case_data['Date']}
    Responsible Department: {case_data['Department']}
    
    -------------------------------------------------------
    INVESTIGATION & RESOLUTION
    -------------------------------------------------------
    Status: {case_data['Status']}
    
    FINAL DECISION: [{case_data['Final_Decision']}]
    
    NOTE TO CUSTOMER:
    {case_data['Resolution_Note']}
    
    -------------------------------------------------------
    ACTION HISTORY:
    {str(case_data['Action_History']).replace(' || ', '\n    - ')}
    
    -------------------------------------------------------
    Thank you for your trust in NS-SUS Quality Standards.
    
    Sincerely,
    Marketing & Customer Service (MCS) Team
    =======================================================
    """
    return report

init_db()

# ==========================================
# 2. ส่วนสมอง AI
# ==========================================
@st.cache_resource
def load_model():
    try:
        data = {
            'text': [
                'สนิมขึ้นที่ขอบเหล็ก', 'สินค้าบุบ', 'ขนาดความหนาไม่ได้ตามสเปค', 
                'ผิวเหล็กเป็นรอยขีดข่วน', 'ความแข็งไม่ได้มาตรฐาน', 'สีเคลือบหลุดร่อน',
                'มีคราบน้ำมันเยอะเกินไป', 'เหล็กยืดตัวไม่ได้', 'ขอบเหล็กคมเกินไป',
                'ค่า Yield Strength ต่ำ', 'Defect ที่ผิว', 'รอยกดทับ', 'เหล็กแตก',
                'ใบ COA ไม่ตรงกับสินค้า', 'เอกสารรับรองคุณภาพผิด', 'หาใบเซอร์ไม่เจอ',
                'ระบุเกรดเหล็กในใบส่งของผิด', 'ไม่ผ่านมาตรฐาน ISO', 'ตรวจสอบย้อนกลับไม่ได้',
                'สเปคในระบบไม่ตรงกับป้าย', 'เอกสารประกอบการเคลมไม่ครบ', 'Label ผิด',
                'ส่งของล่าช้ากว่ากำหนด', 'ติดต่อฝ่ายขายไม่ได้', 'พนักงานขับรถพูดจาไม่สุภาพ',
                'ส่งสินค้าผิดสถานที่', 'แพ็คเกจจิ้งเสียหายจากการขนส่ง', 'ขอใบเสนอราคาช้า',
                'ประสานงานยอดแย่', 'รถขนส่งมาไม่ตรงเวลา', 'แจ้งสถานะสินค้าผิด',
                'ค่าขนส่งแพงเกินไป', 'บริการหลังการขายไม่ดี'
            ],
            'department': [
                'QC', 'QC', 'QC', 'QC', 'QC', 'QC',
                'QC', 'QC', 'QC', 'QC', 'QC', 'QC', 'QC',
                'QA', 'QA', 'QA', 'QA', 'QA', 'QA',
                'QA', 'QA', 'QA',
                'MCS', 'MCS', 'MCS', 'MCS', 'MCS', 'MCS',
                'MCS', 'MCS', 'MCS', 'MCS', 'MCS'
            ]
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
# 3. User Interface
# ==========================================
st.set_page_config(page_title="NS-SUS Smart Claim", layout="wide")

with st.sidebar:
    st.title("🔧 Tools")
    if st.button("🗑️ Reset Database (Clear All)", type="primary"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            init_db()
            st.success("Database Cleared! 🧹")
            time.sleep(1)
            st.rerun()

st.title("NS-SUS Smart Claim & Tracking")

tab1, tab2, tab3, tab4 = st.tabs(["Executive Dashboard", "Submit & Log", "Workflow (Master Control)", "Customer Tracking"])

df = get_all_data()

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab1:
    st.markdown("### Real-time Analytics Dashboard")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        closed = len(df[df['Status'] == 'Case Closed'])
        active = total - closed
        success_rate = (closed / total) * 100 if total > 0 else 0
        
        col1.metric("Total Claims", total)
        col2.metric("Resolved", closed, delta=f"{success_rate:.1f}% Rate")
        col3.metric("Active Issues", active, delta_color="inverse")
        col4.metric("Avg. Resolution", "2.1 Days")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Defects by Dept")
            if 'Department' in df.columns:
                st.bar_chart(df['Department'].value_counts(), color="#FF4B4B")
        with c2:
            st.subheader("Work Status")
            if 'Status' in df.columns:
                st.bar_chart(df['Status'].value_counts(), color="#29B5E8")
    else:
        st.info("Waiting for data stream...")

# --- TAB 2: Submit & Log ---
with tab2:
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("➕ Create New Case")
            col_in1, col_in2 = st.columns([1, 2])
            with col_in1:
                lot_input = st.text_input("Lot No.", placeholder="e.g., LOT-2026-001")
            with col_in2:
                complaint_input = st.text_input("Issue / Complaint", placeholder="Describe the defect...")
            
            if st.button("Process & Save", type="primary", use_container_width=True):
                if lot_input and complaint_input and global_model:
                    with st.spinner("AI Categorizing..."):
                        time.sleep(0.5)
                        predicted_dept = global_model.predict([complaint_input])[0]
                        status = f"Assigned to {predicted_dept}"
                        days = 3
                        if predicted_dept == "QA": days = 1
                        elif predicted_dept == "MCS": days = 2
                        elif predicted_dept == "QC": days = 5
                        save_to_db(lot_input, complaint_input, predicted_dept, status, days)
                    st.success(f"New case assigned to **{predicted_dept}**")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("Please fill in all fields.")
        
        with c2:
            st.write("### 📥 Export Data")
            if not df.empty:
                buffer = io.BytesIO()
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_data,
                    file_name="NSSUS_Report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()
    st.subheader("Operational Log")
    if not df.empty:
        df_display = df.iloc[::-1].copy()
        st.dataframe(df_display[['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Current_Handler']], use_container_width=True, hide_index=True)
    else:
        st.info("No data available.")

# --- TAB 3: Workflow (Master Control) ---
with tab3:
    st.header("Workflow & Action Center")
    user_roles = ["QC", "QA", "MCS"]
    user_dept = st.selectbox("Login As:", user_roles)
    
    subtab_active, subtab_history = st.tabs(["Pending Tasks", "Completed History"])

    if not df.empty:
        completed_tasks = df[df['Status'] == 'Case Closed']
        active_tasks_all = df[df['Status'] != 'Case Closed']
        
        my_active_tasks = pd.DataFrame()
        if user_dept == "MCS":
            my_active_tasks = active_tasks_all
            st.success("MCS Manual Control Override Mode Active")
        else:
            if 'Current_Handler' in df.columns:
                my_active_tasks = active_tasks_all[active_tasks_all['Current_Handler'] == user_dept]

        with subtab_active:
            if not my_active_tasks.empty:
                for index, row in my_active_tasks.iterrows():
                    # ✅ FIXED: ใส่ Suffix เพื่อป้องกัน Duplicate Key
                    unique_suffix = f"_{row['Lot_ID']}_{index}"

                    with st.container(border=True):
                        c1, c2 = st.columns([1.5, 1])
                        with c1:
                            st.markdown(f"#### 📌 {row['Lot_ID']}")
                            st.markdown(f"**Issue:** {row['Complaint']}")
                            st.info(f"**Current Handler:** {row['Current_Handler']}") 
                            with st.expander("History Log"):
                                if pd.notna(row['Action_History']):
                                    for h in str(row['Action_History']).split(' || '):
                                        st.caption(f"• {h}")

                        with c2:
                            st.write("### Action")
                            if user_dept == "MCS":
                                if row['Current_Handler'] == "MCS":
                                    st.markdown("##### Final Decision")
                                    decision = st.selectbox("Outcome", ["Approve", "Compromise", "Reject"], key=f"d{unique_suffix}")
                                    note = st.text_input("Note to Customer", key=f"n{unique_suffix}")
                                    if st.button("Close Case", key=f"btn{unique_suffix}", type="primary"):
                                        update_status(row['Lot_ID'], "Case Closed", f"MCS: {decision}", "Completed", decision, note)
                                        st.rerun()
                                
                                st.markdown("---")
                                st.markdown("##### Manual Control Override")
                                target_depts = ["QC", "QA", "MCS"]
                                new_handler = st.selectbox("Re-assign to:", target_depts, key=f"move{unique_suffix}")
                                if st.button("⚠️ Force Re-assign", key=f"btn_move{unique_suffix}"):
                                    update_status(row['Lot_ID'], f"Re-assigned to {new_handler}", "MCS Manual Control Override", force_handler=new_handler)
                                    st.success(f"Corrected assignment to {new_handler}")
                                    st.rerun()

                            else: 
                                note = st.text_input("Investigation Note", key=f"in{unique_suffix}")
                                if st.button("➡️ Forward to MCS", key=f"fwd{unique_suffix}"):
                                    update_status(row['Lot_ID'], "Investigation Complete", f"{user_dept}: {note}", "MCS")
                                    st.rerun()
            else:
                st.success(f"🎉 No pending tasks for **{user_dept}**")
        
        with subtab_history:
            st.dataframe(completed_tasks, use_container_width=True)

# --- TAB 4: Customer Tracking ---
with tab4:
    st.subheader("Customer Status Check")
    track_id = st.text_input("Enter Lot No.", placeholder="LOT-XXXX-XXX")
    if st.button("Search"):
        df_latest = get_all_data()
        if not df_latest.empty:
            res = df_latest[df_latest['Lot_ID'].astype(str) == str(track_id)]
            if not res.empty:
                r = res.iloc[-1]
                st.success("✅ Found Case")
                st.progress(100 if r['Status'] == 'Case Closed' else 50)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Lot ID:** {r['Lot_ID']}")
                    st.write(f"**Status:** {r['Status']}")
                with c2:
                    st.write(f"**Dept:** {r['Department']}")
                    st.write(f"**Handler:** {r['Current_Handler']}")
                
                if r['Status'] == 'Case Closed':
                    st.divider()
                    st.info(f"**Final Decision:** {r['Final_Decision']}\n\n**Note:** {r['Resolution_Note']}")
                    
                    # === ✨ FEATURE ใหม่: ปุ่มดาวน์โหลด Report ===
                    report_content = generate_customer_report(r)
                    st.download_button(
                        label="📄 Download Official Resolution Report",
                        data=report_content,
                        file_name=f"Resolution_Report_{r['Lot_ID']}.txt",
                        mime="text/plain"
                    )
            else:
                st.error("Not Found")
