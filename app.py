"""
========================================================
HR Attrition Analytics - Streamlit Dashboard (현업용)
조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화
========================================================
서강대학교 AI·SW대학원 | 김현태 (A74032)

실행: streamlit run app.py
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# ─── 페이지 설정 ───
st.set_page_config(page_title="HR Attrition Analytics", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-header{font-size:.95rem;color:#64748b}
div[data-testid="stMetricValue"]{font-size:1.8rem}
.risk-badge{display:inline-block;padding:4px 12px;border-radius:6px;font-weight:700;font-size:13px}
.risk-high{background:#fef2f2;color:#ef4444;border:1px solid #fecaca}
.risk-mid{background:#fffbeb;color:#f59e0b;border:1px solid #fed7aa}
.risk-low{background:#f0fdf4;color:#22c55e;border:1px solid #bbf7d0}
.score-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:8px 0}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 분석 파이프라인 (캐시됨)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(emp_bytes, survey_bytes, training_bytes):
    progress = st.progress(0, text="📂 데이터 로딩 중...")
    
    # 1. 로드
    emp = pd.read_csv(io.BytesIO(emp_bytes), encoding='utf-8-sig')
    survey = pd.read_csv(io.BytesIO(survey_bytes))
    training = pd.read_csv(io.BytesIO(training_bytes))
    progress.progress(10, text="✅ 데이터 로드 완료")
    
    # 2. 전처리
    drop_cols = ['FirstName','LastName','ADEmail','Supervisor']
    emp = emp.drop(columns=[c for c in drop_cols if c in emp.columns])
    
    for col in ['DepartmentType','Division','Title','EmployeeStatus','EmployeeType','TerminationType']:
        if col in emp.columns:
            emp[col] = emp[col].astype(str).str.strip()
    
    emp['Attrition'] = emp['EmployeeStatus'].apply(
        lambda x: 1 if x in ['Voluntarily Terminated','Terminated for Cause'] else 0)
    
    emp['StartDate_parsed'] = pd.to_datetime(emp['StartDate'], errors='coerce', dayfirst=True)
    emp['ExitDate_parsed'] = pd.to_datetime(emp['ExitDate'], errors='coerce', dayfirst=True)
    emp['DOB_parsed'] = pd.to_datetime(emp['DOB'], errors='coerce', dayfirst=True)
    
    ref_date = emp['ExitDate_parsed'].max()
    if pd.isna(ref_date): ref_date = pd.Timestamp.now()
    
    emp['Tenure_Years'] = emp.apply(
        lambda r: (r['ExitDate_parsed']-r['StartDate_parsed']).days/365.25 
        if pd.notna(r['ExitDate_parsed']) 
        else (ref_date-r['StartDate_parsed']).days/365.25, axis=1).round(1)
    emp['Age'] = ((ref_date - emp['DOB_parsed']).dt.days / 365.25).round(0)
    emp['Exit_Year'] = emp['ExitDate_parsed'].dt.year
    emp['Exit_YM'] = emp['ExitDate_parsed'].dt.to_period('M').astype(str)
    
    progress.progress(25, text="✅ 전처리 완료")
    
    # 3. 통합
    merged = emp.merge(survey, left_on='EmpID', right_on='Employee ID', how='left')
    merged = merged.merge(training, left_on='EmpID', right_on='Employee ID', how='left')
    merged = merged.drop(columns=['Employee ID_x','Employee ID_y'], errors='ignore')
    progress.progress(35, text="✅ 데이터 통합 완료")
    
    # 4. 인코딩
    encode_map = {'DepartmentType':'Dept_enc','Title':'Title_enc','GenderCode':'Gender_enc',
                  'EmployeeType':'EmpType_enc','Performance Score':'Perf_enc'}
    encoders = {}
    for col, enc in encode_map.items():
        le = LabelEncoder()
        merged[enc] = le.fit_transform(merged[col].astype(str))
        encoders[col] = le
    
    feature_cols = ['Dept_enc','Title_enc','Gender_enc','EmpType_enc','Perf_enc',
                    'Current Employee Rating','Engagement Score','Satisfaction Score',
                    'Work-Life Balance Score','Training Duration(Days)','Training Cost','LocationCode']
    feature_labels = ['Department','Title','Gender','Employee Type','Performance',
                      'Employee Rating','Engagement','Satisfaction','Work-Life Balance',
                      'Training Duration','Training Cost','Location']
    
    X = merged[feature_cols].fillna(0)
    y = merged['Attrition']
    
    progress.progress(45, text="🔄 SMOTE 처리 중...")
    
    # 5. SMOTE + RF
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    progress.progress(60, text="🧠 Random Forest 학습 중...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['재직','퇴직'], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    
    fi_df = pd.DataFrame({'feature':feature_labels,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    
    progress.progress(75, text="📊 SHAP 분석 중...")
    
    # 6. SHAP
    explainer = shap.TreeExplainer(rf)
    X_shap = X_test[:300]
    X_shap_df = pd.DataFrame(X_shap, columns=feature_labels)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_class1 = shap_vals[:,:,1]
    else:
        shap_class1 = shap_vals
    
    progress.progress(85, text="📋 개별 직원 이탈 스코어링 중...")
    
    # 7. 개별 직원 이탈 위험 스코어링
    X_all = merged[feature_cols].fillna(0)
    merged['Attrition_Prob'] = rf.predict_proba(X_all)[:,1]
    merged['Risk_Score'] = (merged['Attrition_Prob'] * 100).round(1)
    merged['Risk_Level'] = merged['Risk_Score'].apply(
        lambda x: '🔴 높음' if x >= 60 else '🟡 중간' if x >= 30 else '🟢 낮음')
    
    # 8. 집계
    dept_stats = emp.groupby('DepartmentType').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    dept_stats['rate'] = (dept_stats['terminated']/dept_stats['total']*100).round(1)
    
    title_stats = emp.groupby('Title').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    title_stats['rate'] = (title_stats['terminated']/title_stats['total']*100).round(1)
    
    cross_stats = emp.groupby(['DepartmentType','Title']).agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    cross_stats['rate'] = (cross_stats['terminated']/cross_stats['total']*100).round(1)
    
    exited = emp[emp['ExitDate_parsed'].notna()]
    yearly_exits = exited.groupby('Exit_Year')['EmpID'].count().reset_index()
    yearly_exits.columns = ['year','exits']
    yearly_data = []
    for _,row in yearly_exits.iterrows():
        yr = int(row['year'])
        act = len(emp[(emp['StartDate_parsed'].dt.year<=yr)&((emp['ExitDate_parsed'].isna())|(emp['ExitDate_parsed'].dt.year>=yr))])
        yearly_data.append({'year':yr,'exits':int(row['exits']),'active':act,'rate':round(row['exits']/max(act,1)*100,1)})
    yearly_df = pd.DataFrame(yearly_data)
    
    survey_compare = merged.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
    
    progress.progress(100, text="✅ 분석 완료!")
    
    return {
        'emp':emp, 'merged':merged, 'exited':exited,
        'rf':rf, 'fi_df':fi_df, 'cm':cm, 'report':report, 'auc':auc,
        'shap_class1':shap_class1, 'X_shap_df':X_shap_df,
        'dept_stats':dept_stats, 'title_stats':title_stats,
        'cross_stats':cross_stats, 'yearly_df':yearly_df,
        'survey_compare':survey_compare, 'feature_labels':feature_labels,
        'encoders':encoders
    }


# ═══════════════════════════════════════════════════════════════
# PDF 보고서 생성
# ═══════════════════════════════════════════════════════════════
def generate_pdf_report(R, sel_dept):
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica','B',14)
            self.cell(0,10,'HR Attrition Analysis Report',align='C',new_x="LMARGIN",new_y="NEXT")
            self.set_font('Helvetica','',9)
            self.cell(0,6,f'Department: {sel_dept} | Generated by HR Attrition Analytics',align='C',new_x="LMARGIN",new_y="NEXT")
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica','I',8)
            self.cell(0,10,f'Page {self.page_no()}',align='C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Executive Summary
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Executive Summary',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',10)
    
    emp = R['emp']
    total = len(emp)
    termed = emp['Attrition'].sum()
    rate = round(termed/total*100,1)
    
    pdf.cell(0,6,f'Total Employees: {total:,}',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,6,f'Terminated: {termed} ({rate}%)',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,6,f'Model ROC-AUC: {R["auc"]:.4f}',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(5)
    
    # Department Stats
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Department Attrition Rates',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',9)
    
    dept_data = R['dept_stats'].sort_values('rate',ascending=False)
    if sel_dept != 'All':
        dept_data = dept_data[dept_data['DepartmentType']==sel_dept]
    
    # Table header
    pdf.set_font('Helvetica','B',9)
    pdf.cell(60,7,'Department',border=1)
    pdf.cell(30,7,'Total',border=1,align='C')
    pdf.cell(30,7,'Terminated',border=1,align='C')
    pdf.cell(30,7,'Rate (%)',border=1,align='C')
    pdf.ln()
    
    pdf.set_font('Helvetica','',9)
    for _,row in dept_data.iterrows():
        pdf.cell(60,6,str(row['DepartmentType']),border=1)
        pdf.cell(30,6,str(row['total']),border=1,align='C')
        pdf.cell(30,6,str(row['terminated']),border=1,align='C')
        pdf.cell(30,6,str(row['rate']),border=1,align='C')
        pdf.ln()
    
    pdf.ln(5)
    
    # Feature Importance
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Top 5 Feature Importance',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',9)
    for _,row in R['fi_df'].head(5).iterrows():
        pdf.cell(0,6,f"  - {row['feature']}: {row['importance']:.4f}",new_x="LMARGIN",new_y="NEXT")
    
    pdf.ln(5)
    
    # High Risk Employees
    pdf.set_font('Helvetica','B',12)
    pdf.cell(0,8,'Top 10 High-Risk Employees',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',8)
    
    risk_df = R['merged'].nlargest(10, 'Risk_Score')
    pdf.set_font('Helvetica','B',8)
    pdf.cell(20,6,'EmpID',border=1)
    pdf.cell(50,6,'Department',border=1)
    pdf.cell(50,6,'Title',border=1)
    pdf.cell(25,6,'Risk %',border=1,align='C')
    pdf.cell(25,6,'Level',border=1,align='C')
    pdf.ln()
    
    pdf.set_font('Helvetica','',8)
    for _,row in risk_df.iterrows():
        pdf.cell(20,5,str(row['EmpID']),border=1)
        pdf.cell(50,5,str(row['DepartmentType'])[:20],border=1)
        pdf.cell(50,5,str(row['Title'])[:20],border=1)
        pdf.cell(25,5,str(row['Risk_Score']),border=1,align='C')
        level_text = 'HIGH' if row['Risk_Score']>=60 else 'MID' if row['Risk_Score']>=30 else 'LOW'
        pdf.cell(25,5,level_text,border=1,align='C')
        pdf.ln()
    
    return pdf.output()


# ═══════════════════════════════════════════════════════════════
# ChatGPT API 액션 플랜
# ═══════════════════════════════════════════════════════════════
def get_ai_action_plan(api_key, dept_name, dept_rate, dept_total, fi_text, context):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt = f"""당신은 HR 전략 컨설턴트입니다. 아래 분석 결과를 기반으로 
{dept_name} 조직에 대한 맞춤 HR 액션 플랜을 한국어로 작성해주세요.

{context}

{dept_name} 조직: 인원 {dept_total}명, 이탈률 {dept_rate}%
주요 Feature Importance: {fi_text}

[출력 형식]
1. 주요 이탈 원인 분석 (2-3가지)
2. 긴급 조치 사항 (1-2가지)  
3. 중장기 개선 방안 (2-3가지)
4. 기대 효과
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 호출 오류: {e}"


# ═══════════════════════════════════════════════════════════════
# 메인 앱
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">🏢 HR Attrition Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화 · AI·SW대학원 김현태 / A74032</p>', unsafe_allow_html=True)
    st.divider()
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # ─── 사이드바 ───
    with st.sidebar:
        st.header("📂 데이터 업로드")
        emp_file = st.file_uploader("직원 인사정보", type=['csv'], help="employee_data.csv")
        survey_file = st.file_uploader("직원 설문조사", type=['csv'], help="engagement_survey.csv")
        training_file = st.file_uploader("직원 교육정보", type=['csv'], help="training_data.csv")
        
        all_up = emp_file and survey_file and training_file
        if all_up:
            st.success("✅ 3개 파일 업로드 완료")
            if st.button("🧠 AI 이탈 분석 시작", type="primary", use_container_width=True):
                st.session_state.results = run_pipeline(
                    emp_file.getvalue(), survey_file.getvalue(), training_file.getvalue())
                st.rerun()
        else:
            cnt = sum([1 for f in [emp_file, survey_file, training_file] if f])
            st.info(f"📎 {cnt}/3 파일 업로드됨")
        
        if st.session_state.results:
            st.divider()
            st.header("🔍 필터")
            R = st.session_state.results
            depts = ['전체'] + sorted(R['dept_stats']['DepartmentType'].tolist())
            sel_dept = st.selectbox("Department", depts)
            
            if sel_dept == '전체':
                tlist = ['전체'] + sorted(R['title_stats']['Title'].tolist())
            else:
                dt = R['cross_stats'][R['cross_stats']['DepartmentType']==sel_dept]['Title'].unique()
                tlist = ['전체'] + sorted(dt.tolist())
            sel_title = st.selectbox("Job Title", tlist)
            
            st.divider()
            st.header("⚙️ AI 설정")
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    # ─── 분석 전 안내 ───
    if st.session_state.results is None:
        st.markdown("### 👋 시작하기")
        st.markdown("왼쪽 사이드바에서 **3개의 CSV 파일**을 업로드한 후 **AI 이탈 분석 시작** 버튼을 클릭하세요.")
        st.markdown("""
업로드하면 자동으로 수행됩니다:
1. **데이터 전처리** — 결측치 처리, 피처 엔지니어링, 테이블 통합
2. **탐색적 분석** — 조직/직무/연도별 이탈 패턴 시각화
3. **머신러닝 모델링** — Random Forest + SMOTE
4. **SHAP 분석** — 이탈 요인 해석
5. **개별 직원 스코어링** — 이탈 위험도 점수 산출
6. **HR 액션 플랜** — ChatGPT API 연동 자동 생성
""")
        c1,c2,c3 = st.columns(3)
        c1.info("📁 **employee_data.csv**\n\n직원 인사정보")
        c2.info("📁 **engagement_survey.csv**\n\n직원 설문조사")
        c3.info("📁 **training_data.csv**\n\n교육 훈련 정보")
        return
    
    # ═══════════════════════════════════════════════════════════
    # 대시보드
    # ═══════════════════════════════════════════════════════════
    R = st.session_state.results
    sel_dept = st.session_state.get('sel_dept', '전체')
    sel_title = st.session_state.get('sel_title', '전체')
    api_key = st.session_state.get('api_key', '')
    
    # 필터 적용된 데이터
    emp = R['emp']
    if sel_dept != '전체':
        filt = emp[emp['DepartmentType']==sel_dept]
    else:
        filt = emp
    if sel_title != '전체':
        filt = filt[filt['Title']==sel_title]
    
    kpi_total = len(filt)
    kpi_term = filt['Attrition'].sum()
    kpi_rate = round(kpi_term/max(kpi_total,1)*100,1)
    
    # KPI
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("전체 인원", f"{kpi_total:,}명")
    k2.metric("퇴직자 수", f"{kpi_term}명")
    k3.metric("이탈률", f"{kpi_rate}%",
              delta="위험" if kpi_rate>=15 else "주의" if kpi_rate>=10 else "양호",
              delta_color="inverse" if kpi_rate>=10 else "normal")
    k4.metric("재직자", f"{kpi_total-kpi_term:,}명")
    
    st.divider()
    
    # ─── 탭 ───
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 이탈 분석", "🧠 주요 원인 분석", "📈 모델 성능",
        "👤 직원 이탈 스코어링", "🎯 HR 액션 플랜", "📥 보고서 다운로드"
    ])
    
    # ═══ TAB 1: 이탈 분석 ═══
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📉 조직별 이탈률")
            dd = R['dept_stats'].sort_values('rate', ascending=True)
            fig, ax = plt.subplots(figsize=(8,4))
            colors = [('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
            ax.barh(dd['DepartmentType'], dd['rate'], color=colors, height=0.6)
            for i,(rate,total) in enumerate(zip(dd['rate'],dd['total'])):
                ax.text(rate+0.3, i, f'{rate}% ({total}명)', va='center', fontsize=9)
            ax.set_xlabel('이탈률 (%)'); ax.set_title('조직별 이탈률', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        
        with c2:
            st.subheader("📊 직무별 이탈률 Top 10")
            td = R['title_stats'].copy()
            if sel_dept != '전체':
                td = R['cross_stats'][R['cross_stats']['DepartmentType']==sel_dept].copy()
            top10 = td.nlargest(10,'rate').sort_values('rate',ascending=True)
            fig, ax = plt.subplots(figsize=(8,4))
            colors = [('#ef4444' if r>=20 else '#f59e0b' if r>=10 else '#22c55e') for r in top10['rate']]
            ax.barh(top10['Title'], top10['rate'], color=colors, height=0.6)
            for i,(rate,total) in enumerate(zip(top10['rate'],top10['total'])):
                ax.text(rate+0.3, i, f'{rate}% (n={total})', va='center', fontsize=9)
            ax.set_xlabel('이탈률 (%)'); ax.set_title('직무별 이탈률 Top 10', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        
        # 연도별 추이
        st.subheader("📅 연도별 이탈 추이")
        c1,c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(R['yearly_df']['year'], R['yearly_df']['exits'], color='#ef4444', alpha=0.7)
            ax.plot(R['yearly_df']['year'], R['yearly_df']['exits'], 'o-', color='#991b1b', lw=2)
            for x,y in zip(R['yearly_df']['year'], R['yearly_df']['exits']):
                ax.text(x, y+8, str(y), ha='center', fontweight='bold')
            ax.set_title('연도별 퇴직 건수', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(R['yearly_df']['year'], R['yearly_df']['rate'], 'o-', color='#ef4444', lw=2,
                   markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax.fill_between(R['yearly_df']['year'], R['yearly_df']['rate'], alpha=0.1, color='#ef4444')
            for x,y in zip(R['yearly_df']['year'], R['yearly_df']['rate']):
                ax.text(x, y+0.5, f'{y}%', ha='center', fontweight='bold')
            ax.set_title('연도별 이탈률 추이', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        
        # 조직별 연도별
        st.subheader("🏢 조직별 연도별 퇴직 추이")
        dept_yr = R['exited'].groupby(['DepartmentType','Exit_Year'])['EmpID'].count().reset_index()
        dept_yr.columns = ['dept','year','exits']
        fig, ax = plt.subplots(figsize=(14,5))
        cmap = {'Production':'#ef4444','Software Engineering':'#f59e0b','IT/IS':'#3b82f6',
                'Sales':'#22c55e','Admin Offices':'#8b5cf6','Executive Office':'#06b6d4'}
        for dept in cmap:
            sub = dept_yr[dept_yr['dept']==dept]
            if len(sub)>0: ax.plot(sub['year'], sub['exits'], 'o-', label=dept, color=cmap.get(dept,'#999'), lw=2)
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        ax.set_title('조직별 연도별 퇴직 건수', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    
    # ═══ TAB 2: 주요 원인 분석 ═══
    with tab2:
        st.info("📌 전체 데이터 기반 분석 결과입니다.")
        
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("🧠 Feature Importance")
            fi = R['fi_df'].sort_values('importance',ascending=True)
            fig, ax = plt.subplots(figsize=(8,5))
            colors = plt.cm.viridis(np.linspace(0.3,0.9,len(fi)))
            ax.barh(fi['feature'], fi['importance'], color=colors, height=0.6)
            ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        
        with c2:
            st.subheader("🎯 재직자 vs 퇴직자 서베이")
            sc = R['survey_compare']
            sc.index = ['재직자','퇴직자']
            fig, ax = plt.subplots(figsize=(8,4))
            x = np.arange(3); w = 0.35
            ax.bar(x-w/2, sc.iloc[0], w, label='재직자', color='#3b82f6')
            ax.bar(x+w/2, sc.iloc[1], w, label='퇴직자', color='#ef4444')
            ax.set_xticks(x); ax.set_xticklabels(['Engagement','Satisfaction','Work-Life\nBalance'])
            ax.set_ylabel('평균 점수'); ax.legend(); ax.set_ylim(2.5,3.5)
            ax.set_title('서베이 비교', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(sc, use_container_width=True)
        
        # SHAP
        st.subheader("📊 SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(12,6))
        shap.summary_plot(R['shap_class1'], R['X_shap_df'], plot_type='dot', show=False)
        plt.title('SHAP - 퇴직 예측 기여도', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        
        # 교차 이탈률
        st.subheader("🔍 조직 × 직무 교차 이탈률")
        cross_sig = R['cross_stats'][R['cross_stats']['total']>=10].nlargest(15,'rate')
        st.dataframe(cross_sig[['DepartmentType','Title','total','terminated','rate']].rename(
            columns={'DepartmentType':'조직','Title':'직무','total':'인원','terminated':'퇴직','rate':'이탈률(%)'}),
            use_container_width=True, hide_index=True)
    
    # ═══ TAB 3: 모델 성능 ═══
    with tab3:
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("📋 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(R['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['재직 예측','퇴직 예측'], yticklabels=['실제 재직','실제 퇴직'])
            ax.set_title('Confusion Matrix', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.subheader("📊 성능 지표")
            st.metric("ROC-AUC Score", f"{R['auc']:.4f}")
            rpt = pd.DataFrame(R['report']).T
            st.dataframe(rpt.round(3), use_container_width=True)
    
    # ═══ TAB 4: 직원 이탈 스코어링 ═══
    with tab4:
        st.subheader("👤 개별 직원 이탈 위험도 스코어링")
        st.caption("Random Forest 모델이 각 직원의 이탈 확률을 0~100점으로 산출합니다.")
        
        merged_df = R['merged']
        
        # 필터 적용
        score_df = merged_df.copy()
        if sel_dept != '전체':
            score_df = score_df[score_df['DepartmentType']==sel_dept]
        if sel_title != '전체':
            score_df = score_df[score_df['Title']==sel_title]
        
        # 위험도 요약
        c1,c2,c3 = st.columns(3)
        high = len(score_df[score_df['Risk_Score']>=60])
        mid = len(score_df[(score_df['Risk_Score']>=30)&(score_df['Risk_Score']<60)])
        low = len(score_df[score_df['Risk_Score']<30])
        c1.metric("🔴 고위험 (60+)", f"{high}명", delta=f"{high/max(len(score_df),1)*100:.1f}%")
        c2.metric("🟡 중위험 (30-59)", f"{mid}명", delta=f"{mid/max(len(score_df),1)*100:.1f}%")
        c3.metric("🟢 저위험 (0-29)", f"{low}명", delta=f"{low/max(len(score_df),1)*100:.1f}%")
        
        # 위험도 분포 차트
        fig, axes = plt.subplots(1,2, figsize=(14,4))
        axes[0].hist(score_df['Risk_Score'], bins=20, color='#3b82f6', alpha=0.7, edgecolor='white')
        axes[0].axvline(60, color='#ef4444', ls='--', lw=2, label='고위험 기준(60)')
        axes[0].axvline(30, color='#f59e0b', ls='--', lw=2, label='중위험 기준(30)')
        axes[0].set_title('이탈 위험 점수 분포', fontweight='bold')
        axes[0].set_xlabel('Risk Score'); axes[0].set_ylabel('인원수'); axes[0].legend()
        
        dept_risk = score_df.groupby('DepartmentType')['Risk_Score'].mean().sort_values(ascending=True)
        axes[1].barh(dept_risk.index, dept_risk.values, color='#8b5cf6', height=0.6)
        for i,v in enumerate(dept_risk.values):
            axes[1].text(v+0.5, i, f'{v:.1f}', va='center', fontsize=10)
        axes[1].set_title('조직별 평균 위험 점수', fontweight='bold')
        axes[1].set_xlabel('평균 Risk Score')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        
        # 고위험 직원 테이블
        st.subheader("🔴 고위험 직원 목록 (Risk Score ≥ 60)")
        high_risk = score_df[score_df['Risk_Score']>=60].nlargest(50,'Risk_Score')
        
        display_cols = ['EmpID','DepartmentType','Title','Risk_Score','Risk_Level',
                       'Tenure_Years','Performance Score','Current Employee Rating']
        available = [c for c in display_cols if c in high_risk.columns]
        
        if len(high_risk) > 0:
            st.dataframe(
                high_risk[available].rename(columns={
                    'EmpID':'사번','DepartmentType':'조직','Title':'직무',
                    'Risk_Score':'위험점수','Risk_Level':'위험등급',
                    'Tenure_Years':'근속(년)','Performance Score':'성과등급',
                    'Current Employee Rating':'평가점수'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("✅ 선택된 필터 조건에 고위험 직원이 없습니다.")
        
        # 전체 테이블 (검색 가능)
        st.subheader("📋 전체 직원 위험도 테이블")
        search = st.text_input("🔍 사번 또는 직무로 검색", placeholder="예: 1001, Production...")
        
        full_df = score_df[available].rename(columns={
            'EmpID':'사번','DepartmentType':'조직','Title':'직무',
            'Risk_Score':'위험점수','Risk_Level':'위험등급',
            'Tenure_Years':'근속(년)','Performance Score':'성과등급',
            'Current Employee Rating':'평가점수'
        }).sort_values('위험점수', ascending=False)
        
        if search:
            mask = full_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            full_df = full_df[mask]
        
        st.dataframe(full_df, use_container_width=True, hide_index=True, height=400)
    
    # ═══ TAB 5: HR 액션 플랜 ═══
    with tab5:
        st.subheader("🎯 HR 액션 플랜")
        
        fi_text = ', '.join([f"{r['feature']}({r['importance']:.3f})" for _,r in R['fi_df'].head(5).iterrows()])
        context = f"""전체 인원: {len(emp)}명, 퇴직자: {emp['Attrition'].sum()}명, 이탈률: {emp['Attrition'].mean()*100:.1f}%
Feature Importance Top5: {fi_text}"""
        
        dept_list = R['dept_stats'].sort_values('rate', ascending=False)
        if sel_dept != '전체':
            dept_list = dept_list[dept_list['DepartmentType']==sel_dept]
        
        for _, row in dept_list.iterrows():
            rate = row['rate']
            risk_html = f'<span class="risk-badge risk-high">🔴 높음 ({rate}%)</span>' if rate>=15 \
                else f'<span class="risk-badge risk-mid">🟡 중간 ({rate}%)</span>' if rate>=5 \
                else f'<span class="risk-badge risk-low">🟢 낮음 ({rate}%)</span>'
            
            with st.expander(f"📋 {row['DepartmentType']} — 이탈률 {rate}%", expanded=(rate>=10)):
                st.markdown(risk_html, unsafe_allow_html=True)
                
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("인원", f"{row['total']}명")
                mc2.metric("퇴직", f"{int(row['terminated'])}명")
                mc3.metric("이탈률", f"{rate}%")
                
                # API 키가 있으면 ChatGPT 액션 플랜 생성
                if api_key:
                    if st.button(f"🧠 AI 액션 플랜 생성 - {row['DepartmentType']}", key=f"ai_{row['DepartmentType']}"):
                        with st.spinner("ChatGPT가 액션 플랜을 생성 중..."):
                            plan = get_ai_action_plan(api_key, row['DepartmentType'], rate, row['total'], fi_text, context)
                            st.markdown("### 🤖 AI 생성 액션 플랜")
                            st.markdown(plan)
                else:
                    # 프리셋 액션 플랜
                    if rate >= 15:
                        st.markdown("""
**🎯 권장 액션 플랜:**
- 🔴 **[긴급]** 핵심 인재 대상 리텐션 패키지 긴급 도입 (급여 조정 + 성과 보너스)
- 🔴 **[긴급]** 1:1 면담을 통한 이탈 위험 조기 감지 체계 구축
- 🟡 **[높음]** 승진 경로 투명화 및 역량 개발 프로그램 강화
- 🟡 **[높음]** 워라밸 개선 (유연근무, 교대근무 최적화)
- 🔵 **[중간]** 조직 문화 개선 및 소통 채널 확대
""")
                    elif rate >= 5:
                        st.markdown("""
**🎯 권장 액션 플랜:**
- 🟡 **[높음]** 특정 고이탈 직무 대상 맞춤 대응
- 🔵 **[중간]** 멘토링 프로그램 및 내부 직무 전환 경로 마련
- 🔵 **[중간]** 정기 서베이 강화 및 모니터링
""")
                    else:
                        st.markdown("""
**🎯 권장 액션 플랜:**
- 🟢 **[유지]** 현행 인사 정책 유지 및 정기 모니터링
- 🔵 **[중간]** 높은 리텐션 요인 분석 후 타 부서 벤치마킹
""")
                
                if not api_key:
                    st.caption("💡 사이드바에서 OpenAI API Key를 입력하면 ChatGPT가 조직별 맞춤 액션 플랜을 자동 생성합니다.")
    
    # ═══ TAB 6: 보고서 다운로드 ═══
    with tab6:
        st.subheader("📥 분석 보고서 다운로드")
        st.markdown("분석 결과를 PDF 보고서로 다운로드하여 경영진 보고에 활용할 수 있습니다.")
        
        c1,c2 = st.columns(2)
        with c1:
            report_dept = st.selectbox("보고서 대상 조직", ['All'] + sorted(R['dept_stats']['DepartmentType'].tolist()), key="report_dept")
        
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📄 PDF 보고서 생성", type="primary", use_container_width=True):
                with st.spinner("PDF 생성 중..."):
                    pdf_bytes = generate_pdf_report(R, report_dept)
                    st.download_button(
                        label="⬇️ PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"HR_Attrition_Report_{report_dept}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        st.divider()
        
        # CSV 다운로드
        st.subheader("📊 데이터 다운로드")
        c1,c2 = st.columns(2)
        
        with c1:
            risk_csv = R['merged'][['EmpID','DepartmentType','Title','Risk_Score','Risk_Level',
                                    'Tenure_Years','Performance Score','Current Employee Rating','Attrition']].copy()
            csv_data = risk_csv.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ 직원 위험도 스코어링 (CSV)", csv_data,
                             "employee_risk_scores.csv", "text/csv", use_container_width=True)
        
        with c2:
            dept_csv = R['dept_stats'].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ 조직별 이탈 통계 (CSV)", dept_csv,
                             "department_attrition.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
