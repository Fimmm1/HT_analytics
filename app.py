import streamlit as st
st.set_page_config(page_title="HR Attrition Analytics", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

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
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def setup_font():
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)
    for kf in ['NanumGothic','NanumBarunGothic','Malgun Gothic','AppleGothic']:
        if kf in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = kf; break
    plt.rcParams['axes.unicode_minus'] = False
setup_font()

st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-header{font-size:.95rem;color:#64748b}
div[data-testid="stMetricValue"]{font-size:1.8rem}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 파이프라인
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(emp_bytes, survey_bytes, training_bytes):
    progress = st.progress(0, text="📂 데이터 로딩 중...")
    emp = pd.read_csv(io.BytesIO(emp_bytes), encoding='utf-8-sig')
    survey = pd.read_csv(io.BytesIO(survey_bytes))
    training = pd.read_csv(io.BytesIO(training_bytes))
    progress.progress(10, text="✅ 데이터 로드 완료")

    drop_cols = ['FirstName','LastName','ADEmail','Supervisor']
    emp = emp.drop(columns=[c for c in drop_cols if c in emp.columns])
    for col in ['DepartmentType','Division','Title','EmployeeStatus','EmployeeType','TerminationType']:
        if col in emp.columns: emp[col] = emp[col].astype(str).str.strip()
    emp['Attrition'] = emp['EmployeeStatus'].apply(lambda x: 1 if x in ['Voluntarily Terminated','Terminated for Cause'] else 0)
    emp['StartDate_parsed'] = pd.to_datetime(emp['StartDate'], errors='coerce', dayfirst=True)
    emp['ExitDate_parsed'] = pd.to_datetime(emp['ExitDate'], errors='coerce', dayfirst=True)
    emp['DOB_parsed'] = pd.to_datetime(emp['DOB'], errors='coerce', dayfirst=True)
    ref = emp['ExitDate_parsed'].max()
    if pd.isna(ref): ref = pd.Timestamp.now()
    emp['Tenure_Years'] = emp.apply(lambda r: (r['ExitDate_parsed']-r['StartDate_parsed']).days/365.25 if pd.notna(r['ExitDate_parsed']) else (ref-r['StartDate_parsed']).days/365.25, axis=1).round(1)
    emp['Age'] = ((ref - emp['DOB_parsed']).dt.days / 365.25).round(0)
    emp['Exit_Year'] = emp['ExitDate_parsed'].dt.year
    progress.progress(25, text="✅ 전처리 완료")

    merged = emp.merge(survey, left_on='EmpID', right_on='Employee ID', how='left')
    merged = merged.merge(training, left_on='EmpID', right_on='Employee ID', how='left')
    merged = merged.drop(columns=['Employee ID_x','Employee ID_y'], errors='ignore')
    progress.progress(35, text="✅ 데이터 통합 완료")

    # ─── 칼럼 추가 시 encode_map, feature_cols, feature_labels 에 칼럼 추가하고 데이터 전처리가 필요하다면 바로 위에서 처리 ───
    encode_map = {'DepartmentType':'Dept_enc','Title':'Title_enc','GenderCode':'Gender_enc','EmployeeType':'EmpType_enc','Performance Score':'Perf_enc'}
    for col, enc in encode_map.items():
        merged[enc] = LabelEncoder().fit_transform(merged[col].astype(str))
    feature_cols = ['Dept_enc','Title_enc','Gender_enc','EmpType_enc','Perf_enc','Current Employee Rating','Engagement Score','Satisfaction Score','Work-Life Balance Score','Training Duration(Days)','Training Cost','LocationCode']
    feature_labels = ['Department','Title','Gender','Employee Type','Performance','Employee Rating','Engagement','Satisfaction','Work-Life Balance','Training Duration','Training Cost','Location']
    X = merged[feature_cols].fillna(0); y = merged['Attrition']
    progress.progress(45, text="🔄 SMOTE 처리 중...")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    progress.progress(55, text="🧠 Random Forest 학습 중...")

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test); y_proba = rf.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Active','Terminated'], output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    fi_df = pd.DataFrame({'feature':feature_labels,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    progress.progress(65, text="📊 SHAP 분석 중...")

    explainer = shap.TreeExplainer(rf)
    X_shap = X_test[:300]; X_shap_df = pd.DataFrame(X_shap, columns=feature_labels)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list): shap_c1 = shap_vals[1]
    elif shap_vals.ndim == 3: shap_c1 = shap_vals[:,:,1]
    else: shap_c1 = shap_vals

    # ─── 원본 데이터 기반 SHAP (조직별 필터링용) ───
    X_orig = merged[feature_cols].fillna(0)
    sample_size = min(800, len(X_orig))
    sample_idx = np.random.RandomState(42).choice(len(X_orig), sample_size, replace=False)
    X_orig_sample = X_orig.iloc[sample_idx]
    X_orig_sample_df = pd.DataFrame(X_orig_sample.values, columns=feature_labels)
    orig_depts = merged['DepartmentType'].iloc[sample_idx].values
    orig_shap = explainer.shap_values(X_orig_sample.values)
    if isinstance(orig_shap, list): orig_shap_c1 = orig_shap[1]
    elif orig_shap.ndim == 3: orig_shap_c1 = orig_shap[:,:,1]
    else: orig_shap_c1 = orig_shap

    progress.progress(75, text="🏢 조직별 분석 중...")
    
    # ─── 조직별 Feature Importance (퇴직자 20명 이상 조직만) ───
    dept_fi = {}
    for dept in emp['DepartmentType'].unique():
        dept_mask = merged['DepartmentType'] == dept
        X_dept = merged.loc[dept_mask, feature_cols].fillna(0)
        y_dept = merged.loc[dept_mask, 'Attrition']
        if y_dept.sum() >= 15 and len(y_dept) >= 50:
            try:
                sm = SMOTE(random_state=42, k_neighbors=min(5, int(y_dept.sum())-1))
                Xd_res, yd_res = sm.fit_resample(X_dept, y_dept)
                rf_dept = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
                rf_dept.fit(Xd_res, yd_res)
                dept_fi[dept] = pd.DataFrame({'feature':feature_labels,'importance':rf_dept.feature_importances_}).sort_values('importance',ascending=False)
            except:
                pass

    # ─── 조직별 서베이 비교 ───
    dept_survey = {}
    for dept in emp['DepartmentType'].unique():
        sub = merged[merged['DepartmentType']==dept]
        if sub['Attrition'].sum() >= 5:
            sv = sub.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
            if 0 in sv.index and 1 in sv.index:
                dept_survey[dept] = sv

    progress.progress(85, text="📋 직원 스코어링 중...")

    X_all = merged[feature_cols].fillna(0)
    merged['Risk_Score'] = (rf.predict_proba(X_all)[:,1] * 100).round(1)
    merged['Risk_Level'] = merged['Risk_Score'].apply(lambda x: '🔴 High' if x>=60 else '🟡 Medium' if x>=30 else '🟢 Low')

    dept_stats = emp.groupby('DepartmentType').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    dept_stats['rate'] = (dept_stats['terminated']/dept_stats['total']*100).round(1)
    title_stats = emp.groupby('Title').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    title_stats['rate'] = (title_stats['terminated']/title_stats['total']*100).round(1)
    cross_stats = emp.groupby(['DepartmentType','Title']).agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index()
    cross_stats['rate'] = (cross_stats['terminated']/cross_stats['total']*100).round(1)
    exited = emp[emp['ExitDate_parsed'].notna()]
    ye = exited.groupby('Exit_Year')['EmpID'].count().reset_index(); ye.columns=['year','exits']
    yd = []
    for _,r in ye.iterrows():
        yr=int(r['year']); act=len(emp[(emp['StartDate_parsed'].dt.year<=yr)&((emp['ExitDate_parsed'].isna())|(emp['ExitDate_parsed'].dt.year>=yr))])
        yd.append({'year':yr,'exits':int(r['exits']),'active':act,'rate':round(r['exits']/max(act,1)*100,1)})
    yearly_df = pd.DataFrame(yd)
    survey_compare = merged.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
    progress.progress(100, text="✅ 분석 완료!")

    return {'emp':emp,'merged':merged,'exited':exited,'rf':rf,'fi_df':fi_df,'cm':cm,'report':report,'auc':auc,
            'shap_c1':shap_c1,'X_shap_df':X_shap_df,'dept_stats':dept_stats,'title_stats':title_stats,
            'cross_stats':cross_stats,'yearly_df':yearly_df,'survey_compare':survey_compare,
            'feature_labels':feature_labels,'dept_fi':dept_fi,'dept_survey':dept_survey,
            'orig_shap_c1':orig_shap_c1,'X_orig_sample_df':X_orig_sample_df,'orig_depts':orig_depts}
# ═══════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════
def generate_pdf(R, sel_dept):
    from fpdf import FPDF
    import tempfile, os

    emp = R['emp']
    total = len(emp); termed = int(emp['Attrition'].sum()); rate = round(termed/total*100,1)

    # 차트를 이미지로 저장하는 헬퍼
    def save_chart(fig, name):
        path = os.path.join(tempfile.gettempdir(), f'{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return path

    # ─── 차트 생성 ───
    # 1) 조직별 이탈률
    dd = R['dept_stats'].sort_values('rate', ascending=True)
    fig1, ax = plt.subplots(figsize=(8,4))
    colors = [('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
    ax.barh(dd['DepartmentType'], dd['rate'], color=colors, height=0.6)
    for i,(r,t) in enumerate(zip(dd['rate'],dd['total'])): ax.text(r+0.3,i,f'{r}% ({t})',va='center',fontsize=9)
    ax.set_xlabel('Attrition Rate (%)'); ax.set_title('Department Attrition Rate', fontweight='bold')
    plt.tight_layout()
    chart1 = save_chart(fig1, 'dept_rate')

    # 2) Feature Importance
    fi_src = R['dept_fi'].get(sel_dept, R['fi_df']) if sel_dept!='All' else R['fi_df']
    fi = fi_src.sort_values('importance', ascending=True)
    fig2, ax = plt.subplots(figsize=(8,5))
    ax.barh(fi['feature'], fi['importance'], color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi))), height=0.6)
    ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
    plt.tight_layout()
    chart2 = save_chart(fig2, 'fi')

    # 3) 연도별 추이
    fig3, ax = plt.subplots(figsize=(8,4))
    ax.bar(R['yearly_df']['year'], R['yearly_df']['exits'], color='#ef4444', alpha=0.7)
    ax.plot(R['yearly_df']['year'], R['yearly_df']['exits'], 'o-', color='#991b1b', lw=2)
    for x,y in zip(R['yearly_df']['year'], R['yearly_df']['exits']): ax.text(x,y+8,str(y),ha='center',fontweight='bold')
    ax.set_title('Yearly Exit Count', fontweight='bold'); ax.set_xlabel('Year'); ax.set_ylabel('Exits')
    plt.tight_layout()
    chart3 = save_chart(fig3, 'yearly')

    # 4) 위험도 분포
    m = R['merged']
    if sel_dept!='All': m = m[m['DepartmentType']==sel_dept]
    fig4, ax = plt.subplots(figsize=(8,4))
    ax.hist(m['Risk_Score'], bins=20, color='#3b82f6', alpha=0.7, edgecolor='white')
    ax.axvline(60, color='#ef4444', ls='--', lw=2, label='High(60)')
    ax.axvline(30, color='#f59e0b', ls='--', lw=2, label='Mid(30)')
    ax.set_title('Risk Score Distribution', fontweight='bold'); ax.legend()
    plt.tight_layout()
    chart4 = save_chart(fig4, 'risk_dist')

    # ─── PDF 생성 ───
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica','B',10)
            self.set_text_color(100,100,100)
            self.cell(0,8,'HR Attrition Analysis Report | Confidential',align='R',new_x="LMARGIN",new_y="NEXT")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        def footer(self):
            self.set_y(-15); self.set_font('Helvetica','I',8); self.set_text_color(150,150,150)
            self.cell(0,10,f'Page {self.page_no()} | Generated by HR Attrition Analytics',align='C')
        def section_title(self, txt):
            self.set_font('Helvetica','B',13); self.set_text_color(30,30,30)
            self.cell(0,10,txt,new_x="LMARGIN",new_y="NEXT"); self.ln(2)
        def sub_title(self, txt):
            self.set_font('Helvetica','B',10); self.set_text_color(60,60,60)
            self.cell(0,7,txt,new_x="LMARGIN",new_y="NEXT"); self.ln(1)
        def body_text(self, txt):
            self.set_font('Helvetica','',9); self.set_text_color(50,50,50)
            self.multi_cell(0,5,txt); self.ln(2)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ─── 표지 ───
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica','B',28); pdf.set_text_color(30,30,30)
    pdf.cell(0,15,'HR Attrition',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,15,'Analysis Report',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(10)
    pdf.set_font('Helvetica','',12); pdf.set_text_color(100,100,100)
    scope = sel_dept if sel_dept!='All' else 'All Departments'
    pdf.cell(0,8,f'Scope: {scope}',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.cell(0,8,f'Total Employees: {total:,} | Attrition Rate: {rate}%',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(20)
    pdf.set_font('Helvetica','',10)

    # ─── Executive Summary ───
    pdf.add_page()
    pdf.section_title('1. Executive Summary')
    pdf.body_text(f'This report analyzes employee attrition patterns across the organization based on {total:,} employee records. '
                  f'The overall attrition rate is {rate}%, with {termed} employees having left the organization. '
                  f'The analysis employs Random Forest machine learning model (ROC-AUC: {R["auc"]:.4f}) and SHAP explainability framework '
                  f'to identify key attrition drivers and generate targeted retention strategies.')

    # KPI 박스
    pdf.sub_title('Key Metrics')
    pdf.set_font('Helvetica','B',9)
    pdf.cell(47,7,'Total Employees',border=1,align='C'); pdf.cell(47,7,'Terminated',border=1,align='C')
    pdf.cell(47,7,'Attrition Rate',border=1,align='C'); pdf.cell(47,7,'ROC-AUC Score',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',9)
    pdf.cell(47,7,f'{total:,}',border=1,align='C'); pdf.cell(47,7,f'{termed}',border=1,align='C')
    pdf.cell(47,7,f'{rate}%',border=1,align='C'); pdf.cell(47,7,f'{R["auc"]:.4f}',border=1,align='C'); pdf.ln()
    pdf.ln(5)

    # 위험 조직
    pdf.sub_title('High-Risk Departments')
    high_risk = R['dept_stats'][R['dept_stats']['rate']>=10].sort_values('rate',ascending=False)
    if len(high_risk)>0:
        for _,r in high_risk.iterrows():
            level = 'CRITICAL' if r['rate']>=15 else 'WARNING'
            pdf.body_text(f"  [{level}] {r['DepartmentType']}: {r['rate']}% attrition ({int(r['terminated'])}/{r['total']} employees)")
    else:
        pdf.body_text('  No departments above 10% attrition threshold.')

    # ─── 조직별 분석 ───
    pdf.add_page()
    pdf.section_title('2. Department Attrition Analysis')
    pdf.image(chart1, x=10, w=190)
    pdf.ln(5)

    pdf.set_font('Helvetica','B',9)
    pdf.cell(55,7,'Department',border=1); pdf.cell(25,7,'Total',border=1,align='C')
    pdf.cell(25,7,'Terminated',border=1,align='C'); pdf.cell(25,7,'Rate (%)',border=1,align='C')
    pdf.cell(30,7,'Risk Level',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',9)
    for _,r in R['dept_stats'].sort_values('rate',ascending=False).iterrows():
        level = 'HIGH' if r['rate']>=15 else 'MEDIUM' if r['rate']>=5 else 'LOW'
        pdf.cell(55,6,str(r['DepartmentType']),border=1)
        pdf.cell(25,6,str(r['total']),border=1,align='C')
        pdf.cell(25,6,str(int(r['terminated'])),border=1,align='C')
        pdf.cell(25,6,str(r['rate']),border=1,align='C')
        pdf.cell(30,6,level,border=1,align='C'); pdf.ln()
    pdf.ln(5)

    # ─── Feature Importance ───
    pdf.add_page()
    pdf.section_title('3. Key Attrition Drivers')
    pdf.body_text('The following features were identified as the most significant predictors of employee attrition, '
                  'ranked by Random Forest feature importance score.')
    pdf.image(chart2, x=10, w=190)
    pdf.ln(5)

    pdf.sub_title('Top 5 Features')
    pdf.set_font('Helvetica','B',9)
    pdf.cell(10,7,'#',border=1,align='C'); pdf.cell(60,7,'Feature',border=1); pdf.cell(35,7,'Importance',border=1,align='C')
    pdf.cell(80,7,'Interpretation',border=1); pdf.ln()
    pdf.set_font('Helvetica','',8)
    interpretations = {
        'Training Cost': 'Training investment level correlates with retention',
        'Location': 'Geographic factors influence turnover decisions',
        'Title': 'Certain job roles have inherently higher turnover',
        'Engagement': 'Lower engagement scores predict higher attrition',
        'Satisfaction': 'Job satisfaction directly impacts retention',
        'Work-Life Balance': 'Poor work-life balance drives departures',
        'Employee Rating': 'Performance ratings affect career decisions',
        'Training Duration': 'Training program length impacts development',
        'Department': 'Organizational culture varies by department',
        'Performance': 'Performance assessment affects retention',
        'Employee Type': 'Employment type affects stability',
        'Gender': 'Gender-based patterns in attrition'
    }
    for i,(_,r) in enumerate(fi_src.head(5).iterrows()):
        pdf.cell(10,6,str(i+1),border=1,align='C')
        pdf.cell(60,6,str(r['feature']),border=1)
        pdf.cell(35,6,f"{r['importance']:.4f}",border=1,align='C')
        interp = interpretations.get(r['feature'], 'Contributing factor to attrition')
        pdf.cell(80,6,interp[:40],border=1); pdf.ln()

    # ─── 연도별 추이 ───
    pdf.add_page()
    pdf.section_title('4. Yearly Attrition Trend')
    pdf.image(chart3, x=10, w=190)
    pdf.ln(5)
    pdf.set_font('Helvetica','B',9)
    pdf.cell(30,7,'Year',border=1,align='C'); pdf.cell(30,7,'Exits',border=1,align='C')
    pdf.cell(35,7,'Active Est.',border=1,align='C'); pdf.cell(30,7,'Rate (%)',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',9)
    for _,r in R['yearly_df'].iterrows():
        pdf.cell(30,6,str(int(r['year'])),border=1,align='C'); pdf.cell(30,6,str(r['exits']),border=1,align='C')
        pdf.cell(35,6,str(r['active']),border=1,align='C'); pdf.cell(30,6,str(r['rate']),border=1,align='C'); pdf.ln()

    # ─── 위험 직원 ───
    pdf.add_page()
    pdf.section_title('5. Employee Risk Scoring')
    pdf.body_text('Each employee was assigned a risk score (0-100) based on the Random Forest model prediction probability. '
                  'Employees scoring 60+ are classified as High Risk and require immediate attention.')
    pdf.image(chart4, x=10, w=190)
    pdf.ln(5)

    m_scope = R['merged'] if sel_dept=='All' else R['merged'][R['merged']['DepartmentType']==sel_dept]
    high = len(m_scope[m_scope['Risk_Score']>=60])
    mid = len(m_scope[(m_scope['Risk_Score']>=30)&(m_scope['Risk_Score']<60)])
    low = len(m_scope[m_scope['Risk_Score']<30])
    pdf.sub_title('Risk Distribution')
    pdf.body_text(f'  High Risk (60+): {high} employees | Medium (30-59): {mid} employees | Low (0-29): {low} employees')

    pdf.sub_title('Top 20 High-Risk Employees')
    pdf.set_font('Helvetica','B',8)
    pdf.cell(15,6,'ID',border=1); pdf.cell(45,6,'Department',border=1); pdf.cell(45,6,'Title',border=1)
    pdf.cell(20,6,'Risk%',border=1,align='C'); pdf.cell(20,6,'Tenure',border=1,align='C')
    pdf.cell(25,6,'Performance',border=1,align='C'); pdf.cell(15,6,'Rating',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',7)
    for _,r in m_scope.nlargest(20,'Risk_Score').iterrows():
        pdf.cell(15,5,str(r['EmpID']),border=1)
        pdf.cell(45,5,str(r['DepartmentType'])[:22],border=1)
        pdf.cell(45,5,str(r['Title'])[:22],border=1)
        pdf.cell(20,5,str(r['Risk_Score']),border=1,align='C')
        tenure = str(r.get('Tenure_Years','N/A'))
        pdf.cell(20,5,tenure,border=1,align='C')
        perf = str(r.get('Performance Score','N/A'))[:12]
        pdf.cell(25,5,perf,border=1,align='C')
        rating = str(r.get('Current Employee Rating','N/A'))
        pdf.cell(15,5,rating,border=1,align='C'); pdf.ln()

    # ─── 서베이 ───
    pdf.add_page()
    pdf.section_title('6. Survey Analysis')
    sc = R['survey_compare']; sc.index = ['Active','Terminated']
    pdf.set_font('Helvetica','B',9)
    pdf.cell(55,7,'Metric',border=1); pdf.cell(30,7,'Active',border=1,align='C')
    pdf.cell(30,7,'Terminated',border=1,align='C'); pdf.cell(30,7,'Difference',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',9)
    for col in sc.columns:
        diff = sc.loc['Terminated',col] - sc.loc['Active',col]
        sign = '+' if diff>0 else ''
        short = col.replace(' Score','')
        pdf.cell(55,6,short,border=1); pdf.cell(30,6,str(sc.loc['Active',col]),border=1,align='C')
        pdf.cell(30,6,str(sc.loc['Terminated',col]),border=1,align='C')
        pdf.cell(30,6,f'{sign}{diff:.2f}',border=1,align='C'); pdf.ln()

    # ─── 권고사항 ───
    pdf.add_page()
    pdf.section_title('7. Recommendations & Action Plan')
    pdf.body_text('Based on the comprehensive analysis above, the following strategic recommendations are proposed:')

    for _,r in R['dept_stats'].sort_values('rate',ascending=False).iterrows():
        if sel_dept!='All' and r['DepartmentType']!=sel_dept: continue
        rt = r['rate']
        pdf.sub_title(f"{r['DepartmentType']} (Attrition: {rt}%)")
        if rt >= 15:
            pdf.body_text(
                f"[CRITICAL] Immediate retention intervention required.\n"
                f"  1. Deploy emergency retention packages for high-risk employees\n"
                f"  2. Conduct 1:1 stay interviews within 30 days\n"
                f"  3. Review compensation competitiveness vs market\n"
                f"  4. Establish career development roadmap for key talent\n"
                f"  5. Implement flexible work arrangements\n"
                f"  Target: Reduce attrition from {rt}% to {max(rt-5,5):.0f}% within 12 months")
        elif rt >= 5:
            pdf.body_text(
                f"[WARNING] Targeted intervention recommended.\n"
                f"  1. Identify and address high-turnover job roles\n"
                f"  2. Strengthen mentoring and internal mobility programs\n"
                f"  3. Enhance regular engagement surveys and follow-up\n"
                f"  Target: Maintain attrition below {max(rt-2,3):.0f}% within 12 months")
        else:
            pdf.body_text(
                f"[STABLE] Continue current retention practices.\n"
                f"  1. Maintain existing HR policies and monitoring\n"
                f"  2. Analyze retention success factors for cross-department benchmarking")

    # ─── Disclaimer ───
    pdf.ln(10)
    pdf.set_font('Helvetica','I',8); pdf.set_text_color(130,130,130)
    pdf.multi_cell(0,4,'Disclaimer: This report is generated by an AI-powered analytics system and is intended as a decision-support tool. '
                       'Final decisions should incorporate human judgment, organizational context, and qualitative factors not captured in the data. '
                       'All employee data is handled in accordance with data privacy regulations.')

    # 임시 파일 정리
    for f in [chart1,chart2,chart3,chart4]:
        try: os.remove(f)
        except: pass

    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════
# ChatGPT
# ═══════════════════════════════════════════════════════════════
def get_ai_plan(api_key, dept, rate, total, fi_text, ctx):
    try:
        from openai import OpenAI; client = OpenAI(api_key=api_key)
        prompt = f"""당신은 글로벌 HR 컨설팅 펌의 시니어 HR 전략 컨설턴트입니다.
반드시 아래 데이터 수치를 근거로만 답변하세요.

[분석 결과] {ctx}
[대상] {dept}: {total}명, 이탈률 {rate}%
[Feature Importance] {fi_text}

## 📊 {dept} 이탈 분석 리포트
### 1. 현황 진단 (전사 평균 비교, 영향도 정량화)
### 2. 핵심 이탈 원인 3가지 ([근거: 수치] 포함)
### 3. 단기 액션 (0~3개월) - 표: 우선순위|시책|대상|방법|KPI
### 4. 중장기 액션 (3~12개월) - 표 형태
### 5. 기대 효과 (이탈률 목표, 비용 절감)
### ⚠️ 본 분석은 의사결정 참고자료이며 최종 판단은 HR 담당자 검토 필요"""
        resp = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"system","content":"15년 경력 글로벌 HR 전략 컨설턴트. 한국어 답변."},
                      {"role":"user","content":prompt}], max_tokens=2000, temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e: return f"API Error: {e}"

# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">🏢 HR Attrition Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화</p>', unsafe_allow_html=True)
    st.divider()
    if 'results' not in st.session_state: st.session_state.results = None

    with st.sidebar:
        st.header("📂 데이터 업로드")
        st.caption("CSV 파일을 드래그하거나 Browse files로 업로드")
        emp_f = st.file_uploader("① 직원 인사정보", type=['csv'], help="employee_data.csv")
        survey_f = st.file_uploader("② 직원 설문조사", type=['csv'], help="engagement_survey.csv")
        training_f = st.file_uploader("③ 직원 교육정보", type=['csv'], help="training_data.csv")
        if emp_f and survey_f and training_f:
            st.success("✅ 3개 파일 업로드 완료")
            if st.button("🧠 AI 이탈 분석 시작", type="primary", use_container_width=True):
                st.session_state.results = run_pipeline(emp_f.getvalue(), survey_f.getvalue(), training_f.getvalue())
                st.rerun()
        else: st.info(f"📎 {sum([1 for f in [emp_f,survey_f,training_f] if f])}/3")
        if st.session_state.results:
            st.divider(); st.header("🔍 필터"); R=st.session_state.results
            depts=['전체']+sorted(R['dept_stats']['DepartmentType'].tolist())
            sel_dept=st.selectbox("Department", depts, key='sel_dept')
            if sel_dept=='전체': tl=['전체']+sorted(R['title_stats']['Title'].tolist())
            else: tl=['전체']+sorted(R['cross_stats'][R['cross_stats']['DepartmentType']==sel_dept]['Title'].unique().tolist())
            sel_title=st.selectbox("Job Title", tl, key='sel_title')
            st.divider(); st.header("⚙️ AI 설정")
            api_key=st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key='api_key', value=st.secrets.get("OPENAI_API_KEY",""))

    if st.session_state.results is None:
        st.markdown("### 👋 시작하기")
        st.markdown("왼쪽 사이드바에서 **3개의 CSV 파일**을 업로드한 후 **AI 이탈 분석 시작** 버튼을 클릭하세요.")
        c1,c2,c3=st.columns(3)
        c1.info("📁 **employee_data.csv**\n\n직원 인사정보 (26개 변수)")
        c2.info("📁 **engagement_survey.csv**\n\n직원 설문조사 (5개 변수)")
        c3.info("📁 **training_data.csv**\n\n교육 훈련 정보 (9개 변수)")
        st.markdown("---")
        st.markdown("**Pipeline:** 전처리 → EDA → RF + SMOTE → SHAP → 조직별 분석 → 스코어링 → 액션 플랜")
        return

    R=st.session_state.results; emp=R['emp']
    sel_dept=st.session_state.get('sel_dept','전체'); sel_title=st.session_state.get('sel_title','전체')
    api_key=st.session_state.get('api_key','')
    filt=emp.copy()
    if sel_dept!='전체': filt=filt[filt['DepartmentType']==sel_dept]
    if sel_title!='전체': filt=filt[filt['Title']==sel_title]
    kt=len(filt); kterm=int(filt['Attrition'].sum()); kr=round(kterm/max(kt,1)*100,1)

    k1,k2,k3,k4=st.columns(4)
    k1.metric("전체 인원",f"{kt:,}명"); k2.metric("퇴직자 수",f"{kterm}명")
    k3.metric("이탈률",f"{kr}%",delta="위험" if kr>=15 else "주의" if kr>=10 else "양호",delta_color="inverse" if kr>=10 else "normal")
    k4.metric("재직자",f"{kt-kterm:,}명")
    st.divider()

    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 이탈 분석","🧠 주요 원인 분석","📈 모델 성능","👤 직원 스코어링","🎯 HR 액션 플랜","📥 보고서"])

    # ═══ TAB 1: 이탈 분석 (필터 연동) ═══
    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.subheader("📉 조직별 이탈률")
            dd=R['dept_stats'].sort_values('rate',ascending=True)
            fig,ax=plt.subplots(figsize=(8,4))
            colors=[('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
            ax.barh(dd['DepartmentType'],dd['rate'],color=colors,height=0.6)
            for i,(rate,total) in enumerate(zip(dd['rate'],dd['total'])): ax.text(rate+0.3,i,f'{rate}% ({total})',va='center',fontsize=9)
            ax.set_xlabel('Attrition Rate (%)'); ax.set_title('Department Attrition Rate',fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.subheader("📊 직무별 이탈률 Top 10")
            td=R['title_stats'].copy()
            if sel_dept!='전체': td=R['cross_stats'][R['cross_stats']['DepartmentType']==sel_dept].copy()
            top10=td.nlargest(10,'rate').sort_values('rate',ascending=True)
            fig,ax=plt.subplots(figsize=(8,4))
            colors=[('#ef4444' if r>=20 else '#f59e0b' if r>=10 else '#22c55e') for r in top10['rate']]
            ax.barh(top10['Title'],top10['rate'],color=colors,height=0.6)
            for i,(rate,total) in enumerate(zip(top10['rate'],top10['total'])): ax.text(rate+0.3,i,f'{rate}% (n={total})',va='center',fontsize=9)
            ax.set_xlabel('Attrition Rate (%)'); ax.set_title('Title Attrition Rate Top 10',fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.subheader("📅 연도별 이탈 추이")
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(8,4))
            ax.bar(R['yearly_df']['year'],R['yearly_df']['exits'],color='#ef4444',alpha=0.7)
            ax.plot(R['yearly_df']['year'],R['yearly_df']['exits'],'o-',color='#991b1b',lw=2)
            for x,y in zip(R['yearly_df']['year'],R['yearly_df']['exits']): ax.text(x,y+8,str(y),ha='center',fontweight='bold')
            ax.set_title('Yearly Exit Count',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(8,4))
            ax.plot(R['yearly_df']['year'],R['yearly_df']['rate'],'o-',color='#ef4444',lw=2,markersize=8,markerfacecolor='white',markeredgewidth=2)
            ax.fill_between(R['yearly_df']['year'],R['yearly_df']['rate'],alpha=0.1,color='#ef4444')
            for x,y in zip(R['yearly_df']['year'],R['yearly_df']['rate']): ax.text(x,y+0.5,f'{y}%',ha='center',fontweight='bold')
            ax.set_title('Yearly Attrition Rate',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ═══ TAB 2: 주요 원인 분석 (조직별 연동!) ═══
    with tab2:
        # 조직 선택에 따라 다른 분석 표시
        if sel_dept != '전체':
            st.info(f"📌 **{sel_dept}** 조직 분석 결과입니다.")
        else:
            st.info("📌 **전체** 데이터 기반 분석 결과입니다. 사이드바에서 조직을 선택하면 해당 조직 분석으로 전환됩니다.")

        c1,c2=st.columns(2)
        with c1:
            # Feature Importance: 조직별 or 전체
            if sel_dept != '전체' and sel_dept in R['dept_fi']:
                st.subheader(f"🧠 Feature Importance ({sel_dept})")
                fi = R['dept_fi'][sel_dept].sort_values('importance',ascending=True)
            else:
                st.subheader("🧠 Feature Importance (전체)")
                fi = R['fi_df'].sort_values('importance',ascending=True)
                if sel_dept != '전체' and sel_dept not in R['dept_fi']:
                    st.caption(f"⚠️ {sel_dept} 조직은 퇴직자 수가 적어 개별 모델 학습이 어렵습니다. 전체 결과를 표시합니다.")

            fig,ax=plt.subplots(figsize=(8,5))
            ax.barh(fi['feature'],fi['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi))),height=0.6)
            ax.set_title('Feature Importance (Random Forest)',fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            # 서베이 비교: 조직별 or 전체
            if sel_dept != '전체' and sel_dept in R['dept_survey']:
                st.subheader(f"🎯 Survey Comparison ({sel_dept})")
                sc = R['dept_survey'][sel_dept]
                sc.index = ['Active','Terminated']
            else:
                st.subheader("🎯 Survey Comparison (전체)")
                sc = R['survey_compare']
                sc.index = ['Active','Terminated']

            fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3); w=0.35
            ax.bar(x-w/2,sc.iloc[0],w,label='Active',color='#3b82f6')
            ax.bar(x+w/2,sc.iloc[1],w,label='Terminated',color='#ef4444')
            ax.set_xticks(x); ax.set_xticklabels(['Engagement','Satisfaction','Work-Life\nBalance'])
            ax.set_ylabel('Score'); ax.legend(); ax.set_ylim(2.0,4.0)
            ax.set_title('Active vs Terminated Survey',fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # 차이 하이라이트
            diff = sc.iloc[1] - sc.iloc[0]
            st.dataframe(pd.DataFrame({
                'Item': ['Engagement','Satisfaction','Work-Life Balance'],
                'Active': sc.iloc[0].values,
                'Terminated': sc.iloc[1].values,
                'Diff': diff.values
            }), use_container_width=True, hide_index=True)

      # SHAP (조직별 필터링 가능)
        if sel_dept != '전체':
            st.subheader(f"📊 SHAP Summary Plot ({sel_dept})")
            dept_mask = R['orig_depts'] == sel_dept
            if dept_mask.sum() >= 10:
                shap_filtered = R['orig_shap_c1'][dept_mask]
                X_filtered = R['X_orig_sample_df'][dept_mask]
                fig,ax=plt.subplots(figsize=(12,6))
                shap.summary_plot(shap_filtered, X_filtered, plot_type='dot', show=False)
                plt.title(f'SHAP Summary - {sel_dept}', fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                st.caption(f"⚠️ {sel_dept} 샘플이 10건 미만이어서 전체 SHAP을 표시합니다.")
                fig,ax=plt.subplots(figsize=(12,6))
                shap.summary_plot(R['shap_c1'],R['X_shap_df'],plot_type='dot',show=False)
                plt.title('SHAP Summary Plot (All)',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.subheader("📊 SHAP Summary Plot (전체)")
            plt.figure(figsize=(12,6))
            shap.summary_plot(R['orig_shap_c1'],R['X_orig_sample_df'],plot_type='dot',show=False)
            st.pyplot(plt.gcf()); plt.close('all')

        # 교차 이탈률: 조직별 필터링
        if sel_dept != '전체':
            st.subheader(f"🔍 {sel_dept} 직무별 이탈률")
            cs = R['cross_stats'][(R['cross_stats']['DepartmentType']==sel_dept)&(R['cross_stats']['total']>=5)].sort_values('rate',ascending=False)
        else:
            st.subheader("🔍 조직 × 직무 교차 이탈률 (Top 15)")
            cs = R['cross_stats'][R['cross_stats']['total']>=10].nlargest(15,'rate')
        st.dataframe(cs[['DepartmentType','Title','total','terminated','rate']].rename(
            columns={'DepartmentType':'Dept','Title':'Title','total':'Total','terminated':'Term','rate':'Rate(%)'}),
            use_container_width=True,hide_index=True)

    # ═══ TAB 3: 모델 성능 ═══
    with tab3:
        c1,c2=st.columns(2)
        with c1:
            st.subheader("📋 Confusion Matrix")
            fig,ax=plt.subplots(figsize=(6,5))
            sns.heatmap(R['cm'],annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['Active','Term'],yticklabels=['Active','Term'])
            ax.set_title('Confusion Matrix',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.subheader("📊 Performance")
            st.metric("ROC-AUC",f"{R['auc']:.4f}")
            st.dataframe(pd.DataFrame(R['report']).T.round(3),use_container_width=True)

    # ═══ TAB 4: 직원 스코어링 ═══
    with tab4:
        st.subheader("👤 Employee Risk Scoring")
        m=R['merged'].copy()
        if sel_dept!='전체': m=m[m['DepartmentType']==sel_dept]
        if sel_title!='전체': m=m[m['Title']==sel_title]
        c1,c2,c3=st.columns(3)
        h=len(m[m['Risk_Score']>=60]); mid=len(m[(m['Risk_Score']>=30)&(m['Risk_Score']<60)]); lo=len(m[m['Risk_Score']<30])
        c1.metric("🔴 High (60+)",f"{h}"); c2.metric("🟡 Medium (30-59)",f"{mid}"); c3.metric("🟢 Low (0-29)",f"{lo}")
        fig,axes=plt.subplots(1,2,figsize=(14,4))
        axes[0].hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
        axes[0].axvline(60,color='#ef4444',ls='--',lw=2,label='High(60)'); axes[0].axvline(30,color='#f59e0b',ls='--',lw=2,label='Mid(30)')
        axes[0].set_title('Risk Score Distribution',fontweight='bold'); axes[0].legend()
        dr=m.groupby('DepartmentType')['Risk_Score'].mean().sort_values(ascending=True)
        axes[1].barh(dr.index,dr.values,color='#8b5cf6',height=0.6)
        for i,v in enumerate(dr.values): axes[1].text(v+0.5,i,f'{v:.1f}',va='center')
        axes[1].set_title('Avg Risk by Dept',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.subheader("🔴 High Risk Employees")
        cols=['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating']
        avail=[c for c in cols if c in m.columns]
        hr=m[m['Risk_Score']>=60].nlargest(50,'Risk_Score')
        if len(hr)>0: st.dataframe(hr[avail],use_container_width=True,hide_index=True)
        else: st.success("✅ No high-risk employees.")
        st.subheader("📋 All Employees")
        search=st.text_input("🔍 Search")
        full=m[avail].sort_values('Risk_Score',ascending=False)
        if search: full=full[full.astype(str).apply(lambda x: x.str.contains(search,case=False)).any(axis=1)]
        st.dataframe(full,use_container_width=True,hide_index=True,height=400)

    # ═══ TAB 5: 액션 플랜 ═══
    with tab5:
        st.subheader("🎯 HR Action Plan")
        # 선택된 조직의 FI 사용
        if sel_dept!='전체' and sel_dept in R['dept_fi']:
            fi_src = R['dept_fi'][sel_dept]
        else:
            fi_src = R['fi_df']
        fi_text=', '.join([f"{r['feature']}({r['importance']:.3f})" for _,r in fi_src.head(5).iterrows()])
        ctx=f"Total: {len(emp)}, Term: {emp['Attrition'].sum()}, Rate: {emp['Attrition'].mean()*100:.1f}%, Top Features: {fi_text}"
        dl=R['dept_stats'].sort_values('rate',ascending=False)
        if sel_dept!='전체': dl=dl[dl['DepartmentType']==sel_dept]
        for _,r in dl.iterrows():
            rate=r['rate']
            with st.expander(f"📋 {r['DepartmentType']} — {rate}%",expanded=(rate>=10)):
                mc1,mc2,mc3=st.columns(3); mc1.metric("Total",f"{r['total']}"); mc2.metric("Term",f"{int(r['terminated'])}"); mc3.metric("Rate",f"{rate}%")
                if api_key:
                    if st.button(f"🧠 AI Plan - {r['DepartmentType']}",key=f"ai_{r['DepartmentType']}"):
                        with st.spinner("Generating..."): st.markdown(get_ai_plan(api_key,r['DepartmentType'],rate,r['total'],fi_text,ctx))
                else:
                    if rate>=15: st.markdown("🔴**[Urgent]** Retention package 🔴**[Urgent]** 1:1 interviews 🟡**[High]** Career path 🟡**[High]** Work-life balance")
                    elif rate>=5: st.markdown("🟡**[High]** Target high-turnover roles 🔵**[Mid]** Mentoring 🔵**[Mid]** Survey")
                    else: st.markdown("🟢**[Maintain]** Current policy 🔵**[Mid]** Benchmark")
                if not api_key: st.caption("💡 OpenAI API Key 입력 시 AI 맞춤 액션 플랜 생성")

    # ═══ TAB 6: 보고서 ═══
    with tab6:
        st.subheader("📥 Reports")
        c1,c2=st.columns(2)
        with c1:
            rd=st.selectbox("Target",['All']+sorted(R['dept_stats']['DepartmentType'].tolist()),key="rpt")
            if st.button("📄 PDF",type="primary",use_container_width=True):
                with st.spinner("..."): pdf=generate_pdf(R,rd); st.download_button("⬇️ Download",pdf,f"Report_{rd}.pdf","application/pdf",use_container_width=True)
        with c2:
            csv1=R['merged'][['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating','Attrition']].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Risk CSV",csv1,"risk_scores.csv","text/csv",use_container_width=True)
            csv2=R['dept_stats'].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Dept CSV",csv2,"dept_stats.csv","text/csv",use_container_width=True)

if __name__=="__main__": main()
