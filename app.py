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
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ─── 한글 폰트 설정 ───
@st.cache_resource
def setup_font():
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)
    for kf in ['NanumGothic','NanumBarunGothic','Malgun Gothic','AppleGothic']:
        if kf in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = kf
            break
    plt.rcParams['axes.unicode_minus'] = False
setup_font()

# ─── 스타일 ───
st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-header{font-size:.95rem;color:#64748b}
div[data-testid="stMetricValue"]{font-size:1.8rem}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 분석 파이프라인
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

    encode_map = {'DepartmentType':'Dept_enc','Title':'Title_enc','GenderCode':'Gender_enc','EmployeeType':'EmpType_enc','Performance Score':'Perf_enc'}
    for col, enc in encode_map.items():
        merged[enc] = LabelEncoder().fit_transform(merged[col].astype(str))
    feature_cols = ['Dept_enc','Title_enc','Gender_enc','EmpType_enc','Perf_enc','Current Employee Rating','Engagement Score','Satisfaction Score','Work-Life Balance Score','Training Duration(Days)','Training Cost','LocationCode']
    feature_labels = ['Department','Title','Gender','Employee Type','Performance','Employee Rating','Engagement','Satisfaction','Work-Life Balance','Training Duration','Training Cost','Location']
    X = merged[feature_cols].fillna(0)
    y = merged['Attrition']
    progress.progress(45, text="🔄 SMOTE 처리 중...")

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

    explainer = shap.TreeExplainer(rf)
    X_shap = X_test[:300]
    X_shap_df = pd.DataFrame(X_shap, columns=feature_labels)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list): shap_c1 = shap_vals[1]
    elif shap_vals.ndim == 3: shap_c1 = shap_vals[:,:,1]
    else: shap_c1 = shap_vals
    progress.progress(85, text="📋 직원 스코어링 중...")

    X_all = merged[feature_cols].fillna(0)
    merged['Risk_Score'] = (rf.predict_proba(X_all)[:,1] * 100).round(1)
    merged['Risk_Level'] = merged['Risk_Score'].apply(lambda x: '🔴 높음' if x>=60 else '🟡 중간' if x>=30 else '🟢 낮음')

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
            'cross_stats':cross_stats,'yearly_df':yearly_df,'survey_compare':survey_compare,'feature_labels':feature_labels}

# ═══════════════════════════════════════════════════════════════
# PDF 생성
# ═══════════════════════════════════════════════════════════════
def generate_pdf(R, sel_dept):
    from fpdf import FPDF
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica','B',14); self.cell(0,10,'HR Attrition Analysis Report',align='C',new_x="LMARGIN",new_y="NEXT")
            self.set_font('Helvetica','',9); self.cell(0,6,f'Department: {sel_dept}',align='C',new_x="LMARGIN",new_y="NEXT"); self.ln(5)
        def footer(self):
            self.set_y(-15); self.set_font('Helvetica','I',8); self.cell(0,10,f'Page {self.page_no()}',align='C')
    pdf = PDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    emp=R['emp']; total=len(emp); termed=emp['Attrition'].sum(); rate=round(termed/total*100,1)
    pdf.set_font('Helvetica','B',12); pdf.cell(0,8,'Executive Summary',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',10)
    pdf.cell(0,6,f'Total: {total:,} | Terminated: {termed} ({rate}%) | ROC-AUC: {R["auc"]:.4f}',new_x="LMARGIN",new_y="NEXT"); pdf.ln(5)
    pdf.set_font('Helvetica','B',12); pdf.cell(0,8,'Department Stats',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','B',9)
    pdf.cell(60,7,'Department',border=1); pdf.cell(25,7,'Total',border=1,align='C'); pdf.cell(25,7,'Term',border=1,align='C'); pdf.cell(25,7,'Rate%',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',9)
    dd = R['dept_stats'].sort_values('rate',ascending=False)
    if sel_dept!='All': dd=dd[dd['DepartmentType']==sel_dept]
    for _,r in dd.iterrows():
        pdf.cell(60,6,str(r['DepartmentType'])[:25],border=1); pdf.cell(25,6,str(r['total']),border=1,align='C')
        pdf.cell(25,6,str(int(r['terminated'])),border=1,align='C'); pdf.cell(25,6,str(r['rate']),border=1,align='C'); pdf.ln()
    pdf.ln(5)
    pdf.set_font('Helvetica','B',12); pdf.cell(0,8,'Top 5 Features',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','',9)
    for _,r in R['fi_df'].head(5).iterrows(): pdf.cell(0,6,f"  {r['feature']}: {r['importance']:.4f}",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(5)
    pdf.set_font('Helvetica','B',12); pdf.cell(0,8,'Top 10 High-Risk Employees',new_x="LMARGIN",new_y="NEXT")
    pdf.set_font('Helvetica','B',8)
    pdf.cell(20,6,'ID',border=1); pdf.cell(50,6,'Department',border=1); pdf.cell(50,6,'Title',border=1); pdf.cell(25,6,'Risk%',border=1,align='C'); pdf.ln()
    pdf.set_font('Helvetica','',8)
    for _,r in R['merged'].nlargest(10,'Risk_Score').iterrows():
        pdf.cell(20,5,str(r['EmpID']),border=1); pdf.cell(50,5,str(r['DepartmentType'])[:20],border=1)
        pdf.cell(50,5,str(r['Title'])[:20],border=1); pdf.cell(25,5,str(r['Risk_Score']),border=1,align='C'); pdf.ln()
    return pdf.output()

# ═══════════════════════════════════════════════════════════════
# ChatGPT 액션 플랜
# ═══════════════════════════════════════════════════════════════
def get_ai_plan(api_key, dept, rate, total, fi_text, ctx):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""당신은 글로벌 HR 컨설팅 펌(McKinsey, Mercer 수준)의 시니어 HR 전략 컨설턴트입니다.
반드시 아래 제공된 데이터 수치를 근거로만 답변하고, 데이터에 없는 내용은 추측하지 마세요.

[분석 결과] {ctx}
[대상] {dept} 조직: 인원 {total}명, 이탈률 {rate}%
[Feature Importance Top5] {fi_text}

아래 형식으로 작성하세요:
## 📊 {dept} 조직 이탈 분석 리포트
### 1. 현황 진단 (전사 평균과 비교, 비즈니스 영향도 정량화)
### 2. 핵심 이탈 원인 3가지 (각각 [근거: 수치] 포함)
### 3. 단기 액션 플랜 (0~3개월) - 표 형태 (우선순위|시책명|대상|실행방법|KPI)
### 4. 중장기 액션 플랜 (3~12개월) - 표 형태
### 5. 기대 효과 (이탈률 감소 목표, 비용 절감, 생산성 개선)
### ⚠️ 유의사항: 본 분석은 의사결정 참고자료이며, 최종 판단은 HR 담당자의 검토가 필요합니다."""
        resp = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"system","content":"당신은 15년 경력 글로벌 HR 전략 컨설턴트입니다. 한국어로 답변하세요."},
                      {"role":"user","content":prompt}], max_tokens=2000, temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e: return f"API 오류: {e}"

# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">🏢 HR Attrition Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화 · AI·SW대학원 김현태 / A74032</p>', unsafe_allow_html=True)
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
        else:
            st.info(f"📎 {sum([1 for f in [emp_f,survey_f,training_f] if f])}/3 파일 업로드됨")
        if st.session_state.results:
            st.divider(); st.header("🔍 필터")
            R=st.session_state.results
            depts=['전체']+sorted(R['dept_stats']['DepartmentType'].tolist())
            sel_dept=st.selectbox("Department", depts)
            if sel_dept=='전체': tl=['전체']+sorted(R['title_stats']['Title'].tolist())
            else: tl=['전체']+sorted(R['cross_stats'][R['cross_stats']['DepartmentType']==sel_dept]['Title'].unique().tolist())
            sel_title=st.selectbox("Job Title", tl)
            st.divider(); st.header("⚙️ AI 설정")
            api_key=st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    # ─── 분석 전 ───
    if st.session_state.results is None:
        st.markdown("### 👋 시작하기")
        st.markdown("왼쪽 사이드바에서 **3개의 CSV 파일**을 드래그하여 업로드한 후 **AI 이탈 분석 시작** 버튼을 클릭하세요.")
        c1,c2,c3=st.columns(3)
        c1.info("📁 **employee_data.csv**\n\n직원 인사정보 (26개 변수)")
        c2.info("📁 **engagement_survey.csv**\n\n직원 설문조사 (5개 변수)")
        c3.info("📁 **training_data.csv**\n\n교육 훈련 정보 (9개 변수)")
        st.markdown("---")
        st.markdown("""**분석 파이프라인:** 데이터 전처리 → EDA 시각화 → Random Forest + SMOTE → SHAP 분석 → 직원 이탈 스코어링 → HR 액션 플랜 자동 생성""")
        return

    # ═══ 대시보드 ═══
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

    with tab2:
        st.info("📌 전체 데이터 기반 분석 결과입니다.")
        c1,c2=st.columns(2)
        with c1:
            st.subheader("🧠 Feature Importance")
            fi=R['fi_df'].sort_values('importance',ascending=True)
            fig,ax=plt.subplots(figsize=(8,5))
            ax.barh(fi['feature'],fi['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi))),height=0.6)
            ax.set_title('Feature Importance (Random Forest)',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            st.subheader("🎯 Survey Comparison")
            sc=R['survey_compare']; sc.index=['Active','Terminated']
            fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3); w=0.35
            ax.bar(x-w/2,sc.iloc[0],w,label='Active',color='#3b82f6'); ax.bar(x+w/2,sc.iloc[1],w,label='Terminated',color='#ef4444')
            ax.set_xticks(x); ax.set_xticklabels(['Engagement','Satisfaction','Work-Life\nBalance'])
            ax.set_ylabel('Score'); ax.legend(); ax.set_ylim(2.5,3.5); ax.set_title('Active vs Terminated Survey',fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(sc,use_container_width=True)

        st.subheader("📊 SHAP Summary Plot")
        fig,ax=plt.subplots(figsize=(12,6))
        shap.summary_plot(R['shap_c1'],R['X_shap_df'],plot_type='dot',show=False)
        plt.title('SHAP Summary Plot',fontweight='bold'); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("🔍 Cross Attrition Rate (Top 15)")
        cs=R['cross_stats'][R['cross_stats']['total']>=10].nlargest(15,'rate')
        st.dataframe(cs[['DepartmentType','Title','total','terminated','rate']].rename(
            columns={'DepartmentType':'Dept','Title':'Title','total':'Total','terminated':'Term','rate':'Rate(%)'}),use_container_width=True,hide_index=True)

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

    with tab4:
        st.subheader("👤 Employee Attrition Risk Scoring")
        m=R['merged'].copy()
        if sel_dept!='전체': m=m[m['DepartmentType']==sel_dept]
        if sel_title!='전체': m=m[m['Title']==sel_title]
        c1,c2,c3=st.columns(3)
        h=len(m[m['Risk_Score']>=60]); mid=len(m[(m['Risk_Score']>=30)&(m['Risk_Score']<60)]); lo=len(m[m['Risk_Score']<30])
        c1.metric("🔴 High Risk (60+)",f"{h}"); c2.metric("🟡 Medium (30-59)",f"{mid}"); c3.metric("🟢 Low (0-29)",f"{lo}")

        fig,axes=plt.subplots(1,2,figsize=(14,4))
        axes[0].hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
        axes[0].axvline(60,color='#ef4444',ls='--',lw=2,label='High(60)'); axes[0].axvline(30,color='#f59e0b',ls='--',lw=2,label='Mid(30)')
        axes[0].set_title('Risk Score Distribution',fontweight='bold'); axes[0].legend()
        dr=m.groupby('DepartmentType')['Risk_Score'].mean().sort_values(ascending=True)
        axes[1].barh(dr.index,dr.values,color='#8b5cf6',height=0.6)
        for i,v in enumerate(dr.values): axes[1].text(v+0.5,i,f'{v:.1f}',va='center')
        axes[1].set_title('Avg Risk Score by Dept',fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("🔴 High Risk Employees")
        cols=['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating']
        avail=[c for c in cols if c in m.columns]
        hr=m[m['Risk_Score']>=60].nlargest(50,'Risk_Score')
        if len(hr)>0: st.dataframe(hr[avail],use_container_width=True,hide_index=True)
        else: st.success("✅ No high-risk employees in selection.")

        st.subheader("📋 All Employees")
        search=st.text_input("🔍 Search (ID, Title...)")
        full=m[avail].sort_values('Risk_Score',ascending=False)
        if search: full=full[full.astype(str).apply(lambda x: x.str.contains(search,case=False)).any(axis=1)]
        st.dataframe(full,use_container_width=True,hide_index=True,height=400)

    with tab5:
        st.subheader("🎯 HR Action Plan")
        fi_text=', '.join([f"{r['feature']}({r['importance']:.3f})" for _,r in R['fi_df'].head(5).iterrows()])
        ctx=f"Total: {len(emp)}, Terminated: {emp['Attrition'].sum()}, Rate: {emp['Attrition'].mean()*100:.1f}%, Top Features: {fi_text}"
        dl=R['dept_stats'].sort_values('rate',ascending=False)
        if sel_dept!='전체': dl=dl[dl['DepartmentType']==sel_dept]
        for _,r in dl.iterrows():
            rate=r['rate']
            with st.expander(f"📋 {r['DepartmentType']} — {rate}%",expanded=(rate>=10)):
                mc1,mc2,mc3=st.columns(3); mc1.metric("Total",f"{r['total']}"); mc2.metric("Term",f"{int(r['terminated'])}"); mc3.metric("Rate",f"{rate}%")
                if api_key:
                    if st.button(f"🧠 Generate AI Plan - {r['DepartmentType']}",key=f"ai_{r['DepartmentType']}"):
                        with st.spinner("Generating..."): st.markdown(get_ai_plan(api_key,r['DepartmentType'],rate,r['total'],fi_text,ctx))
                else:
                    if rate>=15: st.markdown("**Actions:** 🔴[Urgent] Retention package 🔴[Urgent] 1:1 interviews 🟡[High] Career path 🟡[High] Work-life balance 🔵[Mid] Culture improvement")
                    elif rate>=5: st.markdown("**Actions:** 🟡[High] Target high-turnover roles 🔵[Mid] Mentoring program 🔵[Mid] Regular survey")
                    else: st.markdown("**Actions:** 🟢[Maintain] Current policy 🔵[Mid] Benchmark best practices")
                if not api_key: st.caption("💡 Enter OpenAI API Key in sidebar for AI-generated action plans.")

    with tab6:
        st.subheader("📥 Download Reports")
        c1,c2=st.columns(2)
        with c1:
            rd=st.selectbox("Target Dept",['All']+sorted(R['dept_stats']['DepartmentType'].tolist()),key="rpt_dept")
            if st.button("📄 Generate PDF",type="primary",use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf=generate_pdf(R,rd)
                    st.download_button("⬇️ Download PDF",pdf,f"HR_Report_{rd}.pdf","application/pdf",use_container_width=True)
        with c2:
            csv1=R['merged'][['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating','Attrition']].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Risk Scores CSV",csv1,"risk_scores.csv","text/csv",use_container_width=True)
            csv2=R['dept_stats'].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Dept Stats CSV",csv2,"dept_stats.csv","text/csv",use_container_width=True)

if __name__=="__main__": main()
