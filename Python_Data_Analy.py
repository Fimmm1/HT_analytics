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

# ─── 폰트 ───
@st.cache_resource
def setup_font():
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)
    for kf in ['NanumGothic','NanumBarunGothic','Malgun Gothic','AppleGothic']:
        if kf in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = kf; break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
setup_font()

st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-header{font-size:.95rem;color:#64748b}
div[data-testid="stMetricValue"]{font-size:1.8rem}
.insight-box{background:#f0f4ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0 20px 0;font-size:14px;color:#1e293b}
.warning-box{background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0 20px 0;font-size:14px;color:#1e293b}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 파이프라인
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(eb, sb, tb):
    progress = st.progress(0, text="📂 데이터 로딩 중...")
    emp = pd.read_csv(io.BytesIO(eb), encoding='utf-8-sig')
    survey = pd.read_csv(io.BytesIO(sb))
    training = pd.read_csv(io.BytesIO(tb))
    progress.progress(10, text="✅ 데이터 로드 완료")
    for c in ['DepartmentType','Division','Title','EmployeeStatus','EmployeeType','TerminationType']:
        if c in emp.columns: emp[c]=emp[c].astype(str).str.strip()
    emp.drop(columns=[c for c in ['FirstName','LastName','ADEmail','Supervisor'] if c in emp.columns],inplace=True)
    emp['Attrition']=emp['EmployeeStatus'].apply(lambda x:1 if x in ['Voluntarily Terminated','Terminated for Cause'] else 0)
    emp['StartDate_parsed']=pd.to_datetime(emp['StartDate'],errors='coerce',dayfirst=True)
    emp['ExitDate_parsed']=pd.to_datetime(emp['ExitDate'],errors='coerce',dayfirst=True)
    emp['DOB_parsed']=pd.to_datetime(emp['DOB'],errors='coerce',dayfirst=True)
    ref=emp['ExitDate_parsed'].max(); 
    if pd.isna(ref): ref=pd.Timestamp.now()
    emp['Tenure_Years']=emp.apply(lambda r:(r['ExitDate_parsed']-r['StartDate_parsed']).days/365.25 if pd.notna(r['ExitDate_parsed']) else (ref-r['StartDate_parsed']).days/365.25,axis=1).round(1)
    emp['Age']=((ref-emp['DOB_parsed']).dt.days/365.25).round(0)
    emp['Exit_Year']=emp['ExitDate_parsed'].dt.year
    progress.progress(25,text="✅ 전처리 완료")
    merged=emp.merge(survey,left_on='EmpID',right_on='Employee ID',how='left').merge(training,left_on='EmpID',right_on='Employee ID',how='left')
    merged.drop(columns=['Employee ID_x','Employee ID_y'],inplace=True,errors='ignore')
    progress.progress(35,text="✅ 데이터 통합 완료")
    for c,e in {'DepartmentType':'Dept_enc','Title':'Title_enc','GenderCode':'Gender_enc','EmployeeType':'EmpType_enc','Performance Score':'Perf_enc'}.items():
        merged[e]=LabelEncoder().fit_transform(merged[c].astype(str))
    fc=['Dept_enc','Title_enc','Gender_enc','EmpType_enc','Perf_enc','Current Employee Rating','Engagement Score','Satisfaction Score','Work-Life Balance Score','Training Duration(Days)','Training Cost','LocationCode']
    fl=['Department','Title','Gender','Employee Type','Performance','Employee Rating','Engagement','Satisfaction','Work-Life Balance','Training Duration','Training Cost','Location']
    X=merged[fc].fillna(0);y=merged['Attrition']
    progress.progress(45,text="🔄 SMOTE 처리 중...")
    Xr,yr=SMOTE(random_state=42).fit_resample(X,y)
    Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=0.2,random_state=42,stratify=yr)
    progress.progress(55,text="🧠 Random Forest 학습 중...")
    rf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=5,random_state=42,n_jobs=-1)
    rf.fit(Xtr,ytr)
    yp=rf.predict(Xte);ypr=rf.predict_proba(Xte)[:,1]
    cm=confusion_matrix(yte,yp);rpt=classification_report(yte,yp,target_names=['Active','Terminated'],output_dict=True);auc=roc_auc_score(yte,ypr)
    fi=pd.DataFrame({'feature':fl,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    progress.progress(65,text="📊 SHAP 분석 중...")
    exp=shap.TreeExplainer(rf)
    Xs=Xte[:300];Xsd=pd.DataFrame(Xs,columns=fl)
    sv=exp.shap_values(Xs)
    if isinstance(sv,list):sc1=sv[1]
    elif sv.ndim==3:sc1=sv[:,:,1]
    else:sc1=sv
    ss=min(800,len(merged));si=np.random.RandomState(42).choice(len(merged),ss,replace=False)
    Xos=merged[fc].fillna(0).iloc[si];Xosd=pd.DataFrame(Xos.values,columns=fl);odepts=merged['DepartmentType'].iloc[si].values
    osv=exp.shap_values(Xos.values)
    if isinstance(osv,list):osc1=osv[1]
    elif osv.ndim==3:osc1=osv[:,:,1]
    else:osc1=osv
    progress.progress(75,text="🏢 조직별 분석 중...")
    dfi={}
    for d in emp['DepartmentType'].unique():
        dm=merged['DepartmentType']==d;Xd=merged.loc[dm,fc].fillna(0);yd=merged.loc[dm,'Attrition']
        if yd.sum()>=15 and len(yd)>=50:
            try:
                sm=SMOTE(random_state=42,k_neighbors=min(5,int(yd.sum())-1));Xdr,ydr=sm.fit_resample(Xd,yd)
                rfd=RandomForestClassifier(n_estimators=80,max_depth=8,random_state=42,n_jobs=-1);rfd.fit(Xdr,ydr)
                dfi[d]=pd.DataFrame({'feature':fl,'importance':rfd.feature_importances_}).sort_values('importance',ascending=False)
            except:pass
    dsv={}
    for d in emp['DepartmentType'].unique():
        sub=merged[merged['DepartmentType']==d]
        if sub['Attrition'].sum()>=5:
            sv2=sub.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
            if 0 in sv2.index and 1 in sv2.index:dsv[d]=sv2
    progress.progress(85,text="📋 직원 스코어링 중...")
    Xa=merged[fc].fillna(0);merged['Risk_Score']=(rf.predict_proba(Xa)[:,1]*100).round(1)
    merged['Risk_Level']=merged['Risk_Score'].apply(lambda x:'🔴 High' if x>=60 else '🟡 Medium' if x>=30 else '🟢 Low')
    ds=emp.groupby('DepartmentType').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();ds['rate']=(ds['terminated']/ds['total']*100).round(1)
    ts=emp.groupby('Title').agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();ts['rate']=(ts['terminated']/ts['total']*100).round(1)
    cs=emp.groupby(['DepartmentType','Title']).agg(total=('EmpID','count'),terminated=('Attrition','sum')).reset_index();cs['rate']=(cs['terminated']/cs['total']*100).round(1)
    exited=emp[(emp['Attrition']==1)&(emp['ExitDate_parsed'].notna())]
    ye=exited.groupby('Exit_Year')['EmpID'].count().reset_index();ye.columns=['year','exits']
    yd2=[]
    for _,r in ye.iterrows():
        yr=int(r['year']);act=len(emp[(emp['StartDate_parsed'].dt.year<=yr)&((emp['ExitDate_parsed'].isna())|(emp['ExitDate_parsed'].dt.year>=yr))])
        yd2.append({'year':yr,'exits':int(r['exits']),'active':act,'rate':round(r['exits']/max(act,1)*100,1)})
    ydf=pd.DataFrame(yd2)
    scomp=merged.groupby('Attrition')[['Engagement Score','Satisfaction Score','Work-Life Balance Score']].mean().round(2)
    progress.progress(100,text="✅ 분석 완료!")
    return {'emp':emp,'merged':merged,'exited':exited,'rf':rf,'fi':fi,'cm':cm,'rpt':rpt,'auc':auc,
            'sc1':sc1,'Xsd':Xsd,'osc1':osc1,'Xosd':Xosd,'odepts':odepts,
            'ds':ds,'ts':ts,'cs':cs,'ydf':ydf,'scomp':scomp,'fl':fl,'dfi':dfi,'dsv':dsv}

# ═══════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════
def gen_pdf(R,sd):
    from fpdf import FPDF;import tempfile,os
    emp=R['emp'];total=len(emp);termed=int(emp['Attrition'].sum());rate=round(termed/total*100,1)
    def svc(fig,n):
        p=os.path.join(tempfile.gettempdir(),f'{n}.png');fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);return p
    dd=R['ds'].sort_values('rate',ascending=True)
    f1,ax=plt.subplots(figsize=(10,5));colors=[('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
    ax.barh(dd['DepartmentType'],dd['rate'],color=colors,height=0.6)
    for i,(r,t) in enumerate(zip(dd['rate'],dd['total'])):ax.text(r+0.3,i,f'{r}% ({t})',va='center',fontsize=9)
    ax.set_xlabel('Attrition Rate (%)');ax.set_title('Department Attrition Rate',fontweight='bold');plt.tight_layout();c1=svc(f1,'d')
    fis=R['dfi'].get(sd,R['fi']) if sd!='All' else R['fi'];fiv=fis.sort_values('importance',ascending=True)
    f2,ax=plt.subplots(figsize=(10,6));ax.barh(fiv['feature'],fiv['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fiv))),height=0.6)
    ax.set_title('Feature Importance',fontweight='bold');plt.tight_layout();c2=svc(f2,'f')
    f3,ax=plt.subplots(figsize=(10,5));ax.bar(R['ydf']['year'],R['ydf']['exits'],color='#ef4444',alpha=0.7);ax.plot(R['ydf']['year'],R['ydf']['exits'],'o-',color='#991b1b',lw=2)
    for x,y in zip(R['ydf']['year'],R['ydf']['exits']):ax.text(x,y+3,str(y),ha='center',fontweight='bold')
    ax.set_title('Yearly Exits',fontweight='bold');plt.tight_layout();c3=svc(f3,'y')
    m=R['merged'] if sd=='All' else R['merged'][R['merged']['DepartmentType']==sd]
    f4,ax=plt.subplots(figsize=(10,5));ax.hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
    ax.axvline(60,color='#ef4444',ls='--',lw=2,label='High(60)');ax.axvline(30,color='#f59e0b',ls='--',lw=2,label='Mid(30)')
    ax.set_title('Risk Score Distribution',fontweight='bold');ax.legend();plt.tight_layout();c4=svc(f4,'r')
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica','B',10);self.set_text_color(100,100,100);self.cell(0,8,'HR Attrition Analysis Report | Confidential',align='R',new_x="LMARGIN",new_y="NEXT");self.line(10,self.get_y(),200,self.get_y());self.ln(3)
        def footer(self):
            self.set_y(-15);self.set_font('Helvetica','I',8);self.set_text_color(150,150,150);self.cell(0,10,f'Page {self.page_no()}',align='C')
        def stitle(self,t):self.set_font('Helvetica','B',13);self.set_text_color(30,30,30);self.cell(0,10,t,new_x="LMARGIN",new_y="NEXT");self.ln(2)
        def stitle2(self,t):self.set_font('Helvetica','B',10);self.set_text_color(60,60,60);self.cell(0,7,t,new_x="LMARGIN",new_y="NEXT");self.ln(1)
        def body(self,t):self.set_font('Helvetica','',9);self.set_text_color(50,50,50);self.multi_cell(0,5,t);self.ln(2)
    pdf=PDF();pdf.set_auto_page_break(auto=True,margin=20)
    pdf.add_page();pdf.ln(40);pdf.set_font('Helvetica','B',28);pdf.set_text_color(30,30,30)
    pdf.cell(0,15,'HR Attrition',align='C',new_x="LMARGIN",new_y="NEXT");pdf.cell(0,15,'Analysis Report',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(10);pdf.set_font('Helvetica','',12);pdf.set_text_color(100,100,100)
    sc=sd if sd!='All' else 'All Departments'
    pdf.cell(0,8,f'Scope: {sc}',align='C',new_x="LMARGIN",new_y="NEXT");pdf.cell(0,8,f'Total: {total:,} | Rate: {rate}%',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.ln(20);pdf.set_font('Helvetica','',10);pdf.cell(0,8,'Sogang Univ. AI/SW | Kim Hyuntae (A74032)',align='C',new_x="LMARGIN",new_y="NEXT")
    pdf.add_page();pdf.stitle('1. Executive Summary');pdf.body(f'Total {total:,} employees, {termed} terminated ({rate}%). ROC-AUC: {R["auc"]:.4f}')
    pdf.stitle2('High-Risk Departments')
    for _,r in R['ds'][R['ds']['rate']>=10].sort_values('rate',ascending=False).iterrows():
        lv='CRITICAL' if r['rate']>=15 else 'WARNING';pdf.body(f"  [{lv}] {r['DepartmentType']}: {r['rate']}% ({int(r['terminated'])}/{r['total']})")
    pdf.add_page();pdf.stitle('2. Department Analysis');pdf.image(c1,x=10,w=190);pdf.ln(5)
    pdf.set_font('Helvetica','B',9);pdf.cell(55,7,'Department',border=1);pdf.cell(25,7,'Total',border=1,align='C');pdf.cell(25,7,'Term',border=1,align='C');pdf.cell(25,7,'Rate%',border=1,align='C');pdf.cell(30,7,'Risk',border=1,align='C');pdf.ln()
    pdf.set_font('Helvetica','',9)
    for _,r in R['ds'].sort_values('rate',ascending=False).iterrows():
        lv='HIGH' if r['rate']>=15 else 'MID' if r['rate']>=5 else 'LOW'
        pdf.cell(55,6,str(r['DepartmentType']),border=1);pdf.cell(25,6,str(r['total']),border=1,align='C');pdf.cell(25,6,str(int(r['terminated'])),border=1,align='C');pdf.cell(25,6,str(r['rate']),border=1,align='C');pdf.cell(30,6,lv,border=1,align='C');pdf.ln()
    pdf.add_page();pdf.stitle('3. Key Attrition Drivers');pdf.image(c2,x=10,w=190);pdf.ln(5)
    pdf.add_page();pdf.stitle('4. Yearly Trend');pdf.image(c3,x=10,w=190)
    pdf.add_page();pdf.stitle('5. Risk Scoring');pdf.image(c4,x=10,w=190);pdf.ln(5)
    pdf.stitle2('Top 20 High-Risk');pdf.set_font('Helvetica','B',8)
    pdf.cell(15,6,'ID',border=1);pdf.cell(45,6,'Department',border=1);pdf.cell(45,6,'Title',border=1);pdf.cell(20,6,'Risk%',border=1,align='C');pdf.cell(20,6,'Tenure',border=1,align='C');pdf.cell(25,6,'Perf',border=1,align='C');pdf.ln()
    pdf.set_font('Helvetica','',7)
    for _,r in m.nlargest(20,'Risk_Score').iterrows():
        pdf.cell(15,5,str(r['EmpID']),border=1);pdf.cell(45,5,str(r['DepartmentType'])[:22],border=1);pdf.cell(45,5,str(r['Title'])[:22],border=1)
        pdf.cell(20,5,str(r['Risk_Score']),border=1,align='C');pdf.cell(20,5,str(r.get('Tenure_Years','N/A')),border=1,align='C');pdf.cell(25,5,str(r.get('Performance Score','N/A'))[:12],border=1,align='C');pdf.ln()
    pdf.add_page();pdf.stitle('6. Recommendations')
    for _,r in R['ds'].sort_values('rate',ascending=False).iterrows():
        if sd!='All' and r['DepartmentType']!=sd:continue
        rt=r['rate'];pdf.stitle2(f"{r['DepartmentType']} ({rt}%)")
        if rt>=15:pdf.body(f"[CRITICAL] 1.Retention packages 2.Stay interviews 3.Compensation review 4.Career roadmap 5.Flexible work\nTarget: {rt}% -> {max(rt-5,5):.0f}%")
        elif rt>=5:pdf.body(f"[WARNING] 1.Target high-turnover roles 2.Mentoring 3.Engagement surveys\nTarget: below {max(rt-2,3):.0f}%")
        else:pdf.body("[STABLE] Maintain current policies, benchmark success factors")
    pdf.ln(10);pdf.set_font('Helvetica','I',8);pdf.set_text_color(130,130,130);pdf.multi_cell(0,4,'Disclaimer: AI-powered decision-support tool. Final decisions require HR professional review.')
    for f in [c1,c2,c3,c4]:
        try:import os;os.remove(f)
        except:pass
    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════
# ChatGPT
# ═══════════════════════════════════════════════════════════════
def get_ai_plan(key,dept,rate,total,fi_text,ctx):
    try:
        from openai import OpenAI;client=OpenAI(api_key=key)
        prompt=f"""당신은 글로벌 HR 컨설팅 펌의 시니어 HR 전략 컨설턴트입니다.
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
        resp=client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"system","content":"15년 경력 글로벌 HR 전략 컨설턴트. 한국어 답변."},{"role":"user","content":prompt}],max_tokens=2000,temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e:return f"API Error: {e}"

# ═══════════════════════════════════════════════════════════════
# 차트 + 설명 헬퍼
# ═══════════════════════════════════════════════════════════════
def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>',unsafe_allow_html=True)
def warn_insight(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">🏢 HR Attrition Analytics</h1>',unsafe_allow_html=True)
    st.markdown('<p class="sub-header">조직·직무별 이탈 패턴 분석 및 HR 액션 플랜 자동화 · AI·SW대학원 김현태 / A74032</p>',unsafe_allow_html=True)
    st.divider()
    if 'results' not in st.session_state:st.session_state.results=None
    with st.sidebar:
        st.header("📂 데이터 업로드")
        st.caption("CSV 파일을 드래그하거나 Browse files로 업로드")
        ef=st.file_uploader("① 직원 인사정보",type=['csv'],help="employee_data.csv")
        sf=st.file_uploader("② 직원 설문조사",type=['csv'],help="engagement_survey.csv")
        tf=st.file_uploader("③ 직원 교육정보",type=['csv'],help="training_data.csv")
        if ef and sf and tf:
            st.success("✅ 3개 파일 업로드 완료")
            if st.button("🧠 AI 이탈 분석 시작",type="primary",use_container_width=True):
                st.session_state.results=run_pipeline(ef.getvalue(),sf.getvalue(),tf.getvalue());st.rerun()
        else:st.info(f"📎 {sum([1 for f in[ef,sf,tf]if f])}/3")
        if st.session_state.results:
            st.divider();st.header("🔍 필터");R=st.session_state.results
            sd=st.selectbox("Department",['전체']+sorted(R['ds']['DepartmentType'].tolist()),key='sel_dept')
            if sd=='전체':tl=['전체']+sorted(R['ts']['Title'].tolist())
            else:tl=['전체']+sorted(R['cs'][R['cs']['DepartmentType']==sd]['Title'].unique().tolist())
            st.selectbox("Job Title",tl,key='sel_title')
            st.divider();st.header("⚙️ AI 설정")
            st.text_input("OpenAI API Key",type="password",placeholder="sk-...",key='api_key',value=st.secrets.get("OPENAI_API_KEY","") if hasattr(st,'secrets') else "")

    if st.session_state.results is None:
        st.markdown("### 👋 시작하기")
        st.markdown("왼쪽 사이드바에서 **3개의 CSV 파일**을 업로드한 후 **AI 이탈 분석 시작** 버튼을 클릭하세요.")
        c1,c2,c3=st.columns(3)
        c1.info("📁 **employee_data.csv**\n\n직원 인사정보 (26개 변수)")
        c2.info("📁 **engagement_survey.csv**\n\n직원 설문조사 (5개 변수)")
        c3.info("📁 **training_data.csv**\n\n교육 훈련 정보 (9개 변수)")
        return

    R=st.session_state.results;emp=R['emp']
    sd=st.session_state.get('sel_dept','전체');stl=st.session_state.get('sel_title','전체')
    ak=st.session_state.get('api_key','')
    filt=emp.copy()
    if sd!='전체':filt=filt[filt['DepartmentType']==sd]
    if stl!='전체':filt=filt[filt['Title']==stl]
    kt=len(filt);kterm=int(filt['Attrition'].sum());kr=round(kterm/max(kt,1)*100,1)

    k1,k2,k3,k4=st.columns(4)
    k1.metric("전체 인원",f"{kt:,}명");k2.metric("퇴직자 수",f"{kterm}명")
    k3.metric("이탈률",f"{kr}%",delta="위험" if kr>=15 else "주의" if kr>=10 else "양호",delta_color="inverse" if kr>=10 else "normal")
    k4.metric("재직자",f"{kt-kterm:,}명")
    st.divider()

    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 이탈 분석","🧠 주요 원인 분석","📈 모델 성능","👤 직원 스코어링","🎯 HR 액션 플랜","📥 보고서"])

    # ═══ TAB 1 ═══
    with tab1:
        # 조직별 이탈률
        st.subheader("📉 조직별 이탈률")
        dd=R['ds'].sort_values('rate',ascending=True)
        fig,ax=plt.subplots(figsize=(14,6))
        colors=[('#ef4444' if r>=15 else '#f59e0b' if r>=5 else '#22c55e') for r in dd['rate']]
        ax.barh(dd['DepartmentType'],dd['rate'],color=colors,height=0.5)
        for i,(r,t) in enumerate(zip(dd['rate'],dd['total'])):ax.text(r+0.3,i,f'{r}%  ({t}명)',va='center',fontsize=11,fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)',fontsize=12);ax.set_title('Department Attrition Rate',fontsize=16,fontweight='bold');ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        # 고위험/저위험 조직 분석
        high_depts=R['ds'][R['ds']['rate']>=15]['DepartmentType'].tolist()
        low_depts=R['ds'][R['ds']['rate']<5]['DepartmentType'].tolist()
        if high_depts:
            warn_insight(f"**고위험 조직:** {', '.join(high_depts)} — 전사 평균({R['ds']['rate'].mean():.1f}%)을 크게 상회하며, 즉시 리텐션 대응이 필요합니다.")
        if low_depts:
            insight(f"**안정 조직:** {', '.join(low_depts)} — 이탈률이 5% 미만으로 양호합니다. 해당 조직의 리텐션 성공 요인을 분석하여 고위험 조직에 벤치마킹할 수 있습니다.")

        st.markdown("---")

        # 직무별 이탈률
        st.subheader("📊 직무별 이탈률 Top 10")
        td=R['ts'].copy()
        if sd!='전체':td=R['cs'][R['cs']['DepartmentType']==sd].copy()
        top10=td.nlargest(10,'rate').sort_values('rate',ascending=True)
        fig,ax=plt.subplots(figsize=(14,7))
        colors=[('#ef4444' if r>=20 else '#f59e0b' if r>=10 else '#22c55e') for r in top10['rate']]
        ax.barh(top10['Title'],top10['rate'],color=colors,height=0.5)
        for i,(r,t) in enumerate(zip(top10['rate'],top10['total'])):ax.text(r+0.3,i,f'{r}%  (n={t})',va='center',fontsize=11,fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)',fontsize=12);ax.set_title('Title Attrition Rate Top 10',fontsize=16,fontweight='bold');ax.tick_params(labelsize=10)
        plt.tight_layout();st.pyplot(fig);plt.close()
        top1=td.nlargest(1,'rate').iloc[0] if len(td)>0 else None
        if top1 is not None and top1['rate']>=15:
            warn_insight(f"**최고 위험 직무:** {top1['Title']} ({top1['rate']}%, {int(top1['total'])}명 중 {int(top1['terminated'])}명 퇴직) — 해당 직무의 퇴직 사유와 보상 경쟁력을 집중 점검해야 합니다.")

        st.markdown("---")

        # 연도별 퇴직 건수
        st.subheader("📅 연도별 퇴직 건수")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.bar(R['ydf']['year'],R['ydf']['exits'],color='#ef4444',alpha=0.7,width=0.6)
        ax.plot(R['ydf']['year'],R['ydf']['exits'],'o-',color='#991b1b',lw=2.5,markersize=8)
        for x,y in zip(R['ydf']['year'],R['ydf']['exits']):ax.text(x,y+5,str(y),ha='center',fontweight='bold',fontsize=12)
        ax.set_title('Yearly Exit Count',fontsize=16,fontweight='bold');ax.set_xlabel('Year',fontsize=12);ax.set_ylabel('Exits',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        if len(R['ydf'])>=2:
            first=R['ydf'].iloc[0];last=R['ydf'].iloc[-1]
            insight(f"**연도별 추이:** {int(first['year'])}년 {first['exits']}건 → {int(last['year'])}년 {last['exits']}건으로 퇴직 건수가 변화하고 있습니다. 추세가 증가하고 있다면 조직 내부 변화(구조조정, 정책 변경 등)와의 연관성을 확인해야 합니다.")

        st.markdown("---")

        # 연도별 이탈률 추이
        st.subheader("📅 연도별 이탈률 추이")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.plot(R['ydf']['year'],R['ydf']['rate'],'o-',color='#ef4444',lw=2.5,markersize=10,markerfacecolor='white',markeredgewidth=2.5)
        ax.fill_between(R['ydf']['year'],R['ydf']['rate'],alpha=0.1,color='#ef4444')
        for x,y in zip(R['ydf']['year'],R['ydf']['rate']):ax.text(x,y+0.5,f'{y}%',ha='center',fontweight='bold',fontsize=12)
        ax.set_title('Yearly Attrition Rate',fontsize=16,fontweight='bold');ax.set_xlabel('Year',fontsize=12);ax.set_ylabel('Rate (%)',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        insight(f"**이탈률 추이:** 연도별 이탈률의 증감 추세를 모니터링하여, 급등 시점에 어떤 조직 변화가 있었는지 역추적하는 것이 중요합니다.")

    # ═══ TAB 2 ═══
    with tab2:
        if sd!='전체':st.info(f"📌 **{sd}** 조직 분석 결과입니다.")
        else:st.info("📌 **전체** 데이터 기반 분석 결과입니다. 사이드바에서 조직을 선택하면 해당 조직 분석으로 전환됩니다.")

        # Feature Importance
        if sd!='전체' and sd in R['dfi']:
            st.subheader(f"🧠 Feature Importance — {sd}")
            fi_data=R['dfi'][sd].sort_values('importance',ascending=True)
        else:
            st.subheader("🧠 Feature Importance — 전체")
            fi_data=R['fi'].sort_values('importance',ascending=True)
            if sd!='전체' and sd not in R['dfi']:
                st.caption(f"⚠️ {sd} 조직은 퇴직자 수가 적어 개별 모델 학습이 어렵습니다. 전체 결과를 표시합니다.")
        fig,ax=plt.subplots(figsize=(14,7))
        ax.barh(fi_data['feature'],fi_data['importance'],color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi_data))),height=0.5)
        for i,(f,v) in enumerate(zip(fi_data['feature'],fi_data['importance'])):ax.text(v+0.002,i,f'{v:.3f}',va='center',fontsize=11,fontweight='bold')
        ax.set_title('Feature Importance (Random Forest)',fontsize=16,fontweight='bold');ax.set_xlabel('Importance',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        top3=fi_data.nlargest(3,'importance')
        insight(f"**이탈 예측 핵심 변수 Top 3:** {', '.join([f'{r.feature}({r.importance:.1%})' for _,r in top3.iterrows()])} — 이 변수들이 퇴직 여부를 예측할 때 가장 큰 영향력을 가집니다. 단, 이 수치는 '얼마나 중요한가'만 보여주며, '어떤 방향으로 영향을 미치는가'는 아래 SHAP에서 확인할 수 있습니다.")

        st.markdown("---")

        # 서베이 비교
        if sd!='전체' and sd in R['dsv']:
            st.subheader(f"🎯 재직자 vs 퇴직자 서베이 — {sd}")
            sc=R['dsv'][sd];sc.index=['Active','Terminated']
        else:
            st.subheader("🎯 재직자 vs 퇴직자 서베이 — 전체")
            sc=R['scomp'];sc.index=['Active','Terminated']
        fig,ax=plt.subplots(figsize=(14,6));x=np.arange(3);w=0.3
        bars1=ax.bar(x-w/2,sc.iloc[0],w,label='Active (재직자)',color='#3b82f6')
        bars2=ax.bar(x+w/2,sc.iloc[1],w,label='Terminated (퇴직자)',color='#ef4444')
        ax.set_xticks(x);ax.set_xticklabels(['Engagement\nScore','Satisfaction\nScore','Work-Life\nBalance Score'],fontsize=12)
        ax.set_ylabel('Average Score',fontsize=12);ax.legend(fontsize=12);ax.set_ylim(2.0,4.0)
        for b in bars1:ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{b.get_height():.2f}',ha='center',fontsize=11,fontweight='bold',color='#3b82f6')
        for b in bars2:ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f'{b.get_height():.2f}',ha='center',fontsize=11,fontweight='bold',color='#ef4444')
        ax.set_title('Active vs Terminated Survey Comparison',fontsize=16,fontweight='bold');ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        diff=sc.iloc[1]-sc.iloc[0]
        cols=['Engagement Score','Satisfaction Score','Work-Life Balance Score']
        neg=[c.replace(' Score','') for c,d in zip(cols,diff) if d<0]
        if neg:
            warn_insight(f"**퇴직자 서베이 경고:** 퇴직자는 재직자 대비 {', '.join(neg)} 점수가 낮습니다. 이 항목들의 하락이 이탈 선행 지표로 작동하므로, 정기 서베이에서 해당 점수가 떨어지는 직원/조직을 조기에 감지하는 체계가 필요합니다.")

        st.markdown("---")

        # SHAP
        if sd!='전체':
            st.subheader(f"📊 SHAP Summary Plot — {sd}")
            dm=R['odepts']==sd
            if dm.sum()>=10:
                plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'][dm],R['Xosd'][dm],plot_type='dot',show=False)
                st.pyplot(plt.gcf());plt.close('all')
            else:
                st.caption(f"⚠️ {sd} 샘플 10건 미만, 전체 SHAP 표시")
                plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'],R['Xosd'],plot_type='dot',show=False)
                st.pyplot(plt.gcf());plt.close('all')
        else:
            st.subheader("📊 SHAP Summary Plot — 전체")
            plt.figure(figsize=(14,7));shap.summary_plot(R['osc1'],R['Xosd'],plot_type='dot',show=False)
            st.pyplot(plt.gcf());plt.close('all')
        insight("**SHAP 해석:** 각 점은 하나의 직원입니다. 빨간 점(높은 값)이 오른쪽에 몰려있으면 '해당 변수 값이 높을수록 퇴직 확률 증가', 파란 점(낮은 값)이 오른쪽이면 '값이 낮을수록 퇴직 확률 증가'를 의미합니다. Feature Importance가 '무엇이 중요한가'라면, SHAP은 '어떻게 영향을 미치는가'를 보여줍니다.")

        st.markdown("---")

        # 교차 이탈률
        if sd!='전체':
            st.subheader(f"🔍 {sd} 직무별 이탈률")
            csd=R['cs'][(R['cs']['DepartmentType']==sd)&(R['cs']['total']>=5)].sort_values('rate',ascending=False)
        else:
            st.subheader("🔍 조직 × 직무 교차 이탈률 (Top 15)")
            csd=R['cs'][R['cs']['total']>=10].nlargest(15,'rate')
        st.dataframe(csd[['DepartmentType','Title','total','terminated','rate']].rename(columns={'DepartmentType':'Dept','Title':'Title','total':'Total','terminated':'Term','rate':'Rate(%)'}),use_container_width=True,hide_index=True,height=400)
        insight("**교차 이탈률:** 조직과 직무를 동시에 고려하여 가장 위험한 그룹을 식별합니다. 인원이 많으면서 이탈률도 높은 그룹이 비즈니스 영향이 가장 크므로 우선 대응 대상입니다.")

    # ═══ TAB 3 ═══
    with tab3:
        st.subheader("📋 Confusion Matrix")
        fig,ax=plt.subplots(figsize=(10,8))
        sns.heatmap(R['cm'],annot=True,fmt='d',cmap='Blues',ax=ax,annot_kws={'size':18},xticklabels=['Active Predicted','Terminated Predicted'],yticklabels=['Actual Active','Actual Terminated'])
        ax.set_title('Confusion Matrix',fontsize=16,fontweight='bold');ax.tick_params(labelsize=12)
        plt.tight_layout();st.pyplot(fig);plt.close()
        tn,fp,fn,tp=R['cm'].ravel()
        insight(f"**해석:** 실제 퇴직자 중 모델이 잡아낸 비율(Recall) = {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%. 실제 퇴직자를 놓치는 건수(False Negative) = {fn}건. HR 관점에서 이 수치가 낮을수록 위험 직원을 놓칠 확률이 줄어듭니다.")

        st.markdown("---")

        st.subheader("📊 성능 지표")
        st.metric("ROC-AUC Score",f"{R['auc']:.4f}")
        st.dataframe(pd.DataFrame(R['rpt']).T.round(3),use_container_width=True)
        insight(f"**ROC-AUC {R['auc']:.4f}:** 0.5가 무작위, 1.0이 완벽 예측입니다. 현재 수준은 HR 이탈 예측에서 우수한 성능이며, 조직 수준의 패턴 분석과 위험 그룹 식별에 신뢰할 수 있는 수준입니다.")

    # ═══ TAB 4 ═══
    with tab4:
        st.subheader("👤 Employee Risk Scoring")
        st.caption("Random Forest 모델이 각 직원의 이탈 확률을 0~100점으로 산출합니다.")
        m=R['merged'].copy()
        if sd!='전체':m=m[m['DepartmentType']==sd]
        if stl!='전체':m=m[m['Title']==stl]
        c1,c2,c3=st.columns(3)
        h=len(m[m['Risk_Score']>=60]);mid=len(m[(m['Risk_Score']>=30)&(m['Risk_Score']<60)]);lo=len(m[m['Risk_Score']<30])
        c1.metric("🔴 High Risk (60+)",f"{h}명");c2.metric("🟡 Medium (30-59)",f"{mid}명");c3.metric("🟢 Low (0-29)",f"{lo}명")

        st.markdown("---")

        st.subheader("📊 위험 점수 분포")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.hist(m['Risk_Score'],bins=20,color='#3b82f6',alpha=0.7,edgecolor='white')
        ax.axvline(60,color='#ef4444',ls='--',lw=2.5,label='High Risk (60)');ax.axvline(30,color='#f59e0b',ls='--',lw=2.5,label='Medium Risk (30)')
        ax.set_title('Risk Score Distribution',fontsize=16,fontweight='bold');ax.set_xlabel('Risk Score',fontsize=12);ax.set_ylabel('Count',fontsize=12);ax.legend(fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()
        insight(f"**위험 분포:** 고위험(60+) {h}명, 중위험(30-59) {mid}명, 저위험(0-29) {lo}명. 고위험 직원은 즉시 1:1 면담과 리텐션 조치가 필요하며, 중위험 직원은 정기 모니터링 대상입니다.")

        st.markdown("---")

        st.subheader("🏢 조직별 평균 위험 점수")
        dr=m.groupby('DepartmentType')['Risk_Score'].mean().sort_values(ascending=True)
        fig,ax=plt.subplots(figsize=(14,5))
        ax.barh(dr.index,dr.values,color='#8b5cf6',height=0.5)
        for i,v in enumerate(dr.values):ax.text(v+0.5,i,f'{v:.1f}',va='center',fontsize=12,fontweight='bold')
        ax.set_title('Average Risk Score by Department',fontsize=16,fontweight='bold');ax.set_xlabel('Avg Risk Score',fontsize=12);ax.tick_params(labelsize=11)
        plt.tight_layout();st.pyplot(fig);plt.close()

        st.markdown("---")

        st.subheader("🔴 고위험 직원 목록 (Risk Score ≥ 60)")
        cols=['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating']
        avail=[c for c in cols if c in m.columns]
        hr=m[m['Risk_Score']>=60].nlargest(50,'Risk_Score')
        if len(hr)>0:st.dataframe(hr[avail],use_container_width=True,hide_index=True)
        else:st.success("✅ No high-risk employees.")

        st.markdown("---")

        st.subheader("📋 전체 직원 위험도 테이블")
        search=st.text_input("🔍 사번, 직무 등으로 검색")
        full=m[avail].sort_values('Risk_Score',ascending=False)
        if search:full=full[full.astype(str).apply(lambda x:x.str.contains(search,case=False)).any(axis=1)]
        st.dataframe(full,use_container_width=True,hide_index=True,height=500)

    # ═══ TAB 5 ═══
    with tab5:
        st.subheader("🎯 HR Action Plan")
        if sd!='전체' and sd in R['dfi']:fis=R['dfi'][sd]
        else:fis=R['fi']
        fi_text=', '.join([f"{r['feature']}({r['importance']:.3f})" for _,r in fis.head(5).iterrows()])
        ctx=f"Total: {len(emp)}, Term: {emp['Attrition'].sum()}, Rate: {emp['Attrition'].mean()*100:.1f}%, Top Features: {fi_text}"
        dl=R['ds'].sort_values('rate',ascending=False)
        if sd!='전체':dl=dl[dl['DepartmentType']==sd]
        for _,r in dl.iterrows():
            rate=r['rate']
            with st.expander(f"📋 {r['DepartmentType']} — {rate}%",expanded=(rate>=10)):
                mc1,mc2,mc3=st.columns(3);mc1.metric("Total",f"{r['total']}");mc2.metric("Term",f"{int(r['terminated'])}");mc3.metric("Rate",f"{rate}%")
                if ak:
                    if st.button(f"🧠 AI Plan - {r['DepartmentType']}",key=f"ai_{r['DepartmentType']}"):
                        with st.spinner("Generating..."):st.markdown(get_ai_plan(ak,r['DepartmentType'],rate,r['total'],fi_text,ctx))
                else:
                    if rate>=15:st.markdown("🔴 **[Urgent]** Retention package & 1:1 interviews\n\n🟡 **[High]** Career path transparency & Work-life balance\n\n🔵 **[Mid]** Culture improvement & communication")
                    elif rate>=5:st.markdown("🟡 **[High]** Target high-turnover roles\n\n🔵 **[Mid]** Mentoring & Regular survey")
                    else:st.markdown("🟢 **[Maintain]** Current policy\n\n🔵 **[Mid]** Benchmark best practices")
                if not ak:st.caption("💡 사이드바에서 OpenAI API Key 입력 시 AI 맞춤 액션 플랜 생성")

    # ═══ TAB 6 ═══
    with tab6:
        st.subheader("📥 보고서 다운로드")
        rd=st.selectbox("대상 조직",['All']+sorted(R['ds']['DepartmentType'].tolist()),key="rpt")
        if st.button("📄 PDF 보고서 생성",type="primary",use_container_width=True):
            with st.spinner("PDF 생성 중..."):
                pdf=gen_pdf(R,rd)
                st.download_button("⬇️ PDF 다운로드",pdf,f"Report_{rd}.pdf","application/pdf",use_container_width=True)
        st.markdown("---")
        st.subheader("📊 CSV 다운로드")
        csv1=R['merged'][['EmpID','DepartmentType','Title','Risk_Score','Risk_Level','Tenure_Years','Performance Score','Current Employee Rating','Attrition']].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ 직원 위험도 CSV",csv1,"risk_scores.csv","text/csv",use_container_width=True)
        csv2=R['ds'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ 조직별 통계 CSV",csv2,"dept_stats.csv","text/csv",use_container_width=True)

if __name__=="__main__":main()
