import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import os

# 페이지 설정
st.set_page_config(page_title="KGC 설비 예지보전 시스템", layout="wide")

# --- 1. 데이터 로드 함수 ---
@st.cache_data
def load_data():
    # 데이터는 깃허브 업로드 제한 때문에 외부 URL에서 직접 가져옵니다.
    base_url = "https://raw.githubusercontent.com/vitidm/NASA-Turbofan-Engine-Degradation-Simulation-Data-Set/master/CMAPSSData/"
    train_url = f"{base_url}train_FD001.txt"
    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    
    try:
        df = pd.read_csv(train_url, sep='\s+', header=None).dropna(axis=1)
        df.columns = columns
        return df
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None

# --- 2. 데이터 가공 함수 ---
def process_unit_data(df, unit_id):
    unit_df = df[df['unit'] == unit_id].copy()
    max_cycle = unit_df['cycle'].max()
    current_cycle = int(unit_df['cycle'].iloc[-1])
    current_rul = int(max_cycle - current_cycle)
    # RUL을 기반으로 0~100점 사이의 건강 점수 생성
    health_score = max(5, min(100, int((current_rul / 150) * 100)))
    return unit_df, current_cycle, current_rul, health_score

# --- 3. 실시간 센서 로그 생성 ---
def get_live_logs(unit_df, current_rul):
    logs = ""
    last_s11 = unit_df['s11'].iloc[-1]
    
    if current_rul < 30:
        logs += f'''
        <div style="background: rgba(239, 68, 68, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.3); margin-bottom: 10px;">
            <strong style="color: #ef4444;">🚨 긴급: 설비 교체 주기 도달</strong><br>
            <small style="color: #94a3b8;">잔존 수명이 {current_rul}회 남았습니다. 즉시 점검 필요.</small>
        </div>
        '''
    if last_s11 > 477:
        logs += f'''
        <div style="background: rgba(245, 158, 11, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(245, 158, 11, 0.3); margin-bottom: 10px;">
            <strong style="color: #f59e0b;">⚠️ 센서 11 온도 주의</strong><br>
            <small style="color: #94a3b8;">현재 온도 {last_s11:.2f}ºC. 임계치 초과 징후.</small>
        </div>
        '''
    if not logs:
        logs = '<p style="color: #64748b; text-align: center; padding-top: 20px;">정상 운전 중</p>'
    return logs

# --- 4. 메인 실행부 ---
def main():
    # 1. 데이터 로드 (URL 방식 사용 중)
    df = load_data()
    
    if df is None:
        st.error("데이터를 불러오지 못했습니다. URL 주소나 인터넷 연결을 확인해주세요.")
        return

    # 사이드바
    st.sidebar.title("🛠️ 제어 센터")
    unit_id = st.sidebar.selectbox("모니터링할 유닛 선택", sorted(df['unit'].unique()))
    
    # 데이터 가공
    unit_df, current_cycle, current_rul, health_score = process_unit_data(df, unit_id)
    
    # --- 상단 HTML 대시보드 렌더링 ---
    # 중요: index.html 파일이 반드시 깃허브 저장소에 업로드되어 있어야 합니다.
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html_template = f.read()
            
        status_text = "교체 권고" if current_rul < 40 else "안정"
        risk_level = "HIGH" if current_rul < 40 else ("MEDIUM" if current_rul < 80 else "LOW")
        
        # HTML 템플릿의 변수들을 실제 데이터로 치환
        render_html = html_template.replace("{{SELECTED_UNIT}}", str(unit_id))
        render_html = render_html.replace("{{CURRENT_RUL}}", str(current_rul))
        render_html = render_html.replace("{{CURRENT_CYCLE}}", str(current_cycle))
        render_html = render_html.replace("{{HEALTH_SCORE}}", str(health_score))
        render_html = render_html.replace("{{RISK_LEVEL}}", risk_level)
        render_html = render_html.replace("{{REPLACE_STATUS}}", status_text)
        render_html = render_html.replace("{{SENSOR_LOGS}}", get_live_logs(unit_df, current_rul))

        # HTML 컴포넌트 출력
        components.html(render_html, height=850, scrolling=False)
    else:
        st.warning("📊 상단 대시보드 구성을 위해 'index.html' 파일을 깃허브에 업로드해 주세요.")
        st.info("파일 업로드 후 'Manage app -> Reboot'를 클릭하면 대시보드가 나타납니다.")

    # --- 하단 상세 그래프 ---
    st.markdown("### 📈 유닛 상세 트렌드 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Unit {unit_id} 센서(S11) 변화**")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(unit_df['cycle'], unit_df['s11'], color='#0ea5e9', linewidth=2)
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Sensor Value")
        st.pyplot(fig1)
    with col2:
        st.write(f"**RUL(잔존수명) 감소 추이**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(unit_df['cycle'], unit_df['cycle'].max() - unit_df['cycle'], color='#ef4444', linewidth=2)
        ax2.fill_between(unit_df['cycle'], unit_df['cycle'].max() - unit_df['cycle'], color='#ef4444', alpha=0.1)
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("Remaining Cycles")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
