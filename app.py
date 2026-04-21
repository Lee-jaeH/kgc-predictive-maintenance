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
    # 상단 타이틀 (HTML이 안 뜰 경우를 대비해 스트림릿 타이틀 유지)
    st.title("🫚 KGC 설비 예지보전 실시간 모니터링")

    df = load_data()
    if df is None: return

    # 사이드바
    st.sidebar.title("🛠️ 제어 센터")
    unit_id = st.sidebar.selectbox("모니터링할 유닛 선택", sorted(df['unit'].unique()))
    
    # 데이터 계산
    unit_df, current_cycle, current_rul, health_score = process_unit_data(df, unit_id)
    
    # --- HTML 리포트 렌더링 ---
    # 깃허브 최상단(root)에 index.html이 있어야 합니다.
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html_template = f.read()
            
        status_text = "교체 권고" if current_rul < 40 else "안정"
        risk_level = "HIGH" if current_rul < 40 else ("MEDIUM" if current_rul < 80 else "LOW")
        
        # 데이터 치환
        render_html = html_template.replace("{{SELECTED_UNIT}}", str(unit_id))
        render_html = render_html.replace("{{CURRENT_RUL}}", str(current_rul))
        render_html = render_html.replace("{{CURRENT_CYCLE}}", str(current_cycle))
        render_html = render_html.replace("{{HEALTH_SCORE}}", str(health_score))
        render_html = render_html.replace("{{RISK_LEVEL}}", risk_level)
        render_html = render_html.replace("{{REPLACE_STATUS}}", status_text)
        render_html = render_html.replace("{{SENSOR_LOGS}}", get_live_logs(unit_df, current_rul))

        # HTML 컴포넌트 출력 (scrolling=True로 설정하여 내용이 넘쳐도 볼 수 있게 함)
        components.html(render_html, height=900, scrolling=True)
    else:
        # 파일이 없을 경우 에러 메시지를 확실히 띄움
        st.error("❌ 'index.html' 파일을 찾을 수 없습니다. GitHub 저장소의 루트 경로에 파일이 있는지 확인해주세요.")
        # 현재 경로의 파일 목록을 보여줌 (디버깅용)
        st.write("현재 경로 파일 목록:", os.listdir("."))

    # --- 하단 그래프 ---
    st.markdown("---")
    st.markdown("### 📈 유닛 상세 트렌드 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Unit {unit_id} 센서(S11) 변화**")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(unit_df['cycle'], unit_df['s11'], color='#0ea5e9')
        st.pyplot(fig1)
    with col2:
        st.write(f"**RUL(잔존수명) 감소 추이**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(unit_df['cycle'], unit_df['cycle'].max() - unit_df['cycle'], color='#ef4444')
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
