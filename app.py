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

# --- 1. 데이터 로드 함수 (NASA 공개 데이터셋 URL) ---
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/vitidm/NASA-Turbofan-Engine-Degradation-Simulation-Data-Set/master/CMAPSSData/"
    train_url = f"{base_url}train_FD001.txt"
    
    # 데이터 컬럼 정의
    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    
    try:
        # 실제 데이터를 인터넷에서 읽어옵니다.
        df = pd.read_csv(train_url, sep='\s+', header=None).dropna(axis=1)
        df.columns = columns
        return df
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None

# --- 2. 데이터 가공 (실제 RUL 계산) ---
def process_unit_data(df, unit_id):
    # 선택된 유닛의 데이터만 추출
    unit_df = df[df['unit'] == unit_id].copy()
    max_cycle = unit_df['cycle'].max()
    
    # 현재 시점의 실제 정보 추출
    current_cycle = int(unit_df['cycle'].iloc[-1])
    # RUL 계산: 이 데이터셋의 끝을 '고장'으로 보고 역산
    current_rul = int(max_cycle - current_cycle)
    
    # 건강 점수(Health Score) 계산: 150사이클 이상 남으면 100점, 가까워질수록 0점
    health_score = max(5, min(100, int((current_rul / 150) * 100)))
    
    return unit_df, current_cycle, current_rul, health_score

# --- 3. 실시간 센서 로그 생성 ---
def get_live_logs(unit_df, current_rul):
    logs = ""
    # 센서 11(온도) 수치 확인
    last_s11 = unit_df['s11'].iloc[-1]
    
    if current_rul < 30:
        logs += f'''
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-red/50 alert-row mb-3">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-red/20 text-neon-red rounded-full flex items-center justify-center font-black mr-4">!</div>
            <div>
                <p class="text-sm font-bold text-white">긴급: 설비 교체 주기 도달</p>
                <p class="text-xs text-dark-300 mt-1">잔존 수명이 {current_rul} 사이클 미만입니다. 즉시 점검이 필요합니다.</p>
            </div>
        </div>
        '''
    
    if last_s11 > 477:
        logs += f'''
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-amber/50 mb-3">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-amber/20 text-neon-amber rounded-full flex items-center justify-center font-black mr-4">S11</div>
            <div>
                <p class="text-sm font-bold text-white">센서 11 온도 과열</p>
                <p class="text-xs text-dark-300 mt-1">현재 온도 {last_s11:.2f}ºC. 임계치 초과 징후 포착.</p>
            </div>
        </div>
        '''
        
    if not logs:
        logs = '<p class="text-dark-500 text-sm text-center py-10">정상 운전 중 (이상 징후 없음)</p>'
    return logs

# --- 4. 메인 실행부 ---
def main():
    # 데이터 불러오기
    df = load_data()
    if df is None: return

    # 사이드바 제어
    st.sidebar.title("🛠️ 제어 센터")
    unit_id = st.sidebar.selectbox("모니터링할 유닛 선택", sorted(df['unit'].unique()))
    
    # 데이터 계산
    unit_df, current_cycle, current_rul, health_score = process_unit_data(df, unit_id)
    
    # --- HTML 리포트 데이터 주입 ---
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html_template = f.read()
            
        # HTML 내부의 {{변수}}들을 실제 데이터로 교체 (매우 중요!)
        status_text = "위험 (교체 필요)" if current_rul < 30 else ("주의" if current_rul < 70 else "안정")
        risk_level = "HIGH" if current_rul < 30 else ("MEDIUM" if current_rul < 70 else "LOW")
        
        render_html = html_template.replace("{{SELECTED_UNIT}}", str(unit_id))
        render_html = render_html.replace("{{CURRENT_RUL}}", str(current_rul))
        render_html = render_html.replace("{{CURRENT_CYCLE}}", str(current_cycle))
        render_html = render_html.replace("{{HEALTH_SCORE}}", str(health_score))
        render_html = render_html.replace("{{RISK_LEVEL}}", risk_level)
        render_html = render_html.replace("{{REPLACE_STATUS}}", status_text)
        render_html = render_html.replace("{{SENSOR_LOGS}}", get_live_logs(unit_df, current_rul))

        # 스트림릿 상단에 커스텀 대시보드 표시
        components.html(render_html, height=850)

    # --- 하단 스트림릿 자체 그래프 ---
    st.markdown("### 📈 상세 데이터 분석 (상세 뷰)")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Unit {unit_id} 센서 트렌드**")
        fig, ax = plt.subplots()
        ax.plot(unit_df['cycle'], unit_df['s11'], color='#0ea5e9')
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor 11 Value")
        st.pyplot(fig)
    with c2:
        st.write(f"**잔존 수명(RUL) 감소 그래프**")
        fig, ax = plt.subplots()
        ax.plot(unit_df['cycle'], (unit_df['cycle'].max() - unit_df['cycle']), color='#ef4444')
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
