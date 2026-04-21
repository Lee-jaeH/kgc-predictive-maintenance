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

# --- 2. 데이터 가공 및 RUL 계산 ---
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

# --- 3. 실시간 센서 로그 생성 (HTML용) ---
def get_live_logs(unit_df, current_rul):
    logs = ""
    # 마지막 시점의 센서 11(온도) 수치 확인
    last_s11 = unit_df['s11'].iloc[-1]
    
    # 위험 단계 로그
    if current_rul < 30:
        logs += f'''
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-red/50 alert-row mb-3">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-red/20 text-neon-red rounded-full flex items-center justify-center font-black mr-4 border border-neon-red/30">!</div>
            <div>
                <p class="text-sm font-bold text-white">긴급: 설비 교체 주기 도달</p>
                <p class="text-xs text-dark-300 mt-1">잔존 수명이 {current_rul} 사이클 미만입니다. 즉시 점검이 필요합니다.</p>
            </div>
        </div>
        '''
    
    # 주의 단계 로그 (S11 온도 기준)
    if last_s11 > 477:
        logs += f'''
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-amber/50 mb-3">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-amber/20 text-neon-amber rounded-full flex items-center justify-center font-black mr-4 border border-neon-amber/30">S11</div>
            <div>
                <p class="text-sm font-bold text-white">센서 11 온도 주의</p>
                <p class="text-xs text-dark-300 mt-1">현재 온도 {last_s11:.2f}ºC. 임계치 근접 패턴 포착.</p>
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
    if df is None:
        return

    # 사이드바 제어
    st.sidebar.title("🛠️ 제어 센터")
    unit_id = st.sidebar.selectbox("모니터링할 유닛 선택", sorted(df['unit'].unique()))
    
    # 데이터 계산 (RUL, 건강점수 등)
    unit_df, current_cycle, current_rul, health_score = process_unit_data(df, unit_id)
    
    # --- HTML 리포트 데이터 주입 ---
    if os.path.exists("index.html"):
        try:
            with open("index.html", "r", encoding="utf-8") as f:
                html_template = f.read()
                
            # 상태 텍스트 결정
            status_text = "교체 권고" if current_rul < 40 else "안정"
            risk_level = "HIGH" if current_rul < 40 else ("MEDIUM" if current_rul < 80 else "LOW")
            
            # HTML 내부의 {{변수}}들을 실제 데이터로 교체
            render_html = html_template.replace("{{SELECTED_UNIT}}", str(unit_id))
            render_html = render_html.replace("{{CURRENT_RUL}}", str(current_rul))
            render_html = render_html.replace("{{CURRENT_CYCLE}}", str(current_cycle))
            render_html = render_html.replace("{{HEALTH_SCORE}}", str(health_score))
            render_html = render_html.replace("{{RISK_LEVEL}}", risk_level)
            render_html = render_html.replace("{{REPLACE_STATUS}}", status_text)
            render_html = render_html.replace("{{SENSOR_LOGS}}", get_live_logs(unit_df, current_rul))

            # 스트림릿 상단에 커스텀 대시보드 표시
            components.html(render_html, height=850, scrolling=False)
        except Exception as e:
            st.error(f"HTML 렌더링 오류: {e}")
    else:
        st.warning("⚠️ index.html 파일을 찾을 수 없습니다.")

    # --- 하단 상세 그래프 (스트림릿 기본 차트) ---
    st.markdown("---")
    st.markdown("### 📈 Unit 상세 데이터 트렌드")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Unit {unit_id} 센서(S11) 변화**")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(unit_df['cycle'], unit_df['s11'], color='#0ea5e9', linewidth=2)
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Sensor 11 Value")
        st.pyplot(fig1)
        
    with c2:
        st.write(f"**잔존 수명(RUL) 감소 추이**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        # RUL은 전체 사이클 수에서 현재 사이클을 뺀 값의 흐름
        max_c = unit_df['cycle'].max()
        ax2.plot(unit_df['cycle'], max_c - unit_df['cycle'], color='#ef4444', linewidth=2)
        ax2.fill_between(unit_df['cycle'], max_c - unit_df['cycle'], color='#ef4444', alpha=0.1)
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("RUL")
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
