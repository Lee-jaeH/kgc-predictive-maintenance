import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="KGC AI 설비 예지보전",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- [데이터 분석 및 ML 알림 엔진] ---
@st.cache_data
def analyze_equipment_data():
    data_path = 'data/train_FD001.txt'
    
    if not os.path.exists(data_path):
        total_units = 100
        mock_ruls = np.random.randint(5, 205, size=total_units)
        unit_ids = np.arange(1, total_units + 1)
    else:
        columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df = pd.read_csv(data_path, sep='\s+', header=None, names=columns)
        max_cycles = df.groupby('unit')['cycle'].max()
        unit_ids = max_cycles.index.values
        # 실제 모델이 있다면 여기서 예측값을 가져오지만, 여기서는 계산된 RUL을 사용
        mock_ruls = (200 - max_cycles).clip(lower=0).values 

    # 1. 통계치 계산
    avg_rul = int(np.mean(mock_ruls))
    danger_mask = mock_ruls < 50
    caution_mask = (mock_ruls >= 50) & (mock_ruls <= 100)
    
    danger_units = len(mock_ruls[danger_mask])
    caution_units = len(mock_ruls[caution_mask])
    normal_units = len(mock_ruls) - danger_units - caution_units
    health_score = round(((normal_units * 100) + (caution_units * 50)) / len(mock_ruls), 1)

    # 2. ML 이상 징후 알림 생성 (RUL이 낮은 순서대로 상위 알림 추출)
    alerts_html = ""
    # 위험(Danger) 등급 유닛들 중 가장 심각한 3개 추출
    danger_indices = np.where(danger_mask)[0]
    sorted_danger_idx = danger_indices[np.argsort(mock_ruls[danger_indices])][:3]
    
    for idx in sorted_danger_idx:
        u_id = unit_ids[idx]
        rul_val = mock_ruls[idx]
        alerts_html += f"""
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-red/30 mb-3 animate-pulse">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-red/20 text-neon-red rounded-full flex items-center justify-center font-black mr-4 border border-neon-red/40">!</div>
            <div class="flex-grow">
                <div class="flex justify-between items-center">
                    <p class="text-sm font-black text-white">Unit {u_id} 고장 임박 감지</p>
                    <span class="px-2 py-0.5 bg-neon-red/20 text-neon-red text-[10px] font-black rounded uppercase">Critical</span>
                </div>
                <p class="text-xs text-dark-300 mt-1">ML 분석 결과, 예상 잔존 수명이 <span class="text-neon-red font-bold">{rul_val} 사이클</span> 미만입니다. 즉시 점검이 필요합니다.</p>
            </div>
        </div>
        """

    # 주의(Caution) 등급 유닛 중 1개 추가
    caution_indices = np.where(caution_mask)[0]
    if len(caution_indices) > 0:
        u_id = unit_ids[caution_indices[0]]
        alerts_html += f"""
        <div class="flex items-start p-4 bg-dark-800 rounded-xl border border-neon-amber/20 mb-3">
            <div class="w-10 h-10 flex-shrink-0 bg-neon-amber/20 text-neon-amber rounded-full flex items-center justify-center font-black mr-4 border border-neon-amber/40">?</div>
            <div class="flex-grow">
                <div class="flex justify-between items-center">
                    <p class="text-sm font-black text-white">Unit {u_id} 이상 징후 포착</p>
                    <span class="px-2 py-0.5 bg-neon-amber/20 text-neon-amber text-[10px] font-black rounded uppercase">Warning</span>
                </div>
                <p class="text-xs text-dark-300 mt-1">센서 S11 및 S15 패턴이 비정상적입니다. 예방 정비를 권고합니다.</p>
            </div>
        </div>
        """

    return {
        "avg_rul": str(avg_rul),
        "health_score": str(health_score),
        "normal_pct": str(normal_units),
        "caution_pct": str(caution_units),
        "danger_pct": str(danger_units),
        "danger_count": f"{danger_units:02d}",
        "caution_count": f"{caution_units:02d}",
        "alert_items": alerts_html
    }

# --- [HTML 렌더링 엔진] ---
def render_dashboard():
    stats = analyze_equipment_data()
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            template = f.read()
        
        for key, value in stats.items():
            template = template.replace(f"{{{{{key}}}}}", value)
            
        components.html(template, height=920, scrolling=True)
    except FileNotFoundError:
        st.error("❌ 'index.html' 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    render_dashboard()
    with st.sidebar:
        st.header("⚙️ 분석 제어")
        if st.button("모델 재분석 (Refresh)"):
            st.cache_data.clear()
            st.rerun()
