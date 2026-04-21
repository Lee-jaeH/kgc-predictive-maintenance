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

# --- [데이터 분석 엔진] ---
@st.cache_data
def analyze_equipment_data():
    """
    NASA CMAPSS FD001 데이터 구조를 기반으로 실시간 대시보드 수치를 계산합니다.
    실제 프로젝트 시 'data/train_FD001.txt' 경로에 데이터를 배치하세요.
    """
    data_path = 'data/train_FD001.txt'
    
    # 데이터 파일이 없을 경우 시뮬레이션 데이터 생성 (데모용)
    if not os.path.exists(data_path):
        total_units = 100
        # 랜덤하게 RUL(잔존수명) 분포 생성
        mock_ruls = np.random.randint(5, 205, size=total_units)
    else:
        # 실제 데이터 로드 및 분석 (FD001 구조)
        columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df = pd.read_csv(data_path, sep='\s+', header=None, names=columns)
        
        # 각 유닛별 현재 마지막 사이클 확인
        max_cycles = df.groupby('unit')['cycle'].max()
        # 간단한 RUL 시뮬레이션 (200 사이클 기준 고장 가정)
        mock_ruls = (200 - max_cycles).clip(lower=0).values 

    # 통계치 계산
    avg_rul = int(np.mean(mock_ruls))
    danger_units = len(mock_ruls[mock_ruls < 50])
    caution_units = len(mock_ruls[(mock_ruls >= 50) & (mock_ruls <= 100)])
    normal_units = len(mock_ruls) - danger_units - caution_units
    
    # 건전도 점수 (정상 100점, 주의 50점, 위험 0점 가중치 평균)
    health_score = round(((normal_units * 100) + (caution_units * 50)) / len(mock_ruls), 1)

    return {
        "avg_rul": str(avg_rul),
        "health_score": str(health_score),
        "normal_pct": str(normal_units),
        "caution_pct": str(caution_units),
        "danger_pct": str(danger_units),
        "danger_count": f"{danger_units:02d}",
        "caution_count": f"{caution_units:02d}"
    }

# --- [HTML 렌더링 엔진] ---
def render_dashboard():
    # 1. 데이터 분석 결과 가져오기
    stats = analyze_equipment_data()
    
    # 2. 외부 index.html 파일 읽기
    try:
        # index.html이 app.py와 같은 경로에 있어야 합니다.
        with open("index.html", "r", encoding="utf-8") as f:
            template = f.read()
        
        # 3. HTML 내 플레이스홀더 {{key}}를 실제 값으로 치환
        for key, value in stats.items():
            template = template.replace(f"{{{{{key}}}}}", value)
            
        # 4. 스트림릿 컴포넌트로 HTML 출력 (디자인 그대로 표출)
        components.html(template, height=920, scrolling=True)
        
    except FileNotFoundError:
        st.error("❌ 'index.html' 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜 주세요.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

# --- [메인 실행부] ---
if __name__ == "__main__":
    render_dashboard()
    
    # 사이드바 제어판 (데이터 갱신 등)
    with st.sidebar:
        st.header("⚙️ 분석 시스템")
        if st.button("실시간 데이터 분석 갱신"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("KGC 프로젝트: NASA FD001 데이터셋 기반 예지보전 대시보드입니다.")
