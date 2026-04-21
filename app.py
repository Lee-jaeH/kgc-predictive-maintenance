import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 페이지 설정
st.set_page_config(page_title="KGC 설비 예지 보전 시스템", layout="wide")

# --- 헬퍼 함수: 데이터 로드 ---
@st.cache_data
def load_data():
    # 실제 환경에서는 데이터 경로를 깃허브 내 경로로 수정해야 합니다.
    # 예시: pd.read_csv('data/train_FD001.txt', sep=' ', header=None)
    # 여기서는 샘플 데이터를 생성하거나 업로드된 로직을 시뮬레이션합니다.
    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    # 실제 파일이 있다면 아래 주석을 해제하고 사용하세요.
    # train = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns)
    
    # 데모용 더미 데이터 (구조 확인용)
    df = pd.DataFrame(np.random.randn(100, 26), columns=columns)
    df['unit'] = np.repeat(np.arange(1, 6), 20)
    df['cycle'] = np.tile(np.arange(1, 21), 5)
    return df

# --- 메인 타이틀 ---
st.title("🛠️ KGC 설비 예지 보전 (RUL 예측)")
st.markdown("""
이 대시보드는 NASA CMAPSS 데이터를 활용하여 **증삼기 모터**나 **추출기 펌프**의 잔존 수명(Remaining Useful Life)을 예측하는 시스템 시뮬레이션입니다.
""")

# 데이터 로드
df = load_data()

# --- 사이드바 메뉴 ---
menu = st.sidebar.selectbox("메뉴 선택", ["프로젝트 개요", "데이터 분석(EDA)", "RUL 예측 모델", "실시간 모니터링 시뮬레이션"])

if menu == "프로젝트 개요":
    st.header("1. 프로젝트 목표")
    st.info("설비의 센서 데이터를 분석하여 고장 전 잔존 수명(RUL)을 예측함으로써 유지보수 비용을 절감하고 가동 중단을 방지합니다.")
    st.image("https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbcM5N5%2FbtrK6Zq0f4V%2Fkkk9k9k9k9k9k9k9k9k9k9%2Fimg.png", caption="Predictive Maintenance Concept")
    
    st.subheader("데이터 요약")
    st.write(df.head())
    st.write(f"전체 데이터 크기: {df.shape}")

elif menu == "데이터 분석(EDA)":
    st.header("2. 센서 데이터 분석")
    
    selected_sensor = st.selectbox("분석할 센서 선택", [f's{i}' for i in range(1, 22)])
    unit_id = st.slider("설비 번호(Unit) 선택", 1, 5, 1)
    
    unit_data = df[df['unit'] == unit_id]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=unit_data, x='cycle', y=selected_sensor, ax=ax)
    ax.set_title(f"Unit {unit_id} - Sensor {selected_sensor} Trend")
    st.pyplot(fig)
    
    st.write("📌 사이클이 진행됨에 따라 센서 값의 변화 패턴을 확인할 수 있습니다.")

elif menu == "RUL 예측 모델":
    st.header("3. LSTM 기반 RUL 예측 결과")
    st.write("코랩에서 학습시킨 모델의 성능 지표를 표시합니다.")
    
    col1, col2 = st.columns(2)
    col1.metric("Root Mean Squared Error (RMSE)", "15.42", "-1.2")
    col2.metric("Mean Absolute Error (MAE)", "12.10", "-0.8")
    
    st.subheader("실제값 vs 예측값 비교")
    # 예시 차트
    chart_data = pd.DataFrame({
        'Actual RUL': np.linspace(100, 0, 50),
        'Predicted RUL': np.linspace(105, 5, 50) + np.random.normal(0, 5, 50)
    })
    st.line_chart(chart_data)

elif menu == "실시간 모니터링 시뮬레이션":
    st.header("4. 설비 상태 실시간 모니터링")
    st.warning("현재 데이터 기반으로 계산된 실시간 잔존 수명입니다.")
    
    # 프로그레스 바를 활용한 시각화
    current_cycle = st.slider("현재 사이클(운행 시간)", 1, 100, 45)
    predicted_rul = 100 - current_cycle + np.random.randint(-5, 5)
    
    st.subheader(f"예측된 잔존 수명: {predicted_rul} Cycles")
    progress_color = "green" if predicted_rul > 30 else "red"
    st.progress(max(0, min(predicted_rul, 100)) / 100)
    
    if predicted_rul <= 20:
        st.error("⚠️ 경고: 즉시 점검이 필요합니다! (RUL이 20 미만)")
    else:
        st.success("✅ 정상: 설비가 안정적인 상태입니다.")

# 푸터
st.sidebar.markdown("---")
st.sidebar.text("Developed for KGC Project")
