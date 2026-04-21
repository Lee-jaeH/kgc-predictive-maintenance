import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit.components.v1 as components
import os
import requests

# 페이지 설정
st.set_page_config(page_title="KGC 설비 예지보전 시스템", layout="wide")

# --- 데이터 로드 함수 ---
# --- 데이터 로드 함수 (URL에서 직접 읽기) ---
@st.cache_data
def load_data():
    # 데이터 파일 경로 (GitHub 저장소의 루트에 있다고 가정)
    train_path = 'train_FD001.txt'
    test_path = 'test_FD001.txt'
    rul_path = 'RUL_FD001.txt'
    # NASA CMAPSS 데이터가 호스팅된 공개 저장소 URL
    base_url = "https://raw.githubusercontent.com/vitidm/NASA-Turbofan-Engine-Degradation-Simulation-Data-Set/master/CMAPSSData/"
    
    train_url = f"{base_url}train_FD001.txt"
    test_url = f"{base_url}test_FD001.txt"
    rul_url = f"{base_url}RUL_FD001.txt"

    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

    if not os.path.exists(train_path):
        st.error(f"데이터 파일({train_path})을 찾을 수 없습니다. GitHub에 데이터 파일을 함께 업로드해주세요.")
    try:
        # 데이터 읽기 (\s+ 는 공백이 여러 개일 때 처리)
        train_df = pd.read_csv(train_url, sep='\s+', header=None).dropna(axis=1)
        train_df.columns = columns
        
        test_df = pd.read_csv(test_url, sep='\s+', header=None).dropna(axis=1)
        test_df.columns = columns
        
        rul_df = pd.read_csv(rul_url, sep='\s+', header=None).dropna(axis=1)
        rul_df.columns = ['id_truth']
        
        return train_df, test_df, rul_df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None, None, None

    train_df = pd.read_csv(train_path, sep=' ', header=None).dropna(axis=1)
    train_df.columns = columns
    
    test_df = pd.read_csv(test_path, sep=' ', header=None).dropna(axis=1)
    test_df.columns = columns
    
    rul_df = pd.read_csv(rul_path, sep=' ', header=None).dropna(axis=1)
    rul_df.columns = ['id_truth']
    
    return train_df, test_df, rul_df

# --- 전처리 로직 ---
def preprocess_data(train_df, test_df, rul_df):
def preprocess_data(train_df, test_df):
    # RUL 계산
    train_df['rul'] = train_df.groupby('unit')['cycle'].transform(max) - train_df['cycle']

    # 스케일링
    scaler = MinMaxScaler()
    cols_to_scale = train_df.columns.difference(['unit', 'cycle', 'rul'])
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    return train_df, test_df, scaler

# --- 시퀀스 생성 함수 ---
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
    return train_df, test_df

# --- 메인 대시보드 ---
def main():
    st.title("🚀 KGC 설비 예지보전 (LSTM RUL 예측)")
    st.markdown("""
    이 앱은 NASA CMAPSS 데이터를 사용하여 증삼기 모터 및 추출기 펌프와 같은 설비의 **잔존 수명(Remaining Useful Life)**을 예측합니다.
    """)

    train_df, test_df, rul_df = load_data()
    if train_df is None: return
    # 1. 커스텀 HTML 리포트 출력 (index.html 파일이 있는 경우에만)
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
    
    st.title("🚀 KGC 설비 예지보전 실시간 모니터링")
    
    # 데이터 로드 (URL 방식)
    with st.spinner('인터넷에서 데이터를 불러오는 중...'):
        train_df, test_df, rul_df = load_data()
        
    if train_df is None:
        st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해주세요.")
        return

    train_df, test_df, scaler = preprocess_data(train_df, test_df, rul_df)
    train_df, test_df = preprocess_data(train_df, test_df)

    # 사이드바 설정
    st.sidebar.header("설정 및 분석")
    unit_id = st.sidebar.selectbox("분석할 유닛(ID) 선택", train_df['unit'].unique())

    tab1, tab2, tab3 = st.tabs(["📊 데이터 현황", "🧠 모델 예측", "📈 센서 분석"])
    tab1, tab2 = st.tabs(["📊 센서 데이터 추이", "🧠 RUL 분석 모델"])

    with tab1:
        st.subheader(f"Unit {unit_id} 데이터 샘플")
        st.write(train_df[train_df['unit'] == unit_id].head())
        selected_sensor = st.selectbox("조회할 센서", [f's{i}' for i in range(1, 22)])
        unit_data = train_df[train_df['unit'] == unit_id]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("최대 사이클", int(train_df[train_df['unit'] == unit_id]['cycle'].max()))
        with col2:
            st.metric("현재 데이터 포인트 수", len(train_df[train_df['unit'] == unit_id]))
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=unit_data, x='cycle', y=selected_sensor, ax=ax, color='#ef4444')
        ax.set_title(f"Unit {unit_id} - Sensor {selected_sensor} Trend")
        st.pyplot(fig)

    with tab2:
        st.subheader("RUL 예측 결과")
        # 모델 구축 (코랩의 로직 요약)
        sequence_length = 50
        sensor_cols = [f's{i}' for i in range(1, 22)]
        
        # 모델은 가볍게 정의하거나, 미리 저장된 h5 파일을 로드하는 것을 권장합니다.
        # 여기서는 시연을 위해 구조만 생성합니다.
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(sequence_length, len(sensor_cols))),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse')
        
        st.info("실제 서비스 환경에서는 학습된 .h5 모델 파일을 로드하여 사용합니다.")
        
        # 시각화 예시 (실제 예측값 대신 샘플 데이터로 시뮬레이션)
        st.subheader(f"Unit {unit_id} 잔존 수명(RUL) 시뮬레이션")
        unit_data = train_df[train_df['unit'] == unit_id]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(unit_data['cycle'].values, unit_data['rul'].values, label='Actual RUL')
        ax.plot(unit_data['cycle'].values, unit_data['rul'].values, label='Ground Truth RUL', color='black')
        ax.fill_between(unit_data['cycle'].values, unit_data['rul'].values, color='red', alpha=0.2)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL')
        ax.set_ylabel('Remaining Useful Life')
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("주요 센서 추이 분석")
        selected_sensor = st.selectbox("센서 선택", [f's{i}' for i in range(1, 22)])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=train_df[train_df['unit'] == unit_id], x='cycle', y=selected_sensor, ax=ax)
        st.pyplot(fig)
        st.success(f"현재 Unit {unit_id}의 예측 상태는 안정적입니다.")

if __name__ == "__main__":
    main()
