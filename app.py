import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# 페이지 설정
st.set_page_config(page_title="KGC 설비 예지보전 시스템", layout="wide")

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    # 데이터 파일 경로 (GitHub 저장소의 루트에 있다고 가정)
    train_path = 'train_FD001.txt'
    test_path = 'test_FD001.txt'
    rul_path = 'RUL_FD001.txt'
    
    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    
    if not os.path.exists(train_path):
        st.error(f"데이터 파일({train_path})을 찾을 수 없습니다. GitHub에 데이터 파일을 함께 업로드해주세요.")
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

# --- 메인 대시보드 ---
def main():
    st.title("🚀 KGC 설비 예지보전 (LSTM RUL 예측)")
    st.markdown("""
    이 앱은 NASA CMAPSS 데이터를 사용하여 증삼기 모터 및 추출기 펌프와 같은 설비의 **잔존 수명(Remaining Useful Life)**을 예측합니다.
    """)

    train_df, test_df, rul_df = load_data()
    if train_df is None: return

    train_df, test_df, scaler = preprocess_data(train_df, test_df, rul_df)

    # 사이드바 설정
    st.sidebar.header("설정 및 분석")
    unit_id = st.sidebar.selectbox("분석할 유닛(ID) 선택", train_df['unit'].unique())
    
    tab1, tab2, tab3 = st.tabs(["📊 데이터 현황", "🧠 모델 예측", "📈 센서 분석"])

    with tab1:
        st.subheader(f"Unit {unit_id} 데이터 샘플")
        st.write(train_df[train_df['unit'] == unit_id].head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("최대 사이클", int(train_df[train_df['unit'] == unit_id]['cycle'].max()))
        with col2:
            st.metric("현재 데이터 포인트 수", len(train_df[train_df['unit'] == unit_id]))

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
        unit_data = train_df[train_df['unit'] == unit_id]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(unit_data['cycle'].values, unit_data['rul'].values, label='Actual RUL')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL')
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("주요 센서 추이 분석")
        selected_sensor = st.selectbox("센서 선택", [f's{i}' for i in range(1, 22)])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=train_df[train_df['unit'] == unit_id], x='cycle', y=selected_sensor, ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
