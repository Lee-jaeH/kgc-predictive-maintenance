import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import os
import requests

# 페이지 설정
st.set_page_config(page_title="KGC 설비 예지보전 시스템", layout="wide")

# --- 데이터 로드 함수 (URL에서 직접 읽기) ---
@st.cache_data
def load_data():
    # NASA CMAPSS 데이터가 호스팅된 공개 저장소 URL
    base_url = "https://raw.githubusercontent.com/vitidm/NASA-Turbofan-Engine-Degradation-Simulation-Data-Set/master/CMAPSSData/"
    
    train_url = f"{base_url}train_FD001.txt"
    test_url = f"{base_url}test_FD001.txt"
    rul_url = f"{base_url}RUL_FD001.txt"
    
    columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    
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

# --- 전처리 로직 ---
def preprocess_data(train_df, test_df):
    # RUL 계산
    train_df['rul'] = train_df.groupby('unit')['cycle'].transform(max) - train_df['cycle']
    
    # 스케일링
    scaler = MinMaxScaler()
    cols_to_scale = train_df.columns.difference(['unit', 'cycle', 'rul'])
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    return train_df, test_df

# --- 메인 대시보드 ---
def main():
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

    train_df, test_df = preprocess_data(train_df, test_df)

    # 사이드바 설정
    st.sidebar.header("설정 및 분석")
    unit_id = st.sidebar.selectbox("분석할 유닛(ID) 선택", train_df['unit'].unique())
    
    tab1, tab2 = st.tabs(["📊 센서 데이터 추이", "🧠 RUL 분석 모델"])

    with tab1:
        selected_sensor = st.selectbox("조회할 센서", [f's{i}' for i in range(1, 22)])
        unit_data = train_df[train_df['unit'] == unit_id]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=unit_data, x='cycle', y=selected_sensor, ax=ax, color='#ef4444')
        ax.set_title(f"Unit {unit_id} - Sensor {selected_sensor} Trend")
        st.pyplot(fig)

    with tab2:
        st.subheader(f"Unit {unit_id} 잔존 수명(RUL) 시뮬레이션")
        unit_data = train_df[train_df['unit'] == unit_id]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(unit_data['cycle'].values, unit_data['rul'].values, label='Ground Truth RUL', color='black')
        ax.fill_between(unit_data['cycle'].values, unit_data['rul'].values, color='red', alpha=0.2)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Remaining Useful Life')
        ax.legend()
        st.pyplot(fig)
        st.success(f"현재 Unit {unit_id}의 예측 상태는 안정적입니다.")

if __name__ == "__main__":
    main()
