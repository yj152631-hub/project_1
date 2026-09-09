import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# -----------------------------------------------------------------------------
# 1. 한글 폰트 지원 설정 (Seaborn & Matplotlib 완벽 적용)
# -----------------------------------------------------------------------------
@st.cache_resource
def set_custom_font():
    font_path = 'NanumSquareNeo-Variable.ttf'
    
    if os.path.exists(font_path):
        # 폰트 매니저에 강제 추가
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        
        # Matplotlib 기본 설정
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        
        # Seaborn 기본 테마에 폰트 강제 적용 (폰트 깨짐 방지)
        sns.set_theme(style="white", rc={'font.family': font_name, 'axes.unicode_minus': False})
        return True
    else:
        st.warning(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        return False

set_custom_font()

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (병합 및 파생 변수 생성)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # 데이터 불러오기
        baci_df = pd.read_csv('baci_85_sample.csv')
        codes_df = pd.read_csv('country_codes_sample.csv')
        
        # BACI 데이터와 국가 코드 데이터 병합 (j 컬럼 기준)
        df_merged = pd.merge(baci_df, codes_df, on='j', how='left')
        
        # 직관적인 컬럼명으로 변경
        df_merged.rename(columns={'t': 'Year', 'v': 'Export_Value'}, inplace=True)
        
        # 수출액을 기준으로 3등분하여 'Trade_Grade'(무역액 등급) 파생 변수 생성
        if 'Export_Value' in df_merged.columns:
            df_merged['Trade_Grade'] = pd.qcut(
                df_merged['Export_Value'], 
                q=3, 
                labels=['소', '중', '대'],
                duplicates='drop'
            )
            
        return df_merged
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 'baci_85_sample.csv'와 'country_codes_sample.csv'가 같은 폴더에 있는지 확인해주세요.")
        return None

df_original = load_data()

if df_original is not None:
    
    # -----------------------------------------------------------------------------
    # 3. 사이드바 필터 설정
    # -----------------------------------------------------------------------------
    st.sidebar.header("필터 설정")

    # 국가 선택 필터
    if 'country_name' in df_original.columns:
        country_list = df_original['country_name'].dropna().unique().tolist()
        selected_countries = st.sidebar.multiselect("국가 선택", options=country_list, default=country_list[:5])
    else:
        selected_countries = []
        st.sidebar.warning("데이터 병합 후 'country_name' 컬럼을 찾을 수 없습니다.")

    # 무역액 등급 선택 필터
    selected_grades = st.sidebar.multiselect(
        "무역액 등급 선택", 
        options=['대', '중', '소'], 
        default=['대', '중', '소']
    )

    # 필터 적용
    df_filtered = df_original.copy()
    if selected_countries and 'country_name' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['country_name'].isin(selected_countries)]
        
    if selected_grades and 'Trade_Grade' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Trade_Grade'].isin(selected_grades)]

    # -----------------------------------------------------------------------------
    # 4. 메인 대시보드 화면 구성
    # -----------------------------------------------------------------------------
    st.title("무역 분석 대시보드")

    # [요구사항 2] 결측치 확인
    st.subheader("데이터 결측치 현황")
    missing_values = df_original.isnull().sum()
    if missing_values.sum() > 0:
        st.dataframe(missing_values[missing_values > 0].rename("결측치 개수"))
    else:
        st.write("현재 데이터에 결측치가 없습니다.")

    st.markdown("---")

    # [요구사항 3] 주요 지표 요약 (2열 구조)
    st.subheader("주요 지표")
    col1, col2 = st.columns(2)

    with col1:
        total_transactions = len(df_filtered)
        st.metric("총 거래 건수", f"{total_transactions:,} 건")

    with col2:
        if 'Export_Value' in df_filtered.columns:
            total_export = df_filtered['Export_Value'].sum()
            st.metric("총 수출액(달러)", f"${total_export:,.2f}")
        else:
            st.metric("총 수출액(달러)", "데이터 없음")

    st.markdown("---")

    # [요구사항 4] 시각화 차트 (2열 구조)
    st.subheader("데이터 시각화")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**국가*연도 수출액 히트맵 (상위 8개국)**")
        if 'country_name' in df_filtered.columns and 'Year' in df_filtered.columns and 'Export_Value' in df_filtered.columns:
            top_8_countries = df_filtered.groupby('country_name')['Export_Value'].sum().nlargest(8).index
            df_top8 = df_filtered[df_filtered['country_name'].isin(top_8_countries)]
            
            if not df_top8.empty:
                pivot_df = df_top8.pivot_table(index='country_name', columns='Year', values='Export_Value', aggfunc='sum')
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pivot_df, annot=False, cmap='YlGnBu', ax=ax)
                ax.set_ylabel("국가")
                ax.set_xlabel("연도")
                st.pyplot(fig)
            else:
                st.caption("선택된 필터 조건에 맞는 데이터가 없습니다.")
        else:
            st.write("차트를 그리기 위한 데이터가 부족합니다.")

    with col4:
        st.markdown("**무역액 등급 분포**")
        if 'Trade_Grade' in df_filtered.columns:
            # 카테고리 순서를 '대, 중, 소'로 고정
            grade_counts = df_filtered['Trade_Grade'].value_counts().reindex(['대', '중', '소'])
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.barplot(x=grade_counts.index, y=grade_counts.values, palette='Set2', ax=ax2)
            ax2.set_xlabel("무역액 등급")
            ax2.set_ylabel("건수")
            st.pyplot(fig2)
        else:
            st.write("무역액 등급 분포를 표시할 수 없습니다.")

    st.markdown("---")

    # [요구사항 5] 교차표 (원본 건수 & 정규화 비율)
    st.subheader("상위 5개국 * 무역액 등급 교차표")
    if 'country_name' in df_filtered.columns and 'Trade_Grade' in df_filtered.columns and 'Export_Value' in df_filtered.columns:
        top_5_countries = df_filtered.groupby('country_name')['Export_Value'].sum().nlargest(5).index
        df_top5 = df_filtered[df_filtered['country_name'].isin(top_5_countries)]
        
        if not df_top5.empty:
            crosstab_count = pd.crosstab(df_top5['country_name'], df_top5['Trade_Grade'])
            if not crosstab_count.empty:
                # 존재하는 등급 컬럼만 순서대로 정렬
                cols_count = [col for col in ['대', '중', '소'] if col in crosstab_count.columns]
                crosstab_count = crosstab_count[cols_count]
            
            crosstab_norm = pd.crosstab(df_top5['country_name'], df_top5['Trade_Grade'], normalize='index') * 100
            if not crosstab_norm.empty:
                cols_norm = [col for col in ['대', '중', '소'] if col in crosstab_norm.columns]
                crosstab_norm = crosstab_norm[cols_norm]
            
            st.markdown("**원본 건수**")
            st.dataframe(crosstab_count)
            
            st.markdown("**정규화 비율 (%)**")
            st.dataframe(crosstab_norm.style.format("{:.2f}%"))
        else:
            st.caption("선택된 필터 조건에 맞는 데이터가 없어 교차표를 생성할 수 없습니다.")
    else:
        st.write("교차표를 생성하기 위한 데이터가 부족합니다.")