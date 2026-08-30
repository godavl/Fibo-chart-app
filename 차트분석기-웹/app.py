import warnings
warnings.filterwarnings("ignore")

import urllib.parse
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from google import genai

# ==========================================
# 1. 페이지 기본 설정 및 디자인 스타일링
# ==========================================
st.set_page_config(
    page_title="피보나치 AI 주식/코인 분석기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 피보나치 AI 차트 & 분석 대시보드")
st.caption("한글 종목명, 영문, 티커를 입력하면 차트 분석과 Gemini AI 보고서를 생성합니다.")

# ==========================================
# 2. 핵심 로직 함수 (검색, 차트, AI)
# ==========================================
def search_ticker_by_name(keyword):
    preset_map = {
        "카카오": "035720.KS",
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "네이버": "035420.KS",
        "NAVER": "035420.KS",
        "현대차": "005380.KS",
        "비트코인": "BTC-USD",
        "이더리움": "ETH-USD",
        "리플": "XRP-USD"
    }
    
    clean_keyword = keyword.replace(" ", "")
    if clean_keyword in preset_map:
        return preset_map[clean_keyword]

    # 네이버 금융 자동완성 API
    try:
        naver_url = f"https://ac.finance.naver.com/ac?q={urllib.parse.quote(keyword)}&target=stock"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(naver_url, headers=headers, timeout=8).json()
        items = res.get('items', [])
        if items and len(items) > 0 and len(items[0]) > 0:
            first_match = items[0][0]
            for elem in first_match:
                if isinstance(elem, str) and len(elem) == 6 and elem.isdigit():
                    return f"{elem}.KS"
    except:
        pass

    # 야후 파이낸스 검색 API
    try:
        yahoo_search_url = f"https://query2.finance.yahoo.com/1/finance/search?q={urllib.parse.quote(keyword)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(yahoo_search_url, headers=headers, timeout=8).json()
        quotes = res.get('quotes', [])
        if quotes:
            return quotes[0]['symbol']
    except:
        pass

    return None

def get_chart_data(user_input):
    found_symbol = search_ticker_by_name(user_input)
    symbol = found_symbol if found_symbol else user_input.upper()
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1y&interval=1d"
        res = requests.get(url, headers=headers, timeout=10).json()
        result = res.get('chart', {}).get('result')
        
        if not result and symbol.endswith('.KS'):
            symbol = symbol.replace('.KS', '.KQ')
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1y&interval=1d"
            res = requests.get(url, headers=headers, timeout=10).json()
            result = res.get('chart', {}).get('result')

        if not result or not result[0].get('timestamp'):
            return None, symbol

        timestamps = result[0]['timestamp']
        quote = result[0]['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        }, index=pd.to_datetime(timestamps, unit='s')).dropna()
        
        return df, symbol
    except Exception as e:
        return None, symbol

def plot_fibonacci_chart(df, symbol):
    recent_df = df.tail(120)
    high_val = recent_df['High'].max()
    low_val = recent_df['Low'].min()
    diff = high_val - low_val
    
    levels = {
        "0.000 (High)": high_val,
        "0.236": high_val - 0.236 * diff,
        "0.382": high_val - 0.382 * diff,
        "0.500": high_val - 0.500 * diff,
        "0.618": high_val - 0.618 * diff,
        "0.786": high_val - 0.786 * diff,
        "1.000 (Low)": low_val
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(recent_df.index, recent_df['Close'], label='Close Price', color='#1f77b4', linewidth=1.8)

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b', '#d62728']
    for idx, (level_name, price) in enumerate(levels.items()):
        ax.axhline(y=price, color=colors[idx % len(colors)], linestyle='--', alpha=0.6, label=f'{level_name}: {price:,.2f}')

    ax.set_title(f"[{symbol}] Fibonacci Retracement Chart", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    return fig

# ==========================================
# 3. 사이드바 - 설정 영역
# ==========================================
with st.sidebar:
    st.header("⚙️ 서비스 설정")
    gemini_api_key = st.text_input("Gemini API Key 입력", type="password")
    st.markdown("---")
    st.markdown("💡 **Tip:** API 키를 입력하면 매번 입력하지 않도록 설정할 수도 있습니다.")

# ==========================================
# 4. 메인 화면 - 검색 및 결과
# ==========================================
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("종목명 또는 티커를 입력하세요", placeholder="예: 카카오, 삼성전자, 비트코인, NVDA")
with col2:
    st.write(" ") # 높이 맞춤용
    st.write(" ")
    search_btn = st.button("분석 시작 🚀", use_container_width=True)

if search_btn and user_input:
    if not gemini_api_key:
        st.error("왼쪽 사이드바에 Gemini API Key를 입력해 주세요.")
    else:
        with st.spinner(f"[{user_input}] 데이터 수집 및 차트 생성 중..."):
            df, target_symbol = get_chart_data(user_input)

        if df is None or df.empty:
            st.error(f"[{user_input}] 데이터를 불러올 수 없습니다. 정확한 종목명이나 티커(예: 035720.KS)를 입력해 주세요.")
        else:
            # 2열 레이아웃 배치
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader("📊 피보나치 되돌림 차트")
                fig = plot_fibonacci_chart(df, target_symbol)
                st.pyplot(fig)

            with right_col:
                st.subheader("🤖 Gemini AI 상세 분석 보고서")
                with st.spinner("AI 분석 보고서를 생성하는 중입니다..."):
                    try:
                        client = genai.Client(api_key=gemini_api_key)
                        ohlcv_data = df[["Open", "High", "Low", "Close", "Volume"]].tail(120).to_string()

                        prompt = f"""
너는 피보나치 되돌림 채널 전문 애널리스트다.
제공된 OHLCV 데이터를 바탕으로 분석 보고서를 작성하라.

[목차 구성을 엄격히 준수할 것]:
1. 📌 [3초 핵심 요약] (현재가, 판단 신호, 핵심 지지/저항, 1차/2차 목표가, 손절 기준)
2. 채널 기준점 (Pivot Points)
3. 두 채널 기울기 괴리 및 신뢰도
4. 현재 위치 및 피보나치 되돌림 레벨
5. 핵심 매물대 및 매물 진공 구간
6. 시나리오 분석 (시나리오 A: 현실적 목표, 시나리오 B: 광기 오버슈팅)
7. 무효화 조건 (손절 및 대응 지침)

[대상 자산]: {target_symbol}
[최근 120일 차트 데이터]:
{ohlcv_data}
                        """

                        # 💡 최신 모델 우선 순위 목록 (실패 시 차선책으로 자동 이동)
                        candidate_models = ['gemini-3.6-flash', 'gemini-flash', 'gemini-pro']
                        response = None

                        for model_name in candidate_models:
                            try:
                                response = client.models.generate_content(
                                    model=model_name,
                                    contents=prompt
                                )
                                if response and hasattr(response, 'text'):
                                    break
                            except Exception as model_err:
                                print(f"Model {model_name} 호출 실패, 다음 모델로 시도: {model_err}")
                                continue

                        if response and hasattr(response, 'text'):
                            st.markdown(response.text)
                        else:
                            st.error("모든 Gemini AI 모델 응답에 실패했습니다. API 키나 네트워크를 확인해 주세요.")

                    except Exception as e:
                        st.error(f"AI 분석 생성 실패: {e}")
