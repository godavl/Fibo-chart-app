import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import requests
import google.generativeai as genai

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="통합 글로벌 & KRX AI 피보나치 대시보드", layout="wide")
st.title("📈 글로벌 주식/매크로 & 피보나치 AI 분석 대시보드")

# --- 2. API 키 및 설정 (사이드바) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_KEY", "")  # 공공데이터포털 인증키

with st.sidebar:
    st.header("⚙️ API 키 설정")
    user_gemini_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    user_datago_key = st.text_input("공공데이터포털 API Key", value=DATA_GO_KR_KEY, type="password")
    
    if user_gemini_key:
        GEMINI_API_KEY = user_gemini_key
    if user_datago_key:
        DATA_GO_KR_KEY = user_datago_key

# --- 3. 데이터 수집 함수 (캐싱 적용) ---

# 3-1. 주가 데이터 수집 (yfinance)
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_code: str, period: str = "120d"):
    try:
        df = yf.download(ticker_code, period=period, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"[{ticker_code}] yfinance 데이터 수집 실패: {e}")
        return None

# 3-2. 글로벌 매크로 지표 수집 (환율 및 미국 국채 금리)
@st.cache_data(ttl=3600)
def fetch_macro_indicators():
    macro_data = {}
    try:
        # USD/KRW, USD/JPY, US 10Y Yield (^TNX)
        tickers = {"USD_KRW": "KRW=X", "USD_JPY": "JPY=X", "US_10Y": "^TNX"}
        for name, sym in tickers.items():
            df = yf.download(sym, period="5d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                macro_data[name] = float(df['Close'].iloc[-1])
    except Exception as e:
        print(f"매크로 지표 수집 제한: {e}")
    return macro_data

# 3-3. 한국 주식 공공데이터포털 API 수급 수집
@st.cache_data(ttl=3600)
def fetch_krx_info(ticker_digits: str, service_key: str):
    if not service_key or not ticker_digits.isdigit():
        return None
    
    url = "http://apis.data.go.kr/1160100/service/GetStockMarketInfoService/getStockMarketInfo"
    params = {
        'serviceKey': service_key,
        'numOfRows': '5',
        'pageNo': '1',
        'resultType': 'json',
        'likeShtnIscd': ticker_digits
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            items = res.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
            if items:
                latest = items[0]
                return {
                    "fltRt": latest.get('fltRt', 'N/A'),        # 등락률
                    "clpr": latest.get('clpr', 'N/A'),          # 종가
                    "mrktTotAmt": latest.get('mrktTotAmt', 'N/A')# 시가총액
                }
    except Exception as e:
        print(f"공공데이터포털 수급 조회 실패: {e}")
    return None

# --- 4. 변곡점(P1~P4) 및 피보나치 계산 알고리즘 ---
def detect_pivots_and_fibonacci(df, order=5):
    close_prices = df['Close'].values
    dates = df.index
    
    # scipy를 활용한 지역 고점(Peaks) 및 저점(Troughs) 실제 계산
    high_idx = argrelextrema(close_prices, np.greater, order=order)[0]
    low_idx = argrelextrema(close_prices, np.less, order=order)[0]
    
    pivots = []
    for idx in high_idx:
        pivots.append({'type': 'High', 'date': dates[idx], 'price': float(close_prices[idx]), 'idx': idx})
    for idx in low_idx:
        pivots.append({'type': 'Low', 'date': dates[idx], 'price': float(close_prices[idx]), 'idx': idx})
        
    pivots = sorted(pivots, key=lambda x: x['date'])
    recent_pivots = pivots[-4:] if len(pivots) >= 4 else pivots
    
    # P1~P4 포맷팅
    pivot_dict = {}
    for i, p in enumerate(recent_pivots, 1):
        pivot_dict[f"P{i}"] = {
            "type": p['type'],
            "date": p['date'].strftime('%Y-%m-%d'),
            "price": p['price'],
            "idx": p['idx']
        }
        
    # 피보나치 레벨 계산 (120일 최고/최저 기준)
    high_p = float(df['High'].max())
    low_p = float(df['Low'].min())
    diff = high_p - low_p
    
    fib_levels = {
        "0.000 (High)": high_p,
        "0.236": high_p - 0.236 * diff,
        "0.382": high_p - 0.382 * diff,
        "0.500": high_p - 0.500 * diff,
        "0.618": high_p - 0.618 * diff,
        "0.786": high_p - 0.786 * diff,
        "1.000 (Low)": low_p
    }
    
    return pivot_dict, fib_levels

# --- 5. 차트 시각화 (피봇 포인트 P1~P4 표시 포함) ---
def plot_fibonacci_chart(df, ticker, fib_levels, pivot_dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
    
    # 피보나치 선 그리기
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'black']
    for (label, level), color in zip(fib_levels.items(), colors):
        ax.axhline(level, linestyle='--', alpha=0.5, color=color, label=f"{label}: {level:,.2f}")
        
    # [핵심] 알고리즘이 탐지한 실제 P1~P4 피봇 포인트 차트에 시각화
    for p_name, p_data in pivot_dict.items():
        marker_color = 'red' if p_data['type'] == 'High' else 'green'
        ax.scatter(pd.to_datetime(p_data['date']), p_data['price'], color=marker_color, s=80, zorder=5)
        ax.annotate(
            f"{p_name} ({p_data['price']:,.0f})",
            (pd.to_datetime(p_data['date']), p_data['price']),
            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold'
        )

    ax.set_title(f"[{ticker}] Fibonacci Chart & Detected Pivot Points (P1~P4)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    return fig

# --- 6. Gemini AI 동기화 분석 함수 ---
def get_gemini_analysis(api_key, ticker, df, fib_levels, pivot_dict, volume_ratio, macro_data, krx_info):
    if not api_key:
        return "❌ Gemini API 키를 설정해 주세요."

    genai.configure(api_key=api_key)

    # 동적 Flash 모델 탐색
    candidate_models = []
    try:
        all_models = genai.list_models()
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                candidate_models.append(m.name.replace('models/', ''))
        candidate_models = sorted(list(set(candidate_models)), reverse=True)
    except Exception as e:
        print(f"모델 동적 탐색 실패, 백업 목록 사용: {e}")
        candidate_models = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']

    if not candidate_models:
        candidate_models = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']

    latest_price = float(df['Close'].iloc[-1])
    fib_summary = "\n".join([f"- {k}: {v:,.2f}" for k, v in fib_levels.items()])
    pivot_summary = "\n".join([f"- {k} ({v['type']}): {v['date']}일자 {v['price']:,.2f}원" for k, v in pivot_dict.items()])
    
    macro_str = f"원/달러: {macro_data.get('USD_KRW', 'N/A'):,.1f}원 | 엔/달러: {macro_data.get('USD_JPY', 'N/A'):,.1f}엔 | 미 10년물 금리: {macro_data.get('US_10Y', 'N/A'):.2f}%"
    krx_str = f"등락률: {krx_info.get('fltRt', 'N/A')}% | 시가총액: {krx_info.get('mrktTotAmt', 'N/A')}원" if krx_info else "해당 없음 (외국/코인 종목)"

    prompt = f"""
당신은 월스트리트 수석 기술적 분석가입니다.
제공된 **실제 차트 피봇 데이터(P1~P4)**와 **피보나치 구간**만을 바탕으로 [{ticker}] 정밀 보고서를 작성하세요.

[실제 차트 탐지 피봇 포인트 (P1~P4)] - *이 숫자만 참조하세요*
{pivot_summary}

[피보나치 주요 지지/저항 구간]
{fib_summary}

[수급 및 시장 상황]
- 현재가: {latest_price:,.2f}
- 최근 20일 대비 거래량 배율: {volume_ratio:.2f}배
- 공공데이터 수급 정보: {krx_str}
- 글로벌 매크로 지표: {macro_str}

[작성 지침]
1. 위에서 제공된 P1~P4 피봇 가격 및 날짜만을 정확히 인용하여 현재 가격 흐름을 평가하세요.
2. 매크로 환경(환율, 금리)과 거래량 배율({volume_ratio:.2f}배)이 종목 지지에 미치는 영향을 분석하세요.
3. [실전 대응 지침] 섹션에 목표가, 분할 매수 진입가, 손절가를 명확한 수치로 제시하세요.
"""

    response_text = None
    used_model = None

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                response_text = response.text
                used_model = model_name
                break
        except Exception as e:
            print(f"[{model_name}] 실행 오류: {e}")
            continue

    if response_text:
        return f"*(사용된 모델: `{used_model}`)*\n\n{response_text}"
    else:
        return "❌ Gemini AI 분석 생성에 실패했습니다. API 키 및 연결 상태를 확인해 주세요."

# --- 7. 메인 UI 구성 ---
ticker_input = st.text_input("종목 티커 입력 (예: EWY, NVDA, 035420, 7203.T)", value="035420")

if st.button("통합 분석 시작 🚀"):
    if not ticker_input:
        st.warning("티커를 입력해 주세요.")
    else:
        raw_ticker = ticker_input.strip().upper()
        
        # 한국 숫자 코드 지원 (.KS)
        ticker_code = f"{raw_ticker}.KS" if raw_ticker.isdigit() else raw_ticker

        with st.spinner(f"[{ticker_code}] 멀티 데이터 수집 중..."):
            df = fetch_stock_data(ticker_code)
            macro_data = fetch_macro_indicators()
            krx_info = fetch_krx_info(raw_ticker, DATA_GO_KR_KEY) if raw_ticker.isdigit() else None

            if df is None or df.empty:
                st.error("데이터를 가져오지 못했습니다. 티커명을 확인해 주세요.")
            else:
                # 거래량 배율 계산
                recent_20_vol = float(df['Volume'].tail(20).mean())
                current_vol = float(df['Volume'].iloc[-1])
                volume_ratio = (current_vol / recent_20_vol) if recent_20_vol > 0 else 1.0

                # 피봇 포인트 및 피보나치 계산
                pivot_dict, fib_levels = detect_pivots_and_fibonacci(df)

                # 1. 차트 출력
                st.subheader("📊 피보나치 & P1~P4 피봇 동기화 차트")
                fig = plot_fibonacci_chart(df, ticker_code, fib_levels, pivot_dict)
                st.pyplot(fig)

                # 2. AI 보고서 출력
                st.subheader("🤖 Gemini AI 매크로 & 수급 연동 보고서")
                with st.spinner("Gemini AI 분석 진행 중..."):
                    report = get_gemini_analysis(
                        GEMINI_API_KEY, ticker_code, df, fib_levels, pivot_dict, volume_ratio, macro_data, krx_info
                    )
                    st.markdown(report)
