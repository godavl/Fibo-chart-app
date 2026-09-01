# -*- coding: utf-8 -*-
"""
app.py - 피보나치 & 매크로 Multi-Agent 종합 분석 시스템
- User-Agent 적용 및 10분 캐싱(@st.cache_data)으로 yfinance Rate Limit 방지
- 차트 시각화(종가/거래량) 및 기간 옵션 확장(1m ~ max)
- 2단계 매크로 분석 결과의 카드/메트릭 UI 스타일 적용
- .env 및 st.secrets를 통한 Gemini API 키 자동 로딩
- 1, 2단계 병렬 처리(ThreadPoolExecutor) 및 3단계 확률 조정
"""

import os
import json
import re
import time
import concurrent.futures
from datetime import date

import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="Multi-Agent AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Agent 피보나치 & 매크로 종합 분석 시스템")
st.caption("독립된 기술적 분석과 매크로 수급 분석을 병렬 처리 후 AI 중재자가 최종 시나리오를 조정합니다.")

# ==========================================
# 2. 공통 유틸리티 (User-Agent 수집, 캐싱, 모델 탐색 & JSON 파서)
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data_cached(ticker: str, period: str = "1y", max_retries: int = 3) -> pd.DataFrame:
    """
    User-Agent 헤더 추가, 10분 캐싱 및 재시도(Retry)가 적용된 주가 수집 함수
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })

    delay = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period=period)
            if not df.empty:
                return df
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"데이터 수집 실패 ({max_retries}회 시도 초과): {e}")
        
        time.sleep(delay * (attempt + 1))
        
    raise Exception(f"'{ticker}' 티커의 데이터를 불러올 수 없습니다. 티커명을 확인해 주세요.")


def get_dynamic_flash_models(api_key: str) -> list:
    """Gemini API에서 이용 가능한 최신 Flash 모델 목록을 동적으로 탐색합니다."""
    genai.configure(api_key=api_key)
    try:
        all_models = genai.list_models()
        candidate_models = []
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                candidate_models.append(m.name.replace('models/', ''))
        
        candidate_models = sorted(list(set(candidate_models)), reverse=True)
        return candidate_models if candidate_models else ['gemini-2.5-flash', 'gemini-1.5-flash']
    except Exception as e:
        st.warning(f"모델 실시간 탐색 실패, 백업 모델 사용: {e}")
        return ['gemini-2.5-flash', 'gemini-1.5-flash']


def _parse_json_robust(text: str) -> dict:
    """3단계 방어적 JSON 파서"""
    if not text:
        return {"error": "응답 텍스트가 비어있습니다.", "overall_macro_judgment": "neutral"}

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "parse_error": True,
        "raw_text": text,
        "overall_macro_judgment": "neutral",
        "overall_macro_reasoning": "JSON 파싱 예외 발생으로 기본값 적용"
    }


def guess_asset_type(symbol: str) -> str:
    """티커 포맷으로 주식/암호화폐 구별"""
    s = symbol.upper().strip()
    return "암호화폐" if s.endswith("-USD") or s.startswith("BTC") or "KRW-" in s else "주식"

# ==========================================
# 3. 에이전트 프롬프트 및 실행 함수
# ==========================================

STAGE1_PROMPT_TEMPLATE = """너는 피보나치 되돌림 및 기술적 차트 분석 전문가다.
매크로, 뉴스, 거시경제 지표는 완전히 배제하고 오직 제공된 주가/기술 데이터만으로 분석하라.

분석 대상: {symbol} ({asset_type})
최근 Price Data Summary:
{data_summary}

반드시 아래 형식과 규칙을 포함하여 리포트를 작성하라:
1. 현재 주요 추세 및 피보나치 레벨 분석
2. 시나리오 A (상승/반등) 및 시나리오 B (하강/조정) 제시
3. 반드시 리포트 하단에 정량 확률 수치를 아래 명확한 양식 그대로 표기할 것:
   - [시나리오 A 확률: XX%]
   - [시나리오 B 확률: XX%]
"""

def run_stage1(api_key: str, symbol: str, data_summary: str) -> str:
    genai.configure(api_key=api_key)
    candidate_models = get_dynamic_flash_models(api_key)
    asset_type = guess_asset_type(symbol)
    prompt = STAGE1_PROMPT_TEMPLATE.format(
        symbol=symbol, asset_type=asset_type, data_summary=data_summary
    )

    for model_name in candidate_models:# -*- coding: utf-8 -*-
"""
app.py - 피보나치 & 매크로 Multi-Agent 종합 분석 시스템
- 한글 종목명 지원 (테슬라, 삼성전자, 비트코인 등 -> Ticker 매핑)
- 대표 지수 차트(S&P 500, KOSPI) 탭 추가
- User-Agent 적용 및 10분 캐싱(@st.cache_data)으로 yfinance Rate Limit 방지
- 차트 시각화(종가/거래량) 및 기간 옵션 확장(1m ~ max)
- 2단계 매크로 분석 결과의 카드/메트릭 UI 적용
"""

import os
import json
import re
import time
import concurrent.futures
from datetime import date

import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ==========================================
# 1. 한글 종목명 -> 티커 변환 맵
# ==========================================
KOREAN_TICKER_MAP = {
    # 대표 미국 주식
    "테슬라": "TSLA",
    "엔비디아": "NVDA",
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "아마존": "AMZN",
    "메타": "META",
    "아이온큐": "IONQ",
    "팔란티어": "PLTR",
    "비아디": "BYDDY",
    
    # 대표 한국 주식 (국내 주식은 .KS 또는 .KQ가 붙음)
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "sk하이닉스": "000660.KS",
    "현대차": "005380.KS",
    "현대자동차": "005380.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "LG에너지솔루션": "373220.KS",
    "lg에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "포스코홀딩스": "005490.KS",
    "POSCO홀딩스": "005490.KS",
    "에코프로": "086520.KQ",
    "에코프로비엠": "247540.KQ",

    # 대표 암호화폐 및 지수
    "비트코인": "BTC-USD",
    "이더리움": "ETH-USD",
    "리플": "XRP-USD",
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "나스닥": "^IXIC",
    "S&P500": "^GSPC",
    "에스엔피500": "^GSPC"
}

def resolve_ticker(input_text: str) -> str:
    """한글 이름 및 소문자 입력값을 yfinance 티커 표준 포맷으로 변환"""
    clean_text = input_text.strip()
    # 맵에 직접 존재하는 경우
    if clean_text in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[clean_text]
    
    # 맵의 키를 소문자 제거 후 단순 비교
    for k, v in KOREAN_TICKER_MAP.items():
        if k.lower() == clean_text.lower():
            return v
            
    # 그 외 영어 티커는 대문자로 반환 (예: nvda -> NVDA)
    return clean_text.upper()

# ==========================================
# 2. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="Multi-Agent AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Agent 피보나치 & 매크로 종합 분석 시스템")
st.caption("독립된 기술적 분석과 매크로 수급 분석을 병렬 처리 후 AI 중재자가 최종 시나리오를 조정합니다.")

# ==========================================
# 3. 공통 유틸리티 (yfinance 캐싱, 모델 탐색 & JSON 파서)
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data_cached(ticker: str, period: str = "1y", max_retries: int = 3) -> pd.DataFrame:
    """User-Agent 헤더 추가, 10분 캐싱 및 재시도가 적용된 데이터 수집 함수"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })

    delay = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(period=period)
            if not df.empty:
                return df
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"데이터 수집 실패 ({max_retries}회 시도 초과): {e}")
        
        time.sleep(delay * (attempt + 1))
        
    raise Exception(f"'{ticker}' 티커의 데이터를 불러올 수 없습니다. 티커명이나 한글명을 확인해 주세요.")


def get_dynamic_flash_models(api_key: str) -> list:
    """Gemini API에서 이용 가능한 최신 Flash 모델 목록을 탐색합니다."""
    genai.configure(api_key=api_key)
    try:
        all_models = genai.list_models()
        candidate_models = []
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name.lower():
                candidate_models.append(m.name.replace('models/', ''))
        
        candidate_models = sorted(list(set(candidate_models)), reverse=True)
        return candidate_models if candidate_models else ['gemini-2.5-flash', 'gemini-1.5-flash']
    except Exception as e:
        st.warning(f"모델 실시간 탐색 실패, 백업 모델 사용: {e}")
        return ['gemini-2.5-flash', 'gemini-1.5-flash']


def _parse_json_robust(text: str) -> dict:
    """방어적 JSON 파서"""
    if not text:
        return {"error": "응답 텍스트가 비어있습니다.", "overall_macro_judgment": "neutral"}

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "parse_error": True,
        "raw_text": text,
        "overall_macro_judgment": "neutral",
        "overall_macro_reasoning": "JSON 파싱 예외 발생으로 기본값 적용"
    }


def guess_asset_type(symbol: str) -> str:
    """티커 포맷으로 주식/암호화폐 구별"""
    s = symbol.upper().strip()
    return "암호화폐" if s.endswith("-USD") or s.startswith("BTC") or "KRW-" in s else "주식"

# ==========================================
# 4. 에이전트 프롬프트 및 실행 함수
# ==========================================

STAGE1_PROMPT_TEMPLATE = """너는 피보나치 되돌림 및 기술적 차트 분석 전문가다.
매크로, 뉴스, 거시경제 지표는 완전히 배제하고 오직 제공된 주가/기술 데이터만으로 분석하라.

분석 대상: {symbol} ({asset_type})
최근 Price Data Summary:
{data_summary}

반드시 아래 형식과 규칙을 포함하여 리포트를 작성하라:
1. 현재 주요 추세 및 피보나치 레벨 분석
2. 시나리오 A (상승/반등) 및 시나리오 B (하강/조정) 제시
3. 반드시 리포트 하단에 정량 확률 수치를 아래 명확한 양식 그대로 표기할 것:
   - [시나리오 A 확률: XX%]
   - [시나리오 B 확률: XX%]
"""

def run_stage1(api_key: str, symbol: str, data_summary: str) -> str:
    genai.configure(api_key=api_key)
    candidate_models = get_dynamic_flash_models(api_key)
    asset_type = guess_asset_type(symbol)
    prompt = STAGE1_PROMPT_TEMPLATE.format(
        symbol=symbol, asset_type=asset_type, data_summary=data_summary
    )

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(prompt)
            if res and hasattr(res, 'text') and res.text:
                return res.text
        except Exception:
            continue
    return "1단계 기술적 분석 응답 생성 실패"


STAGE2_PROMPT_TEMPLATE = """너는 거시환경·수급·뉴스만 전문적으로 평가하는 매크로 애널리스트다.
기술적 차트(지지/저항, 피보나치 등)는 완전히 배제하고, 순수 수급 및 뉴스로만 판단하라.

분석 대상: {symbol} ({asset_type})
오늘 날짜: {today}

최신 데이터/뉴스를 검색·참고하여 아래 JSON 구조로만 출력하라. 마크다운 설명 금지.

{{
  "supply_demand": {{
    "volume_spike_detected": boolean,
    "judgment": "accumulation" | "distribution" | "neutral",
    "reasoning": string
  }},
  "macro": {{
    "rate_environment": string,
    "fed_stance": "hawkish" | "dovish" | "neutral",
    "judgment": "favorable" | "neutral" | "unfavorable"
  }},
  "news_events": [
    {{ "headline": string, "type": "noise" | "structural", "summary": string }}
  ],
  "overall_macro_judgment": "favorable" | "neutral" | "unfavorable",
  "overall_macro_reasoning": string
}}
"""

def run_stage2(api_key: str, symbol: str) -> dict:
    genai.configure(api_key=api_key)
    candidate_models = get_dynamic_flash_models(api_key)
    asset_type = guess_asset_type(symbol)
    prompt = STAGE2_PROMPT_TEMPLATE.format(
        symbol=symbol, asset_type=asset_type, today=date.today().isoformat()
    )

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(prompt)
            if res and hasattr(res, 'text') and res.text:
                return _parse_json_robust(res.text)
        except Exception:
            continue
    return _parse_json_robust("")


STAGE3_PROMPT_TEMPLATE = """너는 독립된 ①기술적 분석 리포트와 ②매크로·수급 분석 JSON을 종합 조정하는 중재자다.
①, ②의 원본 데이터를 다시 재해석하지 말고, 규칙에 맞춰 시나리오 확률만 최종 조정하라.

조정 규칙:
- 매크로가 favorable: 시나리오 A 확률 상향 조정
- 매크로가 neutral: 원래 확률 유지
- 매크로가 unfavorable: 시나리오 A 확률 하향 조정 및 리스크 방어 강화

[① 기술적 분석 결과]
{stage1_result}

[② 매크로·수급 분석 결과]
{stage2_result}

반드시 아래 JSON 구조로만 응답하라.
{{
  "scenario_a": {{
    "adjusted_probability_pct": number,
    "reasoning": string
  }},
  "scenario_b": {{
    "adjusted_probability_pct": number,
    "reasoning": string
  }},
  "risk_management_notes": [string],
  "final_recommendation": string
}}
"""

def run_stage3(api_key: str, stage1_text: str, stage2_dict: dict) -> dict:
    genai.configure(api_key=api_key)
    candidate_models = get_dynamic_flash_models(api_key)
    prompt = STAGE3_PROMPT_TEMPLATE.format(
        stage1_result=stage1_text,
        stage2_result=json.dumps(stage2_dict, ensure_ascii=False, indent=2)
    )

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(prompt)
            if res and hasattr(res, 'text') and res.text:
                return _parse_json_robust(res.text)
        except Exception:
            continue
    return _parse_json_robust("")

# ==========================================
# 5. Streamlit 사이드바
# ==========================================
default_api_key = os.getenv("GEMINI_API_KEY", "")
if not default_api_key and "GEMINI_API_KEY" in st.secrets:
    default_api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", value=default_api_key, type="password")
    
    if api_key:
        st.caption("✅ API Key가 설정되었습니다.")
    else:
        st.caption("⚠️ `.env` 파일에 GEMINI_API_KEY를 등록하면 자동 입력됩니다.")
        
    # 한글 지원 검색창
    user_input_symbol = st.text_input("종목명 / 티커 (예: 테슬라, 삼성전자, NVDA, BTC-USD)", value="테슬라")
    
    # 입력된 종목명을 표준 티커로 자동 변환
    symbol = resolve_ticker(user_input_symbol)
    
    if user_input_symbol != symbol:
        st.caption(f"🔍 변환된 티커: **{symbol}**")

    period = st.selectbox(
        "차트 조회 기간", 
        ["1m", "3m", "6m", "1y", "2y", "5y", "max"], 
        index=3  # 기본값: 1y
    )
    
    run_btn = st.button("🚀 종합 분석 실행", use_container_width=True)

# ==========================================
# 6. 메인 분석 파이프라인
# ==========================================
if run_btn:
    if not api_key:
        st.error("Gemini API Key를 입력하거나 .env 파일에 GEMINI_API_KEY를 설정해주세요.")
        st.stop()

    st.info(f"🔍 **{user_input_symbol}** (`{symbol}`) ({period} 기간) 데이터를 수집하고 분석을 시작합니다...")

    # 1. 대상 종목 데이터 수집
    try:
        df = fetch_stock_data_cached(symbol, period=period)
        recent_close = df['Close'].iloc[-1]
        high_val = df['High'].max()
        low_val = df['Low'].min()
        data_summary = f"""- 최근 종가: {recent_close:.2f}
- 선택 기간 내 최고가: {high_val:.2f}
- 선택 기간 내 최저가: {low_val:.2f}
- 최근 5일 거래량 평균: {df['Volume'].tail(5).mean():.0f}
"""
    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        st.stop()

    # 2. 병렬 AI 분석 실행
    with st.spinner("⚡ 1단계(기술적 분석) 및 2단계(매크로 수급) 동시 분석 중..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_stage1 = executor.submit(run_stage1, api_key, symbol, data_summary)
            future_stage2 = executor.submit(run_stage2, api_key, symbol)

            stage1_result_text = future_stage1.result()
            stage2_result_json = future_stage2.result()

    # 3. 3단계 종합 판단 실행
    with st.spinner("🤖 3단계 종합 중재 에이전트가 최종 확률을 조정 중..."):
        stage3_result_json = run_stage3(api_key, stage1_result_text, stage2_result_json)

    # ==========================================
    # 7. 결과 출력 UI (지수 차트 탭 포함)
    # ==========================================
    st.success("✅ 분석 완료!")

    # 3단계 핵심 메트릭 카드
    st.subheader("🎯 3단계: 최종 조정 시나리오 (AI 중재 결과)")
    col1, col2 = st.columns(2)
    with col1:
        sc_a = stage3_result_json.get("scenario_a", {})
        st.metric(
            label="📈 시나리오 A (상승/반등 조정 확률)",
            value=f"{sc_a.get('adjusted_probability_pct', 'N/A')}%"
        )
        st.write(f"**근거:** {sc_a.get('reasoning', '-')}")

    with col2:
        sc_b = stage3_result_json.get("scenario_b", {})
        st.metric(
            label="📉 시나리오 B (하락/조정 조정 확률)",
            value=f"{sc_b.get('adjusted_probability_pct', 'N/A')}%"
        )
        st.write(f"**근거:** {sc_b.get('reasoning', '-')}")

    st.markdown(f"**💡 최종 권고사항:** {stage3_result_json.get('final_recommendation', '-')}")
    
    with st.expander("🛡️ 리스크 관리 가이드라인"):
        notes = stage3_result_json.get("risk_management_notes", [])
        for note in notes:
            st.write(f"- {note}")

    st.divider()

    # 탭 구성: 종목 차트, 시장 지수 차트, 1단계, 2단계
    tab_chart, tab_index, tab1, tab2 = st.tabs([
        f"📉 {user_input_symbol} 차트", 
        "🌐 주요 지수 흐름 (S&P500/KOSPI)", 
        "📊 1단계: 독립 기술적 분석", 
        "🌐 2단계: 매크로 · 수급 분석"
    ])

    # 탭 1: 검색한 종목 차트
    with tab_chart:
        st.subheader(f"{user_input_symbol} ({symbol}) 종가 추이 ({period})")
        st.line_chart(df['Close'], use_container_width=True)
        st.caption("📊 거래량 (Volume)")
        st.bar_chart(df['Volume'], use_container_width=True)

    # 탭 2: 대표 시장 지수 차트 (S&P 500, KOSPI)
    with tab_index:
        st.subheader(f"🌐 대표 시장 지수 추이 ({period})")
        idx_col1, idx_col2 = st.columns(2)
        
        with idx_col1:
            st.markdown("#### 🇺🇸 S&P 500 (`^GSPC`)")
            try:
                sp500_df = fetch_stock_data_cached("^GSPC", period=period)
                st.line_chart(sp500_df['Close'], use_container_width=True)
            except Exception as e:
                st.caption(f"S&P 500 데이터 로딩 실패: {e}")

        with idx_col2:
            st.markdown("#### 🇰🇷 KOSPI (`^KS11`)")
            try:
                kospi_df = fetch_stock_data_cached("^KS11", period=period)
                st.line_chart(kospi_df['Close'], use_container_width=True)
            except Exception as e:
                st.caption(f"KOSPI 데이터 로딩 실패: {e}")

    # 탭 3: 1단계 기술적 분석
    with tab1:
        st.markdown(stage1_result_text)

    # 탭 4: 2단계 매크로 분석
    with tab2:
        st.subheader("🌐 거시경제(Macro) 및 수급 동향 상세 분석")
        
        overall_j = str(stage2_result_json.get("overall_macro_judgment", "neutral")).lower()
        overall_reason = stage2_result_json.get("overall_macro_reasoning", "분석 정보 없음")
        
        j_map = {
            "favorable": ("🟢 우호적 (Favorable)", st.success),
            "neutral": ("🟡 중립 (Neutral)", st.warning),
            "unfavorable": ("🔴 비우호적 (Unfavorable)", st.error)
        }
        text, alert_func = j_map.get(overall_j, ("⚪ 판단 불가", st.info))
        
        st.markdown("#### 🎯 종합 매크로 판단")
        alert_func(f"**{text}**\n\n{overall_reason}")
            
        st.divider()

        col_sd, col_mc = st.columns(2)

        with col_sd:
            st.markdown("### 📊 수급 동향 (Supply & Demand)")
            sd = stage2_result_json.get("supply_demand", {})
            
            sd_j = str(sd.get("judgment", "neutral")).lower()
            sd_map = {
                "accumulation": "🟢 매집 (Accumulation)", 
                "distribution": "🔴 분매 (Distribution)", 
                "neutral": "🟡 중립 (Neutral)"
            }
            
            spike = sd.get("volume_spike_detected", False)
            spike_badge = "🚨 거래량 폭증 감지" if spike else "⚪ 거래량 정상"
            
            st.metric(label="수급 상태", value=sd_map.get(sd_j, sd_j))
            st.caption(f"**거래량 패턴:** {spike_badge}")
            st.write(f"**수급 근거:** {sd.get('reasoning', '-')}")

        with col_mc:
            st.markdown("### 🏦 거시 환경 (Macro Economy)")
            mc = stage2_result_json.get("macro", {})
            
            fed_s = str(mc.get("fed_stance", "neutral")).lower()
            fed_map = {
                "hawkish": "🦅 매파적 (Hawkish)", 
                "dovish": "🕊️ 비둘기파적 (Dovish)", 
                "neutral": "⚖️ 중립 (Neutral)"
            }
            
            st.metric(label="Fed 통화정책 기조", value=fed_map.get(fed_s, fed_s))
            st.markdown(f"**금리 환경:** {mc.get('rate_environment', '-')}")
            st.write(f"**매크로 평가:** {mc.get('judgment', '-')}")

        st.divider()

        st.markdown("### 📰 주요 뉴스 & 영향 평가")
        news_list = stage2_result_json.get("news_events", [])
        
        if news_list and isinstance(news_list, list):
            for item in news_list:
                headline = item.get("headline", "제목 없음")
                n_type = str(item.get("type", "noise")).lower()
                summary = item.get("summary", "-")
                
                tag = "📌 [구조적 변수]" if n_type == "structural" else "🌊 [단기 노이즈]"
                
                with st.expander(f"{tag} {headline}"):
                    st.write(summary)
        else:
            st.info("수집된 주요 뉴스 및 이벤트가 없습니다.")

        with st.expander("🔍 원본 Raw JSON 데이터 보기"):
            st.json(stage2_result_json)
