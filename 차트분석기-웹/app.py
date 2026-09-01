# -*- coding: utf-8 -*-
"""
app.py - 피보나치 & 매크로 Multi-Agent 종합 분석 시스템 (최종 통합 버전)

핵심 기능:
- 초장기 광기 구간(Mania/Secular) + 단기 변동성 구간(Tactical) 이중 피보나치
- 중첩 구간(Confluence Zone) 자동 탐지 및 가중치 부여
- 실제 기술적 지표 계산 후 LLM에 전달
- 3단계 Multi-Agent (기술적 → 매크로 → 중재)
- 야후 파이낸스 IP 차단 대응 강화 (캐시 + 재시도)
"""

import os
import json
import re
import time
import random
import concurrent.futures
from datetime import date
from typing import Optional, Dict, List

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 한글 종목명 / 지수 → 티커 맵
# ==========================================
KOREAN_TICKER_MAP = {
    "코스피": "^KS11", "코스닥": "^KQ11", "나스닥": "^IXIC",
    "S&P500": "^GSPC", "s&p500": "^GSPC", "에스엔피500": "^GSPC",
    "다우존스": "^DJI", "다우": "^DJI", "필라델피아반도체": "^SOX",
    "테슬라": "TSLA", "엔비디아": "NVDA", "애플": "AAPL",
    "마이크로소프트": "MSFT", "구글": "GOOGL", "알파벳": "GOOGL",
    "아마존": "AMZN", "메타": "META", "아이온큐": "IONQ", "팔란티어": "PLTR",
    "SQQQ": "SQQQ", "TQQQ": "TQQQ", "SOXL": "SOXL", "SPY": "SPY", "QQQ": "QQQ",
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "sk하이닉스": "000660.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS",
    "네이버": "035420.KS", "카카오": "035720.KS",
    "LG에너지솔루션": "373220.KS", "lg에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS", "포스코홀딩스": "005490.KS",
    "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ",
    "비트코인": "BTC-USD", "이더리움": "ETH-USD", "리플": "XRP-USD",
}

def resolve_ticker(input_text: str) -> str:
    clean = input_text.strip()
    if clean in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[clean]
    for k, v in KOREAN_TICKER_MAP.items():
        if k.lower() == clean.lower():
            return v
    return clean.upper()

def guess_asset_type(symbol: str) -> str:
    s = symbol.upper().strip()
    if s.startswith("^"):
        return "시장 지수(Index)"
    if s.endswith("-USD") or s.startswith("BTC") or "KRW-" in s:
        return "암호화폐"
    return "주식/ETF"

# ==========================================
# 2. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="Multi-Agent AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Agent 종합 분석 시스템")
st.caption("초장기 광기 구간 + 단기 변동성 멀티 타임프레임 피보나치 · 3단계 AI 분석")

# ==========================================
# 3. 데이터 수집 (야후 차단 대응)
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data_cached(ticker: str, period: str = "1y", max_retries: int = 4) -> pd.DataFrame:
    """야후 파이낸스 데이터 수집 (재시도 + 지수 백오프)"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=True)
            if not df.empty:
                return df
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["too many requests", "rate limit", "429"]):
                wait = (2 ** attempt) * 8 + random.uniform(3, 7)
            else:
                wait = 4 * (attempt + 1) + random.uniform(1, 3)
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise Exception(f"데이터 수집 실패 ({max_retries}회 시도): {e}")
    raise Exception(f"'{ticker}' 데이터를 불러올 수 없습니다.")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_snapshot() -> dict:
    """VIX / 미국 10년물 간단 스냅샷 (실패해도 빈 dict)"""
    result = {}
    for name, ticker in [("vix", "^VIX"), ("us10y", "^TNX")]:
        try:
            df = yf.Ticker(ticker).history(period="5d")
            if not df.empty and len(df) >= 2:
                result[name] = {
                    "last": float(df["Close"].iloc[-1]),
                    "change_pct": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
                }
        except Exception:
            pass
    return result

# ==========================================
# 4. 기술적 지표 + 멀티 타임프레임 피보나치
# ==========================================
FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

def calc_fib_levels(high: float, low: float, prefix: str = "") -> dict:
    if high <= low or high <= 0:
        return {}
    rng = high - low
    levels = {}
    for r in FIB_RATIOS:
        price = high - rng * r
        key = f"{prefix}{r:.3f}" if prefix else f"{r:.3f}"
        levels[key] = round(price, 4)
    return levels

def find_significant_swing(df: pd.DataFrame):
    """초장기 광기 구간의 대표 고점/저점 탐지"""
    if df.empty or len(df) < 30:
        return None, None

    high_series = df["High"]
    low_series = df["Low"]

    mania_high = float(high_series.max())
    high_idx = high_series.idxmax()

    # 최고점 이전의 의미 있는 저점
    pre_high = low_series.loc[:high_idx]
    mania_low = float(pre_high.min()) if len(pre_high) > 10 else float(low_series.min())

    # 최고점 이후 더 큰 스윙이 있으면 보정
    post_high = low_series.loc[high_idx:]
    if len(post_high) > 20:
        post_low = float(post_high.min())
        if (mania_high - post_low) / mania_high > 0.30:
            if (mania_high - post_low) > (mania_high - mania_low):
                mania_low = post_low

    return mania_high, mania_low

def detect_confluence(mania_levels: dict, tactical_levels: dict, threshold_pct: float = 1.8) -> list:
    """두 타임프레임 레벨 중첩 구간 탐지"""
    confluence = []
    for m_key, m_price in mania_levels.items():
        for t_key, t_price in tactical_levels.items():
            if m_price <= 0:
                continue
            diff_pct = abs(m_price - t_price) / m_price * 100
            if diff_pct <= threshold_pct:
                confluence.append({
                    "price_zone": round((m_price + t_price) / 2, 4),
                    "mania_level": m_key,
                    "tactical_level": t_key,
                    "distance_pct": round(diff_pct, 2)
                })
    return sorted(confluence, key=lambda x: x["price_zone"])

def position_comment(price: float, high: float, low: float, label: str) -> str:
    if high <= low or high <= 0:
        return ""
    retrace = (high - price) / (high - low)
    if retrace < 0.236:
        return f"{label}: 매우 얕은 조정 (광기 잔여 에너지 강함)"
    elif retrace < 0.382:
        return f"{label}: 얕은 조정 구간 (23.6~38.2%) — 광기 에너지 잔존 가능"
    elif retrace < 0.5:
        return f"{label}: 중간 조정"
    elif retrace < 0.618:
        return f"{label}: 깊은 조정 진입"
    elif retrace < 0.786:
        return f"{label}: 골든 포켓(61.8%) 테스트 구간 — 핵심 지지/저항"
    else:
        return f"{label}: 강한 추세 훼손 구간 (78.6% 이하)"

def build_multi_timeframe_fib(df_long: pd.DataFrame, df_short: pd.DataFrame, current_price: float) -> dict:
    """초장기 광기 + 단기 변동성 + 중첩 분석"""
    mania_high, mania_low = find_significant_swing(df_long)
    mania_fib = calc_fib_levels(mania_high, mania_low, prefix="mania_") if mania_high and mania_low else {}

    tac_high = float(df_short["High"].max())
    tac_low = float(df_short["Low"].min())
    tactical_fib = calc_fib_levels(tac_high, tac_low, prefix="tac_")

    mania_clean = {k.replace("mania_", ""): v for k, v in mania_fib.items()}
    tac_clean = {k.replace("tac_", ""): v for k, v in tactical_fib.items()}
    confluence = detect_confluence(mania_clean, tac_clean, threshold_pct=1.8)

    return {
        "mania": {
            "high": round(mania_high, 4) if mania_high else None,
            "low": round(mania_low, 4) if mania_low else None,
            "fib_levels": mania_fib,
            "comment": position_comment(current_price, mania_high, mania_low, "초장기") if mania_high else ""
        },
        "tactical": {
            "high": round(tac_high, 4),
            "low": round(tac_low, 4),
            "fib_levels": tactical_fib,
            "comment": position_comment(current_price, tac_high, tac_low, "단기")
        },
        "confluence_zones": confluence,
        "current_price": round(current_price, 4)
    }

def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    """보조 기술적 지표"""
    if df.empty or len(df) < 5:
        return {"error": "데이터 부족"}

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None
    current = float(close.iloc[-1])

    sma = {}
    for w in [20, 50, 200]:
        if len(close) >= w:
            sma[f"sma_{w}"] = round(float(close.rolling(w).mean().iloc[-1]), 4)

    rsi_val = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None

    def pct_change(n):
        if len(close) > n:
            return round(float((close.iloc[-1] / close.iloc[-n - 1] - 1) * 100), 2)
        return None

    vol_info = {}
    if volume is not None and volume.dropna().sum() > 0:
        vol_ma20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        recent_vol = volume.tail(5).mean()
        vol_info = {
            "avg_5d": round(float(recent_vol), 0),
            "ratio_5d_vs_20d": round(float(recent_vol / vol_ma20), 2) if vol_ma20 and vol_ma20 > 0 else None
        }

    return {
        "current_price": round(current, 4),
        "sma": sma,
        "rsi_14": rsi_val,
        "perf_5d_pct": pct_change(5),
        "perf_20d_pct": pct_change(20),
        "perf_60d_pct": pct_change(60),
        "volume": vol_info,
    }

def build_fib_data_summary(mtf: dict, indicators: dict, symbol: str, asset_type: str) -> str:
    """1단계 에이전트용 멀티 타임프레임 요약"""
    lines = [
        f"분석 대상: {symbol} ({asset_type})",
        f"현재가: {mtf['current_price']}",
        "",
        "=== ① 초장기 광기 구간 (Secular / Mania Horizon) ===",
        f"광기 고점: {mtf['mania']['high']}",
        f"광기 저점: {mtf['mania']['low']}",
        f"해석: {mtf['mania']['comment']}",
        "피보나치 레벨:"
    ]
    for k, v in mtf["mania"]["fib_levels"].items():
        lines.append(f"  - {k.replace('mania_', '')}: {v}")

    lines += [
        "",
        "=== ② 단기 변동성 구간 (Tactical Horizon) ===",
        f"단기 고점: {mtf['tactical']['high']}",
        f"단기 저점: {mtf['tactical']['low']}",
        f"해석: {mtf['tactical']['comment']}",
        "피보나치 레벨:"
    ]
    for k, v in mtf["tactical"]["fib_levels"].items():
        lines.append(f"  - {k.replace('tac_', '')}: {v}")

    if mtf["confluence_zones"]:
        lines += ["", "=== ③ 중첩 구간 (Confluence Zones) — 가장 중요한 레벨 ==="]
        for z in mtf["confluence_zones"]:
            lines.append(
                f"  ★ {z['price_zone']}  (초장기 {z['mania_level']} + 단기 {z['tactical_level']}, 거리 {z['distance_pct']}%)"
            )
    else:
        lines += ["", "=== ③ 중첩 구간: 현재 뚜렷한 중첩 없음 ==="]

    lines += [
        "",
        "=== 보조 기술적 지표 ===",
        f"RSI(14): {indicators.get('rsi_14')}",
        f"5일 수익률: {indicators.get('perf_5d_pct')}%",
        f"20일 수익률: {indicators.get('perf_20d_pct')}%",
    ]
    for k, v in indicators.get("sma", {}).items():
        lines.append(f"{k.upper()}: {v}")

    vol = indicators.get("volume", {})
    if vol.get("ratio_5d_vs_20d"):
        lines.append(f"거래량 5일/20일 비율: {vol['ratio_5d_vs_20d']}")

    return "\n".join(lines)

# ==========================================
# 5. Gemini 유틸
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_dynamic_flash_models(api_key: str) -> list:
    genai.configure(api_key=api_key)
    try:
        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods and "flash" in m.name.lower():
                models.append(m.name.replace("models/", ""))
        models = sorted(set(models), reverse=True)
        return models if models else ["gemini-2.0-flash", "gemini-1.5-flash"]
    except Exception:
        return ["gemini-2.0-flash", "gemini-1.5-flash"]

def _parse_json_robust(text: str) -> dict:
    if not text:
        return {"error": "응답 텍스트가 비어있습니다.", "overall_macro_judgment": "neutral"}

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "parse_error": True,
        "raw_text": text[:800],
        "overall_macro_judgment": "neutral",
        "overall_macro_reasoning": "JSON 파싱 예외로 기본값 적용"
    }

def call_gemini(api_key: str, prompt: str, models: list) -> Optional[str]:
    genai.configure(api_key=api_key)
    for name in models:
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            if res and hasattr(res, "text") and res.text:
                return res.text
        except Exception:
            continue
    return None

# ==========================================
# 6. 에이전트 프롬프트 & 실행
# ==========================================
STAGE1_PROMPT = """너는 멀티 타임프레임 피보나치 분석 전문가다.
초장기 광기 구간(Mania/Secular Horizon)과 단기 변동성 구간(Tactical Horizon)을 결합한 이중 구조 분석을 수행한다.

아래에 **이미 계산된 실제 수치**만 사용해라. 숫자를 임의로 바꾸거나 무시하지 마라.

{data_summary}

분석 시 반드시 지켜야 할 원칙:

1. 초장기 광기 구간의 피보나치 레벨은 시장의 '심리적·구조적 뼈대'다. 
   기관·세력이 장기 포지션을 조절할 때 참고하는 거시 이정표로 해석하라.

2. 단기 변동성 구간 레벨은 실전 매매·진입 타이밍용이다.

3. 두 타임프레임 레벨이 겹치는 **중첩 구간(Confluence Zone)**이 있으면 
   그 가격대를 '절대 깨지면 안 되는 요새' 또는 '폭발적 반등 가능 변곡점'으로 가중치를 크게 부여하라.

4. 광기 해석:
   - 23.6%~38.2% 사이의 얕은 조정 → 광기 잔여 에너지가 아직 강함
   - 61.8% 골든 포켓 → 가장 중요한 지지/저항 테스트 구간
   - 78.6% 이탈 → 추세 구조 훼손 가능성 높음

5. 시나리오 작성:
   - 시나리오 A (상승/반등): 주요 지지(특히 중첩 구간·골든 포켓)를 지키는 경우의 목표가와 조건
   - 시나리오 B (하락/조정): 핵심 지지 이탈 시 패닉 매도 가능성과 다음 지지 레벨

6. 리포트 하단에 반드시 아래 형식으로 확률을 명시하라.
   - [시나리오 A 확률: XX%]
   - [시나리오 B 확률: XX%]

확률은 중첩 구간의 강도, 현재가 위치, 광기 잔여 에너지를 종합해 합리적으로 배분하고 합이 100%에 가깝게 맞춰라.
"""

def run_stage1(api_key: str, data_summary: str, models: list) -> str:
    prompt = STAGE1_PROMPT.format(data_summary=data_summary)
    text = call_gemini(api_key, prompt, models)
    return text or "1단계 기술적 분석 응답 생성 실패"

STAGE2_PROMPT = """너는 거시환경·수급을 전문적으로 평가하는 매크로 애널리스트다.
기술적 차트 해석은 하지 말고, 아래 제공된 데이터와 일반적인 매크로 지식을 바탕으로 판단하라.

분석 대상: {symbol} ({asset_type})
오늘 날짜: {today}

[제공된 시장 데이터]
{market_context}

반드시 아래 JSON만 출력하라. 마크다운 금지.

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

def run_stage2(api_key: str, symbol: str, asset_type: str, market_context: str, models: list) -> dict:
    prompt = STAGE2_PROMPT.format(
        symbol=symbol,
        asset_type=asset_type,
        today=date.today().isoformat(),
        market_context=market_context
    )
    text = call_gemini(api_key, prompt, models)
    return _parse_json_robust(text or "")

STAGE3_PROMPT = """너는 ①기술적 분석 리포트와 ②매크로 분석 JSON을 종합하는 중재자다.
원본 데이터를 다시 해석하지 말고, 아래 규칙에 따라 확률만 최종 조정하라.

조정 규칙:
- 매크로 favorable → 시나리오 A 확률 상향
- 매크로 neutral → 기존 확률 유지
- 매크로 unfavorable → 시나리오 A 확률 하향 + 리스크 강조

[① 기술적 분석]
{stage1_result}

[② 매크로 분석]
{stage2_result}

반드시 아래 JSON만 출력하라.

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

def run_stage3(api_key: str, stage1_text: str, stage2_dict: dict, models: list) -> dict:
    prompt = STAGE3_PROMPT.format(
        stage1_result=stage1_text,
        stage2_result=json.dumps(stage2_dict, ensure_ascii=False, indent=2)
    )
    text = call_gemini(api_key, prompt, models)
    return _parse_json_robust(text or "")

# ==========================================
# 7. 사이드바
# ==========================================
default_api_key = os.getenv("GEMINI_API_KEY", "")
if not default_api_key and "GEMINI_API_KEY" in st.secrets:
    default_api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.header("⚙️ 분석 설정")
    api_key = st.text_input("Gemini API Key", value=default_api_key, type="password")
    if api_key:
        st.caption("✅ API Key 설정됨")
    else:
        st.caption("⚠️ GEMINI_API_KEY를 설정하세요.")

    user_input_symbol = st.text_input(
        "검색 대상 (주식 / 지수 / 코인)",
        value="S&P500",
        help="예: S&P500, 코스피, 테슬라, 삼성전자, BTC-USD"
    )
    symbol = resolve_ticker(user_input_symbol)
    asset_type = guess_asset_type(symbol)
    st.caption(f"📌 감지된 티커: **{symbol}** ({asset_type})")

    period = st.selectbox(
        "단기 분석 기간 (Tactical)",
        ["3m", "6m", "1y", "2y"],
        index=2
    )
    run_btn = st.button("🚀 AI 분석 실행", use_container_width=True)

# ==========================================
# 8. 메인 파이프라인
# ==========================================
if run_btn:
    if not api_key:
        st.error("Gemini API Key를 입력해주세요.")
        st.stop()

    st.info(f"🔍 **{user_input_symbol}** (`{symbol}`) 데이터 수집 및 멀티 타임프레임 분석 중...")

    # 1) 단기 + 초장기 데이터
    try:
        df_short = fetch_stock_data_cached(symbol, period=period)
        # 초장기는 max 대신 5y로 제한해 야후 부하와 차단 확률을 낮춤
        df_long = fetch_stock_data_cached(symbol, period="5y")
    except Exception as e:
        st.error(f"데이터 수집 실패: {e}")
        st.stop()

    current_price = float(df_short["Close"].iloc[-1])
    indicators = calculate_technical_indicators(df_short)
    if "error" in indicators:
        st.error(indicators["error"])
        st.stop()

    mtf = build_multi_timeframe_fib(df_long, df_short, current_price)
    data_summary = build_fib_data_summary(mtf, indicators, symbol, asset_type)

    # 2) 매크로 컨텍스트
    macro_snap = fetch_macro_snapshot()
    market_context_parts = [
        f"최근 5일 수익률: {indicators.get('perf_5d_pct')}%",
        f"최근 20일 수익률: {indicators.get('perf_20d_pct')}%",
        f"RSI(14): {indicators.get('rsi_14')}",
    ]
    vol = indicators.get("volume", {})
    if vol.get("ratio_5d_vs_20d"):
        market_context_parts.append(f"거래량 5일/20일 비율: {vol['ratio_5d_vs_20d']}")
    if macro_snap.get("vix"):
        market_context_parts.append(
            f"VIX: {macro_snap['vix']['last']:.2f} (5일 변화 {macro_snap['vix']['change_pct']:.1f}%)"
        )
    if macro_snap.get("us10y"):
        market_context_parts.append(
            f"미국 10년물 금리: {macro_snap['us10y']['last']:.2f}% (5일 변화 {macro_snap['us10y']['change_pct']:.1f}%)"
        )
    market_context = "\n".join(market_context_parts)

    models = get_dynamic_flash_models(api_key)

    # 3) 1·2단계 병렬
    with st.spinner("⚡ 1단계(멀티 타임프레임 피보나치) & 2단계(매크로) 동시 분석 중..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(run_stage1, api_key, data_summary, models)
            f2 = executor.submit(run_stage2, api_key, symbol, asset_type, market_context, models)
            stage1_text = f1.result()
            stage2_json = f2.result()

    # 4) 3단계
    with st.spinner("🤖 3단계 종합 중재 중..."):
        stage3_json = run_stage3(api_key, stage1_text, stage2_json, models)

    # ==========================================
    # 9. 결과 UI
    # ==========================================
    st.success(f"✅ [{user_input_symbol}] 분석 완료")

    # 계산된 핵심 수치
    with st.expander("📐 멀티 타임프레임 피보나치 & 지표 (LLM에 전달된 수치)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"{mtf['current_price']}")
        c2.metric("RSI(14)", f"{indicators.get('rsi_14', 'N/A')}")
        c3.metric("5일 수익률", f"{indicators.get('perf_5d_pct', 'N/A')}%")
        c4.metric("20일 수익률", f"{indicators.get('perf_20d_pct', 'N/A')}%")

        st.markdown("**초장기 광기 구간**")
        st.write(mtf["mania"]["comment"])
        if mtf["mania"]["fib_levels"]:
            st.dataframe(
                pd.DataFrame([{"레벨": k.replace("mania_", ""), "가격": v} for k, v in mtf["mania"]["fib_levels"].items()]),
                use_container_width=True, hide_index=True
            )

        st.markdown("**단기 변동성 구간**")
        st.write(mtf["tactical"]["comment"])
        if mtf["tactical"]["fib_levels"]:
            st.dataframe(
                pd.DataFrame([{"레벨": k.replace("tac_", ""), "가격": v} for k, v in mtf["tactical"]["fib_levels"].items()]),
                use_container_width=True, hide_index=True
            )

        if mtf["confluence_zones"]:
            st.markdown("**★ 중첩 구간 (Confluence)**")
            st.dataframe(pd.DataFrame(mtf["confluence_zones"]), use_container_width=True, hide_index=True)

    # 최종 시나리오
    st.subheader(f"🎯 최종 AI 시나리오 — {user_input_symbol}")
    col1, col2 = st.columns(2)
    with col1:
        sc_a = stage3_json.get("scenario_a", {})
        st.metric("📈 시나리오 A (상승/반등)", f"{sc_a.get('adjusted_probability_pct', 'N/A')}%")
        st.write(f"**근거:** {sc_a.get('reasoning', '-')}")
    with col2:
        sc_b = stage3_json.get("scenario_b", {})
        st.metric("📉 시나리오 B (하락/조정)", f"{sc_b.get('adjusted_probability_pct', 'N/A')}%")
        st.write(f"**근거:** {sc_b.get('reasoning', '-')}")

    st.markdown(f"**💡 최종 권고:** {stage3_json.get('final_recommendation', '-')}")

    with st.expander("🛡️ 리스크 관리 가이드"):
        for note in stage3_json.get("risk_management_notes", []):
            st.write(f"- {note}")

    st.divider()

    tab_chart, tab1, tab2 = st.tabs([
        f"📉 {user_input_symbol} 차트",
        "📊 1단계: 멀티 타임프레임 피보나치",
        "🌐 2단계: 매크로·수급"
    ])

    with tab_chart:
        st.subheader(f"{user_input_symbol} (`{symbol}`) — 단기 {period} / 초장기 5y")
        st.line_chart(df_short["Close"], use_container_width=True)
        if "Volume" in df_short.columns and df_short["Volume"].sum() > 0:
            st.caption("거래량")
            st.bar_chart(df_short["Volume"], use_container_width=True)

    with tab1:
        st.markdown(stage1_text)

    with tab2:
        overall_j = str(stage2_json.get("overall_macro_judgment", "neutral")).lower()
        overall_reason = stage2_json.get("overall_macro_reasoning", "-")
        j_map = {
            "favorable": ("🟢 우호적", st.success),
            "neutral": ("🟡 중립", st.warning),
            "unfavorable": ("🔴 비우호적", st.error),
        }
        text, alert = j_map.get(overall_j, ("⚪ 판단 불가", st.info))
        st.markdown("#### 종합 매크로 판단")
        alert(f"**{text}**\n\n{overall_reason}")

        col_sd, col_mc = st.columns(2)
        with col_sd:
            st.markdown("### 수급")
            sd = stage2_json.get("supply_demand", {})
            st.metric("판단", sd.get("judgment", "-"))
            st.write(sd.get("reasoning", "-"))
        with col_mc:
            st.markdown("### 거시 환경")
            mc = stage2_json.get("macro", {})
            st.metric("Fed 기조", mc.get("fed_stance", "-"))
            st.write(f"금리 환경: {mc.get('rate_environment', '-')}")

        st.markdown("### 관련 이슈")
        for item in stage2_json.get("news_events", []) or []:
            tag = "📌 구조적" if item.get("type") == "structural" else "🌊 단기"
            with st.expander(f"{tag} | {item.get('headline', '-')}"):
                st.write(item.get("summary", "-"))

        with st.expander("Raw JSON"):
            st.json(stage2_json)