# -*- coding: utf-8 -*-
"""
research/event_inventory.py

📋 Breakout Alpha Study — Phase A: Event Inventory（事件盤點）

⚠️ 這支程式回答的問題只有一個：
    「過去（目前資料涵蓋範圍內），箱型突破（Box Breakout）事件到底有多少、
      分佈在哪些股票、有量確認/無量確認各占多少？」

它**不**計算 Forward Return、不做 CAR、不分 Bull/Bear regime 統計——那些是
Phase B/C/D 的工作，刻意留到看過這裏的結果之後才決定要不要做（Stop Rule）。

── 為什麼不能直接沿用 BreakoutEngine 現成的「即時判斷」邏輯 ──
`engines/breakout_engine.py` 的 `analyze_consolidation_box()` /
`analyze_breakout()` 是設計來回答「用到今天為止的資料，現在算不算在
突破狀態」，也就是只看「最新一天」。如果直接把這兩個函式套在一段長期
歷史資料上跑一次，只會得到「最後一天」的答案，看不到過去兩年內「到底
發生過幾次」突破。

這裡的做法是**逐日滾動**（walk-forward）：在每一個交易日 t，只用「t 那天
以前」的資料重新呼叫一次 BreakoutEngine 的函式（避免用到未來資料，也就是
避免 look-ahead bias），判斷「t 那天」是不是「第一次」從非突破狀態轉為
突破狀態（edge-triggered：只在 False→True 的那一天記一次事件，之後只要
`breakout_confirmed` 持續是 True，就不會重複計數，直到它先變回 False，
才有可能觸發下一次事件）。這樣可以確保 30 天的延續突破只算 1 個事件，
不是 30 個。

── 為什麼 Regime 只當 metadata，不當分組依據（本輪與使用者對齊的決定）──
`MarketRegimeEngine.classify()` 只依賴大盤指數（^TWII），跟個股資料無關，
所以可以先算出一份「日期 → regime」的時間序列（只需要對大盤跑一次逐日
滾動，不需要對每檔股票各跑一次）。這裡把這份 regime 時間序列直接標註在
每一筆事件上，但**不用它來分組統計**——要不要分組，等這支程式跑出真實
事件數之後再決定（事件數不夠多，分組只會製造假象的精確度）。

── Stop Rule（研究是否值得繼續的前置條件）──
本程式最後印出的 summary report 會明確標示：
  - 事件總數 < 100  → 建議停止，不要進入 Phase B（統計力不足）
  - 100 <= 事件總數 < 300 → 可以做，但結果只能當初步觀察，不能下結論
  - 事件總數 >= 300 → 統計力大致足夠，可以進入 Phase B
這個門檻是研究設計上的經驗法則，不是統計顯著性的正式檢定，僅供決定
「值不值得投入下一階段成本」，请勿當成任何形式的顯著性保證。

── 使用方式 ──
    python -m research.event_inventory --universe watchlist --limit 20
    python -m research.event_inventory --universe full --resume

⚠️ 這支程式需要對外部網路（yfinance / TWSE）發送請求，本開發沙盒環境
沒有對外網路權限，只完成了語法檢查與邏輯設計，並用合成資料做過自我測試
（見 research/test_event_inventory.py），**尚未在真實資料上實際執行過**。
請在你自己有網路權限的環境跑一次，如果 TWSE/yfinance 的欄位格式跟預期
不同，多半需要對照錯誤訊息微調。

⚠️ 效能提醒：全市場（約 1000~1700 檔上市櫃）× 約 500 個交易日 × 逐日
重算箱型的成本，保守估計是數十分鐘到超過一小時的等級（純 Python 迴圈，
不是網路請求延遲，網路延遲另計）。建議：
  1. 先用 --universe watchlist 或 --limit 20~30 檔跑一次，確認邏輯與
     資料格式沒問題、抓一下單檔平均耗時
  2. 全市場跑之前，確認 --resume（checkpoint）機制正常，避免中途失敗
     要從頭重來
  3. 股價資料本身走 DataEngine/DatabaseEngine 既有快取（6小時新鮮期），
     多次執行不會重複打 yfinance API，只有第一次真的需要下載

── 專業投資角度補強（這次新增，Phase A 範圍內、刻意只加「記錄」不加「分析」）──

這幾項補強的共同原則跟 Regime 一樣：**只當作事件的附加欄位記錄下來，
不用來做任何篩選或分組統計**——要不要用這些欄位切分析，等 Phase B 看過
樣本數之後再決定，這裡不搶先下結論。

1. **流動性標記（liquidity_level_at_event）**：直接呼叫既有的
   `RiskEngine.add_liquidity_metrics()`（沿用它的門檻：日均成交值
   <1,000萬🔴極低流動性 / <5,000萬🟡流動性偏低 / 其餘🟢正常），把突破
   當天的流動性等級記錄下來。這是本專案自己在多處文件反覆點名的風險：
   台股中小型股「線型漂亮但量能稀薄」的陷阱——一個技術面很乾淨的箱型
   突破，如果發生在極低流動性的股票上，實際上根本進不去出不來，這種
   事件不應該跟高流動性股票的突破被平等看待。這裡不是在 Inventory
   階段就排除低流動性事件（那是搶先下結論），而是先誠實記錄下來，
   讓 Phase B 可以事後決定要不要拆開看。

2. **突破幅度（breakout_magnitude_pct）**：收盤價超出箱頂的百分比。
   一檔股票剛好壓線收在箱頂 0.51%（buffer_pct 門檻剛好過）跟收在箱頂
   之上 8% 的乾脆度差很多，直覺上前者更接近雜訊觸發、後者更像真的
   有資金決心推動，但這只是「達人視角」的直覺，不是驗證過的結論——
   記錄下來，交給 Phase B 用資料檢驗這個直覺對不對。

3. **是否為 ETF（is_etf）**：透過 `StockDirectoryEngine` 對照表標記。
   ETF 的「突破」本質上是追蹤指數的技術現象，沒有個股基本面/籌碼面
   的資金動能可以解釋，跟一般個股的箱型突破可能不是同一件事——這個
   專案在 `canslim_engine.py` 已經有 ETF 特殊處理的先例（排除不適用
   的子項而非直接砍分），這裡延續同樣的誠實揭露精神，先標記，不代表
   自動排除。⚠️ 若 `stock_directory` 表尚未透過
   `StockDirectoryEngine.refresh_all()` 建立，這欄位會是 `None`
   （無法判斷），不是「不是 ETF」。

4. **同股票事件序號（event_seq_for_ticker）**：這是同一檔股票在樣本
   期間內的第幾次突破事件。達人視角常見的直覺是「壓力區被測試/突破
   越多次，這個價位的意義會被市場重新定價，第一次突破跟第三次突破
   的『真實度』可能不一樣（甚至可能反過來，多次測試後才是真突破）」，
   這同樣只是假設，記錄下來供 Phase B 檢驗，不預設答案。

5. **是否為漲停突破（is_limit_up_day，本輪新增）**：台股有 ±10% 漲跌停
   限制。如果突破當天同時是漲停(或貼近漲停)，代表這天很可能根本
   buy不到(漲停鎖死、委買排隊)，Phase B 若假設「事件當天收盤價可以
   進場」，對這類事件不成立；而且漲停鎖死當天的「突破」訊號品質可能
   跟一般技術面量價堆疊出來的突破不是同一件事(更可能是消息面/公告
   直接跳空鎖死)。這是跟 liquidity_level_at_event 同一類的 Execution
   Constraint，不是新的 Alpha 因子，只記錄不篩選。
   ⚠️ 判斷用近似門檻(收盤價相較前一日漲幅 >= 9.5%)，不是官方精確的
   跳動點位公式(實際漲停價會因為股價級距四捨五入，精確值可能是
   9.8%~10.0%之間，不是剛好10%)，這裡刻意保守取 9.5% 避免漏判，但
   不保證跟交易所公告的精確漲停價一致。

⚠️ 已知但這次刻意不處理的限制(誠實揭露，非本次範圍)：股票的除權息、
私募、現金增資等公司行動可能造成價格不連續(跳空)，這種跳空如果被
BreakoutEngine 誤判為「真突破」，會混入雜訊。yfinance 的 auto_adjust
只處理一般除權息，私募/現金增資的調整未必完整，尤其台股中小型股。
要正確處理需要額外對接 TWSE 的公司行動公告資料，目前程式碼沒有這個
資料來源，不在這次的範圍內新增(避免又是「先蓋工程再等有沒有用得上」)，
先誠實記錄這個限制，供 Phase B 解讀資料時參考。
"""


import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# 確保可以用 `python -m research.event_inventory` 或直接執行都找得到 engines/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engines.logging_config import get_logger
from engines.breakout_engine import BreakoutEngine
from engines.market_regime_engine import MarketRegimeEngine
from engines.risk_engine import RiskEngine

logger = get_logger(__name__)

# 全市場代碼→是否為ETF的對照表，第一次用到時延遲載入並快取在模組層級，
# 避免每個事件都重新查一次資料庫。若 stock_directory 表是空的（尚未
# refresh_all()），維持 None，呼叫端會誠實記錄「無法判斷」而不是猜測。
_ETF_LOOKUP_CACHE = None


def _get_etf_lookup() -> dict:
    global _ETF_LOOKUP_CACHE
    if _ETF_LOOKUP_CACHE is not None:
        return _ETF_LOOKUP_CACHE
    try:
        from engines.stock_directory_engine import StockDirectoryEngine
        df = StockDirectoryEngine.list_universe()
        if df.empty or "code" not in df.columns or "is_etf" not in df.columns:
            _ETF_LOOKUP_CACHE = {}
        else:
            _ETF_LOOKUP_CACHE = dict(zip(df["code"].astype(str), df["is_etf"].astype(bool)))
    except Exception:
        logger.exception("建立 ETF 對照表失敗，is_etf 欄位將一律記為 None（無法判斷）")
        _ETF_LOOKUP_CACHE = {}
    return _ETF_LOOKUP_CACHE


def _is_etf(ticker: str):
    """回傳 True/False/None。None 代表 stock_directory 表沒有這檔股票的
    資料（多半是尚未 refresh_all()），不代表「已確認不是 ETF」。"""
    lookup = _get_etf_lookup()
    return lookup.get(str(ticker).strip(), None)

# ==========================================
# 參數設定
# ==========================================

# 至少需要這麼多天資料才開始嘗試偵測（min_box_days + exclude_recent_days 之後
# 還要留一些緩衝，避免資料不足時箱型判斷本身就不可靠）
MIN_HISTORY_DAYS = 90

# 與 BreakoutEngine 現行預設一致，刻意不調整，確保這裡測的是「系統實際會
# 標記出來的突破」，不是研究者另外發明的一套參數
BOX_KWARGS = dict(min_box_days=20, max_box_days=60, tight_threshold_pct=12.0, exclude_recent_days=10)
BREAKOUT_KWARGS = dict(buffer_pct=0.5, confirm_window=10)
VOLUME_LOOKBACK = 30

# 台股漲跌停為 ±10%，但實際跳動點位會因股價級距四捨五入，精確值通常
# 落在 9.8%~10.0% 之間，不是剛好 10%。這裡用保守門檻 9.5% 判斷「當天
# 收盤價相較前一日是否為漲停/貼近漲停」，寧可略為寬鬆多抓一些邊界案例，
# 也不要因為門檻抓太緊漏掉真正的漲停鎖死事件。
LIMIT_UP_THRESHOLD_PCT = 9.5

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "research", "output")
EVENTS_CSV = os.path.join(OUTPUT_DIR, "breakout_events_raw.csv")
CHECKPOINT_JSON = os.path.join(OUTPUT_DIR, "event_inventory_checkpoint.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "event_inventory_summary.json")
ERRORS_CSV = os.path.join(OUTPUT_DIR, "event_inventory_errors.csv")

# ⚠️ 命名說明：這兩個數字是 Research v1 Pilot Study 的「工作流程門檻」
# （workflow thresholds），用來決定值不值得投入下一階段的工程成本，
# 不是經過統計檢定驗證過的顯著性臨界值，也不代表任何理論上的「正確」
# 樣本數。目前維持 100 / 300 這組既有數值——沒有新證據支持改成別的
# 數字之前，不調整（見 research/ 的討論紀錄：第一輪 Pilot Study 前，
# 樣本分布、事件集中度都還未知，此時改動門檻本身也只是換一組經驗值，
# 不是資料驅動的決定）。等第一輪、第二輪 Event Inventory 都有真實
# 分布可以參考後，再回頭評估這組數字要不要調整。
PILOT_STOP_EVENTS = 100
PILOT_GO_EVENTS = 300


# ==========================================
# Regime 時間序列（只對大盤算一次，逐日滾動）
# ==========================================

def build_regime_timeline(benchmark_df: pd.DataFrame, min_history: int = 65) -> pd.DataFrame:
    """
    對大盤指數（^TWII）做逐日滾動的 regime 分類，回傳
    columns=['date', 'regime'] 的 DataFrame，供事件標註用。

    ⚠️ 只依賴大盤資料，跟任何個股無關，所以只需要對這一份資料跑一次，
    不需要在每檔股票的迴圈裡重算——這是 Regime 相對 Breakout Inventory
    工程量小很多的原因（regime 是 O(交易日數)，breakout 是
    O(交易日數 × 股票數)）。
    """
    df = benchmark_df.copy().sort_values("date").reset_index(drop=True)
    records = []
    for t in range(min_history, len(df)):
        window = df.iloc[: t + 1]
        result = MarketRegimeEngine.classify(window)
        records.append({
            "date": df["date"].iloc[t],
            "regime": result.get("regime", "N/A（大盤資料不足）"),
        })
    return pd.DataFrame(records)


def _regime_lookup(regime_timeline: pd.DataFrame, event_date) -> str:
    if regime_timeline is None or regime_timeline.empty:
        return "N/A（無 regime 資料）"
    match = regime_timeline[regime_timeline["date"] == event_date]
    if match.empty:
        return "N/A（無對應日期）"
    return match["regime"].iloc[0]


# ==========================================
# 單一股票：逐日滾動偵測箱型突破事件
# ==========================================

def detect_breakout_events(df: pd.DataFrame, ticker: str, regime_timeline: pd.DataFrame = None) -> list:
    """
    對單一股票的完整歷史 OHLCV，逐日重算 BreakoutEngine 的箱型突破判斷，
    用「edge-triggered」（False→True 那一天才算一次事件）方式找出所有
    歷史事件，回傳 event dict 的 list。

    df 需要 columns: date, open, high, low, close, volume（DataEngine 的
    標準格式），且已依日期排序。

    每個事件除了箱型本身的欄位之外，還會附加幾個「專業投資角度」的
    metadata（見本檔案開頭 docstring 的說明）：liquidity_level_at_event、
    breakout_magnitude_pct、is_etf、event_seq_for_ticker、
    is_limit_up_day。這些欄位只是記錄，這支程式不會用它們做任何篩選或
    分組——會不會用來切分析，是 Phase B 的決定。
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    events = []

    if n < MIN_HISTORY_DAYS:
        return events

    # 流動性是整檔股票的既有價量欄位算出來的（20日滾動平均成交值），
    # 不需要逐日重算，一次算好整條時間序列即可，用事件當天的索引直接
    # 查表，避免在主迴圈裡重複呼叫。
    try:
        liquidity_df = RiskEngine.add_liquidity_metrics(df)
    except Exception:
        logger.exception(f"[{ticker}] 流動性指標計算失敗，liquidity_level_at_event 將一律記為 None")
        liquidity_df = None

    is_etf_flag = _is_etf(ticker)
    ticker_event_seq = 0

    prev_confirmed = False
    for t in range(MIN_HISTORY_DAYS, n):
        window = df.iloc[: t + 1]
        try:
            box_info = BreakoutEngine.analyze_consolidation_box(window, **BOX_KWARGS)
            breakout_info = BreakoutEngine.analyze_breakout(window, box_info, **BREAKOUT_KWARGS)
        except Exception:
            # 單一天資料異常（例如缺值）不應該讓整檔股票的迴圈中斷，
            # 視為當天無法判斷，狀態沿用前一天，繼續下一天。
            logger.exception(f"[{ticker}] day index {t} 箱型/突破判斷失敗，略過此日")
            continue

        confirmed = bool(breakout_info.get("breakout_confirmed", False))

        if confirmed and not prev_confirmed:
            # 邊緣觸發：偵測到「新的」突破事件
            event_date = df["date"].iloc[t]
            try:
                volume_info = BreakoutEngine.analyze_volume_resonance(
                    window, box_info=box_info, lookback=VOLUME_LOOKBACK
                )
                volume_confirmed = bool(volume_info.get("breakout_volume_surge_confirmed", False))
            except Exception:
                logger.exception(f"[{ticker}] day index {t} 量能共振判斷失敗，volume_confirmed 記為 None")
                volume_confirmed = None

            box_high = box_info.get("box_high")
            latest_close = float(df["close"].iloc[t])
            breakout_magnitude_pct = (
                round((latest_close - box_high) / box_high * 100, 2)
                if box_high else None
            )

            liquidity_level = None
            if liquidity_df is not None:
                try:
                    liquidity_level = liquidity_df["liquidity_level"].iloc[t]
                except Exception:
                    liquidity_level = None

            is_limit_up_day = None
            if t > 0:
                try:
                    prev_close = float(df["close"].iloc[t - 1])
                    if prev_close > 0:
                        day_change_pct = (latest_close - prev_close) / prev_close * 100
                        is_limit_up_day = bool(day_change_pct >= LIMIT_UP_THRESHOLD_PCT)
                except Exception:
                    logger.exception(f"[{ticker}] day index {t} 漲停判斷失敗，is_limit_up_day 記為 None")
                    is_limit_up_day = None

            ticker_event_seq += 1

            events.append({
                "ticker": ticker,
                "event_date": event_date.strftime("%Y-%m-%d") if hasattr(event_date, "strftime") else str(event_date),
                "box_high": box_high,
                "box_low": box_info.get("box_low"),
                "box_days": box_info.get("box_days"),
                "box_range_pct": box_info.get("range_pct"),
                "volume_confirmed": volume_confirmed,
                "regime_at_event": _regime_lookup(regime_timeline, event_date),
                "breakout_magnitude_pct": breakout_magnitude_pct,
                "liquidity_level_at_event": liquidity_level,
                "is_etf": is_etf_flag,
                "event_seq_for_ticker": ticker_event_seq,
                "is_limit_up_day": is_limit_up_day,
            })

        prev_confirmed = confirmed

    return events


# ==========================================
# 全市場/清單 orchestration + checkpoint
# ==========================================

def get_universe(scope: str, limit: int = None) -> list:
    """
    scope: 'watchlist' 使用 ScannerEngine.DEFAULT_WATCHLIST（現成、少量、
           現在就能跑，適合先驗證邏輯）；'full' 使用
           StockDirectoryEngine.list_universe()（需要先在有網路權限的環境
           跑過 StockDirectoryEngine.refresh_all()，否則回傳空清單）。
    """
    from engines.scanner_engine import ScannerEngine

    if scope == "watchlist":
        tickers = list(ScannerEngine.DEFAULT_WATCHLIST)
    elif scope == "full":
        from engines.stock_directory_engine import StockDirectoryEngine
        df = StockDirectoryEngine.list_universe(exclude_etf=True)
        if df.empty:
            logger.warning(
                "stock_directory 表是空的——請先在有網路權限的環境執行"
                " StockDirectoryEngine.refresh_all() 建立全市場代碼清單，"
                "目前先退回使用 DEFAULT_WATCHLIST。"
            )
            tickers = list(ScannerEngine.DEFAULT_WATCHLIST)
        else:
            tickers = df["code"].astype(str).tolist()
    else:
        raise ValueError(f"未知的 universe scope: {scope}")

    if limit:
        tickers = tickers[:limit]
    return tickers


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_JSON):
        try:
            with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("讀取 checkpoint 失敗，視為沒有 checkpoint 重新開始")
    return {"processed_tickers": []}


def _save_checkpoint(state: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_events_csv(events: list):
    if not events:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(events)
    header = not os.path.exists(EVENTS_CSV)
    df.to_csv(EVENTS_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")


def _append_errors_csv(ticker: str, error: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame([{"ticker": ticker, "error": error, "timestamp": datetime.now().isoformat()}])
    header = not os.path.exists(ERRORS_CSV)
    df.to_csv(ERRORS_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")


def run_inventory(scope: str = "watchlist", limit: int = None, resume: bool = True,
                   sleep_sec: float = 0.3, benchmark_period_days: int = None,
                   progress_callback=None):
    """
    主流程：
      1. 取得股票清單
      2. 建立一次性的 regime 時間序列（只對大盤跑一次）
      3. 逐檔股票：抓歷史資料 → 逐日滾動偵測事件 → 附加寫入 CSV → 更新 checkpoint
      4. 全部跑完後，統計 Phase A summary report
    """
    from engines.data_engine import DataEngine

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = get_universe(scope, limit=limit)
    logger.info(f"Event Inventory 開始，universe scope={scope}，共 {len(tickers)} 檔股票")

    checkpoint = _load_checkpoint() if resume else {"processed_tickers": []}
    processed = set(checkpoint.get("processed_tickers", []))

    # 只需要對大盤資料建立一次 regime 時間序列
    logger.info("建立大盤 regime 時間序列（只需要跑一次，跟個股迴圈無關）...")
    try:
        benchmark_df = DataEngine.get_benchmark_data("^TWII", use_cache=True)
        regime_timeline = build_regime_timeline(benchmark_df) if benchmark_df is not None and not benchmark_df.empty else pd.DataFrame()
    except Exception:
        logger.exception("取得大盤資料或建立 regime 時間序列失敗，事件仍會記錄，但 regime_at_event 會是 N/A")
        regime_timeline = pd.DataFrame()

    total = len(tickers)
    for i, ticker in enumerate(tickers):
        ticker = str(ticker).strip()
        if not ticker:
            continue
        if ticker in processed:
            logger.info(f"[{i+1}/{total}] {ticker} 已處理過（checkpoint），略過")
            continue

        try:
            df = DataEngine.get_stock_data(ticker, use_cache=True)
            if df is None or df.empty:
                logger.warning(f"[{ticker}] 沒有股價資料，略過")
                _append_errors_csv(ticker, "empty_price_data")
            else:
                events = detect_breakout_events(df, ticker, regime_timeline=regime_timeline)
                _append_events_csv(events)
                logger.info(f"[{i+1}/{total}] {ticker} 完成，偵測到 {len(events)} 個事件")
        except Exception as e:
            logger.exception(f"[{ticker}] 處理失敗")
            _append_errors_csv(ticker, str(e))

        processed.add(ticker)
        checkpoint["processed_tickers"] = list(processed)
        _save_checkpoint(checkpoint)

        if progress_callback:
            progress_callback(i + 1, total, ticker)

        if sleep_sec:
            time.sleep(sleep_sec)  # 避免對 yfinance/TWSE 短時間內大量請求

    logger.info("Event Inventory 全部股票處理完畢，開始產出 summary report")
    return summarize_events()


# ==========================================
# Phase A Summary：事件數、分布、集中度、Stop Rule
# ==========================================

def summarize_events(events_csv_path: str = None) -> dict:
    """
    讀取累積下來的事件 CSV，產出 Phase A 需要的所有統計量：
      - 事件總數 / 每年平均事件數
      - 每檔股票事件數的分布（中位數、分位數）
      - 有量確認 / 無量確認 各占比
      - 前 10 檔股票的事件數集中度（HHI）
      - Stop Rule 判定結果
    """
    path = events_csv_path or EVENTS_CSV
    if not os.path.exists(path):
        summary = {"status": "no_events_file", "total_events": 0,
                   "stop_rule_verdict": "STOP（找不到事件檔案，等同於 0 個事件）"}
        _save_summary(summary)
        return summary

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        summary = {"status": "empty", "total_events": 0,
                   "stop_rule_verdict": "STOP（事件數為 0，統計力不足，不建議進入 Phase B）"}
        _save_summary(summary)
        return summary

    df["event_date"] = pd.to_datetime(df["event_date"])
    total_events = len(df)

    date_span_days = (df["event_date"].max() - df["event_date"].min()).days
    if date_span_days < 30:
        # 資料期間過短（例如只涵蓋一天或幾天），年化換算沒有意義，
        # 硬算只會得到荒謬的巨大數字（除以趨近於0的年數），改標示為 None
        # 並附上原始天數，讓使用者知道為什麼看不到年化數字。
        events_per_year = None
        events_per_year_note = f"資料期間僅 {date_span_days} 天，過短無法可靠估計年化事件數"
    else:
        years_span = date_span_days / 365.25
        events_per_year = round(total_events / years_span, 1)
        events_per_year_note = None

    per_ticker_counts = df.groupby("ticker").size().sort_values(ascending=False)
    events_per_ticker_median = float(per_ticker_counts.median())
    events_per_ticker_p25 = float(per_ticker_counts.quantile(0.25))
    events_per_ticker_p75 = float(per_ticker_counts.quantile(0.75))

    vol_confirmed_count = int((df["volume_confirmed"] == True).sum())  # noqa: E712
    vol_not_confirmed_count = int((df["volume_confirmed"] == False).sum())  # noqa: E712
    vol_unknown_count = total_events - vol_confirmed_count - vol_not_confirmed_count
    vol_confirmed_pct = round(vol_confirmed_count / total_events * 100, 1) if total_events else 0.0

    # 前10檔集中度
    top10 = per_ticker_counts.head(10)
    top10_share_pct = round(top10.sum() / total_events * 100, 1) if total_events else 0.0
    shares = per_ticker_counts / total_events
    hhi = round(float((shares ** 2).sum()) * 10000, 1)  # 傳統 HHI 用 0~10000 表示，10000=完全壟斷

    if total_events < PILOT_STOP_EVENTS:
        verdict = (f"STOP（事件總數 {total_events} < {PILOT_STOP_EVENTS}，統計力不足，"
                   f"不建議投入 Phase B 的 Event Study，除非先擴大 universe 或延長資料期間）")
    elif total_events < PILOT_GO_EVENTS:
        verdict = (f"CAUTION（事件總數 {total_events} 介於 {PILOT_STOP_EVENTS}~"
                   f"{PILOT_GO_EVENTS} 之間，可以做 Phase B，但結果只能當初步觀察，"
                   f"不宜下確定性結論）")
    else:
        verdict = f"GO（事件總數 {total_events} >= {PILOT_GO_EVENTS}，可以進入 Phase B）"

    concentration_warning = None
    if top10_share_pct >= 50:
        concentration_warning = (
            f"⚠️ 前10檔股票就占了 {top10_share_pct}% 的事件數，此研究結果高度反映"
            f"少數（可能是高流動性/權值）股票的突破行為，不宜直接外推至全市場。"
        )

    # ── 專業投資角度補強：以下統計皆為描述性統計，不是分組依據 ──
    # （呼應 Regime 的原則：只記錄、不預先分組，分不分組留給 Phase B 決定）

    # 1. 時間分布：跟 top10 集中度是同一種風險的另一個面向——如果事件
    #    高度集中在少數幾個月，代表這批「事件庫存」可能只反映某一段
    #    特定行情（例如一次噴出的大多頭），不是跨時期都存在的現象。
    monthly_counts = df["event_date"].dt.to_period("M").astype(str).value_counts().sort_index()
    monthly_distribution = monthly_counts.to_dict()
    top_month_share_pct = round(monthly_counts.max() / total_events * 100, 1) if total_events else 0.0
    temporal_concentration_warning = None
    if top_month_share_pct >= 30:
        temporal_concentration_warning = (
            f"⚠️ 單一個月就占了 {top_month_share_pct}% 的事件數，事件庫存可能高度"
            f"集中在特定一段行情，之後的統計結果可能只反映那段時期，不宜當作"
            f"「箱型突破」跨時期普遍有效的證據。"
        )

    # 2. 流動性標記分布（只做計數，不排除任何事件）
    liquidity_breakdown = None
    low_liquidity_pct = None
    if "liquidity_level_at_event" in df.columns:
        liquidity_breakdown = df["liquidity_level_at_event"].fillna("未知").value_counts().to_dict()
        low_liquidity_count = df["liquidity_level_at_event"].astype(str).str.contains(
            "極低流動性|流動性偏低", na=False, regex=True
        ).sum()
        low_liquidity_pct = round(low_liquidity_count / total_events * 100, 1) if total_events else 0.0

    # 3. ETF 占比
    etf_event_count = None
    etf_event_pct = None
    if "is_etf" in df.columns:
        etf_event_count = int((df["is_etf"] == True).sum())  # noqa: E712
        etf_event_pct = round(etf_event_count / total_events * 100, 1) if total_events else 0.0

    # 4. 首次突破 vs 重複突破
    repeat_event_pct = None
    if "event_seq_for_ticker" in df.columns:
        repeat_count = int((df["event_seq_for_ticker"] > 1).sum())
        repeat_event_pct = round(repeat_count / total_events * 100, 1) if total_events else 0.0

    # 5. 漲停突破占比（本輪新增，Execution Constraint 的另一個面向）
    limit_up_event_count = None
    limit_up_event_pct = None
    if "is_limit_up_day" in df.columns:
        limit_up_event_count = int((df["is_limit_up_day"] == True).sum())  # noqa: E712
        limit_up_event_pct = round(limit_up_event_count / total_events * 100, 1) if total_events else 0.0

    summary = {
        "status": "ok",
        "total_events": total_events,
        "date_range": [df["event_date"].min().strftime("%Y-%m-%d"), df["event_date"].max().strftime("%Y-%m-%d")],
        "events_per_year": events_per_year,
        "events_per_year_note": events_per_year_note,
        "distinct_tickers_with_events": int(df["ticker"].nunique()),
        "events_per_ticker_median": events_per_ticker_median,
        "events_per_ticker_p25": events_per_ticker_p25,
        "events_per_ticker_p75": events_per_ticker_p75,
        "volume_confirmed_count": vol_confirmed_count,
        "volume_not_confirmed_count": vol_not_confirmed_count,
        "volume_unknown_count": vol_unknown_count,
        "volume_confirmed_pct": vol_confirmed_pct,
        "top10_tickers": top10.to_dict(),
        "top10_share_pct": top10_share_pct,
        "hhi": hhi,
        "concentration_warning": concentration_warning,
        "monthly_distribution": monthly_distribution,
        "top_month_share_pct": top_month_share_pct,
        "temporal_concentration_warning": temporal_concentration_warning,
        "liquidity_breakdown": liquidity_breakdown,
        "low_liquidity_pct": low_liquidity_pct,
        "etf_event_count": etf_event_count,
        "etf_event_pct": etf_event_pct,
        "repeat_event_pct": repeat_event_pct,
        "limit_up_event_count": limit_up_event_count,
        "limit_up_event_pct": limit_up_event_pct,
        "stop_rule_verdict": verdict,
    }
    _save_summary(summary)
    return summary


def _save_summary(summary: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("Breakout Event Inventory — Phase A Summary")
    print("=" * 60)
    if summary.get("status") != "ok":
        print(f"狀態: {summary.get('status')}")
        print(f"Stop Rule 判定: {summary.get('stop_rule_verdict')}")
        print("=" * 60 + "\n")
        return

    print(f"事件總數: {summary['total_events']}")
    print(f"資料期間: {summary['date_range'][0]} ~ {summary['date_range'][1]}")
    print(f"每年平均事件數: {summary['events_per_year'] if summary['events_per_year'] is not None else 'N/A'}")
    if summary.get("events_per_year_note"):
        print(f"  ⚠️ {summary['events_per_year_note']}")
    print(f"有事件的股票數: {summary['distinct_tickers_with_events']}")
    print(f"每檔股票事件數（中位數 / P25 / P75）: "
          f"{summary['events_per_ticker_median']} / {summary['events_per_ticker_p25']} / {summary['events_per_ticker_p75']}")
    print(f"有量確認: {summary['volume_confirmed_count']} 筆 "
          f"({summary['volume_confirmed_pct']}%)，無量確認: {summary['volume_not_confirmed_count']} 筆，"
          f"無法判斷: {summary['volume_unknown_count']} 筆")
    print(f"前10檔股票事件數: {summary['top10_tickers']}")
    print(f"前10檔集中度: {summary['top10_share_pct']}%，HHI: {summary['hhi']}")
    if summary.get("concentration_warning"):
        print(summary["concentration_warning"])

    print(f"\n單月最高占比: {summary['top_month_share_pct']}%")
    if summary.get("temporal_concentration_warning"):
        print(summary["temporal_concentration_warning"])

    if summary.get("liquidity_breakdown") is not None:
        print(f"\n流動性分布: {summary['liquidity_breakdown']}")
        print(f"低流動性事件占比(🔴極低+🟡偏低): {summary['low_liquidity_pct']}%")
    else:
        print("\n流動性分布: N/A（liquidity_level_at_event 欄位缺失，可能是舊版事件檔案）")

    if summary.get("etf_event_count") is not None:
        print(f"ETF 事件數: {summary['etf_event_count']} ({summary['etf_event_pct']}%)")
    else:
        print("ETF 占比: N/A（is_etf 欄位缺失，或 stock_directory 尚未 refresh_all()）")

    if summary.get("repeat_event_pct") is not None:
        print(f"重複突破(同股票第2次以上)占比: {summary['repeat_event_pct']}%")

    if summary.get("limit_up_event_pct") is not None:
        print(f"漲停突破占比: {summary['limit_up_event_count']} 筆 ({summary['limit_up_event_pct']}%)"
              f"（這些事件當天很可能無法實際成交進場，Phase B 需另外處理）")

    print(f"\nStop Rule 判定: {summary['stop_rule_verdict']}")
    print("=" * 60 + "\n")


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Breakout Event Inventory (Phase A)")
    parser.add_argument("--universe", choices=["watchlist", "full"], default="watchlist",
                         help="watchlist=現成觀察名單(快)，full=全市場(需先跑過"
                              " StockDirectoryEngine.refresh_all())")
    parser.add_argument("--limit", type=int, default=None, help="限制處理股票數（測試用）")
    parser.add_argument("--no-resume", action="store_true", help="忽略既有 checkpoint，從頭開始")
    parser.add_argument("--sleep", type=float, default=0.3, help="每檔股票之間的休息秒數，避免打太快")
    args = parser.parse_args()

    summary = run_inventory(
        scope=args.universe,
        limit=args.limit,
        resume=not args.no_resume,
        sleep_sec=args.sleep,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
