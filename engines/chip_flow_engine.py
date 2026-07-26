"""
engines/chip_flow_engine.py

📦 ChipFlowEngine — 三大法人／融資融券「連續天數」追蹤層

⚠️ 定位說明：這個檔案跟 chip_engine.py 是互補、不是取代。
    chip_engine.py    —— 單一股票查詢（get_institutional_snapshot）、
                          單日全市場排行榜（get_market_wide_institutional_ranking）
    chip_flow_engine.py —— 全市場、跨多日累積，計算「連買/連賣天數」這種
                          需要時間序列才能算出來的東西

── 為什麼不能直接複用 chip_engine.get_margin_trend() ──
那支函式的 docstring 已經寫明：TWSE 沒有「單一股票區間查詢」端點，只有
「單日全市場」端點，逐日呼叫是唯一取得序列資料的方式，且「不建議放進
批次掃描」。這裡要做的是「全市場每檔股票的連續天數」，正確做法是
複用 chip_engine.get_market_wide_institutional_ranking() 已經在用的
「逐日抓一次全市場快照、篩選欄位」模式，對每一天呼叫一次
_fetch_institutional_single_day() / _fetch_margin_single_day()，
在記憶體裡按股票代碼累積成時間序列，再從序列算連續天數——
呼叫次數是 O(天數)，不是 O(天數 × 股票數)。

── 誠實揭露：借券（證券借貸）資料 ──
TWSE 官方確實有「借券資訊」頁面（https://www.twse.com.tw/zh/page/trading/
exchange/TWT72U.html），支援 CSV 下載，代表這份資料存在。但這個頁面是
JavaScript 動態表單，实际的機器可讀 API 端點（像 T86/融資那樣的
openapi.twse.com.tw 路徑）需要用瀏覽器開發者工具攔截網路請求才能確認，
這件事沒辦法在沒有瀏覽器、沒有網路的環境完成。下面的
fetch_securities_lending_single_day() 刻意只是一個回傳
status="not_implemented" 的骨架，不是假裝做完的空殼——呼叫端會清楚
知道這塊還沒有真正的資料來源，而不是誤以為有資料但其實是 0。
若確認了正確端點，把該函式內的 TODO 替換掉即可，其餘的累積/連續天數
邏輯可以直接複用（see StreakAccumulator）。

── 誠實揭露：Chip Score 是未經驗證的假設性公式 ──
COMPUTE_CHIP_SCORE 用的權重（外資連買+25、投信連買+35...）是主觀指定，
沒有任何回測或統計驗證支撐，跟本專案在 research/ 模組建立的「沒有證據
不下結論」原則不一致。之所以還是把它做出來，是因為使用者在被清楚告知
這個限制後，仍選擇要現在就要，而不是等驗證。為了不讓這個限制被忽略：
  1. 回傳值裡永遠帶 "is_validated": False 與完整的權重明細，不只給
     一個總分數字，讓呼叫端有能力自行決定要不要信任、要不要重新加權。
  2. 不直接混入 decision_engine.py 既有的 AI Score 投票邏輯（那裡的
     設計刻意避免加權合成單一分數）。這裡是用跟 news_report 的
     event_confidence 相同的模式接進 build_consensus()：當作一則
     「有明確標示未驗證」的額外註記，不改變任何投票權重。
  3. 若之後要驗證這組權重，建議走 research/ 模組同樣的 Event Inventory
     → Event Study 流程：先追蹤連買天數與未來報酬的關係，有證據支撐
     後再回頭調整或移除這裡的權重。
"""
import datetime
from collections import defaultdict

import pandas as pd

from engines.chip_engine import ChipEngine
from engines.logging_config import get_logger

logger = get_logger(__name__)


class ChipFlowEngine:

    # ==========================================
    # 內部：把某個資料源逐日抓成 {日期: 全市場 DataFrame} 的字典
    # ==========================================
    @staticmethod
    def _collect_daily_snapshots(fetch_fn, lookback_days: int, source_name: str) -> dict:
        """
        逐日呼叫 fetch_fn(date_str)，收集近 lookback_days 個「有資料」的
        交易日全市場快照，回傳 {date_str(YYYY-MM-DD): DataFrame}（由舊到新）。

        ⚠️ 呼叫次數是 O(lookback_days)，不是 O(lookback_days × 股票數)——
        這是這個檔案存在的意義，不要在外層再包一個「對每檔股票各呼叫一次」
        的迴圈，那樣就失去了設計這個函式的目的。
        """
        snapshots = {}
        max_calendar_lookback = int(lookback_days * 1.6) + 5  # 涵蓋假日/非交易日緩衝
        for delta in range(max_calendar_lookback):
            if len(snapshots) >= lookback_days:
                break
            d = datetime.datetime.now() - datetime.timedelta(days=delta)
            df = fetch_fn(d.strftime("%Y%m%d"))
            if df is None or df.empty:
                continue
            snapshots[d.strftime("%Y-%m-%d")] = df

        if len(snapshots) < lookback_days:
            logger.warning(
                f"[{source_name}] 只取得 {len(snapshots)}/{lookback_days} 個交易日的全市場快照"
                f"（可能是連假、TWSE服務異常，或 lookback_days 設定超過近期實際交易日數）"
            )
        # 由舊到新排序，方便後續逐日累積計算連續天數
        return dict(sorted(snapshots.items()))

    # ==========================================
    # 三大法人連買/連賣天數（全市場）
    # ==========================================
    @staticmethod
    def compute_institutional_streaks(lookback_days: int = 30) -> dict:
        """
        回傳格式：
            {
                "status": "ok",
                "as_of_date": "2026-07-22",
                "days_used": 28,
                "streaks": {
                    "2330": {
                        "foreign": {"direction": "buy", "days": 7, "cumulative_shares": 12345000},
                        "trust":   {"direction": "buy", "days": 12, "cumulative_shares": 8560000},
                        "dealer":  {"direction": "sell", "days": 4, "cumulative_shares": -620000},
                        "total":   {"direction": "buy", "days": 9, "cumulative_shares": 14560000},
                    },
                    ...
                },
            }
        「連買/連賣天數」定義：從最新交易日往回數，方向（買超為正/賣超為負，
        以0為界）連續一致的天數；一旦出現反方向或掛零就中斷（跟
        research/event_inventory.py 的 edge-triggered 去重是同一種
        「只算連續同狀態」的邏輯，不是巧合，是同一套設計慣例）。
        """
        snapshots = ChipFlowEngine._collect_daily_snapshots(
            ChipEngine._fetch_institutional_single_day, lookback_days, "institutional"
        )
        if not snapshots:
            return {"status": "unavailable",
                    "message": "⚠️ 近期無法取得任何全市場三大法人快照，無法計算連續天數。"}

        # ⚠️ 名稱對照表：直接用 TWSE 當天回傳的官方名稱建表（不用 NameEngine.NAME_MAP，
        # 那份只涵蓋本專案內建觀察名單的幾十檔，這裡涵蓋全部上市股票），只從最新一天
        # 的快照取一次，股票名稱短期內不會變，不需要每天重複解析。
        names = ChipFlowEngine._extract_name_map(snapshots)

        # 逐日快照 -> 逐股票的時間序列
        per_ticker_series = defaultdict(list)  # ticker -> [(date, foreign, trust, dealer, total), ...]
        for date_str, df in snapshots.items():
            code_col = ChipEngine._find_col(df.columns, "證券代號")
            if code_col is None:
                continue
            foreign_col = ChipEngine._find_col(df.columns, "外陸資買賣超股數") \
                or ChipEngine._find_col(df.columns, "外資及陸資買賣超股數") \
                or ChipEngine._find_col(df.columns, "外資買賣超股數")
            trust_col = ChipEngine._find_col(df.columns, "投信買賣超股數")
            dealer_col = ChipEngine._find_col(df.columns, "自營商買賣超股數(自行買賣)") \
                or ChipEngine._find_col(df.columns, "自營商買賣超股數")
            total_col = ChipEngine._find_col(df.columns, "三大法人買賣超股數")
            if not all([foreign_col, trust_col, dealer_col]):
                continue

            for _, row in df.iterrows():
                code = str(row[code_col]).strip()
                if not code or not code[0].isdigit():
                    continue
                foreign = ChipEngine._to_int(row[foreign_col])
                trust = ChipEngine._to_int(row[trust_col])
                dealer = ChipEngine._to_int(row[dealer_col])
                total = ChipEngine._to_int(row[total_col]) if total_col else (foreign + trust + dealer)
                per_ticker_series[code].append((date_str, foreign, trust, dealer, total))

        streaks = {}
        for code, series in per_ticker_series.items():
            series.sort(key=lambda r: r[0])  # 由舊到新
            streaks[code] = {
                "foreign": ChipFlowEngine._streak_from_series([r[1] for r in series]),
                "trust": ChipFlowEngine._streak_from_series([r[2] for r in series]),
                "dealer": ChipFlowEngine._streak_from_series([r[3] for r in series]),
                "total": ChipFlowEngine._streak_from_series([r[4] for r in series]),
            }

        return {
            "status": "ok",
            "as_of_date": max(snapshots.keys()),
            "days_used": len(snapshots),
            "streaks": streaks,
            "names": names,
        }

    @staticmethod
    def _extract_name_map(snapshots: dict) -> dict:
        """從全市場快照的最新一天，抓「證券代號 -> 證券名稱」對照表。
        任何一種全市場快照（三大法人或融資融券）都有這兩欄，共用同一支函式。"""
        if not snapshots:
            return {}
        latest_date = max(snapshots.keys())
        df = snapshots[latest_date]
        code_col = ChipEngine._find_col(df.columns, "證券代號") or ChipEngine._find_col(df.columns, "代號")
        name_col = ChipEngine._find_col(df.columns, "證券名稱") or ChipEngine._find_col(df.columns, "名稱")
        if code_col is None or name_col is None:
            return {}
        names = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            if code and code[0].isdigit():
                names[code] = str(row[name_col]).strip()
        return names

    @staticmethod
    def _streak_from_series(values: list) -> dict:
        """values 由舊到新排列的每日買賣超股數，回傳從最新一天往回數的
        連續同方向天數與這段期間的累積股數。"""
        if not values:
            return {"direction": "flat", "days": 0, "cumulative_shares": 0}

        latest = values[-1]
        if latest > 0:
            direction = "buy"
        elif latest < 0:
            direction = "sell"
        else:
            direction = "flat"

        if direction == "flat":
            return {"direction": "flat", "days": 0, "cumulative_shares": 0}

        days = 0
        cumulative = 0
        for v in reversed(values):
            same_direction = (v > 0 and direction == "buy") or (v < 0 and direction == "sell")
            if not same_direction:
                break
            days += 1
            cumulative += v
        return {"direction": direction, "days": days, "cumulative_shares": cumulative}

    # ==========================================
    # 融資融券連續增減天數（全市場）
    # ==========================================
    @staticmethod
    def compute_margin_streaks(lookback_days: int = 30) -> dict:
        """
        回傳格式跟 compute_institutional_streaks 對稱：
            {
                "status": "ok",
                "as_of_date": "...",
                "days_used": ...,
                "streaks": {
                    "2330": {
                        "margin_balance": {"direction": "decreasing", "days": 6, "change": -1250000},
                        "short_balance":  {"direction": "increasing", "days": 3, "change": 280000},
                    },
                    ...
                },
            }
        ⚠️ 這裡定義的「連續」是「餘額逐日增加/減少」（跟前面三大法人的
        「買賣超方向」不同概念）——融資融券本來就沒有「買超/賣超」，
        只有「餘額增加/減少」，這是這兩種資料類型本質上的差異，不是
        寫法不一致。
        """
        snapshots = ChipFlowEngine._collect_daily_snapshots(
            ChipEngine._fetch_margin_single_day, lookback_days, "margin"
        )
        if not snapshots:
            return {"status": "unavailable",
                    "message": "⚠️ 近期無法取得任何全市場融資融券快照，無法計算連續天數。"}

        names = ChipFlowEngine._extract_name_map(snapshots)

        per_ticker_series = defaultdict(list)
        for date_str, df in snapshots.items():
            code_col = ChipEngine._find_col(df.columns, "代號") or ChipEngine._find_col(df.columns, "證券代號")
            if code_col is None:
                continue
            margin_col = ChipEngine._find_col(df.columns, "融資今日餘額") or ChipEngine._find_col(df.columns, "融資餘額")
            short_col = ChipEngine._find_col(df.columns, "融券今日餘額") or ChipEngine._find_col(df.columns, "融券餘額")
            if margin_col is None and short_col is None:
                continue

            for _, row in df.iterrows():
                code = str(row[code_col]).strip()
                if not code or not code[0].isdigit():
                    continue
                margin_bal = ChipEngine._to_int(row[margin_col]) if margin_col else None
                short_bal = ChipEngine._to_int(row[short_col]) if short_col else None
                per_ticker_series[code].append((date_str, margin_bal, short_bal))

        streaks = {}
        for code, series in per_ticker_series.items():
            series.sort(key=lambda r: r[0])
            margin_vals = [r[1] for r in series if r[1] is not None]
            short_vals = [r[2] for r in series if r[2] is not None]
            streaks[code] = {
                "margin_balance": ChipFlowEngine._streak_from_balance_series(margin_vals),
                "short_balance": ChipFlowEngine._streak_from_balance_series(short_vals),
            }

        return {
            "status": "ok",
            "as_of_date": max(snapshots.keys()),
            "days_used": len(snapshots),
            "streaks": streaks,
            "names": names,
        }

    @staticmethod
    def _streak_from_balance_series(values: list) -> dict:
        """values 是「餘額」（不是逐日增減量），逐日算差值後判斷連續
        增加/減少天數。至少需要2個資料點才算得出方向。"""
        if len(values) < 2:
            return {"direction": "flat", "days": 0, "change": 0}

        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        latest_diff = diffs[-1]
        if latest_diff > 0:
            direction = "increasing"
        elif latest_diff < 0:
            direction = "decreasing"
        else:
            direction = "flat"

        if direction == "flat":
            return {"direction": "flat", "days": 0, "change": 0}

        days = 0
        for d in reversed(diffs):
            same_direction = (d > 0 and direction == "increasing") or (d < 0 and direction == "decreasing")
            if not same_direction:
                break
            days += 1
        change = values[-1] - values[-1 - days]
        return {"direction": direction, "days": days, "change": change}

    # ==========================================
    # 借券（誠實揭露：端點未確認，這裡是骨架，不是完成品）
    # ==========================================
    @staticmethod
    def fetch_securities_lending_single_day(date_str: str) -> pd.DataFrame:
        """
        ⚠️ 尚未實作：TWSE 官方「借券資訊」頁面
        (https://www.twse.com.tw/zh/page/trading/exchange/TWT72U.html)
        確認存在且支援 CSV 下載，但實際的機器可讀 API 端點路徑需要用
        瀏覽器開發者工具攔截該頁查詢表單送出的請求才能確認（不是
        openapi.twse.com.tw 常見的 exchangeReport 路徑，這頁是舊式的
        report-query 表單，端點格式未知）。

        這個函式故意回傳空的 DataFrame，而不是猜測一個可能是錯的 URL
        假裝能動——那樣的風險是「安靜地回傳假資料或直接連線失敗」，
        比「明確告訴呼叫端這裡還沒做」更糟。

        TODO：確認端點後，比照 ChipEngine._fetch_institutional_single_day()
        的模式實作（regex/開發者工具找出實際 URL、參數、回傳格式），
        接上後 compute_lending_streaks() 可以直接複用上面
        _collect_daily_snapshots() + _streak_from_balance_series()，
        不需要重新設計累積邏輯。
        """
        logger.warning(
            f"[securities_lending] 端點尚未確認，fetch_securities_lending_single_day"
            f"({date_str}) 直接回傳空結果，未發送任何網路請求。"
        )
        return pd.DataFrame()

    @staticmethod
    def compute_lending_streaks(lookback_days: int = 30) -> dict:
        return {
            "status": "not_implemented",
            "message": (
                "⚠️ 借券（證券借貸）資料的機器可讀 API 端點尚未確認，"
                "此功能目前為空骨架。TWSE 官方頁面確認資料存在"
                "（https://www.twse.com.tw/zh/page/trading/exchange/TWT72U.html，"
                "支援 CSV 下載），但需要有人用瀏覽器開發者工具找出實際"
                "查詢端點後才能接上。"
            ),
        }

    # ==========================================
    # Chip Score（⚠️ 未經驗證的主觀權重，見檔案開頭誠實揭露）
    # ==========================================
    CHIP_SCORE_WEIGHTS = {
        "foreign_buy_streak": 25,
        "trust_buy_streak": 35,
        "institutional_total_buy_streak": 20,
        "margin_decreasing": 10,
        "short_covering": 10,
    }

    @staticmethod
    def compute_chip_score(institutional_streaks: dict, margin_streaks: dict,
                            streak_days_for_full_score: int = 10) -> dict:
        """
        ⚠️⚠️⚠️ 這是一個主觀指定、未經任何回測驗證的複合分數，不是統計上
        證實有效的 Alpha 訊號。使用前請閱讀本檔案開頭的「誠實揭露」段落。

        計分邏輯（線性給分，連續天數達到 streak_days_for_full_score 給滿分，
        之間線性內插，方向不符不給分）：
          - 外資連買天數 / streak_days_for_full_score × 25 分（上限25）
          - 投信連買天數 / streak_days_for_full_score × 35 分（上限35）
          - 三大法人合計連買天數 / streak_days_for_full_score × 20 分（上限20）
          - 融資連續減少天數 / streak_days_for_full_score × 10 分（上限10）
          - 融券連續減少（回補）天數 / streak_days_for_full_score × 10 分（上限10）

        回傳格式（刻意攤平每一項的原始貢獻，不是只給一個總分）：
            {
                "chip_score": 78.5,
                "is_validated": False,
                "breakdown": {
                    "foreign_buy_streak": {"days": 7, "points": 17.5, "max_points": 25},
                    ...
                },
            }
        """
        result = {}
        for code, streak in institutional_streaks.get("streaks", {}).items():
            breakdown = {}

            foreign = streak.get("foreign", {})
            foreign_days = foreign.get("days", 0) if foreign.get("direction") == "buy" else 0
            breakdown["foreign_buy_streak"] = {
                "days": foreign_days,
                "points": round(min(foreign_days / streak_days_for_full_score, 1.0)
                                 * ChipFlowEngine.CHIP_SCORE_WEIGHTS["foreign_buy_streak"], 1),
                "max_points": ChipFlowEngine.CHIP_SCORE_WEIGHTS["foreign_buy_streak"],
            }

            trust = streak.get("trust", {})
            trust_days = trust.get("days", 0) if trust.get("direction") == "buy" else 0
            breakdown["trust_buy_streak"] = {
                "days": trust_days,
                "points": round(min(trust_days / streak_days_for_full_score, 1.0)
                                 * ChipFlowEngine.CHIP_SCORE_WEIGHTS["trust_buy_streak"], 1),
                "max_points": ChipFlowEngine.CHIP_SCORE_WEIGHTS["trust_buy_streak"],
            }

            total = streak.get("total", {})
            total_days = total.get("days", 0) if total.get("direction") == "buy" else 0
            breakdown["institutional_total_buy_streak"] = {
                "days": total_days,
                "points": round(min(total_days / streak_days_for_full_score, 1.0)
                                 * ChipFlowEngine.CHIP_SCORE_WEIGHTS["institutional_total_buy_streak"], 1),
                "max_points": ChipFlowEngine.CHIP_SCORE_WEIGHTS["institutional_total_buy_streak"],
            }

            margin_info = margin_streaks.get("streaks", {}).get(code, {})
            margin_bal = margin_info.get("margin_balance", {})
            margin_days = margin_bal.get("days", 0) if margin_bal.get("direction") == "decreasing" else 0
            breakdown["margin_decreasing"] = {
                "days": margin_days,
                "points": round(min(margin_days / streak_days_for_full_score, 1.0)
                                 * ChipFlowEngine.CHIP_SCORE_WEIGHTS["margin_decreasing"], 1),
                "max_points": ChipFlowEngine.CHIP_SCORE_WEIGHTS["margin_decreasing"],
            }

            short_bal = margin_info.get("short_balance", {})
            short_days = short_bal.get("days", 0) if short_bal.get("direction") == "decreasing" else 0
            breakdown["short_covering"] = {
                "days": short_days,
                "points": round(min(short_days / streak_days_for_full_score, 1.0)
                                 * ChipFlowEngine.CHIP_SCORE_WEIGHTS["short_covering"], 1),
                "max_points": ChipFlowEngine.CHIP_SCORE_WEIGHTS["short_covering"],
            }

            chip_score = round(sum(item["points"] for item in breakdown.values()), 1)
            result[code] = {
                "chip_score": chip_score,
                "is_validated": False,  # ⚠️ 永遠是 False，這組權重從未經過回測驗證
                "breakdown": breakdown,
            }

        return result

    # ==========================================
    # Top N 排行榜（把上面幾個 compute_* 回傳的全市場字典排序、取前N名）
    # ==========================================
    @staticmethod
    def _label(code: str, names: dict) -> str:
        name = names.get(code, "") if names else ""
        return f"[{code}] {name}" if name else f"[{code}]"

    @staticmethod
    def rank_institutional_streaks(streaks_result: dict, top_n: int = 20) -> dict:
        """
        把 compute_institutional_streaks() 的回傳結果，依「外資／投信／自營商／
        三大法人合計」四個類別，各自排出「連買前N名」「連賣前N名」。

        排序鍵是「連續天數」（不是單日或累積張數），因為這個引擎的定位就是
        「連續性比單日更有參考價值」——累積張數只當作同分時的第二排序鍵。

        回傳格式：
            {
                "外資": {"連買前N名": df, "連賣前N名": df},
                "投信": {...}, "自營商": {...}, "三大法人合計": {...},
            }
        每個 df 欄位：標的 / 連續天數 / 累積張數
        """
        if streaks_result.get("status") != "ok":
            return {}

        streaks = streaks_result.get("streaks", {})
        names = streaks_result.get("names", {})
        categories = {"外資": "foreign", "投信": "trust", "自營商": "dealer", "三大法人合計": "total"}

        rankings = {}
        for label, key in categories.items():
            buy_rows, sell_rows = [], []
            for code, cat_streaks in streaks.items():
                info = cat_streaks.get(key, {})
                days = info.get("days", 0)
                if days <= 0:
                    continue
                row = {
                    "標的": ChipFlowEngine._label(code, names),
                    "代碼": code,
                    "連續天數": days,
                    "累積張數": info.get("cumulative_shares", 0) // 1000,
                }
                if info.get("direction") == "buy":
                    buy_rows.append(row)
                elif info.get("direction") == "sell":
                    sell_rows.append(row)

            buy_df = pd.DataFrame(buy_rows).sort_values(
                ["連續天數", "累積張數"], ascending=[False, False]).head(top_n).reset_index(drop=True) \
                if buy_rows else pd.DataFrame(columns=["標的", "代碼", "連續天數", "累積張數"])
            sell_df = pd.DataFrame(sell_rows).sort_values(
                ["連續天數", "累積張數"], ascending=[False, True]).head(top_n).reset_index(drop=True) \
                if sell_rows else pd.DataFrame(columns=["標的", "代碼", "連續天數", "累積張數"])

            rankings[label] = {"連買前N名": buy_df, "連賣前N名": sell_df}

        return rankings

    @staticmethod
    def rank_margin_streaks(streaks_result: dict, top_n: int = 20) -> dict:
        """
        把 compute_margin_streaks() 的回傳結果排出四張榜：
        融資連增／融資連減、融券連增（放空增加）／融券連減（回補，空方壓力下降）。

        回傳格式：
            {
                "融資": {"連增前N名": df, "連減前N名": df},
                "融券": {"連增前N名": df, "連減前N名": df},
            }
        每個 df 欄位：標的 / 連續天數 / 期間變動張數
        """
        if streaks_result.get("status") != "ok":
            return {}

        streaks = streaks_result.get("streaks", {})
        names = streaks_result.get("names", {})

        def _build(key: str) -> dict:
            up_rows, down_rows = [], []
            for code, s in streaks.items():
                info = s.get(key, {})
                days = info.get("days", 0)
                if days <= 0:
                    continue
                row = {
                    "標的": ChipFlowEngine._label(code, names),
                    "代碼": code,
                    "連續天數": days,
                    "期間變動張數": info.get("change", 0) // 1000,
                }
                if info.get("direction") == "increasing":
                    up_rows.append(row)
                elif info.get("direction") == "decreasing":
                    down_rows.append(row)
            up_df = pd.DataFrame(up_rows).sort_values("連續天數", ascending=False).head(top_n).reset_index(drop=True) \
                if up_rows else pd.DataFrame(columns=["標的", "代碼", "連續天數", "期間變動張數"])
            down_df = pd.DataFrame(down_rows).sort_values("連續天數", ascending=False).head(top_n).reset_index(drop=True) \
                if down_rows else pd.DataFrame(columns=["標的", "代碼", "連續天數", "期間變動張數"])
            return {"連增前N名": up_df, "連減前N名": down_df}

        return {
            "融資": _build("margin_balance"),
            "融券": _build("short_balance"),
        }

    @staticmethod
    def rank_chip_score(chip_score_result: dict, names: dict, top_n: int = 20) -> pd.DataFrame:
        """
        把 compute_chip_score() 的回傳結果排出 Top N。

        ⚠️ 排序用的是這個未經驗證的複合分數，排行榜本身不代表「這些股票
        已被證實籌碼最強」，只是把同一個主觀分數由高到低排列——分項天數
        欄位刻意保留展開，不只顯示總分，讓使用者能自己判斷要不要信任。
        """
        rows = []
        for code, info in chip_score_result.items():
            bd = info.get("breakdown", {})
            rows.append({
                "標的": ChipFlowEngine._label(code, names),
                "代碼": code,
                "Chip Score": info.get("chip_score", 0),
                "外資連買天數": bd.get("foreign_buy_streak", {}).get("days", 0),
                "投信連買天數": bd.get("trust_buy_streak", {}).get("days", 0),
                "三大法人合計連買天數": bd.get("institutional_total_buy_streak", {}).get("days", 0),
                "融資連減天數": bd.get("margin_decreasing", {}).get("days", 0),
                "融券連減(回補)天數": bd.get("short_covering", {}).get("days", 0),
            })
        if not rows:
            return pd.DataFrame(columns=["標的", "代碼", "Chip Score"])
        return pd.DataFrame(rows).sort_values("Chip Score", ascending=False).head(top_n).reset_index(drop=True)

    # ==========================================
    # 籌碼戰情室：一次算完全部（供 app.py 單一按鈕呼叫）
    # ==========================================
    @staticmethod
    def build_chip_flow_dashboard(lookback_days: int = 30, top_n: int = 20,
                                   use_cache: bool = True, max_age_hours: float = 6,
                                   db_path: str = None) -> dict:
        """
        整合 compute_institutional_streaks() + compute_margin_streaks() +
        compute_chip_score()，並排出所有 Top N 排行榜，給「📊 籌碼戰情室」
        頁面一次呼叫。

        ⚠️ 快取設計：跟 ChipEngine.get_market_wide_institutional_ranking() 同一套
        模式——原始的逐日累積結果（streaks/names）只跟「查詢當下的交易日」有關，
        跟 top_n 無關，所以只快取原始結果（預設6小時新鮮期，比單日排行榜的4小時
        略長，因為這裡是30天回溯運算，重算成本更高，不需要更頻繁的重新抓取）。
        Top N 排序與切片每次都重新計算，允許同一份快取搭配不同 top_n 使用。
        use_cache=False 可強制略過快取重新抓取整段歷史。

        回傳格式：
            {
                "status": "ok",
                "as_of_date": "2026-07-24",
                "days_used": 28,
                "institutional_rankings": {...},   # rank_institutional_streaks() 的結果
                "margin_rankings": {...},            # rank_margin_streaks() 的結果
                "chip_score_ranking": df,            # rank_chip_score() 的結果
                "lending_status": "not_implemented",
                "lending_message": "...",
            }
        """
        from engines.db_engine import DatabaseEngine

        cache_key = f"chip_flow_dashboard_raw_{lookback_days}"
        inst_result = None
        margin_result = None

        if use_cache:
            try:
                cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours, db_path=db_path)
                if cached:
                    inst_result = cached["payload"]["institutional"]
                    margin_result = cached["payload"]["margin"]
            except Exception:
                inst_result = None
                margin_result = None

        if inst_result is None or margin_result is None:
            inst_result = ChipFlowEngine.compute_institutional_streaks(lookback_days=lookback_days)
            margin_result = ChipFlowEngine.compute_margin_streaks(lookback_days=lookback_days)
            if inst_result.get("status") == "ok" and margin_result.get("status") == "ok":
                try:
                    DatabaseEngine.set_cache(
                        cache_key,
                        {"institutional": inst_result, "margin": margin_result},
                        db_path=db_path,
                    )
                except Exception:
                    pass  # 快取寫入失敗不影響本次回傳結果，只是下次會重抓

        if inst_result.get("status") != "ok":
            return {"status": "unavailable",
                    "message": inst_result.get("message", "⚠️ 三大法人資料無法取得，無法計算籌碼戰情室。")}

        lending_result = ChipFlowEngine.compute_lending_streaks(lookback_days=lookback_days)

        chip_score_result = {}
        if margin_result.get("status") == "ok":
            chip_score_result = ChipFlowEngine.compute_chip_score(inst_result, margin_result)

        return {
            "status": "ok",
            "as_of_date": inst_result["as_of_date"],
            "days_used": inst_result["days_used"],
            "institutional_rankings": ChipFlowEngine.rank_institutional_streaks(inst_result, top_n=top_n),
            "margin_rankings": ChipFlowEngine.rank_margin_streaks(margin_result, top_n=top_n)
            if margin_result.get("status") == "ok" else {},
            "margin_status_message": margin_result.get("message"),
            "chip_score_ranking": ChipFlowEngine.rank_chip_score(
                chip_score_result, inst_result.get("names", {}), top_n=top_n),
            "lending_status": lending_result.get("status"),
            "lending_message": lending_result.get("message"),
            # 原始逐股票結果一併帶出，供「個股深度分析」頁面查單一股票時直接
            # 從這份已抓好的全市場資料裡取自己的份，不需要為了一檔股票再重抓一次。
            "institutional_streaks_raw": inst_result.get("streaks", {}),
            "chip_score_raw": chip_score_result,
        }


