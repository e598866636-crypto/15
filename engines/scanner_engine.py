import pandas as pd

from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.risk_engine import RiskEngine
from engines.divergence_engine import DivergenceEngine
from engines.strategy_engine import StrategyEngine
from engines.momentum_engine import MomentumEngine
from engines.evidence_engine import EvidenceEngine
from engines.name_engine import NameEngine
from engines.timeframe_engine import TimeframeEngine
from engines.stock_academy_engine import StockAcademyEngine
from engines.pattern_engine import PatternEngine
from engines.rs_rating_engine import RSRatingEngine
from engines.canslim_engine import CanslimEngine
from engines.stage_engine import StageEngine

class ScannerEngine:
    """
    📡 全台股掃描引擎 (Market Scanner) - 前沿科技特化版

    ⚠️ 修正說明（v2.7）：
    1. Pipeline 新增 DivergenceEngine（背離/誘多誘空防禦）與 MomentumEngine
       （飆股七層過濾＋100分評分）兩個階段。DivergenceEngine 必須放在
       StrategyEngine 之前（避免 StrategyEngine/MomentumEngine 用到還沒算
       好的背離欄位），MomentumEngine 則放在 StrategyEngine 之後、
       EvidenceEngine 之前均可（兩者互不相依）。
    2. scan() 回傳結果新增 [代碼][名稱] 全系統標籤化欄位（來自 NameEngine），
       以及「飆股評分／飆股等級／誘盤警報」三個新欄位，供 Dashboard 的
       「A級飆股候選」與「誘盤警報雷達」區塊使用。
    3. DEFAULT_WATCHLIST 新增兩檔常見 ETF（0050、006208）作為 ETF 擴充功能
       的示範標的；ETF 沒有「產業」可言，IndustryEngine 已將其獨立歸類為
       「ETF」類別，不會污染其他產業的平均分數。

    ⚠️ 修正說明（v2.8 新增）：
    4. Pipeline 在 EvidenceEngine 之後新增 TimeframeEngine，針對每檔股票
       產出「短線／波段／長線」三週期判斷與「未來走向」情境推演。
       TimeframeEngine 是純讀取既有欄位的報告產生器（不新增 df 欄位、
       不重算任何指標），所以放在 pipeline 最後、只在 scan() 組裝結果列
       時呼叫即可，不影響其他引擎。
    5. scan() 回傳結果新增「短線建議／波段建議／長線建議／未來走向」四個
       欄位，並同步更新 StrategyEngine 的 entry_signal/exit_signal 買賣點
       欄位到「買進訊號／賣出訊號」，取代原本只顯示連續分數的呈現方式。

    ⚠️ 修正說明（v2.9 新增）：
    6. Pipeline 在 EvidenceEngine 之後新增 StockAcademyEngine（選股學院
       五維度評分：市場面／基本面／技術面／籌碼面／財務面）。全市場批次
       掃描時**刻意不**替每一檔股票額外呼叫 ChipEngine／FundamentalEngine
       （兩者背後分別是 TWSE 網頁與 yfinance `.info`，都是較慢、易有速率
       限制的外部端點，跟 K 線 `.history()` 不是同一組後端），沿用
       momentum_engine.py 對 ChipEngine 的同樣考量——避免全市場掃描被拖慢
       或被限速。因此批次掃描下的「選股評級」，籌碼面會退化成 OBV/量能
       代理評估、基本面與財務面會顯示中性分數＋「資料不足」，只有在
       app.py 的「個股深度分析」頁面才會傳入完整的 chip_report /
       fundamental_report 算出真實的五維度評分。scan() 回傳結果新增
       「選股評級／選股評分／評級標籤／市場面評分／基本面評分／技術面
       評分／籌碼面評分／財務面評分」欄位；並新增 get_academy_top_n()、
       get_dimension_weakest()、get_multi_signal_consensus() 三個輔助方法。


    ⚠️ 修正說明（v2.9.5 新增，把先前孤兒引擎接進實際運作的 pipeline）：
    7. Pipeline 在 StructureEngine 之後新增 PatternEngine.add_patterns()
       （頭肩頂/雙底雙頂/缺口分類/旗形），這個模組原本已經寫好但從未被
       任何地方 import，程式碼是「死的」，現在正式接上。介面跟其他引擎
       一致（df進df出），風險低。
    8. scan() 新增兩階段計算：第一階段掃描迴圈跟 v2.9 一樣，額外用
       RSRatingEngine.compute_raw_score() 算出每檔股票的「原始加權動能
       分數」；第二階段（掃描迴圈跑完後）用 RSRatingEngine.rank_universe()
       把這批原始分數轉成 1~99 的 RS Rating——這是跨股票的百分位排名，
       必須等所有股票都掃完才能算，所以拆成兩階段，不能像其他欄位一樣
       在單一股票的迴圈裡就地算完。⚠️ 這個 RS Rating 的排名母體是「這次
       掃描的股票池」，不是全市場，母體越小統計意義越弱，UI 必須顯示
       universe_size。
    9. scan() 新增呼叫 CanslimEngine.analyze()，組合出 CAN SLIM 評分。
       批次模式沿用 v2.9 對 ChipEngine/FundamentalEngine 的既有取捨
       （不對外呼叫，避免拖慢/限速），所以批次模式下 C/A/I 三項會顯示
       「資料不足」而非真實分數，只有 N/S/L/M 四項（技術面+RS Rating+
       大盤方向）會有實質評分；真正完整的七項評分要到「個股深度分析」
       頁面才會帶入 chip_report/fundamental_report 算出完整版本，這跟
       StockAcademyEngine 批次/個股雙軌的設計哲學一致。
    """

    # 專屬觀察名單：整合跨界生醫、生物運算、大盤權值指標與常見 ETF
    DEFAULT_WATCHLIST = [
        # --- 跨界生醫與生物電腦 ---
        "2330", "3374", "6223", "3711", # 生物晶片與先進封測
        "6841", "2382", "3231", "2356", # AI醫療與智慧生醫
        "6472", "6901", "4743", "6712", # 合成生物與前沿創投
        
        # --- 宏觀資金動向對照組 (保留部分關鍵權值股觀察大盤健康度) ---
        "2317", "2454", "2308", "2379", "3034", "2412", "2881", "2603", "1515", "1101",

        # --- ETF 示範標的（v2.7 興櫃/ETF資料擴充）---
        "0050", "006208",
    ]

    @staticmethod
    def _run_single_pipeline(ticker: str, use_cache: bool = True, max_age_hours: float = 6):
        df = DataEngine.get_stock_data(ticker, use_cache=use_cache, max_age_hours=max_age_hours)
        df = IndicatorEngine.add_indicators(df)
        df = StructureEngine.add_swing_points(df)
        df = PatternEngine.add_patterns(df)
        df = StageEngine.add_stage_analysis(df)
        df = RiskEngine.add_risk_metrics(df)
        df = RiskEngine.add_liquidity_metrics(df)  # v2.9.7 新增：流動性風險，僅用既有 close/volume，不增加外部請求
        df = DivergenceEngine.add_defense_signals(df)
        df = StrategyEngine.generate_signals(df)
        df = MomentumEngine.add_momentum_score(df)
        df = EvidenceEngine.add_evidence(df)
        # 快速模式：不傳 chip_report / fundamental_report，籌碼面退化為
        # OBV/量能代理、基本面與財務面顯示中性分數（見上方 v2.9 說明）。
        df = StockAcademyEngine.add_academy_score(df)
        return df

    @staticmethod
    def _run_timeframe_report(df: pd.DataFrame) -> dict:
        """包一層 try/except：TimeframeEngine 是報告產生器，任何欄位缺漏
        都不應該讓整條掃描 pipeline 中斷（沿用本專案一貫的防禦性設計）。"""
        try:
            return TimeframeEngine.build_report(df)
        except Exception:
            return {}

    @staticmethod
    def scan(tickers=None, use_cache: bool = True, max_age_hours: float = 6, progress_callback=None):
        tickers = tickers or ScannerEngine.DEFAULT_WATCHLIST
        results = []
        errors = []
        # v2.9.5 新增：暫存每檔股票的 df/latest/tf_report，第一輪迴圈跑完、
        # 拿到完整股票池的 RS 原始分數之後，才能算出跨股票排名，所以拆成
        # 「先收集」與「後面組裝結果列」兩段，而不是像其他欄位一樣單檔
        # 就地算完。
        _pending = []
        _raw_rs_scores = {}
        _raw_rvol = {}  # v2.9.6 新增：收集各股最新 RVOL，供跨股票相對成交量排名使用

        for i, ticker in enumerate(tickers):
            ticker = str(ticker).strip()
            if not ticker:
                continue
            try:
                df = ScannerEngine._run_single_pipeline(ticker, use_cache=use_cache, max_age_hours=max_age_hours)
                latest = df.iloc[-1]

                close_val = latest["close"]
                if hasattr(close_val, "iloc"):
                    close_val = close_val.iloc[0]

                trap_flag = bool(latest.get("trap_alert", False))
                tf_report = ScannerEngine._run_timeframe_report(df)

                rs_raw = RSRatingEngine.compute_raw_score(df)
                _raw_rs_scores[ticker] = rs_raw.get('raw_score', float('nan')) if rs_raw.get('status') == 'ok' else float('nan')

                rvol_latest = latest.get('rvol', float('nan'))
                _raw_rvol[ticker] = float(rvol_latest) if pd.notna(rvol_latest) else float('nan')

                _pending.append({
                    'ticker': ticker, 'df': df, 'latest': latest,
                    'close_val': close_val, 'trap_flag': trap_flag, 'tf_report': tf_report,
                })
            except Exception as e:
                errors.append({"代碼": ticker, "錯誤訊息": str(e)})

            if progress_callback:
                progress_callback(i + 1, len(tickers), ticker)

        # 第二輪：股票池已收集完畢，計算跨股票 RS Rating 排名與相對成交量排名，
        # 再組裝每一列結果（包含 RS Rating、批次模式下的 CAN SLIM 評分，以及
        # v2.9.6 新增：把這兩個跨股票排名真正併入 momentum_score 重新計算）。
        rs_rankings = RSRatingEngine.rank_universe(_raw_rs_scores)
        rvol_rankings = RSRatingEngine.rank_universe(_raw_rvol)  # 通用百分位排名工具，非RS專屬

        for item in _pending:
            ticker, df, latest = item['ticker'], item['df'], item['latest']
            close_val, trap_flag, tf_report = item['close_val'], item['trap_flag'], item['tf_report']

            rs_info = rs_rankings.get(ticker, {'rs_rating': None, 'universe_size': 0})
            rs_rating = rs_info.get('rs_rating')
            rvol_info = rvol_rankings.get(ticker, {'percentile': None})
            rvol_percentile = rvol_info.get('percentile')

            # v2.9.6：用完整的跨股票排名重新計算 momentum_score（覆蓋掉
            # _run_single_pipeline 裡算出的「不完整評分」，見 MomentumEngine
            # docstring 的誠實揭露）。只需要重算 MomentumEngine 這一層，
            # 不需要重跑整條 pipeline——後面 StockAcademyEngine 用到的
            # momentum_grade 也會因此拿到更新後的完整版本。
            try:
                df = MomentumEngine.add_momentum_score(df, rs_rating=rs_rating, relative_volume_percentile=rvol_percentile)
                df = StockAcademyEngine.add_academy_score(df)
                latest = df.iloc[-1]
            except Exception:
                pass  # 重算失敗時沿用第一輪的不完整評分，不中斷整列結果

            try:
                canslim_report = CanslimEngine.analyze(
                    df, fundamental_report=None, chip_report=None,
                    market_regime=latest.get("market_regime"), rs_rating=rs_rating,
                )
            except Exception:
                canslim_report = None

            results.append({
                "代碼": ticker,
                "名稱": NameEngine.get_name(ticker),
                "標的": NameEngine.get_tag(ticker),
                "市場別": NameEngine.get_market_type(ticker),
                "收盤價": round(float(close_val), 2),
                "市場狀態": latest.get("market_regime", "N/A"),
                "AI Score": round(float(latest.get("ai_score", 0)), 1),
                "飆股評分": round(float(latest.get("momentum_score", 0)), 1),
                "飆股等級": latest.get("momentum_grade", "D"),
                "買進訊號": latest.get("entry_signal", "⚪ 無明確買進訊號"),
                "賣出訊號": latest.get("exit_signal", "⚪ 無明確賣出訊號"),
                "誘盤警報": "⚠️ 是" if trap_flag else "—",
                "信心度": round(float(latest.get("confidence_pct", 0)), 0),
                "資料品質": round(float(latest.get("data_quality_pct", 0)), 0),
                "操作建議": latest.get("action_guide", "N/A"),
                "短線建議": tf_report.get("short_term", {}).get("view", "N/A"),
                "波段建議": tf_report.get("swing", {}).get("view", "N/A"),
                "長線建議": tf_report.get("long_term", {}).get("view", "N/A"),
                "未來走向": tf_report.get("outlook", {}).get("bias", "N/A"),
                "年化波動率": round(float(latest.get("volatility_annualized", float("nan"))), 1)
                    if pd.notna(latest.get("volatility_annualized", float("nan"))) else None,
                "60日回撤": round(float(latest.get("rolling_mdd_60d", float("nan"))), 1)
                    if pd.notna(latest.get("rolling_mdd_60d", float("nan"))) else None,
                "選股評級": latest.get("academy_grade", "F"),
                "選股評分": int(latest.get("academy_total_score", 0)),
                "評級標籤": latest.get("academy_label", ""),
                "市場面評分": int(latest.get("academy_market_score", 0)),
                "基本面評分": int(latest.get("academy_fundamental_score", 0)),
                "技術面評分": int(latest.get("academy_technical_score", 0)),
                "籌碼面評分": int(latest.get("academy_chip_score", 0)),
                "財務面評分": int(latest.get("academy_financial_score", 0)),
                # v2.9.5 新增欄位
                "RS Rating": rs_rating if rs_rating is not None else None,
                "RS母體數": rs_info.get("universe_size", 0),
                "相對成交量排名": round(rvol_percentile, 0) if rvol_percentile is not None else None,
                "飆股評分完整": bool(latest.get("momentum_score_complete", False)),
                "CANSLIM評分": canslim_report.get("total_score") if canslim_report else None,
                "CANSLIM等級": canslim_report.get("grade") if canslim_report else "N/A（計算失敗）",
                "頭肩型態": ("🔴 頭肩頂確認" if bool(latest.get("hs_top_confirmed", False))
                          else ("🟢 頭肩底確認" if bool(latest.get("hs_bottom_confirmed", False)) else "—")),
                "雙頂雙底": ("🔴 M頭確認" if bool(latest.get("double_top_confirmed", False))
                          else ("🟢 W底確認" if bool(latest.get("double_bottom_confirmed", False)) else "—")),
                "缺口類型": latest.get("gap_type_confirmed", "") or "—",
                "階段分析": latest.get("stage_label", "N/A"),
                # v2.9.7 新增：流動性風險（見 risk_engine.py add_liquidity_metrics 說明），
                # 只用既有 close/volume 計算，不增加額外外部請求。
                "流動性": latest.get("liquidity_level", "資料不足"),
                "日均成交值(百萬)": round(float(latest.get("avg_trading_value_20d", float("nan"))) / 1e6, 1)
                    if pd.notna(latest.get("avg_trading_value_20d", float("nan"))) else None,
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("AI Score", ascending=False).reset_index(drop=True)
            result_df.insert(0, "排名", range(1, len(result_df) + 1))

        error_df = pd.DataFrame(errors)
        return result_df, error_df

    @staticmethod
    def get_top_n(result_df: pd.DataFrame, n: int = 10):
        if result_df is None or result_df.empty:
            return result_df
        return result_df.head(n)

    @staticmethod
    def get_a_grade_candidates(result_df: pd.DataFrame):
        """
        🚀 A級飆股候選 (Momentum A-Grade Candidates)

        篩選 MomentumEngine 七層過濾＋100分評分系統中，飆股等級為 A
        （總分 >= 85，且已通過年線濾網硬性關卡）的標的，依飆股評分排序。
        """
        if result_df is None or result_df.empty or "飆股等級" not in result_df.columns:
            return pd.DataFrame()
        a_grade = result_df[result_df["飆股等級"] == "A"].copy()
        a_grade = a_grade.sort_values("飆股評分", ascending=False).reset_index(drop=True)
        return a_grade

    @staticmethod
    def get_trap_alerts(result_df: pd.DataFrame):
        """
        🛡️ 誘盤警報雷達 (Fake-Signal / Trap Radar)

        篩選近期觸發 DivergenceEngine 背離或假突破/假跌破警報的標的，
        依 AI Score 排序（分數越高但同時觸發警報的標的，代表『看起來還不錯
        但動能可能已經在減弱』，最值得優先留意）。
        """
        if result_df is None or result_df.empty or "誘盤警報" not in result_df.columns:
            return pd.DataFrame()
        alerts = result_df[result_df["誘盤警報"] != "—"].copy()
        alerts = alerts.sort_values("AI Score", ascending=False).reset_index(drop=True)
        return alerts

    @staticmethod
    def get_academy_top_n(result_df: pd.DataFrame, n: int = 10, min_grade: str = "B"):
        """
        🎓 選股大師 Top-N (依「選股評級」由高到低篩選)

        參數：
            min_grade : 最低評級門檻（例如 "B"：只看 B 等以上），對應
                        StockAcademyEngine.GRADE_SCALE 的等級順序。

        ⚠️ 提醒：批次掃描的「選股評級」是快速模式估算（籌碼面用OBV代理、
        基本面/財務面為中性分數），僅供初步篩選；決定下單前建議點進
        「個股深度分析」頁面查看該股完整的五維度評分。
        """
        if result_df is None or result_df.empty or "選股評級" not in result_df.columns:
            return pd.DataFrame()

        grade_order = {g[1]: i for i, g in enumerate(StockAcademyEngine.GRADE_SCALE)}
        min_rank = grade_order.get(min_grade, len(StockAcademyEngine.GRADE_SCALE))
        filtered = result_df[result_df["選股評級"].map(lambda x: grade_order.get(x, len(StockAcademyEngine.GRADE_SCALE))) <= min_rank]

        filtered = filtered.sort_values("選股評分", ascending=False).reset_index(drop=True)
        return filtered.head(n)

    @staticmethod
    def get_dimension_weakest(result_df: pd.DataFrame):
        """
        🎓 找出本次掃描名單中，五維度裡平均分數最低／最高的維度。

        用途：了解整體觀察名單在選股學院框架下的共同優勢與弱點（例如：
        「這批股票技術面普遍很強，但財務面普遍偏弱」）。
        """
        if result_df is None or result_df.empty:
            return {}

        dims = ["市場面評分", "基本面評分", "技術面評分", "籌碼面評分", "財務面評分"]
        available_dims = [d for d in dims if d in result_df.columns]
        if not available_dims:
            return {}

        avg_scores = {d: round(float(result_df[d].mean()), 1) for d in available_dims}
        sorted_dims = sorted(avg_scores.items(), key=lambda x: x[1])

        return {
            "全市場平均維度評分": avg_scores,
            "最弱維度": sorted_dims[0][0],
            "最弱維度平均分": sorted_dims[0][1],
            "最強維度": sorted_dims[-1][0],
            "最強維度平均分": sorted_dims[-1][1],
        }

    @staticmethod
    def get_multi_signal_consensus(result_df: pd.DataFrame):
        """
        🎓 三信號共識：找出「AI Score（短期）、飆股評分（動能）、選股評級
        （中長期，快速模式）」三者都看好的標的，三套評分系統各司其職，
        同時共識最強代表短中長期角度一致。
        """
        if result_df is None or result_df.empty:
            return pd.DataFrame()

        required_cols = {"AI Score", "飆股等級", "選股評級", "飆股評分", "選股評分"}
        if not required_cols.issubset(result_df.columns):
            return pd.DataFrame()

        consensus = result_df[
            (result_df["AI Score"] >= 70) &
            (result_df["飆股等級"].isin(["A", "B"])) &
            (result_df["選股評級"].isin(["A+", "A", "B+", "B"]))
        ].copy()

        if consensus.empty:
            return consensus

        consensus["共識強度"] = (
            (consensus["AI Score"] / 100) * 0.35 +
            (consensus["飆股評分"] / 100) * 0.30 +
            (consensus["選股評分"] / 100) * 0.35
        ) * 100

        return consensus.sort_values("共識強度", ascending=False).reset_index(drop=True)

    # ==========================================
    # 🚦 Hard Gate 篩選漏斗 (v2.9.10 新增)
    # ==========================================
    # 動機：使用者（以「機構等級升級建議」文件回饋）提出的「Universe → Market
    # → Liquidity → Fundamental → RS → Pattern → 候選名單」漏斗式流程。
    # ⚠️ 誠實範圍界定：這裡是「對已經跑完 scan() 的 result_df 做事後依序
    # 篩選並統計每關淘汰多少檔」，不是重寫 scan() 內部的計算順序去做真正
    # 的短路計算（跳過還沒篩到的股票、不去算後面關卡的指標）。原因：
    # scan() 現在的兩階段設計（第一輪收集全部股票的原始指標，第二輪才能
    # 算出跨股票的 RS Rating／相對成交量排名）本身有這樣設計的理由——RS
    # Rating 需要「整個母體」都算完才能排名，沒辦法對單一股票提早算完就
    # 短路跳過；真的要做「早期淘汰、後面關卡直接不算」的效能優化，等於要
    # 把 RS/CANSLIM 這類跨股票或多層計算拆解成可以分階段執行的版本，
    # 是更大範圍的重構，風險（可能引入新 bug）與效益（yfinance 網路請求
    # 才是主要耗時來源，不是這些計算本身）不成比例，這次不做。這裡做的
    # 「事後依序篩選＋統計」已經足以達成使用者要的核心價值：清楚看到
    # 「為什麼從 2450 檔剩下 5 檔」的漏斗，而不需要冒風險重寫已經除錯過
    # 很多輪的核心計算流程。
    @staticmethod
    def apply_hard_gates(
        result_df: pd.DataFrame,
        require_bullish_market: bool = True,
        exclude_illiquid: bool = True,
        min_canslim_score: float = 60,
        min_rs_rating: float = 70,
        min_momentum_grade: str = "B",
    ) -> dict:
        """
        依序套用五道關卡，任何一關未通過就從候選名單移除，並記錄「在哪一關
        被淘汰」。任一關的判斷欄位缺失時，該關卡對缺值的股票一律「放行」
        （不淘汰），因為「資料不足」不等於「不合格」，這裡不假裝知道答案。

        回傳：
            {
                'funnel': [{'gate': 'Universe', 'count': 2450}, ...],
                'passed': DataFrame（通過全部關卡的候選股）,
                'rejected': DataFrame（含新增的「淘汰關卡」欄位，說明在哪關被刷掉）,
            }
        """
        if result_df is None or result_df.empty:
            return {'funnel': [], 'passed': pd.DataFrame(), 'rejected': pd.DataFrame()}

        df = result_df.copy()
        df["淘汰關卡"] = None
        funnel = [{"gate": "Universe（掃描全部標的）", "count": len(df)}]

        def _apply_gate(mask_pass: pd.Series, gate_name: str):
            nonlocal df
            still_alive = df["淘汰關卡"].isna()
            newly_rejected = still_alive & (~mask_pass)
            df.loc[newly_rejected, "淘汰關卡"] = gate_name
            alive_count = int((df["淘汰關卡"].isna()).sum())
            funnel.append({"gate": gate_name, "count": alive_count})

        # Gate 1：市場趨勢（該股本身的市場狀態非空頭）
        if require_bullish_market and "市場狀態" in df.columns:
            mask = ~df["市場狀態"].astype(str).str.contains("空頭", na=False)
            _apply_gate(mask, "Gate 1：市場趨勢（排除空頭）")

        # Gate 2：流動性（排除極低流動性）
        if exclude_illiquid and "流動性" in df.columns:
            mask = ~df["流動性"].astype(str).str.contains("🔴", na=False)
            _apply_gate(mask, "Gate 2：流動性（排除極低流動性）")

        # Gate 3：基本面 / CAN SLIM（缺值放行，不評分不代表不合格）
        if "CANSLIM評分" in df.columns:
            mask = df["CANSLIM評分"].isna() | (df["CANSLIM評分"] >= min_canslim_score)
            _apply_gate(mask, f"Gate 3：基本面（CAN SLIM ≥ {min_canslim_score}）")

        # Gate 4：相對強度（缺值放行）
        if "RS Rating" in df.columns:
            mask = df["RS Rating"].isna() | (df["RS Rating"] >= min_rs_rating)
            _apply_gate(mask, f"Gate 4：相對強度（RS Rating ≥ {min_rs_rating}）")

        # Gate 5：型態動能（飆股等級達標，A/B優於C/D；缺值放行）
        grade_rank = {"A": 4, "B": 3, "C": 2, "D": 1}
        min_rank = grade_rank.get(min_momentum_grade, 3)
        if "飆股等級" in df.columns:
            mask = df["飆股等級"].map(lambda g: grade_rank.get(g, min_rank) >= min_rank if pd.notna(g) else True)
            _apply_gate(mask, f"Gate 5：型態動能（飆股等級 ≥ {min_momentum_grade}）")

        passed = df[df["淘汰關卡"].isna()].drop(columns=["淘汰關卡"]).reset_index(drop=True)
        rejected = df[df["淘汰關卡"].notna()].reset_index(drop=True)

        return {"funnel": funnel, "passed": passed, "rejected": rejected}