import numpy as np
import pandas as pd

from engines.industry_engine import IndustryEngine


class SectorRotationEngine:
    """
    🔁 產業輪動引擎 (Sector Rotation Engine) - TQAI Pro v2.9

    對應使用者提供的多因子決策文件「四、產業輪動」章節：判斷資金目前正在
    流向哪個產業、哪個產業動能正在加速或衰退。

    ⚠️ 設計取捨與誠實揭露（避免功能包裝過度）：
      1. 這裡用的是「等權重」平均報酬曲線（每檔成分股權重相同），不是真正
         market-cap weighted 的產業指數。真正的產業指數（例如可以對照的
         申萬/中信產業指數）通常是市值加權，單一小型股大漲在這裡會跟
         權值股一樣影響「產業平均」，代表性跟實際資金流向的影響力不完全
         成正比，僅供近似參考，不是嚴謹的產業研究工具。
      2. 樣本僅涵蓋呼叫端傳入的 tickers（預設是 ScannerEngine.DEFAULT_WATCHLIST，
         或使用者這次掃描的自訂清單），而 IndustryEngine.INDUSTRY_MAP 目前
         也只對這個觀察名單做了分類——不是全市場產業輪動，部分產業可能
         只有 1~2 檔成分股，代表性有限。
      3. 沒有另外呼叫新的外部資料源：直接複用 DataEngine.get_stock_data()
         已經抓回來（且通常已經有 SQLite 快取）的歷史股價，不會增加額外
         對外請求負擔。

    ⚠️ 因果安全性：報酬曲線全部由歷史收盤價計算（不使用任何未來資料），
    沒有 look-ahead bias 疑慮；但「輪動訊號」是排名變化的方向性描述，
    不是精確的資金流向偵測（本專案沒有分點籌碼等真正的資金流向資料源，
    見 macro_engine.py 對「籌碼面」限制的同樣說明）。
    """

    # ETF／未分類的代碼不適合併入單一產業做輪動比較（ETF本身就是一籃子
    # 股票，「未分類」代表 IndustryEngine 沒有替它分類），一律排除。
    _EXCLUDED_INDUSTRIES = {"ETF", "其他/未分類"}

    @staticmethod
    def build_industry_return_curves(price_data: dict) -> pd.DataFrame:
        """
        price_data: {code: df}，df 需含 'date' 與 'close' 欄位
                    （即 DataEngine.get_stock_data() 的回傳格式）。

        回傳：寬表，index=date，columns=產業名稱，值=該產業內成分股「等權重
        平均」的累積報酬指數（以該股票自己歷史資料的第一天為基準=100，
        逐股正規化後再取橫向平均）。
        """
        if not price_data:
            return pd.DataFrame()

        normalized_series = {}
        industry_of = {}
        for code, df in price_data.items():
            if df is None or df.empty or "close" not in df.columns or "date" not in df.columns:
                continue
            industry = IndustryEngine.get_industry(code)
            if industry in SectorRotationEngine._EXCLUDED_INDUSTRIES:
                continue

            s = df.set_index("date")["close"].astype(float).sort_index()
            s = s[s.notna()]
            if s.empty or s.iloc[0] == 0:
                continue
            normalized_series[code] = s / s.iloc[0] * 100
            industry_of[code] = industry

        if not normalized_series:
            return pd.DataFrame()

        # pd.DataFrame(dict_of_series) 會自動用 index 聯集對齊，不同股票停牌/
        # 資料起始日不同時，缺值會是 NaN，下面 groupby 橫向平均時用
        # skipna（.mean 預設行為）自然跳過，不需要額外處理。
        wide = pd.DataFrame(normalized_series).sort_index()

        industry_cols = {}
        for industry in sorted(set(industry_of.values())):
            codes_in_industry = [c for c, ind in industry_of.items() if ind == industry]
            industry_cols[industry] = wide[codes_in_industry].mean(axis=1)

        result = pd.DataFrame(industry_cols).dropna(how="all")
        return result

    @staticmethod
    def rank_rotation(return_curve_df: pd.DataFrame) -> pd.DataFrame:
        """
        依產業報酬曲線計算 5日/20日/60日報酬率，並比較「5日排名」相對
        「20日排名」的變化，近似判斷資金是否正在輪入/輪出這個產業：
          - 5日排名比20日排名進步很多（數字變小）→ 近期動能明顯轉強，
            疑似資金輪入
          - 5日排名比20日排名退步很多 → 近期動能明顯轉弱，疑似資金輪出
        這是排名變化的方向性描述，不是精確的資金流向量測。
        """
        if return_curve_df is None or return_curve_df.empty:
            return pd.DataFrame()

        latest = return_curve_df.iloc[-1]

        def _chg(n_days):
            if len(return_curve_df) <= n_days:
                return pd.Series(np.nan, index=return_curve_df.columns)
            base = return_curve_df.iloc[-1 - n_days]
            return (latest / base - 1) * 100

        chg_5d = _chg(5)
        chg_20d = _chg(20)
        chg_60d = _chg(60)

        result = pd.DataFrame({
            "產業": return_curve_df.columns,
            "5日報酬%": chg_5d.values,
            "20日報酬%": chg_20d.values,
            "60日報酬%": chg_60d.values,
        })

        result["20日排名"] = result["20日報酬%"].rank(ascending=False, method="min")
        result["5日排名"] = result["5日報酬%"].rank(ascending=False, method="min")
        result["排名變化"] = result["20日排名"] - result["5日排名"]

        def _flag(row):
            if pd.isna(row["排名變化"]):
                return "ℹ️ 資料不足（歷史資料不到5或20個交易日）"
            if row["排名變化"] >= 3:
                return "🔄 疑似資金輪入（近5日排名明顯進步）"
            elif row["排名變化"] <= -3:
                return "📤 疑似資金輪出（近5日排名明顯退步）"
            return "➖ 排名相對穩定"

        result["輪動訊號"] = result.apply(_flag, axis=1)
        result = result.sort_values("20日報酬%", ascending=False, na_position="last").reset_index(drop=True)
        result.insert(0, "排名", range(1, len(result) + 1))
        return result

    @staticmethod
    def build_rotation_report(tickers=None, use_cache: bool = True, max_age_hours: float = 6) -> dict:
        """
        整合方法：抓取 tickers 的歷史股價（優先吃 DataEngine 既有快取，
        通常不會產生額外對外請求）、依 IndustryEngine 分類、計算等權重
        產業報酬曲線與輪動排名。

        回傳：
            {
                'status': 'ok' / 'unavailable',
                'return_curves': DataFrame (index=date, columns=產業),
                'rotation_table': DataFrame（見 rank_rotation() 說明）,
                'failed_tickers': [...],  # 歷史股價抓取失敗的股票代碼
            }
        單一股票抓取失敗不影響其他股票（防禦性設計，沿用本專案風格）。
        """
        from engines.data_engine import DataEngine

        if not tickers:
            from engines.scanner_engine import ScannerEngine
            tickers = ScannerEngine.DEFAULT_WATCHLIST

        price_data = {}
        failed = []
        for t in tickers:
            t = str(t).strip()
            if not t:
                continue
            try:
                df = DataEngine.get_stock_data(t, use_cache=use_cache, max_age_hours=max_age_hours)
                price_data[t] = df
            except Exception:
                failed.append(t)

        return_curves = SectorRotationEngine.build_industry_return_curves(price_data)
        if return_curves.empty:
            return {
                "status": "unavailable",
                "return_curves": pd.DataFrame(),
                "rotation_table": pd.DataFrame(),
                "failed_tickers": failed,
            }

        rotation_table = SectorRotationEngine.rank_rotation(return_curves)
        return {
            "status": "ok",
            "return_curves": return_curves,
            "rotation_table": rotation_table,
            "failed_tickers": failed,
        }
