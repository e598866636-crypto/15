import numpy as np
import pandas as pd
import yfinance as yf

from engines.logging_config import get_logger

logger = get_logger(__name__)


class MacroEngine:
    """
    🌍 總體經濟與跨市場關聯引擎 (Macro & Cross-Market Engine) - TQAI Pro v2.9

    ⚠️ 範圍界定（誠實揭露，避免功能包裝過度）：
    使用者提供的「多因子 AI 決策模型」文件列出了非常完整的清單：總體經濟、
    央行政策、外資資金流、產業輪動、基本面、技術面、籌碼面、選擇權、
    國際市場、期貨、商品、匯率、新聞情緒、社群情緒、法人報告、季節循環、
    事件驅動、AI 融合模型、風險指標……這是一份完整的量化研究藍圖，但其中
    不少項目（新聞情緒 NLP/FinBERT、社群情緒爬蟲、選擇權完整鏈 Greeks、
    券商分點籌碼、法人目標價報告、LSTM/GNN 融合模型）需要付費資料源、
    額外的爬蟲基礎設施，或大型模型推論資源——這些都不是能在沒有對應資料
    源/金鑰的情況下誠實交付的範圍。硬是拼湊一個「看起來有」但資料是規則
    拼湊或假造的版本，比不做還糟，容易讓使用者誤信虛假訊號。

    這個引擎鎖定「用 yfinance 就能真實取得、且跟現有 DataEngine 架構一致」
    的子集合，對應文件裡：
      - 十九、風險指標（VIX 恐慌指數）
      - 十二、匯率（美元/台幣）
      - 十一、商品市場（黃金、原油）
      - 九、國際市場（那斯達克、標普500、費城半導體指數 SOX）
      - 一、總體經濟（美國十年期公債殖利率——資金成本代理指標、美元指數）

    其餘項目（央行決策文字、新聞/社群情緒、選擇權、法人報告、AI融合模型）
    若要繼續擴充，建議先討論資料源可行性，而不是一次性假造。

    ⚠️ 因果安全性：這裡全部是「當下可觀察」的市場報價資料（跟個股一樣
    來自 yfinance 歷史行情），不涉及任何需要未來才能確認的轉折判斷，沒有
    look-ahead bias 疑慮，可以放心當作背景資訊使用。

    ⚠️ 這個引擎產出的是「背景總經環境」的參考資訊，不直接餵進
    StrategyEngine/MomentumEngine 影響個股 ai_score/momentum_score
    ——总經環境對個股的實際影響因產業、營收結構而異，不應該用同一組
    總經訊號無差別套用到所有股票上，避免又一次「同一組事實被不同 Agent
    各自計分、疊加放大」的問題（沿用 strategy_engine.py 的既有教訓）。
    這裡設計成獨立的參考儀表板，由使用者自行綜合判斷。
    """

    # yfinance 代碼：涵蓋美元指數、美元兌台幣、VIX、黃金、原油、
    # 美國十年期公債殖利率、那斯達克、標普500、費城半導體指數(SOX)、台灣加權指數
    #
    # ⚠️ 單位確認（v2.9.1）：^TNX（美國十年期公債殖利率）在 Yahoo Finance／
    # yfinance 上直接是殖利率本身的數值（例如 4.54 代表 4.54%），不是乘以10
    # 的版本——這裡曾經有過這個疑慮沒有查證，已透過網路搜尋確認 Yahoo
    # Finance官方頁面顯示的數字（例如 Previous Close 4.5390）就是殖利率
    # 本身，程式碼不需要額外除以10，維持現狀即可。
    #
    # ⚠️ 單位查證：^TNX（美國十年期公債殖利率）在 Yahoo Finance 上顯示的
    # 數值就是直接的殖利率百分比本身（例如顯示 4.54 代表 4.54%），不需要
    # 額外除以10——這點已於 2026-07 實際查證 Yahoo Finance 頁面確認，
    # 下面 latest 欄位可以直接當百分比讀。
    TICKERS = {
        "美元指數 (DXY)": "DX-Y.NYB",
        "美元/台幣": "TWD=X",
        "VIX 恐慌指數": "^VIX",
        "黃金 (Gold Futures)": "GC=F",
        "原油 (WTI Crude)": "CL=F",
        "美國十年期公債殖利率": "^TNX",
        "那斯達克指數": "^IXIC",
        "標普500指數": "^GSPC",
        "費城半導體指數 (SOX)": "^SOX",
        "台灣加權指數": "^TWII",
    }

    _last_fetch_error = {}

    @staticmethod
    def _record_fetch_error(symbol: str, message: str):
        """統一記錄抓取失敗原因：同時寫進 _last_fetch_error（Dashboard顯示用）
        跟本地 log 檔案（見 logging_config.py），避免每次都要靠使用者截圖
        重現錯誤才能排查。"""
        MacroEngine._last_fetch_error[symbol] = message
        logger.warning(f"[{symbol}] {message}")

    @staticmethod
    def _fetch_one(symbol: str, period: str = "6mo"):
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                MacroEngine._record_fetch_error(symbol, "yfinance回應內容為空（連線成功但沒有資料）")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [str(c).strip().lower() for c in df.columns]
            df = df.dropna(subset=['close']).reset_index(drop=True)
            if df.empty:
                MacroEngine._record_fetch_error(symbol, "資料下載成功但清理後沒有有效收盤價")
                return None
            MacroEngine._last_fetch_error.pop(symbol, None)
            return df
        except Exception as e:
            MacroEngine._record_fetch_error(symbol, f"{type(e).__name__}: {e}")
            return None

    @staticmethod
    def get_snapshot(period: str = "6mo", use_cache: bool = True, max_age_hours: float = 1) -> dict:
        """
        回傳每個總經代理指標的最新值、日變動%、5日變動%、20日變動%。
        任何一檔查詢失敗都不影響其他項目（防禦性設計，沿用本專案風格），
        該項目會標記 status='unavailable'，並附上技術細節（例外類型/訊息，
        比照 chip_engine.py／options_engine.py 已驗證有效的診斷模式），
        方便判斷是網路問題還是 yfinance 代碼失效。

        ⚠️ 修正說明（v2.9.1）：原本完全沒有走快取，每次點擊「抓取最新
        總經數據」都對 yfinance 發送 10 次獨立請求。這裡新增快取
        （預設1小時新鮮期限——比個股/籌碼資料短，因為總經跨市場報價
        變動較快，但仍遠高於即時，避免短時間內重複點擊時對 yfinance
        發送過多請求）。
        """
        from engines.db_engine import DatabaseEngine

        cache_key = f"macro_snapshot_{period}"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    return cached["payload"]
                except Exception:
                    pass

        results = {}
        for name, symbol in MacroEngine.TICKERS.items():
            df = MacroEngine._fetch_one(symbol, period=period)
            if df is None:
                detail = MacroEngine._last_fetch_error.get(symbol, "未知原因")
                results[name] = {"status": "unavailable", "symbol": symbol, "message": f"技術細節：{detail}"}
                continue

            close = df['close']
            latest = float(close.iloc[-1])
            prev_1d = float(close.iloc[-2]) if len(close) >= 2 else np.nan
            prev_5d = float(close.iloc[-6]) if len(close) >= 6 else np.nan
            prev_20d = float(close.iloc[-21]) if len(close) >= 21 else np.nan

            chg_1d = (latest / prev_1d - 1) * 100 if pd.notna(prev_1d) and prev_1d != 0 else np.nan
            chg_5d = (latest / prev_5d - 1) * 100 if pd.notna(prev_5d) and prev_5d != 0 else np.nan
            chg_20d = (latest / prev_20d - 1) * 100 if pd.notna(prev_20d) and prev_20d != 0 else np.nan

            results[name] = {
                "status": "ok",
                "symbol": symbol,
                "latest": round(latest, 2),
                "chg_1d_pct": round(chg_1d, 2) if pd.notna(chg_1d) else None,
                "chg_5d_pct": round(chg_5d, 2) if pd.notna(chg_5d) else None,
                "chg_20d_pct": round(chg_20d, 2) if pd.notna(chg_20d) else None,
            }

        if use_cache:
            try:
                DatabaseEngine.set_cache(cache_key, results, db_path=None)
            except Exception:
                pass

        return results

    @staticmethod
    def build_macro_flags(snapshot: dict = None, period: str = "6mo") -> list:
        """
        依市場上常見的經驗法則，把總經快照轉成人類可讀的訊號清單
        （供 Dashboard 顯示）。

        ⚠️ 這些規則是常見的方向性經驗法則（例如美元走強對亞股/出口股偏空、
        VIX 過高代表恐慌情緒），不是嚴謹的統計顯著性檢定結果，僅供參考，
        不構成投資建議，也不會用來計算任何股票的 ai_score。
        """
        snapshot = snapshot or MacroEngine.get_snapshot(period=period)
        flags = []

        def g(name):
            item = snapshot.get(name, {})
            return item if item.get("status") == "ok" else None

        vix = g("VIX 恐慌指數")
        if vix:
            if vix["latest"] >= 30:
                flags.append(f"🔴 VIX={vix['latest']}，市場恐慌情緒偏高，波動風險上升")
            elif vix["latest"] <= 15:
                flags.append(f"🟢 VIX={vix['latest']}，市場情緒平穩偏樂觀")

        dxy = g("美元指數 (DXY)")
        if dxy and dxy["chg_20d_pct"] is not None:
            if dxy["chg_20d_pct"] > 2:
                flags.append(f"⚠️ 美元指數近20日走強 {dxy['chg_20d_pct']}%，經驗上對亞洲股市（含台股）與電子/出口股偏空")
            elif dxy["chg_20d_pct"] < -2:
                flags.append(f"ℹ️ 美元指數近20日走弱 {dxy['chg_20d_pct']}%，對亞洲股市與出口股相對有利")

        tnx = g("美國十年期公債殖利率")
        if tnx and tnx["chg_20d_pct"] is not None and tnx["chg_20d_pct"] > 5:
            flags.append(f"⚠️ 美國十年期公債殖利率近20日快速上升 {tnx['chg_20d_pct']}%，資金成本上升，經驗上對成長股估值偏空")

        sox = g("費城半導體指數 (SOX)")
        twii = g("台灣加權指數")
        if sox and twii and sox["chg_5d_pct"] is not None and twii["chg_5d_pct"] is not None:
            if sox["chg_5d_pct"] - twii["chg_5d_pct"] > 3:
                flags.append(f"ℹ️ 費半近5日 {sox['chg_5d_pct']}% 明顯領先台股同期 {twii['chg_5d_pct']}%，留意連動性/補漲是否即將顯現")

        gold = g("黃金 (Gold Futures)")
        if gold and gold["chg_20d_pct"] is not None and gold["chg_20d_pct"] > 5:
            flags.append(f"ℹ️ 黃金近20日大漲 {gold['chg_20d_pct']}%，市場避險需求上升的訊號之一")

        if not flags:
            flags.append("✅ 目前總經代理指標未觸發內建的示警規則，屬於中性狀態")

        return flags
