import io

import pandas as pd
import requests

from engines.logging_config import get_logger

logger = get_logger(__name__)


class OptionsEngine:
    """
    📐 選擇權籌碼引擎 (Options Sentiment Engine) - TQAI Pro v2.9

    對應多因子決策文件「八、選擇權資料」。資料源：臺灣期貨交易所(TAIFEX)
    官方公開網頁「臺指選擇權Put/Call比」
    (https://www.taifex.com.tw/cht/3/pcRatioExcel)——公開資料、免費、
    不需要金鑰，已實際抓取驗證過欄位格式與資料屬實（非憑印象假設）。

    ⚠️ 誠實揭露（範圍限制，務必先讀完再使用）：
      1. 只涵蓋「臺指選擇權(TXO)」這個大盤指數選擇權商品的 Put/Call Ratio
         （成交量與未平倉量兩種），不含個股選擇權、不含 Greeks
         （Delta/Gamma/Vega/Theta）、不含隱含波動率(IV)——那些需要完整的
         選擇權鏈報價資料，TAIFEX雖然也有公開部分資料（例如選擇權每日
         Delta值），但格式較複雜、這次沒有實際抓取驗證過，暫不支援，
         避免用沒驗證過的假設格式硬做。
      2. 這個頁面沒有提供公開的JSON/CSV純資料API，是TAIFEX官方的HTML
         表格頁面，這裡用pandas.read_html解析表格——如果TAIFEX改版頁面
         結構，這裡的解析邏輯需要同步調整（沿用stock_directory_engine.py
         解析TWSE ISIN頁面的同樣技巧與同樣的維護風險，已有防禦性檢查：
         解析不出預期欄位時回傳空表，不會用錯位欄位硬算）。
      3. 預設頁面只顯示最近約20個交易日，不是任意long-range歷史查詢；
         若要更長歷史需要另外處理TAIFEX的日期區間查詢參數，目前未驗證，
         暫不支援，只呈現頁面預設的近期窗口。
      4. Put/Call Ratio 的解讀本身在市場上就有分歧：PCR偏高，有人解讀為
         「避險/看空需求較高」，也有人解讀為「情緒過度悲觀的反轉訊號」；
         PCR偏低同樣有兩種相反的解讀角度。本引擎只呈現數字與雙向的常見
         解讀方向，不做絕對的多空判定，也不構成投資建議。
      5. PCR反映的是大盤（臺指）選擇權市場的整體氣氛，不是個股訊號，
         不應該直接套用到個股的進出場判斷上。

    ⚠️ 因果安全性：這裡只用官方已公告的歷史交易日資料，不涉及任何需要
    未來才能確認的判斷，沒有 look-ahead bias 疑慮。
    """

    PCR_URL = "https://www.taifex.com.tw/cht/3/pcRatioExcel"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    }

    _COLUMN_RENAME = {
        "日期": "date",
        "賣權成交量": "put_volume",
        "買權成交量": "call_volume",
        "買賣權成交量比率%": "pcr_volume_pct",
        "賣權未平倉量": "put_oi",
        "買權未平倉量": "call_oi",
        "買賣權未平倉量比率%": "pcr_oi_pct",
    }

    _last_fetch_error = None

    @staticmethod
    def _record_fetch_error(message):
        """統一記錄抓取失敗原因：同時寫進 _last_fetch_error（Dashboard顯示用）
        跟本地 log 檔案（見 logging_config.py），避免每次都要靠使用者截圖
        重現錯誤才能排查。"""
        OptionsEngine._last_fetch_error = message
        logger.warning(message)

    @staticmethod
    def fetch_txo_put_call_ratio(use_cache: bool = True, max_age_hours: float = 4) -> pd.DataFrame:
        """
        抓取並解析 TAIFEX 臺指選擇權Put/Call比頁面。任何解析失敗都回傳空
        DataFrame，不拋例外中斷呼叫端（沿用本專案一貫的防禦性設計）。
        快取新鮮期限預設4小時（比籌碼資料的快取期限短，因為盤中/盤後
        資料更新頻率較高，但仍遠高於即時，不需要每次都重抓）。
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "taifex_txo_pcr"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    return pd.DataFrame(cached["payload"]["records"])
                except Exception:
                    pass

        try:
            resp = requests.get(OptionsEngine.PCR_URL, headers=OptionsEngine.HEADERS, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
        except Exception as e:
            OptionsEngine._record_fetch_error(f"{type(e).__name__}: {e}")
            return pd.DataFrame()

        df = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("賣權成交量" in c for c in cols):
                df = t
                break
        if df is None or df.empty:
            OptionsEngine._record_fetch_error("頁面解析成功但找不到PCR表格（TAIFEX可能已改版頁面結構）")
            return pd.DataFrame()

        df.columns = [str(c).strip() for c in df.columns]
        if not set(OptionsEngine._COLUMN_RENAME.keys()).issubset(set(df.columns)):
            # 欄位跟預期不符（TAIFEX可能改版了頁面結構），如實回傳空表，
            # 不要用錯位的欄位硬算，避免產生看似正常、實則張冠李戴的數字。
            OptionsEngine._record_fetch_error(f"欄位格式與預期不符，實際欄位：{list(df.columns)[:10]}")
            return pd.DataFrame()

        OptionsEngine._last_fetch_error = None

        df = df.rename(columns=OptionsEngine._COLUMN_RENAME)
        numeric_cols = ["put_volume", "call_volume", "pcr_volume_pct", "put_oi", "call_oi", "pcr_oi_pct"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        df = df.dropna(subset=["pcr_oi_pct"]).reset_index(drop=True)

        # TAIFEX頁面預設是「新到舊」排序，反轉成「舊到新」方便畫時間序列折線圖
        df = df.iloc[::-1].reset_index(drop=True)

        if use_cache and not df.empty:
            try:
                DatabaseEngine.set_cache(cache_key, {"records": df.to_dict(orient="records")}, db_path=None)
            except Exception:
                pass

        return df

    @staticmethod
    def build_pcr_report(use_cache: bool = True) -> dict:
        """
        整合方法：抓取近期臺指選擇權Put/Call Ratio，組成附帶雙向解讀提醒
        的報告，供 app.py 顯示。
        """
        df = OptionsEngine.fetch_txo_put_call_ratio(use_cache=use_cache)
        if df.empty:
            detail = OptionsEngine._last_fetch_error or "未知原因"
            return {"status": "unavailable", "message": f"⚠️ 暫時無法取得臺指選擇權Put/Call Ratio資料。技術細節：{detail}"}

        latest = df.iloc[-1]
        pcr_oi = float(latest["pcr_oi_pct"])
        pcr_vol = float(latest["pcr_volume_pct"])

        flags = []
        if pcr_oi >= 140:
            flags.append(
                f"🔴 未平倉量PCR={pcr_oi:.1f}%，賣權未平倉部位明顯高於買權。"
                f"市場常見解讀：避險/看空需求較高；但也有人視PCR過高為情緒過度悲觀的反轉訊號，兩種解讀並存。"
            )
        elif pcr_oi <= 80:
            flags.append(
                f"🟡 未平倉量PCR={pcr_oi:.1f}%，買權未平倉部位明顯高於賣權。"
                f"市場常見解讀：偏多情緒較高；但也有人視PCR過低為情緒過度樂觀的反轉訊號，兩種解讀並存。"
            )
        else:
            flags.append(f"ℹ️ 未平倉量PCR={pcr_oi:.1f}%，多空未平倉部位相對均衡。")

        flags.append(f"參考：當日成交量PCR={pcr_vol:.1f}%（未平倉量PCR通常被認為比成交量PCR更能反映持續性的多空氣氛）。")
        flags.append("⚠️ PCR反映的是臺指（大盤）選擇權市場整體氣氛，不是個股訊號，不應直接套用到個股的進出場判斷，也不構成投資建議。")

        return {
            "status": "ok",
            "date": str(latest["date"]),
            "pcr_oi_pct": round(pcr_oi, 2),
            "pcr_volume_pct": round(pcr_vol, 2),
            "history": df,
            "flags": flags,
        }
