import pandas as pd
import yfinance as yf

from engines.logging_config import get_logger

logger = get_logger(__name__)


class SeasonalityEngine:
    """
    📅 季節循環引擎 (Seasonality Engine) - TQAI Pro v2.9

    對應多因子決策文件「十六、季節循環」：一月效應、除權息行情、年底作帳、
    Q4旺季、聖誕行情、選舉行情等日曆效應。

    ⚠️ 誠實揭露（這項功能特別容易被誤用，務必講清楚範圍與限制）：
      1. 這裡只做「月份別歷史報酬統計」（哪個月份歷史平均表現較好/較差），
         不做「除權息行情」「選舉行情」「聖誕行情」這類需要精確對應特定
         日期/事件的分析——那些需要額外的股利發放日曆、選舉日期等外部
         事件資料源，本引擎沒有這些資料，做了會是編造，不如不做。
      2. 樣本數極小、統計檢定力低：即使抓 10 年資料，每個月份也只有
         約 10 個獨立觀察值（10年 = 10次1月、10次2月……），這遠低於一般
         統計顯著性檢定需要的樣本數。任何「規律」都可能只是雜訊，不是
         真正穩定、可複製的效應——這裡刻意把每一項統計結果都標上樣本數
         (n=幾年)，讓使用者自己判斷這個樣本數夠不夠支撐這個結論。
      3. 存活者偏差／公司體質改變風險：個股過去 10 年可能歷經產業轉型、
         股本更迭、經營權更換等，10 年前的股價行為不代表現在的公司體質，
         歷史規律也不必然適用於單一個股的未來。
      4. 「歷史平均較強的月份」不是「這個月一定會漲」，只是多一個參考
         角度，絕對不能單獨依此做進出場決策，必須搭配基本面/技術面/
         籌碼面等其他分析一起看。

    ⚠️ 因果安全性：這裡用的是完整已收盤的歷史月份資料（不含當月未收完
    的月份，見 build_monthly_seasonality 的排除邏輯），不涉及任何需要
    未來才能確認的判斷，沒有 look-ahead bias 疑慮。

    ⚠️ 與 DataEngine 的關係：這裡另外獨立抓一份更長天期（預設10年）的
    歷史資料，刻意不寫回 DatabaseEngine 的股票快取——那個快取表與新鮮度
    機制是設計給 DataEngine.get_stock_data()（預設2年、給技術指標/策略
    引擎用）使用的，混用不同天期的資料寫入同一張快取表容易造成後續指標
    計算誤用到不一致的資料範圍，因此這裡完全獨立處理，不影響既有快取。
    """

    _SUFFIX_CANDIDATES = [".TW", ".TWO"]
    _last_fetch_error = None

    @staticmethod
    def _record_fetch_error(message: str):
        """統一記錄抓取失敗原因：同時寫進 _last_fetch_error（Dashboard顯示用）
        跟本地 log 檔案（見 logging_config.py），避免每次都要靠使用者截圖
        重現錯誤才能排查。"""
        SeasonalityEngine._last_fetch_error = message
        logger.warning(message)

    @staticmethod
    def fetch_long_history(ticker: str, period: str = "10y", use_cache: bool = True, max_age_hours: float = 24) -> pd.DataFrame:
        """獨立抓取更長天期的歷史股價，專門給季節性統計使用。任何查詢失敗
        都回傳空 DataFrame，不拋例外中斷呼叫端（沿用本專案一貫的防禦性
        設計），並記錄技術細節供 Dashboard 顯示（比照 chip_engine.py／
        options_engine.py 已驗證有效的診斷模式）。

        ⚠️ 修正說明（v2.9.1）：原本完全沒有快取，每次點擊「執行季節循環
        分析」都重新下載一次10年歷史資料。這裡新增快取（預設24小時新鮮
        期限——月份統計本來就不需要每天更新，10年資料也相對穩定）。
        """
        from engines.db_engine import DatabaseEngine
        from engines.data_engine import DataEngine

        ticker = str(ticker).strip()
        cache_key = f"seasonality_history_{ticker}_{period}"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    df = pd.DataFrame(cached["payload"]["records"])
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    return df
                except Exception:
                    pass

        try:
            if DataEngine.is_tw_code(ticker):
                df = pd.DataFrame()
                last_err = None
                for suffix in SeasonalityEngine._SUFFIX_CANDIDATES:
                    candidate = ticker + suffix
                    try:
                        cand_df = yf.download(candidate, period=period, interval="1d",
                                               progress=False, auto_adjust=True)
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"
                        cand_df = pd.DataFrame()
                    if not cand_df.empty:
                        df = cand_df
                        last_err = None
                        break
                if df.empty and last_err:
                    SeasonalityEngine._record_fetch_error(last_err)
            else:
                df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        except Exception as e:
            SeasonalityEngine._record_fetch_error(f"{type(e).__name__}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            if not SeasonalityEngine._last_fetch_error:
                SeasonalityEngine._record_fetch_error("yfinance回應內容為空（連線成功但沒有資料，可能是代碼錯誤或上市時間過短）")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        if df.empty:
            SeasonalityEngine._record_fetch_error("資料下載成功但清理後沒有有效收盤價")
            return df

        SeasonalityEngine._last_fetch_error = None

        if use_cache:
            try:
                save_df = df.copy()
                save_df["date"] = save_df["date"].astype(str)
                DatabaseEngine.set_cache(cache_key, {"records": save_df.to_dict(orient="records")}, db_path=None)
            except Exception:
                pass

        return df

    @staticmethod
    def build_monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
        """
        把每個曆年的每個月當成一個獨立樣本，計算「該月第一個交易日
        → 該月最後一個交易日」的收盤報酬率，再依月份分組算平均報酬率、
        中位數、勝率(正報酬次數/總次數)、樣本數(n=幾年)。

        ⚠️ 排除「當月尚未走完」的最後一筆資料所在月份（例如今天是7月10日，
        7月還沒收完，不能把「7月至今」的報酬跟其他完整月份放在一起比較，
        會低估/高估波動、且不是同一種統計量），確保每個樣本都是完整月份。
        """
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        d = df[["date", "close"]].dropna().copy()
        d["year"] = d["date"].dt.year
        d["month"] = d["date"].dt.month

        # 排除當前尚未走完的月份（用資料最後一天所在的年月判斷）
        last_year, last_month = d["year"].iloc[-1], d["month"].iloc[-1]

        records = []
        for (year, month), group in d.groupby(["year", "month"]):
            if year == last_year and month == last_month:
                continue
            group = group.sort_values("date")
            if len(group) < 2:
                continue
            start_price = float(group["close"].iloc[0])
            end_price = float(group["close"].iloc[-1])
            if start_price <= 0:
                continue
            records.append({"year": year, "month": month, "return_pct": (end_price / start_price - 1) * 100})

        if not records:
            return pd.DataFrame()

        monthly_df = pd.DataFrame(records)
        summary = monthly_df.groupby("month").agg(
            平均報酬pct=("return_pct", "mean"),
            中位數報酬pct=("return_pct", "median"),
            勝率pct=("return_pct", lambda s: (s > 0).mean() * 100),
            樣本數n=("return_pct", "count"),
        ).reset_index()

        summary["月份"] = summary["month"].apply(lambda m: f"{int(m)}月")
        for col in ["平均報酬pct", "中位數報酬pct", "勝率pct"]:
            summary[col] = summary[col].round(2)

        summary = summary.sort_values("month").reset_index(drop=True)
        return summary[["月份", "平均報酬pct", "中位數報酬pct", "勝率pct", "樣本數n"]]

    @staticmethod
    def build_seasonality_report(ticker: str, period: str = "10y") -> dict:
        """
        整合方法：抓長天期歷史資料 → 計算月份別統計 → 組成附帶樣本數與
        誠實提醒的報告，供 app.py 顯示。

        回傳：
            {
                'status': 'ok' / 'unavailable',
                'years_covered': 實際涵蓋的曆年數（= 樣本數上限）,
                'monthly_table': DataFrame（月份/平均報酬%/中位數報酬%/勝率%/樣本數n）,
                'flags': [str, ...] 人類可讀的提醒與重點摘要,
            }
        """
        df = SeasonalityEngine.fetch_long_history(ticker, period=period)
        if df.empty:
            detail = SeasonalityEngine._last_fetch_error or "未知原因"
            return {"status": "unavailable", "message": f"⚠️ 無法取得足夠的長期歷史資料進行季節性分析。技術細節：{detail}"}

        monthly = SeasonalityEngine.build_monthly_seasonality(df)
        if monthly.empty:
            return {"status": "unavailable", "message": "⚠️ 歷史資料不足以計算月份別統計（可能是上市時間太短，不足一個完整月份）。"}

        years_covered = int(df["date"].dt.year.nunique())
        best_month = monthly.loc[monthly["平均報酬pct"].idxmax()]
        worst_month = monthly.loc[monthly["平均報酬pct"].idxmin()]

        flags = []
        if years_covered < 5:
            flags.append(
                f"⚠️ 只有約 {years_covered} 年歷史資料可用，每個月份只有約 {years_covered} 個獨立觀察值，"
                f"樣本數過少，統計結果可信度低，僅供參考。"
            )
        else:
            flags.append(
                f"ℹ️ 共涵蓋約 {years_covered} 個曆年，每個月份約有 {years_covered} 個獨立觀察值。"
                f"即使如此，這在統計上仍屬小樣本，任何「規律」都可能是雜訊，不是穩定可複製的效應。"
            )

        flags.append(
            f"📈 歷史平均報酬最高的月份：{best_month['月份']}"
            f"（平均 {best_month['平均報酬pct']}%，勝率 {best_month['勝率pct']}%，n={int(best_month['樣本數n'])}）"
        )
        flags.append(
            f"📉 歷史平均報酬最低的月份：{worst_month['月份']}"
            f"（平均 {worst_month['平均報酬pct']}%，勝率 {worst_month['勝率pct']}%，n={int(worst_month['樣本數n'])}）"
        )
        flags.append("⚠️ 以上僅為歷史統計，不是未來保證；不含除權息行情、選舉行情等需要額外事件資料源的分析，請勿單獨依此做進出場決策。")

        return {
            "status": "ok",
            "years_covered": years_covered,
            "monthly_table": monthly,
            "flags": flags,
        }
