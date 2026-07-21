import numpy as np
import pandas as pd


class MarketRegimeEngine:
    """
    🌍 大盤市場狀態引擎 (Market Regime Engine) — 真正的「市場」狀態

    ⚠️ 這是 v2.9.5 第二輪新增，修正一個先前沒發現、比較隱蔽的方法論問題：
    `strategy_engine.py` 裡的 `market_regime` 欄位，名稱聽起來像是「大盤現在
    處於什麼狀態」，但實際計算方式是用**這檔股票自己的** sma_60 斜率與波動率
    比率去分類——那是「這檔股票自己的趨勢狀態」，不是市場狀態。兩檔完全
    不相關的股票，一檔多頭噴出、一檔破底下跌，畫面上卻可能同時顯示「大盤
    狀態：📈 穩健多頭趨勢」，這在邏輯上是矛盾的（大盤在同一天不可能兩種
    狀態都對），只是因為欄位名稱造成的誤導。

    這個新引擎改用「加權指數（^TWII）本身」的技術結構來判斷真正的大盤狀態，
    刻意跟 strategy_engine.py 的個股層級欄位分開命名（`true_market_regime`
    vs `market_regime`），並在說明文件與 UI 上都清楚標示兩者的差異，
    不假設使用者會自己發現這個細節。

    ⚠️ 誠實揭露：完整的市場寬度 (Market Breadth，例如「上漲家數/下跌家數」
    「創新高/創新低家數」) 需要全市場所有股票的即時漲跌統計，這裡受限於
    資料來源（僅有加權指數本身的 OHLCV），沒有真正的寬度指標，只用指數
    自己的趨勢與波動率結構做分類，本質上是「大盤指數的技術面狀態」，
    不是嚴謹定義下的「市場寬度」。
    """

    @staticmethod
    def classify(benchmark_df: pd.DataFrame) -> dict:
        """
        輸入 DataEngine.get_benchmark_data() 的回傳結果（加權指數 OHLCV），
        回傳大盤狀態分類。與個股 df 完全無關，同一天呼叫、對誰呼叫都應該
        得到相同結果（這是「市場狀態」的基本定義：跟你在看哪一檔股票無關）。
        """
        if benchmark_df is None or benchmark_df.empty or len(benchmark_df) < 65:
            return {'status': 'insufficient_data', 'regime': 'N/A（大盤資料不足）'}

        df = benchmark_df.copy().sort_values('date').reset_index(drop=True)
        close = df['close']

        sma_60 = close.rolling(60, min_periods=20).mean()
        sma_60_slope = (sma_60 - sma_60.shift(5)) / (sma_60.shift(5) + 1e-9) * 100

        returns = close.pct_change()
        vol_20 = returns.rolling(20, min_periods=5).std() * np.sqrt(252) * 100
        vol_mean_60 = vol_20.rolling(60, min_periods=10).mean()

        current_close = float(close.iloc[-1])
        current_sma60 = float(sma_60.iloc[-1]) if pd.notna(sma_60.iloc[-1]) else np.nan
        current_slope = float(sma_60_slope.iloc[-1]) if pd.notna(sma_60_slope.iloc[-1]) else np.nan
        current_vol = float(vol_20.iloc[-1]) if pd.notna(vol_20.iloc[-1]) else np.nan
        avg_vol = float(vol_mean_60.iloc[-1]) if pd.notna(vol_mean_60.iloc[-1]) else np.nan

        if pd.isna(current_sma60) or pd.isna(current_slope):
            return {'status': 'insufficient_data', 'regime': 'N/A（大盤資料不足）'}

        is_high_vol = pd.notna(current_vol) and pd.notna(avg_vol) and current_vol > avg_vol * 1.3
        above_sma = current_close > current_sma60

        if is_high_vol:
            regime = "⚠️ 大盤高波動震盪區"
            guidance = "系統性風險偏高，個股訊號的可信度普遍下降，建議降低整體曝險部位。"
        elif above_sma and current_slope > 0.4:
            regime = "📈 大盤穩健多頭趨勢"
            guidance = "順勢操作的勝率環境較佳，個股的多頭訊號可信度較高。"
        elif (not above_sma) and current_slope < -0.4:
            regime = "📉 大盤弱勢空頭格局"
            guidance = "逆勢做多風險較高，即使個股技術面看似不錯，也建議提高戒心或降低部位。"
        else:
            regime = "🔄 大盤低波盤整區"
            guidance = "方向不明確，個股表現可能受族群/題材輪動影響更大於大盤方向。"

        return {
            'status': 'ok',
            'regime': regime,
            'guidance': guidance,
            'close': current_close,
            'sma_60': round(current_sma60, 1),
            'sma_60_slope_pct': round(current_slope, 2),
            'volatility_annualized': round(current_vol, 1) if pd.notna(current_vol) else None,
        }
