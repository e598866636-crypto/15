import pandas as pd
import numpy as np

class IndicatorEngine:
    """
    ⚠️ 修正說明（v2.8 新增）：
    新增「選股學院」技術面選股法提到、但原本系統沒有涵蓋的兩個經典指標：
    MTM 動量指標、寶塔線。兩者都只用過去資料計算（不看未來），不會有
    前視偏誤問題，可以放心當作 StrategyEngine/EvidenceEngine 的特徵。

    1. MTM 動量指標 (Momentum)：
       公式：MTM_n(t) = C(t) - C(t-n)，衡量股價相對 n 天前的移動幅度，
       是價格的「先行指標」——理論上動能會先於價格轉向。這裡另外算了
       文件提到的 OSC 震盪量指標 OSC_n(t) = C(t)/C(t-n)*100，用比率
       表示同樣的動能概念，方便跨股票比較幅度。
       ⚠️ 誠實揭露（沿用選股學院文件本身列出的缺點）：MTM 經常在0上下
       反覆穿越，訊號容易過於頻繁／雜訊多，建議搭配趨勢型指標
       （如均線排列、MACD）一起看，不要單獨用 MTM 產生買賣訊號。
       n 預設 10（沿用市場常見參數），可依需求調整。

    2. 寶塔線 (Pagoda/Tower Line)：
       規則（皆用「前n日」不含當天，用 shift(1) 排除當天，避免前視偏誤）：
         - 收盤價 > 前n日每日收盤價的最高者 → 翻紅（趨勢向上確立，買進訊號）
         - 收盤價 < 前n日每日收盤價的最低者 → 翻黑（趨勢向下確立，賣出訊號）
       翻紅/翻黑後會持續維持該狀態，直到出現反方向的翻轉訊號才切換
       （文件原意：「未得到翻紅訊號前不進場；未得到翻黑訊號前持股續抱」），
       這裡用 pagoda_trend 欄位表示目前所處的持續狀態（'red'/'black'/None），
       pagoda_flip_up / pagoda_flip_down 則是「當天剛發生翻轉」的事件旗標。
       ⚠️ 誠實揭露（沿用選股學院文件列出的缺點）：寶塔線只有等趨勢已經
       確立才會翻轉，本質上是落後指標，抓不到最低點/最高點，且訊號會
       比真正的反轉點慢（n=3操作法約落後3天）；在盤整格局容易出現
       小紅小黑、進出頻繁但無獲利的情況。n 預設 4（文件建議 3~5 皆可，
       視操作週期長短調整，數字越大越不敏感、越能過濾雜訊）。
    """
    @staticmethod
    def add_indicators(df: pd.DataFrame, mtm_window: int = 10, pagoda_window: int = 4):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        
        # 1. 均線系統
        df["ema_8"] = c.ewm(span=8, adjust=False).mean()
        df["ema_21"] = c.ewm(span=21, adjust=False).mean()
        df["sma_20"] = c.rolling(20).mean()
        df["sma_60"] = c.rolling(60).mean()
        df["sma_120"] = c.rolling(120).mean()
        df["sma_200"] = c.rolling(200).mean()
        df["vma_20"] = v.rolling(20).mean()

        # 2. 布林通道與 RSI
        std = c.rolling(20).std()
        df["bb_upper"] = df["sma_20"] + std * 2
        df["bb_lower"] = df["sma_20"] - std * 2
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        # 3. KD 與 MACD
        rsv = ((c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min() + 1e-9)) * 100
        df["k_9"] = rsv.ewm(alpha=1/3).mean()
        df["d_9"] = df["k_9"].ewm(alpha=1/3).mean()
        
        df["macd_dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

        # 4. ATR (真實波動幅度)
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # 5. OBV (能量潮指標)
        df['obv'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()

        # ==========================================
        # 🚀 TQAI Pro 升級：市場特徵與狀態識別因子
        # ==========================================
        # 波動率特徵 (歷史波動率近似)
        df['volatility_ratio'] = (df['atr_14'] / c) * 100 
        
        # 季線斜率 (判斷宏觀趨勢動能)
        df['sma_60_slope'] = (df['sma_60'] - df['sma_60'].shift(5)) / (df['sma_60'].shift(5) + 1e-9) * 100
        
        # 量能特徵 (RVOL - 相對成交量)
        df['rvol'] = v / (df['vma_20'] + 1e-9)

        # ==========================================
        # 6. MTM 動量指標（v2.8 新增，見上方 class docstring 說明）
        # ==========================================
        df[f'mtm_{mtm_window}'] = c - c.shift(mtm_window)
        df[f'mtm_oscillator_{mtm_window}'] = (c / (c.shift(mtm_window) + 1e-9)) * 100
        # 方便下游（EvidenceEngine等）用固定欄位名稱存取目前參數下的 MTM，
        # 不需要知道 mtm_window 實際數值
        df['mtm'] = df[f'mtm_{mtm_window}']

        # ==========================================
        # 7. 寶塔線（v2.8 新增，見上方 class docstring 說明）
        # ==========================================
        prior_high_close = c.rolling(pagoda_window).max().shift(1)
        prior_low_close = c.rolling(pagoda_window).min().shift(1)

        flip_up = (c > prior_high_close) & prior_high_close.notna()
        flip_down = (c < prior_low_close) & prior_low_close.notna()

        df['pagoda_flip_up'] = flip_up
        df['pagoda_flip_down'] = flip_down

        # 持續狀態：一旦翻紅/翻黑就維持該狀態，直到出現反方向翻轉訊號
        # （用 forward-fill 實現「未翻黑前持股續抱」的邏輯，皆為因果安全）
        trend_raw = pd.Series(
            np.where(flip_up, "red", np.where(flip_down, "black", None)),
            index=df.index, dtype=object
        )
        df['pagoda_trend'] = trend_raw.ffill()

        return df