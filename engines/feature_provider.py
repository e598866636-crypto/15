import numpy as np
import pandas as pd


class FeatureProvider:
    """
    📦 Feature Provider（共用特徵計算層）— Phase 1 P0

    設計原則（依討論後的 Option 3 決議）：
    - 這是 IndicatorEngine 與 BreakoutEngine（以及未來任何其他 Engine）
      共同依賴的底層，取代「Engine 互相依賴」。正確的依賴方向是：
          IndicatorEngine ──┐
                            ├──> FeatureProvider
          BreakoutEngine ───┘
      而不是 BreakoutEngine → IndicatorEngine。
    - Lazy Compute + Cache-by-column：欄位已存在於傳入的 df 就直接回傳，
      不存在才計算並寫入。這裡的「cache」就是 DataFrame 的欄位本身，
      沒有引入額外的全域狀態或雜湊層——避免在還沒有實際需求前，
      過度設計 Dependency Graph / Cache Contract 這類基礎設施。
    - 目前版本刻意不做的事（依 Phase 1 範圍）：
        ✗ 不修改 IndicatorEngine.add_indicators() 的既有邏輯或欄位名稱
        ✗ 不修改 BreakoutEngine._compute_internal_indicators()
        ✗ 不引入任何全域 cache / 版本號 / hash 依賴追蹤
      這些留給 Phase 2（逐步遷移）與 Phase 0（Architecture Invariants
      文件）之後再視實際需要決定，避免在沒有驗證的情況下同時改變
      多個 Engine 的行為。
    - 所有方法皆為 staticmethod、單向資料流（傳入 df，回傳補齊欄位後的
      df 或單一 Series），不讀寫任何 Engine 的內部狀態。

    ⚠️ 誠實揭露（更新於 Phase 2D）：ensure_ema / ensure_sma / ensure_macd
    已經接上 IndicatorEngine（Phase 2B/2C/2D 依序遷移）。ensure_rsi /
    ensure_atr / ensure_obv / ensure_bbands 這幾個仍是「已經寫好但還沒
    接上」的狀態——IndicatorEngine.add_indicators() 目前仍是自己內聯
    計算 RSI/ATR/OBV/布林通道，還沒有改成呼叫這幾個函式，等後續 Phase
    （2E RSI／2F ATR／2G Bollinger／2H OBV）逐一遷移時才會真正被使用。
    另外，engines/breakout_engine.py 也有自己獨立的 EMA/MACD 計算，
    目前完全沒有依賴 FeatureProvider，是本檔案開頭「取代 Engine 互相
    依賴」這個設計目標尚未完成的部分，留待後續視需要決定是否納入
    遷移範圍。
    """

    # ------------------------------------------------------------
    # 均線 (EMA / SMA)
    # ------------------------------------------------------------
    @staticmethod
    def ensure_ema(df: pd.DataFrame, span: int, source_col: str = "close") -> pd.DataFrame:
        col = f"ema_{span}"
        if col not in df.columns:
            df[col] = df[source_col].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def ensure_sma(df: pd.DataFrame, window: int, source_col: str = "close",
                    min_periods: int | None = None) -> pd.DataFrame:
        col = f"sma_{window}"
        if col not in df.columns:
            df[col] = df[source_col].rolling(window, min_periods=min_periods).mean()
        return df

    # ------------------------------------------------------------
    # MACD（固定慣例參數 12/26/9，欄位名稱沿用 IndicatorEngine 既有慣例）
    # ------------------------------------------------------------
    @staticmethod
    def ensure_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                     signal: int = 9, source_col: str = "close") -> pd.DataFrame:
        if not {"macd_dif", "macd_dea", "macd_hist"}.issubset(df.columns):
            c = df[source_col]
            dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
            dea = dif.ewm(span=signal, adjust=False).mean()
            df["macd_dif"] = dif
            df["macd_dea"] = dea
            df["macd_hist"] = dif - dea
        return df

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------
    @staticmethod
    def ensure_rsi(df: pd.DataFrame, period: int = 14, source_col: str = "close") -> pd.DataFrame:
        col = f"rsi_{period}"
        if col not in df.columns:
            delta = df[source_col].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            df[col] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        return df

    # ------------------------------------------------------------
    # ATR (真實波動幅度)
    # ------------------------------------------------------------
    @staticmethod
    def ensure_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        col = f"atr_{period}"
        if col not in df.columns:
            h, l, c = df["high"], df["low"], df["close"]
            tr = pd.concat(
                [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
            ).max(axis=1)
            df[col] = tr.rolling(period).mean()
        return df

    # ------------------------------------------------------------
    # OBV (能量潮指標)
    # ------------------------------------------------------------
    @staticmethod
    def ensure_obv(df: pd.DataFrame, sma_window: int = 20) -> pd.DataFrame:
        if "obv" not in df.columns:
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        obv_sma_col = "obv_sma" if sma_window == 20 else f"obv_sma_{sma_window}"
        if obv_sma_col not in df.columns:
            df[obv_sma_col] = df["obv"].rolling(sma_window).mean()
        return df

    # ------------------------------------------------------------
    # 布林通道（依賴 SMA，內部呼叫 ensure_sma 確保一致）
    # ------------------------------------------------------------
    @staticmethod
    def ensure_bbands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0,
                       source_col: str = "close") -> pd.DataFrame:
        upper_col, lower_col = f"bb_upper_{window}", f"bb_lower_{window}"
        if not {upper_col, lower_col}.issubset(df.columns):
            df = FeatureProvider.ensure_sma(df, window, source_col=source_col)
            std = df[source_col].rolling(window).std()
            df[upper_col] = df[f"sma_{window}"] + std * num_std
            df[lower_col] = df[f"sma_{window}"] - std * num_std
            # 沿用 IndicatorEngine 目前的無後綴命名慣例（window=20 為預設情境）
            if window == 20:
                df["bb_upper"] = df[upper_col]
                df["bb_lower"] = df[lower_col]
        return df
