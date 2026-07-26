import pandas as pd
import numpy as np


class EvidenceEngine:
    """
    🧾 Evidence Engine 證據引擎 + Confidence Engine 信心模型

    把 Judge Agent 的單一 AI Score 拆解成「看得到的證據」：
      1. evidence_list   : 每一筆訊號至少 3 個有星等(★)的證據，含多/空/風險極性
      2. confidence_pct   : 模型對目前判斷的信心程度（多空證據是否一致、強度是否足夠）
      3. data_quality_pct : 計算當下指標所用資料的完整度（避免假訊號建立在缺值資料上）

    設計理念對應規劃文件：
        AI Score 90 / Confidence 84% / 資料品質 95%
    """

    # 用來檢查資料完整度的核心欄位
    _REQUIRED_COLS = [
        "close", "atr_14", "ema_8", "sma_20", "sma_60", "sma_200",
        "macd_hist", "rsi_14", "k_9", "d_9", "rvol",
        "volatility_ratio", "obv", "obv_sma",
    ]

    @staticmethod
    def _stars(value, breakpoints):
        """依數值大小，將證據強度映射為 1~5 顆星"""
        stars = 1
        for bp in breakpoints:
            if value >= bp:
                stars += 1
        return int(np.clip(stars, 1, 5))

    # ==========================================
    # 1. 主要進入點：對整張 df 計算證據相關欄位
    # ==========================================
    @staticmethod
    def add_evidence(df: pd.DataFrame):
        df = df.copy()
        c = df["close"]

        # ---- 資料品質：檢查近 5 根 K 棒，核心欄位的非缺值比例 ----
        existing_cols = [col for col in EvidenceEngine._REQUIRED_COLS if col in df.columns]
        completeness = df[existing_cols].notna().mean(axis=1) * 100
        # 近 5 筆滾動平均，避免單一筆雜訊造成品質分數震盪
        df["data_quality_pct"] = completeness.rolling(5, min_periods=1).mean().clip(0, 100)

        # ---- 多空證據強度評分（用於信心模型） ----
        bull_score = df["bull_score"] if "bull_score" in df.columns else pd.Series(50, index=df.index)
        bear_score = df["bear_score"] if "bear_score" in df.columns else pd.Series(50, index=df.index)
        ai_score = df["ai_score"] if "ai_score" in df.columns else pd.Series(50, index=df.index)

        # 多空雙方共識程度：bull_score 與 (100 - bear_score) 越接近，代表兩個 Agent 看法一致 → 信心越高
        agreement = 100 - (bull_score - (100 - bear_score)).abs()

        # AI Score 越極端（接近 0 或 100），代表判斷越果斷 → 信心加分；越接近 50（曖昧不明）→ 信心扣分
        decisiveness = (ai_score - 50).abs() * 2

        raw_confidence = (agreement * 0.5) + (decisiveness * 0.3) + (df["data_quality_pct"] * 0.2)
        df["confidence_pct"] = raw_confidence.clip(0, 100)

        df["confidence_label"] = np.select(
            [df["confidence_pct"] >= 80, df["confidence_pct"] >= 55],
            ["🟢 高信心", "🟡 中性信心"],
            default="🔴 低信心（證據分歧或資料不足）",
        )

        return df

    # ==========================================
    # 2. 取得單一時間點（預設最新一筆）的證據清單
    # ==========================================
    @staticmethod
    def get_evidence_list(df: pd.DataFrame, idx: int = -1):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1] if abs(idx) < len(df) else row

        def val(col, default=np.nan):
            return row[col] if col in row.index else default

        def prev_val(col, default=np.nan):
            return prev[col] if col in prev.index else default

        evidence = []

        def add(label, active, stars, polarity, detail=""):
            if active:
                evidence.append({
                    "label": label,
                    "stars": int(stars),
                    "polarity": polarity,   # "bull" | "bear" | "risk"
                    "detail": detail,
                })

        close = val("close")
        atr = val("atr_14", np.nan)

        # --- 多頭證據 ---
        macd_hist = val("macd_hist")
        macd_prev = prev_val("macd_hist")
        if pd.notna(macd_hist):
            add("MACD 動能轉強", macd_hist > 0,
                EvidenceEngine._stars(abs(macd_hist) / (atr + 1e-9) * 100, [1, 3, 6, 10]),
                "bull", f"柱狀體 {macd_hist:.3f}，{'持續放大' if pd.notna(macd_prev) and macd_hist > macd_prev else '由負翻正'}")

        rvol = val("rvol")
        if pd.notna(rvol):
            add("成交量明顯放大 (RVOL)", rvol > 1.3,
                EvidenceEngine._stars(rvol, [1.3, 1.6, 2.0, 3.0]),
                "bull", f"相對量能 {rvol:.2f} 倍")

        sma_60 = val("sma_60")
        if pd.notna(sma_60) and pd.notna(close):
            add("站上季線 (60MA)", close > sma_60,
                EvidenceEngine._stars((close - sma_60) / sma_60 * 100, [1, 3, 6, 10]),
                "bull", f"乖離 {((close - sma_60) / sma_60 * 100):.1f}%")

        k9, d9 = val("k_9"), val("d_9")
        k9_prev, d9_prev = prev_val("k_9"), prev_val("d_9")
        if pd.notna(k9) and pd.notna(d9) and pd.notna(k9_prev) and pd.notna(d9_prev):
            add("KD 黃金交叉", (k9 > d9) and (k9_prev <= d9_prev), 4, "bull", f"K={k9:.1f} D={d9:.1f}")

        obv, obv_sma = val("obv"), val("obv_sma")
        if pd.notna(obv) and pd.notna(obv_sma):
            add("OBV 資金流入 (能量潮走高)", obv > obv_sma, 3, "bull")

        # --- MTM 動量指標與寶塔線（v2.8 新增，見 indicator_engine.py 說明） ---
        # ⚠️ 沿用選股學院文件對 MTM 的提醒：訊號在0上下反覆穿越、雜訊多，
        # 這裡只把「動能由負轉正/由正轉負」這種相對明確的反轉時刻列為證據，
        # 不是每天都會觸發，且僅供輔助參考，不單獨構成買賣依據。
        mtm = val("mtm")
        mtm_prev = prev_val("mtm")
        if pd.notna(mtm) and pd.notna(mtm_prev) and pd.notna(atr) and atr > 0:
            add("MTM動能由負轉正 (動能反轉向上)", (mtm > 0) and (mtm_prev <= 0),
                EvidenceEngine._stars(abs(mtm) / (atr + 1e-9) * 100, [1, 3, 6, 10]),
                "bull", f"MTM={mtm:.2f}")
            add("MTM動能由正轉負 (動能反轉向下)", (mtm < 0) and (mtm_prev >= 0),
                EvidenceEngine._stars(abs(mtm) / (atr + 1e-9) * 100, [1, 3, 6, 10]),
                "bear", f"MTM={mtm:.2f}")

        # ⚠️ 沿用選股學院文件對寶塔線的提醒：這是落後指標，翻紅/翻黑代表
        # 「趨勢已經確立」而非預測轉折，訊號會比真正的反轉點慢，僅供
        # 輔助參考。這裡只在「當天剛翻轉」時列為證據，不是整段紅/黑期間
        # 每天都顯示，避免同一個事件被重複列入證據清單。
        pagoda_up = val("pagoda_flip_up", False)
        pagoda_down = val("pagoda_flip_down", False)
        add("寶塔線翻紅 (趨勢向上確立)", bool(pagoda_up), 3, "bull")
        add("寶塔線翻黑 (趨勢向下確立)", bool(pagoda_down), 3, "bear")

        rsi = val("rsi_14")
        if pd.notna(rsi):
            add("RSI 強勢區", rsi > 55, EvidenceEngine._stars(rsi, [55, 60, 65, 70]), "bull", f"RSI={rsi:.1f}")
            add("RSI 弱勢區", rsi < 45, EvidenceEngine._stars(100 - rsi, [55, 60, 65, 70]), "bear", f"RSI={rsi:.1f}")

        # --- 空頭 / 風險證據 ---
        ema_8 = val("ema_8")
        if pd.notna(ema_8) and pd.notna(close):
            add("短均線蓋頭 (跌破 EMA8)", close < ema_8,
                EvidenceEngine._stars((ema_8 - close) / ema_8 * 100, [1, 3, 6, 10]),
                "bear", f"乖離 {((close - ema_8) / ema_8 * 100):.1f}%")

        if pd.notna(k9) and pd.notna(d9) and pd.notna(k9_prev) and pd.notna(d9_prev):
            add("KD 死亡交叉", (k9 < d9) and (k9_prev >= d9_prev), 4, "bear", f"K={k9:.1f} D={d9:.1f}")

        sma_200 = val("sma_200")
        if pd.notna(sma_200) and pd.notna(close):
            add("跌破年線 (200MA)", close < sma_200, 5, "bear", "長線多頭結構轉弱")

        if pd.notna(sma_60) and pd.notna(close):
            bias_60 = (close - sma_60) / sma_60 * 100
            add("乖離過大／均值回歸風險", abs(bias_60) > 15,
                EvidenceEngine._stars(abs(bias_60), [15, 20, 25, 30]),
                "risk", f"季線乖離 {bias_60:.1f}%")

        vol_ratio = val("volatility_ratio")
        if pd.notna(vol_ratio):
            vol_mean = df["volatility_ratio"].iloc[max(0, idx - 60 if idx >= 0 else len(df) + idx - 60):idx].mean() \
                if len(df) > 60 else df["volatility_ratio"].mean()
            if pd.notna(vol_mean) and vol_mean > 0:
                add("波動率異常放大", vol_ratio > vol_mean * 1.3,
                    EvidenceEngine._stars(vol_ratio / vol_mean, [1.3, 1.6, 2.0, 2.5]),
                    "risk", f"目前波動率為均值的 {vol_ratio / vol_mean:.1f} 倍")

        # 依星等排序（強證據優先），同星等則多頭優先顯示
        polarity_order = {"bull": 0, "bear": 1, "risk": 2}
        evidence.sort(key=lambda e: (-e["stars"], polarity_order.get(e["polarity"], 9)))
        return evidence