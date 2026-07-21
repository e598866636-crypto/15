import numpy as np
import pandas as pd


class StageEngine:
    """
    📊 階段分析引擎 (Stage Analysis Engine — Stan Weinstein 四階段模型)

    ⚠️ v2.9.6 新增：這是先前報告裡明確點名、但完全沒有實作的一塊
    （VCP / CAN SLIM / Market Regime 都已補齊，唯獨 Stage Analysis 之前
    只停留在文件建議層級）。Weinstein 原始定義用「30週均線」（約等於
    150個交易日），本專案既有欄位是 sma_120 / sma_200（IndicatorEngine
    已算好），這裡用 sma_200 作為主要長期均線代理（更接近年線的保守版本），
    sma_60 作為中期確認，避免額外新增一條全新均線造成欄位膨脹。

    四階段定義（簡化版，依均線位置與斜率判斷）：
      Stage 1（築底期）  : 價格在 sma_200 附近盤整，sma_200 走平（斜率接近0）
      Stage 2（上升期）  : 價格站上 sma_60 且 sma_60 > sma_200，sma_200 上揚
                          ——這是 Weinstein/Minervini 建議進場的階段
      Stage 3（做頭期）  : 價格開始跌破 sma_60 但仍在 sma_200 之上，或
                          sma_200 由上揚轉為走平/下彎
      Stage 4（下跌期）  : 價格跌破 sma_200 且 sma_200 走平或下彎
                          ——建議避開或出場的階段

    ⚠️ 誠實揭露：真實的 Stage Analysis 還會參考成交量型態與相對大盤強弱
    做更細緻的判斷（例如 Stage 1 末端的爆量、Stage 2 初期的量價齊揚），
    這裡的簡化版只用均線位置與斜率，屬於「結構性代理」，不是 Weinstein
    原書的完整判斷流程，適合當作快速篩選的第一層濾網，不建議單獨依賴。
    """

    _FLAT_SLOPE_THRESHOLD = 0.15  # sma_200 五日內斜率絕對值小於此百分比視為「走平」

    @staticmethod
    def add_stage_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        主要進入點：加入 stage / stage_label / stage_note 三個欄位。
        需求欄位：close, sma_60, sma_200（需已跑過 IndicatorEngine）。
        資料不足 200 日均線時，一律標示 Stage 為 None（不要用不成熟的
        均線硬猜階段，這比錯誤分類更誠實）。
        """
        df = df.copy()
        if 'sma_200' not in df.columns or 'sma_60' not in df.columns or 'close' not in df.columns:
            df['stage'] = np.nan
            df['stage_label'] = "⚠️ 資料不足（需先跑 IndicatorEngine）"
            df['stage_note'] = ""
            return df

        c = df['close']
        sma60 = df['sma_60']
        sma200 = df['sma_200']
        sma200_slope_pct = (sma200 - sma200.shift(5)) / (sma200.shift(5).abs() + 1e-9) * 100

        above_200 = c > sma200
        above_60 = c > sma60
        sixty_above_200 = sma60 > sma200
        rising_200 = sma200_slope_pct > StageEngine._FLAT_SLOPE_THRESHOLD
        falling_200 = sma200_slope_pct < -StageEngine._FLAT_SLOPE_THRESHOLD
        flat_200 = ~rising_200 & ~falling_200

        stage2 = above_60 & sixty_above_200 & (rising_200 | flat_200)
        stage4 = ~above_200 & (falling_200 | flat_200)
        stage1 = above_200 & flat_200 & ~stage2
        # 其餘（站上年線但均線排列走弱、或年線由多翻空初期）歸為 Stage 3
        stage3 = above_200 & ~stage1 & ~stage2 & ~stage4

        stage = np.select(
            [stage2, stage4, stage1, stage3],
            [2, 4, 1, 3],
            default=np.nan,
        )
        # sma_200 資料不足（前199天必為 NaN）時，一律不判斷
        stage = np.where(sma200.isna(), np.nan, stage)

        label_map = {
            1: "🔵 Stage 1（築底期，觀察為主）",
            2: "🟢 Stage 2（上升期，適合進場）",
            3: "🟡 Stage 3（做頭期，避免加碼、考慮減碼）",
            4: "🔴 Stage 4（下跌期，避免進場）",
        }
        df['stage'] = stage
        df['stage_label'] = [label_map.get(s, "⚠️ 資料不足") for s in stage]

        note = np.select(
            [stage2, stage4, stage1, stage3],
            [
                "價格站上季線且季線在年線之上，年線走平或上揚，趨勢結構偏多。",
                "價格跌破年線，年線走平或下彎，趨勢結構偏空。",
                "價格在年線附近整理，年線走平，尚未確立方向。",
                "價格仍在年線之上，但均線排列開始轉弱，留意是否轉入下跌期。",
            ],
            default="",
        )
        df['stage_note'] = note

        return df

    @staticmethod
    def get_stage_summary(df: pd.DataFrame) -> dict:
        """回傳最新一筆的階段判斷，供 UI 直接使用。"""
        if df is None or df.empty or 'stage' not in df.columns:
            return {'stage': None, 'stage_label': "⚠️ 尚未計算", 'stage_note': ""}
        latest = df.iloc[-1]
        stage = latest.get('stage', np.nan)
        return {
            'stage': int(stage) if pd.notna(stage) else None,
            'stage_label': latest.get('stage_label', "⚠️ 資料不足"),
            'stage_note': latest.get('stage_note', ""),
        }
