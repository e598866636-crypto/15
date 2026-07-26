import numpy as np
import pandas as pd


class DivergenceEngine:
    """
    🧭 背離與誘盤防禦引擎 (Divergence & Trap Defense Engine) - TQAI Pro v2.7

    提供兩組互補的「動能失效」偵測：
      1. MACD 背離 (Divergence)：價格創新高/新低，但 MACD (DIF) 動能未同步
         創新高/新低，暗示目前趨勢動能正在衰竭，是常見的反轉領先訊號。
      2. 假突破／誘盤 (Fake Breakout / Bull-Bear Trap)：價格一度帶量突破
         近期高點（或跌破近期低點），但隨後幾根K棒內又收回原本的區間，
         代表這次突破/跌破可能只是誘多出貨或誘空洗盤，而非真正的趨勢延續。

    ⚠️ 因果安全設計（避免前視偏誤 look-ahead bias）：
    跟 StructureEngine 的 zigzag 完全一樣的道理 —— 「這是不是一個轉折高/低點」
    本質上要等未來走勢反轉超過偏離度之後才能定案。如果把「轉折發生的那一天」
    直接標記為背離/誘多訊號，等於在回測或即時策略判斷的當下，使用了「要靠
    未來資料才能確認」的資訊，構成嚴重的前視偏誤，這正是 structure_engine.py
    docstring 明確警告、且刻意不讓 zigzag 進入 StrategyEngine 的原因。

    因此這裡的兩個偵測函式都刻意把訊號標記在「確認的當下 (confirmation bar)」，
    而不是回填到轉折/突破發生的那一天：
      - detect_macd_divergence()：bearish_divergence / bullish_divergence
        只會在「第二個轉折點被確認」的那一根K棒設為 True，且比較對象一律是
        「已經確認」的前一個同方向轉折點，不會用到尚未確認的最新暫定極值。
      - detect_fake_breakout()：bull_trap_confirmed / bear_trap_confirmed
        只會在「價格真的跌破/漲破回去」的那一根K棒設為 True，判斷當下只往
        回看歷史資料 (prior_high/prior_low 皆用 shift(1) 排除當天)，不使用
        任何當天以後才存在的資訊。

    因此本引擎輸出的所有布林欄位都可以安全餵給 StrategyEngine、
    MomentumEngine 或 BacktestEngine 當作特徵，不會有 repaint 問題。
    """

    # ==========================================
    # 1. MACD 背離偵測（因果安全版本）
    # ==========================================
    @staticmethod
    def detect_macd_divergence(df: pd.DataFrame, deviation: float = 0.04):
        """
        沿用 StructureEngine「動態極值＋反轉確認」狀態機的精神，但獨立一份
        實作專門服務背離偵測，刻意不直接複用 structure_engine 產出的
        zigzag 欄位 —— 一方面維持兩個引擎的職責邊界（zigzag 僅供視覺化），
        一方面確保訊號可以精準標記在「確認當天」。

        欄位輸出：
          bearish_divergence : bool，當天確認「頂背離」(價格新高、MACD未新高)
          bullish_divergence : bool，當天確認「底背離」(價格新低、MACD未新低)
          divergence_note     : str，人類可讀說明（供 Evidence/Dashboard 顯示）
        """
        df = df.copy()
        n = len(df)
        df['bearish_divergence'] = False
        df['bullish_divergence'] = False
        df['divergence_note'] = ""

        if n < 10 or 'macd_dif' not in df.columns:
            return df

        c = df['close'].to_numpy(dtype=float)
        macd = df['macd_dif'].to_numpy(dtype=float)

        bearish_flags = np.zeros(n, dtype=bool)
        bullish_flags = np.zeros(n, dtype=bool)
        notes = [""] * n

        state = 0  # 0=初始判定, 1=尋找波段高點, -1=尋找波段低點
        pivot_idx = 0
        pivot_val = c[0]

        last_high = None   # 最近一個已確認高點 (idx, price, macd)
        prev_high = None   # 前一個已確認高點
        last_low = None
        prev_low = None

        for i in range(1, n):
            price = c[i]
            dev = (price - pivot_val) / (pivot_val + 1e-9)

            if state == 0:
                if dev > deviation:
                    state = 1
                    pivot_idx, pivot_val = i, price
                elif dev < -deviation:
                    state = -1
                    pivot_idx, pivot_val = i, price

            elif state == 1:
                if price > pivot_val:
                    pivot_idx, pivot_val = i, price
                elif dev < -deviation:
                    # 高點於「本日 i」確認（反轉超過偏離度）
                    prev_high, last_high = last_high, (pivot_idx, pivot_val, macd[pivot_idx])
                    if prev_high is not None:
                        _, p_price, p_macd = prev_high
                        _, c_price, c_macd = last_high
                        if c_price > p_price and c_macd < p_macd:
                            bearish_flags[i] = True
                            notes[i] = (
                                f"🔴 頂背離：價格創新高 {c_price:.2f}（前高 {p_price:.2f}），"
                                f"但MACD動能未跟上 {c_macd:.3f}（前值 {p_macd:.3f}），留意上漲動能減弱"
                            )
                    state = -1
                    pivot_idx, pivot_val = i, price

            elif state == -1:
                if price < pivot_val:
                    pivot_idx, pivot_val = i, price
                elif dev > deviation:
                    prev_low, last_low = last_low, (pivot_idx, pivot_val, macd[pivot_idx])
                    if prev_low is not None:
                        _, p_price, p_macd = prev_low
                        _, c_price, c_macd = last_low
                        if c_price < p_price and c_macd > p_macd:
                            bullish_flags[i] = True
                            notes[i] = (
                                f"🟢 底背離：價格創新低 {c_price:.2f}（前低 {p_price:.2f}），"
                                f"但MACD動能未跟上 {c_macd:.3f}（前值 {p_macd:.3f}），留意下跌動能減弱"
                            )
                    state = 1
                    pivot_idx, pivot_val = i, price

        df['bearish_divergence'] = bearish_flags
        df['bullish_divergence'] = bullish_flags
        df['divergence_note'] = notes
        return df

    # ==========================================
    # 2. 假突破／誘盤防禦偵測（因果安全版本）
    # ==========================================
    @staticmethod
    def detect_fake_breakout(df: pd.DataFrame, lookback: int = 20, confirm_bars: int = 3, min_rvol: float = 1.2):
        """
        誘多／誘空防禦偵測 (Bull/Bear Trap)。

        邏輯（全程只往回看歷史資料，可安全用於策略/回測特徵）：
          1. 若當天收盤價 > 前 `lookback` 天（不含當天，用 shift(1) 排除）
             的最高收盤價，且量能達標 (rvol >= min_rvol)，標記為當天的
             breakout_up 候選（帶量突破）。
          2. 在 breakout_up 候選日之後的 `confirm_bars` 天內，若收盤價又
             跌破當初突破所依據的前高水準 → 在『跌破確認的那一天』標記
             bull_trap_confirmed = True（誘多出貨、突破失敗）。
          3. bear_trap_confirmed 為對稱邏輯：帶量跌破前低後，confirm_bars
             天內又收回前低水準之上，視為誘空洗盤。

        欄位輸出：
          breakout_up / breakout_down : 當天是否觸發帶量突破/跌破候選
          bull_trap_confirmed         : 稍後確認「這次向上突破是誘多」
          bear_trap_confirmed         : 稍後確認「這次向下跌破是誘空」
          trap_note                   : 人類可讀說明（標記在確認日）
        """
        df = df.copy()
        n = len(df)
        df['breakout_up'] = False
        df['breakout_down'] = False
        df['bull_trap_confirmed'] = False
        df['bear_trap_confirmed'] = False
        df['trap_note'] = ""

        if n < lookback + 2:
            return df

        c = df['close'].to_numpy(dtype=float)
        rvol = df['rvol'].to_numpy(dtype=float) if 'rvol' in df.columns else np.ones(n)
        has_date = 'date' in df.columns

        prior_high = df['close'].rolling(lookback).max().shift(1).to_numpy(dtype=float)
        prior_low = df['close'].rolling(lookback).min().shift(1).to_numpy(dtype=float)

        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)

        for i in range(n):
            if np.isnan(prior_high[i]):
                continue
            if c[i] > prior_high[i] and rvol[i] >= min_rvol:
                breakout_up[i] = True
            if c[i] < prior_low[i] and rvol[i] >= min_rvol:
                breakout_down[i] = True

        bull_trap = np.zeros(n, dtype=bool)
        bear_trap = np.zeros(n, dtype=bool)
        notes = [""] * n

        def _label(idx):
            if has_date:
                try:
                    return df['date'].iloc[idx].strftime('%m/%d')
                except Exception:
                    return str(idx)
            return str(idx)

        for i in range(n):
            if breakout_up[i]:
                level = prior_high[i]
                for j in range(i + 1, min(i + 1 + confirm_bars, n)):
                    if c[j] < level:
                        bull_trap[j] = True
                        notes[j] = (
                            f"⚠️ 誘多警報：{_label(i)} 帶量突破 {level:.2f} 後，"
                            f"{confirm_bars}天內於 {_label(j)} 又收回至 {c[j]:.2f}，疑似假突破出貨"
                        )
                        break
            if breakout_down[i]:
                level = prior_low[i]
                for j in range(i + 1, min(i + 1 + confirm_bars, n)):
                    if c[j] > level:
                        bear_trap[j] = True
                        notes[j] = (
                            f"⚠️ 誘空警報：{_label(i)} 帶量跌破 {level:.2f} 後，"
                            f"{confirm_bars}天內於 {_label(j)} 又收回至 {c[j]:.2f}，疑似假跌破洗盤"
                        )
                        break

        df['breakout_up'] = breakout_up
        df['breakout_down'] = breakout_down
        df['bull_trap_confirmed'] = bull_trap
        df['bear_trap_confirmed'] = bear_trap
        df['trap_note'] = notes
        return df

    # ==========================================
    # 3. 一次執行（供 pipeline 呼叫）
    # ==========================================
    @staticmethod
    def add_defense_signals(df: pd.DataFrame, deviation: float = 0.04,
                             lookback: int = 20, confirm_bars: int = 3, min_rvol: float = 1.2):
        """一次執行背離偵測＋假突破防禦，回傳補齊全部欄位後的 df。"""
        df = DivergenceEngine.detect_macd_divergence(df, deviation=deviation)
        df = DivergenceEngine.detect_fake_breakout(
            df, lookback=lookback, confirm_bars=confirm_bars, min_rvol=min_rvol
        )
        return df