import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class BreakoutEngine:
    """
    🚀 飆股評分引擎 (Breakout Engine) — 獨立模組

    實作使用者提供的「波段實戰七層過濾模型」+ 100 分飆股評分系統，
    並內建「假訊號防禦引擎 (False Signal / Baiting Defense)」防止誘多/誘空陷阱。

    設計原則（依需求刻意獨立）：
    - 本引擎獨立於 StrategyEngine / IndicatorEngine 之外運作，不共用、不污染
      既有的多空 ai_score 決策邏輯，避免兩套評分系統互相混淆。
    - 只需要標準 OHLCV（含 date 可選）欄位即可運作，內部自行計算
      EMA20 / SMA20 / SMA240 / MACD，不依賴 IndicatorEngine 是否已跑過。
    - 若 df 剛好含有 StructureEngine 產生的 `zigzag` / `zigzag_confirmed` 欄位，
      HH+HL 結構判斷會優先採用那組更準確的轉折點；沒有的話會退回內建的
      局部極值近似算法，不會因此報錯。

    ⚠️ 誠實揭露（重要）：
    「產業題材」(20分) 與「基本面 3新2益」(15分) 這兩項，本質上需要新聞/法說會/
    財報等非結構化資訊才能客觀判定，純粹的價量技術分析無法產生有意義的分數。
    這裡把它們設計成外部輸入參數（theme_score / fundamental_score），預設為 0。
    如果沒有外部提供，代表這檔股票在這兩項上完全沒有加分，分數會偏低，
    這是「資料不足」而不是「這檔股票真的沒題材」，回傳結果的
    `score_breakdown` 會明確標示每一項的分數來源，避免誤讀。
    """

    YEAR_LINE_WINDOW = 240  # 台股慣例年線 MA240

    # ==========================================
    # 0. 內部技術指標（獨立計算，不依賴 IndicatorEngine）
    # ==========================================
    @staticmethod
    def _compute_internal_indicators(df: pd.DataFrame) -> pd.DataFrame:
        c, v = df['close'], df['volume']
        out = pd.DataFrame(index=df.index)
        out['ema_20'] = c.ewm(span=20, adjust=False).mean()
        out['sma_20'] = c.rolling(20).mean()
        out['sma_60'] = c.rolling(60).mean()
        out['sma_240'] = c.rolling(BreakoutEngine.YEAR_LINE_WINDOW, min_periods=120).mean()
        out['vol_ma5'] = v.rolling(5).mean()
        out['macd_dif'] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        out['macd_dea'] = out['macd_dif'].ewm(span=9, adjust=False).mean()
        out['macd_hist'] = out['macd_dif'] - out['macd_dea']
        return out

    # ==========================================
    # 第零層：年線濾網（絕對濾網，S 級淘汰規則）
    # ==========================================
    @staticmethod
    def year_line_filter(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
        close = float(df['close'].iloc[-1])
        sma240 = ind['sma_240'].iloc[-1]
        sma240_prev = ind['sma_240'].iloc[-6] if len(ind) > 6 else np.nan

        if pd.isna(sma240):
            return {'pass': None, 'year_line_rising': None, 'close': close,
                    'sma_240': None, 'reason': '資料不足 240 個交易日，無法計算年線，暫不淘汰'}

        above = close > sma240
        rising = pd.notna(sma240_prev) and sma240 > sma240_prev

        if above and rising:
            reason = '✅ 站上年線且年線翻揚，符合絕對濾網'
        elif above:
            reason = '⚠️ 站上年線但年線走平/向下，濾網邊緣通過'
        else:
            reason = '🚫 跌破年線，依規則直接淘汰（S級）'

        return {
            'pass': bool(above),
            'year_line_rising': bool(rising) if pd.notna(sma240_prev) else None,
            'close': close,
            'sma_240': float(sma240),
            'reason': reason,
        }

    # ==========================================
    # 假訊號防禦引擎：MACD 背離偵測（誘多／誘空）
    # ==========================================
    @staticmethod
    def detect_macd_divergence(df: pd.DataFrame, ind: pd.DataFrame, lookback: int = 60, order: int = 3) -> dict:
        """
        頂背離（誘多警告）：價格創新高，但 MACD DIF 的高點卻走低 → bearish divergence
        底背離（誘空／反轉觀察）：價格創新低，但 MACD DIF 的低點卻走高 → bullish divergence

        優先使用 StructureEngine 的 zigzag_confirmed 轉折點（若存在）以提升準確度，
        否則退回用 scipy.signal.argrelextrema 對收盤價做局部極值近似偵測。
        """
        result = {'bearish_divergence': False, 'bullish_divergence': False, 'detail': ''}
        n = len(df)
        window = min(lookback, n)
        if window < order * 2 + 5:
            result['detail'] = '資料不足，無法進行背離偵測'
            return result

        sub_close = df['close'].values[-window:]
        sub_macd = ind['macd_dif'].values[-window:]

        if 'zigzag' in df.columns and 'zigzag_confirmed' in df.columns:
            sub_df = df.iloc[-window:]
            confirmed = sub_df[sub_df['zigzag'].notna() & (sub_df['zigzag_confirmed'] == True)]  # noqa: E712
            if len(confirmed) >= 2:
                positions = [df.index.get_loc(i) - (n - window) for i in confirmed.index]
                vals = confirmed['zigzag'].values
                # ⚠️ 修正說明：原本這裡把 high_idx / low_idx 都設成同一份「全部
                # 轉折點混在一起」的清單（註解寫「下面統一比對相鄰同極性者」，
                # 但程式碼從未真的做這個過濾），導致下面直接取 [-2]/[-1] 時，
                # 比較到的常常是一個高點配一個低點（zigzag 轉折點本質上高低
                # 交替出現），而不是「兩個高點」或「兩個低點」，背離判斷因此
                # 沒有意義。
                #
                # zigzag 轉折點必定高低交替，因此可以從相鄰兩點的數值大小
                # 反推每個點是高點還是低點：後一點數值較高 → 該點是高點、
                # 前一點是低點，依此交替標記整個序列，再依「同極性」分別
                # 收進 high_idx / low_idx，讓下面的 [-2]/[-1] 比較的一定是
                # 同一種轉折點（兩個高點比頂背離、兩個低點比底背離）。
                if len(vals) >= 2:
                    kinds = [None] * len(vals)
                    kinds[0] = 'L' if vals[0] < vals[1] else 'H'
                    for k in range(1, len(vals)):
                        kinds[k] = 'H' if kinds[k - 1] == 'L' else 'L'
                    high_idx = [p for p, k in zip(positions, kinds) if k == 'H']
                    low_idx = [p for p, k in zip(positions, kinds) if k == 'L']
                else:
                    high_idx, low_idx = BreakoutEngine._fallback_extrema(sub_close, order)
            else:
                high_idx, low_idx = BreakoutEngine._fallback_extrema(sub_close, order)
        else:
            high_idx, low_idx = BreakoutEngine._fallback_extrema(sub_close, order)

        if len(high_idx) >= 2:
            i1, i2 = high_idx[-2], high_idx[-1]
            if (pd.notna(sub_macd[i1]) and pd.notna(sub_macd[i2])
                    and sub_close[i2] > sub_close[i1] and sub_macd[i2] < sub_macd[i1]):
                result['bearish_divergence'] = True
                result['detail'] += (f"⚠️ 頂背離(疑似誘多)：價格創新高({sub_close[i1]:.2f}→{sub_close[i2]:.2f})，"
                                      f"但MACD走低({sub_macd[i1]:.3f}→{sub_macd[i2]:.3f})。 ")

        if len(low_idx) >= 2:
            j1, j2 = low_idx[-2], low_idx[-1]
            if (pd.notna(sub_macd[j1]) and pd.notna(sub_macd[j2])
                    and sub_close[j2] < sub_close[j1] and sub_macd[j2] > sub_macd[j1]):
                result['bullish_divergence'] = True
                result['detail'] += (f"🔍 底背離(潛在反轉)：價格創新低({sub_close[j1]:.2f}→{sub_close[j2]:.2f})，"
                                      f"但MACD走高({sub_macd[j1]:.3f}→{sub_macd[j2]:.3f})。")

        return result

    @staticmethod
    def _fallback_extrema(close_arr: np.ndarray, order: int):
        def _dedupe(idxs, min_gap):
            cleaned = []
            for i in idxs:
                if not cleaned or i - cleaned[-1] > min_gap:
                    cleaned.append(i)
            return cleaned

        high_idx = _dedupe(list(argrelextrema(close_arr, np.greater_equal, order=order)[0]), order)
        low_idx = _dedupe(list(argrelextrema(close_arr, np.less_equal, order=order)[0]), order)
        return high_idx, low_idx

    # ==========================================
    # 第三層：修正箱理論
    # ==========================================
    @staticmethod
    def analyze_consolidation_box(df: pd.DataFrame, min_box_days: int = 20, max_box_days: int = 60,
                                   tight_threshold_pct: float = 12.0, exclude_recent_days: int = 10) -> dict:
        """
        在「排除最近 exclude_recent_days 天」之後的區間裡，尋找
        [min_box_days, max_box_days] 範圍內波動幅度 (max-min)/mid <= tight_threshold_pct
        的最長整理區間，判定修正箱品質。

        ⚠️ 修正說明：原本箱型視窗直接取「最近 N 天」，如果最近幾天剛好是價格
        急拉突破的走勢，只要百分比波動還在門檻內，會被誤算進箱型本身，導致
        box_high 被突破走勢墊高，後續 analyze_breakout 永遠判定不到突破
        （因為箱頂已經跟最新價格一樣高）。現在把最近 exclude_recent_days 天
        排除在箱型計算之外，箱型代表「突破前」真正需要被突破的壓力區間，
        排除掉的最近幾天則留給 analyze_breakout 判斷是否真的站上箱頂。
        """
        c = df['close']
        n = len(c)
        usable = c.iloc[:-exclude_recent_days] if exclude_recent_days > 0 else c
        m = len(usable)
        upper_days = min(max_box_days, m - 1)
        if upper_days < min_box_days:
            return {'has_box': False, 'reason': '資料不足以判斷修正箱'}

        best = None
        for days in range(min_box_days, upper_days + 1):
            window = usable.iloc[-days:]
            box_high, box_low = float(window.max()), float(window.min())
            mid = (box_high + box_low) / 2 if (box_high + box_low) != 0 else 1e-9
            range_pct = (box_high - box_low) / mid * 100
            if range_pct <= tight_threshold_pct:
                if best is None or days > best['box_days']:
                    best = {'box_days': days, 'box_high': box_high, 'box_low': box_low,
                             'range_pct': round(range_pct, 2)}

        if best is None:
            window = usable.iloc[-min_box_days:]
            box_high, box_low = float(window.max()), float(window.min())
            mid = (box_high + box_low) / 2 if (box_high + box_low) != 0 else 1e-9
            range_pct = (box_high - box_low) / mid * 100
            return {
                'has_box': False, 'box_quality': '壞箱(震盪過大)',
                'box_days': min_box_days, 'box_high': box_high, 'box_low': box_low,
                'range_pct': round(range_pct, 2),
                'reason': f"近{min_box_days}日波動幅度{range_pct:.1f}%，超過{tight_threshold_pct}%門檻，籌碼不夠穩定",
            }

        best['has_box'] = True
        best['box_quality'] = '優質修正箱'
        best['reason'] = (f"近{best['box_days']}日於{best['box_low']:.2f}~{best['box_high']:.2f}"
                           f"窄幅整理(波動{best['range_pct']:.1f}%)，籌碼沉澱穩定")
        return best

    # ==========================================
    # 第四層：月線敏感帶（EMA20 vs SMA20）
    # ==========================================
    @staticmethod
    def analyze_ema_sma_band(ind: pd.DataFrame) -> dict:
        ema20, sma20 = ind['ema_20'], ind['sma_20']
        is_bullish = ema20 > sma20

        streak = 0
        for val in is_bullish.iloc[::-1]:
            if pd.isna(val):
                break
            if val:
                streak += 1
            else:
                break

        def _gap_pct(idx):
            if len(ind) <= abs(idx):
                return np.nan
            e, s = ema20.iloc[idx], sma20.iloc[idx]
            if pd.isna(e) or pd.isna(s) or s == 0:
                return np.nan
            return (e - s) / s * 100

        gap_now, gap_prev = _gap_pct(-1), _gap_pct(-6)
        widening = pd.notna(gap_now) and pd.notna(gap_prev) and gap_now > gap_prev

        return {
            'ema20_gt_sma20': bool(is_bullish.iloc[-1]) if pd.notna(is_bullish.iloc[-1]) else False,
            'golden_cross_streak_days': int(streak),
            'gap_pct': None if pd.isna(gap_now) else round(float(gap_now), 2),
            'gap_widening': bool(widening),
        }

    # ==========================================
    # 第五層：量價共振（三段量能驗證）
    # ==========================================
    @staticmethod
    def analyze_volume_resonance(df: pd.DataFrame, box_info: dict = None, lookback: int = 30) -> dict:
        """
        ⚠️ 修正說明：原本用「量能爆量 + 當時創新高」自行找突破日，但在一般
        上升趨勢中，價格幾乎每天都在創新高，只要剛好某天量能隨機偏高就會被
        誤判為「突破日」，導致三段共振的起點根本不是真正的箱型突破，而是
        隨機雜訊。現在改為：優先使用 analyze_consolidation_box 算出的
        box_high，把「收盤價第一次站上 box_high」的那一天定義為突破日，
        這樣三段共振（爆量突破 → 量縮整理 → 再次放量）才真正對應同一個
        技術事件，而不是各自獨立判斷出兩件不相干的事。
        """
        v, c = df['volume'], df['close']
        n = len(df)
        window = min(lookback, n)
        sub_v = v.iloc[-window:].reset_index(drop=True)
        sub_c = c.iloc[-window:].reset_index(drop=True)
        vol_ma5_prior = v.rolling(5).mean().shift(1)

        breakout_day_idx = None
        if box_info and box_info.get('has_box'):
            box_high = box_info['box_high']
            for i in range(window):
                if sub_c.iloc[i] > box_high:
                    breakout_day_idx = i
                    break
        else:
            # 沒有箱型資訊時，退回舊版簡化heuristic（準確度較低）
            for i in range(5, window):
                actual_idx = n - window + i
                avg5 = vol_ma5_prior.iloc[actual_idx]
                if pd.isna(avg5) or avg5 <= 0:
                    continue
                if sub_v.iloc[i] >= 2.0 * avg5 and sub_c.iloc[i] >= sub_c.iloc[:i + 1].max() - 1e-9:
                    breakout_day_idx = i

        if breakout_day_idx is None or breakout_day_idx >= window - 2:
            return {'resonance_confirmed': False, 'stage': '尚未偵測到有效的箱型突破起漲點'}

        breakout_vol = float(sub_v.iloc[breakout_day_idx])
        avg5_at_breakout = vol_ma5_prior.iloc[n - window + breakout_day_idx]
        breakout_vol_ok = pd.notna(avg5_at_breakout) and avg5_at_breakout > 0 and breakout_vol >= 2.0 * avg5_at_breakout

        contraction_slice = sub_v.iloc[breakout_day_idx + 1: window - 1]
        if len(contraction_slice) < 2:
            return {'resonance_confirmed': False, 'stage': '突破後整理天數不足'}

        contraction_avg = float(contraction_slice.mean())
        contraction_ratio = contraction_avg / breakout_vol if breakout_vol > 0 else np.nan
        contraction_ok = pd.notna(contraction_ratio) and 0.2 <= contraction_ratio <= 0.5

        latest_vol = float(sub_v.iloc[-1])
        reexpansion_ratio = latest_vol / contraction_avg if contraction_avg > 0 else np.nan
        reexpansion_ok = pd.notna(reexpansion_ratio) and reexpansion_ratio >= 1.5

        return {
            'resonance_confirmed': bool(breakout_vol_ok and contraction_ok and reexpansion_ok),
            'breakout_volume': breakout_vol,
            'breakout_volume_surge_confirmed': bool(breakout_vol_ok),
            'contraction_avg_volume': contraction_avg,
            'contraction_ratio': None if pd.isna(contraction_ratio) else round(float(contraction_ratio), 2),
            'latest_volume': latest_vol,
            'reexpansion_ratio': None if pd.isna(reexpansion_ratio) else round(float(reexpansion_ratio), 2),
            'stage': '三段量價共振確認' if (breakout_vol_ok and contraction_ok and reexpansion_ok) else '尚未完成三段共振驗證',
        }

    # ==========================================
    # 第六層：技術突破 + 首次回測進場點
    # ==========================================
    @staticmethod
    def analyze_breakout(df: pd.DataFrame, box_info: dict, buffer_pct: float = 0.5,
                          confirm_window: int = 10) -> dict:
        """
        confirm_window 應與 analyze_consolidation_box 的 exclude_recent_days 一致
        （預設都是 10），代表「箱型排除掉、留給突破+量縮+再突破序列發展」的天數。
        """
        if not box_info.get('has_box'):
            return {'breakout_confirmed': False, 'reason': '無有效修正箱可供突破判定'}

        box_high = box_info['box_high']
        c = df['close']
        latest_close = float(c.iloc[-1])
        recent_window = c.iloc[-confirm_window:-1] if confirm_window > 1 else c.iloc[0:0]
        already_broke = bool((recent_window > box_high).any())
        breakout_confirmed = bool(latest_close > box_high * (1 + buffer_pct / 100) or already_broke)

        retest_zone_low, retest_zone_high = box_high, box_high * 1.02
        is_first_retest = bool(already_broke and retest_zone_low <= latest_close <= retest_zone_high * 1.01)

        return {
            'breakout_confirmed': breakout_confirmed,
            'box_high': box_high,
            'is_first_retest_entry': is_first_retest,
            'suggested_entry_zone': [round(retest_zone_low, 2), round(retest_zone_high, 2)],
            'suggested_stop': box_info.get('box_low'),
        }

    # ==========================================
    # 第七層：強者恆強（HH + HL 結構）
    # ==========================================
    @staticmethod
    def analyze_hh_hl_structure(df: pd.DataFrame) -> dict:
        """
        優先使用 StructureEngine 的 zigzag_confirmed 轉折點；
        沒有的話退回用局部極值近似判斷（準確度較低，僅供參考）。

        ⚠️ 簡化說明：完整的 HH/HL 判斷需要分別追蹤「高點序列」與「低點序列」
        是否個別遞增，這裡採用簡化版（近 4 個轉折點整體遞增 + 首尾比較），
        遇到複雜的震盪走勢時可能誤判，建議搭配圖表人工確認。
        """
        if 'zigzag' in df.columns and 'zigzag_confirmed' in df.columns:
            pivots = df[df['zigzag'].notna() & (df['zigzag_confirmed'] == True)]  # noqa: E712
            vals = pivots['zigzag'].tolist()
            source = 'StructureEngine zigzag_confirmed'
        else:
            c = df['close'].values
            hi = argrelextrema(c, np.greater_equal, order=5)[0]
            lo = argrelextrema(c, np.less_equal, order=5)[0]
            combined = sorted([(int(i), float(c[i])) for i in hi] + [(int(i), float(c[i])) for i in lo])
            vals = [v for _, v in combined]
            source = '內建局部極值近似(建議搭配StructureEngine提升準確度)'

        if len(vals) < 4:
            return {'hh_hl_confirmed': False, 'reason': '轉折點不足，無法判斷HH+HL結構', 'source': source}

        last4 = vals[-4:]
        increasing = all(last4[i] <= last4[i + 2] for i in range(len(last4) - 2))
        trend_up = last4[-1] > last4[0]
        confirmed = bool(increasing and trend_up)

        return {
            'hh_hl_confirmed': confirmed,
            'recent_pivots': [round(v, 2) for v in last4],
            'source': source,
            'reason': '近期轉折呈現高點/低點同步墊高的多頭結構' if confirmed else '結構尚未確立或已轉弱',
        }

    @staticmethod
    def check_new_swing_high(df: pd.DataFrame, lookback: int = 60) -> dict:
        c = df['close']
        if len(c) < 2:
            return {'new_high': False}
        ref_window = c.iloc[-(lookback + 1):-1] if len(c) > lookback else c.iloc[:-1]
        if ref_window.empty:
            return {'new_high': False}
        recent_max = float(ref_window.max())
        return {'new_high': bool(float(c.iloc[-1]) >= recent_max), 'recent_max': recent_max}

    # ==========================================
    # 第二層：法人/主力籌碼分數（需外部 ChipEngine 資料）
    # ==========================================
    @staticmethod
    def score_institutional(chip_report: dict = None) -> dict:
        """
        chip_report: ChipEngine.build_chip_report() 的回傳結果。
        法人籌碼(10分)：外資今日買超為正即給分。
        主力籌碼(10分)：融資減、融券增（軋空吃貨型態）給分。
        沒有籌碼資料時，此項以 0 分計，並在 detail 中明確說明原因，
        避免使用者誤以為「該股籌碼真的很差」。
        """
        if not chip_report or chip_report.get('status') != 'ok':
            return {'score': 0, 'max_score': 20, 'detail': ['籌碼資料不可用（可能為上櫃/興櫃或TWSE服務異常），此項以0分計']}

        score, detail = 0, []
        inst = chip_report.get('institutional')
        margin = chip_report.get('margin')

        if inst:
            foreign_net = inst.get('foreign_net', 0)
            if foreign_net > 0:
                score += 10
                detail.append(f"外資買超 {foreign_net:,} 股")
            else:
                detail.append("外資今日未買超")

        if margin:
            if margin.get('margin_change', 0) < 0 and margin.get('short_change', 0) > 0:
                score += 10
                detail.append("融資減、融券增，典型軋空吃貨型態")
            else:
                detail.append("未觀察到融資減券增的軋空型態")

        return {'score': min(score, 20), 'max_score': 20, 'detail': detail}

    # ==========================================
    # 主要進入點：完整七層評分（100分制）
    # ==========================================
    @staticmethod
    def analyze(df: pd.DataFrame, theme_score: float = 0, fundamental_score: float = 0,
                chip_report: dict = None, min_box_days: int = 20, macd_lookback: int = 60) -> dict:
        """
        對單一股票最新狀態執行完整「飆股七層評分」。

        Parameters
        ----------
        df : 標準 OHLCV DataFrame（需含 date, open, high, low, close, volume）。
             若已跑過 StructureEngine.add_swing_points，會自動用其 zigzag_confirmed
             轉折點提升 HH+HL 與背離偵測的準確度。
        theme_score : 0~20，外部提供的產業題材分數（需人工/新聞判定），預設 0
        fundamental_score : 0~15，外部提供的 3新2益基本面分數，預設 0
        chip_report : 可選，傳入 ChipEngine.build_chip_report() 結果以計算法人/主力籌碼分數

        Returns
        -------
        dict，含 total_score / grade / 各層明細 / 假訊號警示。
        """
        if df is None or len(df) < 30:
            return {'error': '資料筆數不足，無法進行飆股評分（建議至少30個交易日）'}

        df = df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df.reset_index(drop=True)
        ind = BreakoutEngine._compute_internal_indicators(df)

        year_line = BreakoutEngine.year_line_filter(df, ind)
        divergence = BreakoutEngine.detect_macd_divergence(df, ind, lookback=macd_lookback)
        box_info = BreakoutEngine.analyze_consolidation_box(df, min_box_days=min_box_days)
        band_info = BreakoutEngine.analyze_ema_sma_band(ind)
        resonance = BreakoutEngine.analyze_volume_resonance(df, box_info=box_info)
        breakout_info = BreakoutEngine.analyze_breakout(df, box_info, confirm_window=10)
        structure_info = BreakoutEngine.analyze_hh_hl_structure(df)
        new_high_info = BreakoutEngine.check_new_swing_high(df)
        chip_score_info = BreakoutEngine.score_institutional(chip_report)

        scores = {
            '產業題材': {'score': float(np.clip(theme_score, 0, 20)), 'max': 20, 'source': '外部輸入(需人工/新聞判定)'},
            '基本面3新2益': {'score': float(np.clip(fundamental_score, 0, 15)), 'max': 15, 'source': '外部輸入(需人工/財報判定)'},
            '法人主力籌碼': {'score': chip_score_info['score'], 'max': 20,
                          'source': 'ChipEngine' if (chip_report and chip_report.get('status') == 'ok') else '無資料,以0分計'},
            '均線動能(EMA20>SMA20)': {
                'score': 10 if (band_info['ema20_gt_sma20'] and band_info.get('gap_widening'))
                else (5 if band_info['ema20_gt_sma20'] else 0), 'max': 10},
            '修正箱整理': {'score': 10 if box_info.get('has_box') else 0, 'max': 10},
            '技術突破': {'score': 10 if breakout_info.get('breakout_confirmed') else 0, 'max': 10},
            '量價結構': {'score': 10 if resonance.get('resonance_confirmed') else 0, 'max': 10},
            '趨勢延續(HH+HL/創新高)': {
                'score': 5 if (structure_info.get('hh_hl_confirmed') or new_high_info.get('new_high')) else 0, 'max': 5},
        }

        total_score = sum(item['score'] for item in scores.values())
        max_possible = sum(item['max'] for item in scores.values())

        baiting_alert = None
        if divergence.get('bearish_divergence'):
            baiting_alert = ("🚨 高危險誘盤警告：偵測到 MACD 頂背離，價格創新高但動能背離，"
                              "疑似主力誘多出貨，建議不要在此追價。")
            # 假訊號防禦：無論其他分數多高，強制封頂，避免誘多股被誤判為 A/B 級
            total_score = min(total_score, 55)

        if year_line.get('pass') is False:
            grade = 'S級(淘汰)'
        elif total_score >= 90:
            grade = 'A級(飆股候選)'
        elif total_score >= 80:
            grade = 'B級(重點追蹤)'
        elif total_score >= 70:
            grade = 'C級(觀察名單)'
        else:
            grade = 'D級(不具操作價值)'

        return {
            'total_score': round(total_score, 1),
            'max_possible_score': max_possible,
            'grade': grade,
            'year_line_filter': year_line,
            'baiting_alert': baiting_alert,
            'macd_divergence': divergence,
            'consolidation_box': box_info,
            'ema_sma_band': band_info,
            'volume_resonance': resonance,
            'breakout': breakout_info,
            'structure_hh_hl': structure_info,
            'new_high': new_high_info,
            'score_breakdown': scores,
        }