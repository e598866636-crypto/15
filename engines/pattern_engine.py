import numpy as np
import pandas as pd


class PatternEngine:
    """
    📐 型態辨識引擎 (Chart Pattern Engine) - TQAI Pro v2.8

    依據《型態學與量價結構技術分析大師講義》新增型態辨識，涵蓋講義五大核心
    模組中的四種（箱型整理本質上等同於「盤整區間突破」，已有相近概念散落
    在 divergence_engine.py 的 breakout_up/breakout_down 與 momentum_engine.py
    的修正箱過濾邏輯中，故本引擎不重複實作，聚焦在講義中真正尚未被涵蓋的
    四種型態）。優先順序依「複雜度／因果難度」由難到易開發：

      1. 頭肩頂 / 頭肩底 (Head & Shoulders Top / Bottom) —— 最先完成
         講義原文：「頭肩型態是比M頭和W底更為宏觀、複雜的結構」。

      2. 跳空缺口分類 (Gap Classification：普通/突破/逃逸(測量)/竭盡) —— 次完成
         四種缺口中，「竭盡缺口」的定義本質上依賴「未來是否被迅速回補」，
         「突破缺口」依賴「未來 3-5 天是否未被回補」—— 這兩種分類都無法在
         缺口發生的當下就判定，是本次新增型態中最容易不小心引入前視偏誤
         (look-ahead bias) 的部分，因此優先處理，並把因果安全邊界設計清楚。

      3. M頭 (雙頂) / W底 (雙底) (Double Top / Double Bottom) —— 本次新增
         結構比頭肩型態單純（只需 3 個轉折點、頸線為單一水平線而非斜線），
         講義原文亦明確指出頭肩「比M頭和W底更為宏觀、複雜」，故排在其後。

      4. 旗形 / 三角旗形 (Flag / Pennant) —— 本次新增
         判斷「旗桿→量縮旗面→帶量再突破」三階段，需要動態追蹤整理區間的
         上下界並持續檢查是否失敗（跌破最大回檔容忍度），是四種新增型態中
         實作邏輯最長、且需要額外一個「型態失敗」分支的一種，故放在最後。

    ⚠️ 因果安全設計（沿用 structure_engine.py / divergence_engine.py 既有的
    設計邊界，未來擴充型態時請比照辦理，勿破壞此邊界）：

    任何需要「未來走勢」才能定案的型態，都不可以把訊號往回標記到型態實際
    發生（或開始形成）的那一天，只能在『確認的當下』那根K棒標記，且確認
    邏輯只能往回看已經發生的歷史資料，不可使用confirm當下尚未存在的未來
    資訊。本引擎的兩個偵測函式都遵循這個原則：

      - detect_head_shoulders()：左肩/頭部/右肩三個高低點，一律採用跟
        divergence_engine.py 相同的「動態極值＋反轉確認」狀態機取得，且
        只使用「已確認」的轉折點（不使用最後一筆仍在動態更新、尚未反轉的
        暫定極值——這正是 structure_engine.py docstring 警告 zigzag 不可
        直接用於策略訊號的原因）。頭肩頂/底只在「頸線被跌破/突破確認」的
        那一根K棒標記為 True，且頸線延伸與突破掃描只往後看已發生的資料。

      - classify_gaps()：缺口「發生當天」只標記 gap_up / gap_down 兩個
        當下立即可判斷的原始事實欄位；缺口的「分類」則必須等待
        confirm_window 天觀察期滿（或提前被回補）才能定案，分類結果只
        寫在『確認的那一天』，不回填到缺口發生當天。

    因此本引擎輸出的 hs_top_confirmed / hs_bottom_confirmed 與
    gap_type_confirmed 皆可安全餵給 StrategyEngine 或 BacktestEngine 當
    特徵使用，不會有 repaint 問題；但跟 StructureEngine 的 zigzag 一樣，
    「型態確認」本質上永遠會比「型態開始形成」晚幾根K棒才能拍板，這是型態
    學方法論本身的極限，不是本引擎的臭蟲(bug)。

    ⚠️ 目前尚未接入 pipeline：
    比照 structure_engine.py 的既有邊界，本引擎目前是獨立模組，
    ScannerEngine._run_single_pipeline 尚未呼叫它。是否要把
    hs_top_confirmed / hs_bottom_confirmed / gap_type_confirmed 接入
    StrategyEngine 或 MomentumEngine 作為額外加分/扣分項，建議先在少量
    標的上驗證型態辨識的準確度與觸發頻率後，再決定是否/如何接入，避免
    重蹈 strategy_engine.py 過去雙重計分問題的覆轍。
    """

    # ==========================================
    # 0. 共用工具：動態極值＋反轉確認 狀態機（只回傳「已確認」的轉折點）
    # ==========================================
    @staticmethod
    def _find_confirmed_pivots(close: np.ndarray, deviation: float = 0.04):
        """
        沿用 structure_engine.py / divergence_engine.py 的狀態機邏輯，
        但只回傳「已經被反轉確認」的轉折點（不含最後一筆仍在動態更新中的
        暫定極值），供頭肩型態辨識使用。

        回傳：List[dict]，每筆 {'idx': int, 'price': float, 'kind': 'H'|'L'}
        依時間先後排序，且 kind 必定交替出現 (H, L, H, L, ...)。
        """
        n = len(close)
        pivots = []
        if n < 3:
            return pivots

        state = 0  # 0=初始判定, 1=尋找波段高點, -1=尋找波段低點
        pivot_idx = 0
        pivot_val = close[0]

        for i in range(1, n):
            price = close[i]
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
                    # 高點於本日反轉確認
                    pivots.append({'idx': pivot_idx, 'price': pivot_val, 'kind': 'H'})
                    state = -1
                    pivot_idx, pivot_val = i, price

            elif state == -1:
                if price < pivot_val:
                    pivot_idx, pivot_val = i, price
                elif dev > deviation:
                    # 低點於本日反轉確認
                    pivots.append({'idx': pivot_idx, 'price': pivot_val, 'kind': 'L'})
                    state = 1
                    pivot_idx, pivot_val = i, price

        return pivots

    # ==========================================
    # 1. 頭肩頂 / 頭肩底 (Head & Shoulders)
    # ==========================================
    @staticmethod
    def detect_head_shoulders(df: pd.DataFrame, deviation: float = 0.04,
                               shoulder_tolerance: float = 0.12,
                               neckline_break_buffer: float = 0.0):
        """
        偵測頭肩頂 (Head & Shoulders Top) 與頭肩底 (Head & Shoulders Bottom)。

        頭肩頂結構（依講義定義）：左肩(高) → 頸線點1(低) → 頭部(最高) →
        頸線點2(低) → 右肩(高，與左肩略同) → 跌破頸線（連接頸線點1、2的
        斜線）確立型態。頭肩底為對稱結構（低點版本）。

        參數：
          deviation             轉折點偵測的最小反轉幅度（同 zigzag 邏輯）
          shoulder_tolerance    左右肩高度差可容忍的相對誤差（預設12%）
          neckline_break_buffer 跌破/突破頸線的緩衝百分比，避免頸線價位附近
                                 的雜訊假突破（預設0，不緩衝）

        欄位輸出：
          hs_top_confirmed / hs_bottom_confirmed : bool，頸線跌破/突破確認當天
          hs_neckline_price : float，確認當天的頸線水位（斜線頸線的內插值）
          hs_target_price    : float，依講義「等幅測量」公式算出的滿足點
          hs_note             : str，人類可讀說明（僅在確認當天有值）
        """
        df = df.copy()
        n = len(df)
        df['hs_top_confirmed'] = False
        df['hs_bottom_confirmed'] = False
        df['hs_neckline_price'] = np.nan
        df['hs_target_price'] = np.nan
        df['hs_note'] = ""

        if n < 30 or 'close' not in df.columns:
            return df

        close = df['close'].to_numpy(dtype=float)
        pivots = PatternEngine._find_confirmed_pivots(close, deviation=deviation)
        if len(pivots) < 5:
            return df

        notes = [""] * n
        top_flags = np.zeros(n, dtype=bool)
        bottom_flags = np.zeros(n, dtype=bool)
        neckline_arr = np.full(n, np.nan)
        target_arr = np.full(n, np.nan)

        # 避免同一天被多組候選型態重複標記造成欄位互相覆蓋
        used_break_idx = set()

        # 依序檢查每一組連續 5 個轉折點是否構成頭肩結構；
        # 一旦右肩(第5個轉折點)已確認，就從那個時間點開始只往後掃描頸線
        # 是否被跌破/突破，不使用尚未發生的資料。
        for k in range(len(pivots) - 4):
            p0, p1, p2, p3, p4 = pivots[k:k + 5]

            # ---- 頭肩頂：H, L, H, L, H，頭部最高，兩肩高度相近 ----
            if p0['kind'] == 'H' and p1['kind'] == 'L' and p2['kind'] == 'H' \
               and p3['kind'] == 'L' and p4['kind'] == 'H':
                left_s, head, right_s = p0['price'], p2['price'], p4['price']
                if head > left_s and head > right_s:
                    shoulder_diff = abs(left_s - right_s) / ((left_s + right_s) / 2 + 1e-9)
                    if shoulder_diff <= shoulder_tolerance:
                        neck1_idx, neck1_price = p1['idx'], p1['price']
                        neck3_idx, neck3_price = p3['idx'], p3['price']
                        slope = (neck3_price - neck1_price) / (neck3_idx - neck1_idx) \
                            if neck3_idx > neck1_idx else 0.0

                        start_scan = p4['idx'] + 1
                        for j in range(start_scan, n):
                            if j in used_break_idx:
                                continue
                            neckline_j = neck1_price + slope * (j - neck1_idx)
                            if close[j] < neckline_j * (1 - neckline_break_buffer):
                                top_flags[j] = True
                                neckline_arr[j] = neckline_j
                                height = head - neckline_j
                                target_arr[j] = neckline_j - height
                                notes[j] = (
                                    f"🔴 頭肩頂確認：左肩{left_s:.2f} / 頭部{head:.2f} / 右肩{right_s:.2f}，"
                                    f"跌破頸線 {neckline_j:.2f}，等幅測量目標價約 {target_arr[j]:.2f}"
                                )
                                used_break_idx.add(j)
                                break

            # ---- 頭肩底：L, H, L, H, L，頭部最低，兩肩高度相近 ----
            if p0['kind'] == 'L' and p1['kind'] == 'H' and p2['kind'] == 'L' \
               and p3['kind'] == 'H' and p4['kind'] == 'L':
                left_s, head, right_s = p0['price'], p2['price'], p4['price']
                if head < left_s and head < right_s:
                    shoulder_diff = abs(left_s - right_s) / ((left_s + right_s) / 2 + 1e-9)
                    if shoulder_diff <= shoulder_tolerance:
                        neck1_idx, neck1_price = p1['idx'], p1['price']
                        neck3_idx, neck3_price = p3['idx'], p3['price']
                        slope = (neck3_price - neck1_price) / (neck3_idx - neck1_idx) \
                            if neck3_idx > neck1_idx else 0.0

                        start_scan = p4['idx'] + 1
                        for j in range(start_scan, n):
                            if j in used_break_idx:
                                continue
                            neckline_j = neck1_price + slope * (j - neck1_idx)
                            if close[j] > neckline_j * (1 + neckline_break_buffer):
                                bottom_flags[j] = True
                                neckline_arr[j] = neckline_j
                                height = neckline_j - head
                                target_arr[j] = neckline_j + height
                                notes[j] = (
                                    f"🟢 頭肩底確認：左肩{left_s:.2f} / 頭部{head:.2f} / 右肩{right_s:.2f}，"
                                    f"突破頸線 {neckline_j:.2f}，等幅測量目標價約 {target_arr[j]:.2f}"
                                )
                                used_break_idx.add(j)
                                break

        df['hs_top_confirmed'] = top_flags
        df['hs_bottom_confirmed'] = bottom_flags
        df['hs_neckline_price'] = neckline_arr
        df['hs_target_price'] = target_arr
        df['hs_note'] = notes
        return df

    # ==========================================
    # 2. 跳空缺口分類 (Gap Classification)
    # ==========================================
    @staticmethod
    def classify_gaps(df: pd.DataFrame, exhaustion_window: int = 3, confirm_window: int = 5,
                       min_rvol: float = 1.5, trend_slope_threshold: float = 0.4):
        """
        缺口四大分類：普通缺口 / 突破缺口 / 逃逸(測量)缺口 / 竭盡缺口。

        因果安全設計：缺口發生當天只標記 gap_up / gap_down（這是當天就能
        確認的原始事實）；「分類」則必須等待 confirm_window 天（或提前
        回補）才能定案，分類結果只寫在「確認的那一天」，不回填到缺口發生
        當天，避免用到缺口發生當下還不存在的未來資訊。

        分類邏輯（confirm_window 天觀察期滿或提前回補時判定）：
          1. 竭盡缺口：缺口在 exhaustion_window 天內就被回補 → 判定為竭盡
             缺口（趨勢末端動能衰竭的敗象，講義：「缺口會於短期內(1~3天)
             被迅速填補」）。
          2. exhaustion_window < 回補天數 <= confirm_window：普通缺口
             （回補速度不夠快到算竭盡，但終究還是被回補，不具備趨勢預測
             力，講義：「此類缺口不具備預測趨勢的能力，實戰中應視為雜
             訊」）。
          3. 觀察期滿仍未回補：
             a. 缺口發生前市場處於低波動盤整 (|sma_60_slope| < 門檻)，
                且量能達標 → 突破缺口（盤整後帶量表態）。
             b. 缺口發生前已處於明確趨勢中 (|sma_60_slope| >= 門檻)，
                且量能達標 → 逃逸/測量缺口（主升段/主跌段中繼）。
             c. 未達量能門檻者，保守歸類為普通缺口（雜訊型缺口，不強行
                套入突破或逃逸的積極解讀）。

        欄位輸出：
          gap_up / gap_down     : 缺口發生當天（原始事實，立即可知）
          gap_size_pct          : 缺口大小（相對前一日收盤價的百分比）
          gap_type_confirmed    : str，僅在確認當天有值："突破缺口"/"逃逸缺口"/
                                    "竭盡缺口"/"普通缺口"
          gap_confirm_day       : bool，本日是否為某個缺口的分類確認日
          gap_note              : str，人類可讀說明（寫在確認當天）
        """
        df = df.copy()
        n = len(df)
        df['gap_up'] = False
        df['gap_down'] = False
        df['gap_size_pct'] = np.nan
        df['gap_type_confirmed'] = ""
        df['gap_confirm_day'] = False
        df['gap_note'] = ""

        required = ['open', 'high', 'low', 'close']
        if n < confirm_window + 2 or any(c not in df.columns for c in required):
            return df

        h = df['high'].to_numpy(dtype=float)
        l = df['low'].to_numpy(dtype=float)
        c = df['close'].to_numpy(dtype=float)
        rvol = df['rvol'].to_numpy(dtype=float) if 'rvol' in df.columns else np.ones(n)
        slope = df['sma_60_slope'].to_numpy(dtype=float) if 'sma_60_slope' in df.columns else np.zeros(n)

        gap_up = np.zeros(n, dtype=bool)
        gap_down = np.zeros(n, dtype=bool)
        gap_size = np.full(n, np.nan)

        for i in range(1, n):
            if l[i] > h[i - 1]:
                gap_up[i] = True
                gap_size[i] = (l[i] - h[i - 1]) / (c[i - 1] + 1e-9) * 100
            elif h[i] < l[i - 1]:
                gap_down[i] = True
                gap_size[i] = (h[i - 1] - l[i]) / (c[i - 1] + 1e-9) * 100

        gap_type = [""] * n
        confirm_day = np.zeros(n, dtype=bool)
        notes = [""] * n

        for i in range(1, n):
            if not (gap_up[i] or gap_down[i]):
                continue

            trend_up = bool(gap_up[i])
            # 缺口區間邊界：向上缺口為 (前日高, 今日低)；向下缺口為 (今日高, 前日低)
            gap_low = h[i - 1] if trend_up else h[i]
            gap_high = l[i] if trend_up else l[i - 1]

            filled_at = None
            for j in range(i + 1, min(i + 1 + confirm_window, n)):
                if trend_up and l[j] <= gap_low:
                    filled_at = j
                    break
                if (not trend_up) and h[j] >= gap_high:
                    filled_at = j
                    break

            pre_slope = slope[i - 1] if not np.isnan(slope[i - 1]) else 0.0
            is_trending_before = abs(pre_slope) >= trend_slope_threshold
            volume_ok = rvol[i] >= min_rvol

            if filled_at is not None and (filled_at - i) <= exhaustion_window:
                confirm_idx = filled_at
                gap_type[confirm_idx] = "竭盡缺口"
                confirm_day[confirm_idx] = True
                notes[confirm_idx] = (
                    f"⚠️ 竭盡缺口確認：{'向上' if trend_up else '向下'}缺口於 {filled_at - i} 天內即被回補，"
                    f"暗示趨勢末端動能衰竭"
                )
            elif filled_at is not None:
                confirm_idx = filled_at
                gap_type[confirm_idx] = "普通缺口"
                confirm_day[confirm_idx] = True
                notes[confirm_idx] = (
                    f"ℹ️ 普通缺口確認：缺口於 {filled_at - i} 天後被回補，不具備趨勢預測力"
                )
            else:
                confirm_idx = min(i + confirm_window, n - 1)
                if not volume_ok:
                    gap_type[confirm_idx] = "普通缺口"
                    notes[confirm_idx] = (
                        f"ℹ️ 普通缺口確認：觀察 {confirm_window} 天內未回補，但量能未達標，保守歸類為雜訊缺口"
                    )
                elif is_trending_before:
                    gap_type[confirm_idx] = "逃逸缺口"
                    notes[confirm_idx] = (
                        f"🚀 逃逸/測量缺口確認：缺口發生前已處於明確趨勢中，觀察 {confirm_window} 天內"
                        f"未被回補且量能達標，可作為波段測量依據"
                    )
                else:
                    gap_type[confirm_idx] = "突破缺口"
                    notes[confirm_idx] = (
                        f"🎯 突破缺口確認：缺口發生前市場處於低波動盤整，觀察 {confirm_window} 天內"
                        f"未被回補且量能達標，趨勢強烈確立"
                    )
                confirm_day[confirm_idx] = True

        df['gap_up'] = gap_up
        df['gap_down'] = gap_down
        df['gap_size_pct'] = gap_size
        df['gap_type_confirmed'] = gap_type
        df['gap_confirm_day'] = confirm_day
        df['gap_note'] = notes
        return df

    # ==========================================
    # 3. M頭 (雙頂) / W底 (雙底) (Double Top / Double Bottom)
    # ==========================================
    @staticmethod
    def detect_double_top_bottom(df: pd.DataFrame, deviation: float = 0.04,
                                  peak_tolerance: float = 0.05,
                                  neckline_break_buffer: float = 0.0):
        """
        偵測M頭(雙頂) 與 W底(雙底)。

        結構（依講義定義，比頭肩型態單純，只需 3 個轉折點）：
          M頭：高點1 → 中間低點(頸線) → 高點2 → 跌破頸線確立。
               講義原文：「無論是左肩高、右肩高或同高，皆屬M頭範疇」，
               因此不像頭肩頂那樣要求「頭部最高」，只要求兩個高點彼此
               相近（peak_tolerance）即可。
          W底：低點1 → 中間高點(頸線) → 低點2 → 突破頸線確立，對稱邏輯。

        跟 detect_head_shoulders 一樣，只使用「已確認」的轉折點，頸線為
        單一水平線（中間轉折點的價位），確認只在頸線被跌破/突破的那一天
        往後掃描標記，不使用尚未發生的資料。

        欄位輸出：
          double_top_confirmed / double_bottom_confirmed : bool，頸線
              跌破/突破確認當天
          double_neckline_price : float，確認當天引用的頸線水位
          double_target_price    : float，依講義「等幅測量」公式：
              W底目標價 = 頸線 + (頸線 - 底部最低價)
              M頭目標價 = 頸線 - (頭部最高價 - 頸線)
          double_note             : str，人類可讀說明（僅在確認當天有值）
        """
        df = df.copy()
        n = len(df)
        df['double_top_confirmed'] = False
        df['double_bottom_confirmed'] = False
        df['double_neckline_price'] = np.nan
        df['double_target_price'] = np.nan
        df['double_note'] = ""

        if n < 20 or 'close' not in df.columns:
            return df

        close = df['close'].to_numpy(dtype=float)
        pivots = PatternEngine._find_confirmed_pivots(close, deviation=deviation)
        if len(pivots) < 3:
            return df

        top_flags = np.zeros(n, dtype=bool)
        bottom_flags = np.zeros(n, dtype=bool)
        neckline_arr = np.full(n, np.nan)
        target_arr = np.full(n, np.nan)
        notes = [""] * n
        used_break_idx = set()

        for k in range(len(pivots) - 2):
            p0, p1, p2 = pivots[k:k + 3]

            # ---- M頭：H, L, H，兩個高點高度相近 ----
            if p0['kind'] == 'H' and p1['kind'] == 'L' and p2['kind'] == 'H':
                peak1, neckline, peak2 = p0['price'], p1['price'], p2['price']
                diff = abs(peak1 - peak2) / ((peak1 + peak2) / 2 + 1e-9)
                if diff <= peak_tolerance:
                    top_price = max(peak1, peak2)
                    start_scan = p2['idx'] + 1
                    for j in range(start_scan, n):
                        if j in used_break_idx:
                            continue
                        if close[j] < neckline * (1 - neckline_break_buffer):
                            top_flags[j] = True
                            neckline_arr[j] = neckline
                            target_arr[j] = neckline - (top_price - neckline)
                            notes[j] = (
                                f"🔴 M頭(雙頂)確認：高點1 {peak1:.2f} / 高點2 {peak2:.2f}，"
                                f"跌破頸線 {neckline:.2f}，等幅測量目標價約 {target_arr[j]:.2f}"
                            )
                            used_break_idx.add(j)
                            break

            # ---- W底：L, H, L，兩個低點高度相近 ----
            if p0['kind'] == 'L' and p1['kind'] == 'H' and p2['kind'] == 'L':
                trough1, neckline, trough2 = p0['price'], p1['price'], p2['price']
                diff = abs(trough1 - trough2) / ((trough1 + trough2) / 2 + 1e-9)
                if diff <= peak_tolerance:
                    bottom_price = min(trough1, trough2)
                    start_scan = p2['idx'] + 1
                    for j in range(start_scan, n):
                        if j in used_break_idx:
                            continue
                        if close[j] > neckline * (1 + neckline_break_buffer):
                            bottom_flags[j] = True
                            neckline_arr[j] = neckline
                            target_arr[j] = neckline + (neckline - bottom_price)
                            notes[j] = (
                                f"🟢 W底(雙底)確認：低點1 {trough1:.2f} / 低點2 {trough2:.2f}，"
                                f"突破頸線 {neckline:.2f}，等幅測量目標價約 {target_arr[j]:.2f}"
                            )
                            used_break_idx.add(j)
                            break

        df['double_top_confirmed'] = top_flags
        df['double_bottom_confirmed'] = bottom_flags
        df['double_neckline_price'] = neckline_arr
        df['double_target_price'] = target_arr
        df['double_note'] = notes
        return df

    # ==========================================
    # 4. 旗形 / 三角旗形 (Flag / Pennant)
    # ==========================================
    @staticmethod
    def detect_flag_pennant(df: pd.DataFrame, pole_window: int = 5, pole_min_pct: float = 0.15,
                             pole_min_rvol: float = 1.3, min_consolidation: int = 5,
                             max_consolidation: int = 40, max_retrace_pct: float = 0.5,
                             consolidation_vol_ratio: float = 1.0, breakout_min_rvol: float = 1.5,
                             breakout_buffer: float = 0.0):
        """
        偵測旗形/三角旗形中繼型態（多頭與空頭對稱），對應講義「型態生命
        週期三大階段」：
          ① 旗桿：pole_window 天內漲跌幅達 pole_min_pct 以上，且平均量能
             達 pole_min_rvol 倍以上（強力噴出）。
          ② 旗面：旗桿結束後，持續追蹤整理區間的上界(多頭)/下界(空頭)，
             要求整理期間平均量能明顯低於旗桿期間（consolidation_vol_ratio，
             對應「窒息式量縮」），且價格不可回檔超過旗桿高度的
             max_retrace_pct（超過視為型態失敗，非旗形整理，停止追蹤該
             候選旗桿，這是本引擎新增的「型態失敗」分支）。
          ③ 再發動：整理天數達到 min_consolidation 以上後，價格帶量
             (breakout_min_rvol) 突破整理區間上界(多頭)/下界(空頭)，確認
             型態，目標價 = 突破價 ± 旗桿高度（等長幅度測量）。

        因果安全設計：整理區間的上下界（consolidation_high/low）採用逐日
        累積的方式建立（只用「今天以前」已發生的資料），第 j 天是否構成
        突破，只跟第 j 天以前累積出的區間比較，不使用第 j 天當天或未來的
        資料回頭定義區間邊界，符合本引擎一貫的因果安全原則。

        欄位輸出：
          flag_bull_confirmed / flag_bear_confirmed : bool，帶量突破確認當天
          flag_target_price                          : float，旗桿等幅測量目標價
          flag_note                                    : str，人類可讀說明
        """
        df = df.copy()
        n = len(df)
        df['flag_bull_confirmed'] = False
        df['flag_bear_confirmed'] = False
        df['flag_target_price'] = np.nan
        df['flag_note'] = ""

        required = ['close', 'high', 'low']
        if n < pole_window + min_consolidation + 2 or any(c not in df.columns for c in required):
            return df

        close = df['close'].to_numpy(dtype=float)
        high = df['high'].to_numpy(dtype=float)
        low = df['low'].to_numpy(dtype=float)
        rvol = df['rvol'].to_numpy(dtype=float) if 'rvol' in df.columns else np.ones(n)

        bull_flags = np.zeros(n, dtype=bool)
        bear_flags = np.zeros(n, dtype=bool)
        target_arr = np.full(n, np.nan)
        notes = [""] * n
        used_confirm = set()

        for i in range(pole_window, n):
            pole_start = i - pole_window
            pct = (close[i] - close[pole_start]) / (close[pole_start] + 1e-9)
            avg_rvol_pole = rvol[pole_start + 1:i + 1].mean() if i > pole_start else rvol[i]

            # ---- 多頭旗形候選：強力上漲旗桿 ----
            if pct >= pole_min_pct and avg_rvol_pole >= pole_min_rvol:
                pole_height = close[i] - close[pole_start]
                invalidate_level = close[i] - pole_height * max_retrace_pct
                consolidation_high = close[i]
                consolidation_rvols = []

                for j in range(i + 1, min(i + 1 + max_consolidation, n)):
                    # 型態失敗判定：跌破最大回檔容忍度，中止此候選旗桿
                    if low[j] < invalidate_level:
                        break
                    days_in = j - i
                    if days_in >= min_consolidation and j not in used_confirm:
                        avg_cons_rvol = np.mean(consolidation_rvols) if consolidation_rvols else rvol[j]
                        if (close[j] > consolidation_high * (1 + breakout_buffer)
                                and rvol[j] >= breakout_min_rvol
                                and avg_cons_rvol < consolidation_vol_ratio * avg_rvol_pole):
                            bull_flags[j] = True
                            target_arr[j] = close[j] + pole_height
                            notes[j] = (
                                f"🚀 多頭旗形確認：旗桿漲幅 {pct * 100:.1f}%，經 {days_in} 天量縮整理後"
                                f"帶量突破 {consolidation_high:.2f}，目標價（旗桿等幅）約 {target_arr[j]:.2f}"
                            )
                            used_confirm.add(j)
                            break
                    consolidation_rvols.append(rvol[j])
                    consolidation_high = max(consolidation_high, close[j])

            # ---- 空頭旗形候選：強力下跌旗桿 ----
            if pct <= -pole_min_pct and avg_rvol_pole >= pole_min_rvol:
                pole_height = close[pole_start] - close[i]
                invalidate_level = close[i] + pole_height * max_retrace_pct
                consolidation_low = close[i]
                consolidation_rvols = []

                for j in range(i + 1, min(i + 1 + max_consolidation, n)):
                    if high[j] > invalidate_level:
                        break
                    days_in = j - i
                    if days_in >= min_consolidation and j not in used_confirm:
                        avg_cons_rvol = np.mean(consolidation_rvols) if consolidation_rvols else rvol[j]
                        if (close[j] < consolidation_low * (1 - breakout_buffer)
                                and rvol[j] >= breakout_min_rvol
                                and avg_cons_rvol < consolidation_vol_ratio * avg_rvol_pole):
                            bear_flags[j] = True
                            target_arr[j] = close[j] - pole_height
                            notes[j] = (
                                f"🔻 空頭旗形確認：旗桿跌幅 {abs(pct) * 100:.1f}%，經 {days_in} 天量縮整理後"
                                f"帶量跌破 {consolidation_low:.2f}，目標價（旗桿等幅）約 {target_arr[j]:.2f}"
                            )
                            used_confirm.add(j)
                            break
                    consolidation_rvols.append(rvol[j])
                    consolidation_low = min(consolidation_low, close[j])

        df['flag_bull_confirmed'] = bull_flags
        df['flag_bear_confirmed'] = bear_flags
        df['flag_target_price'] = target_arr
        df['flag_note'] = notes
        return df

    # ==========================================
    # 5. 一次執行（供 pipeline 呼叫；目前尚未接入 ScannerEngine，見類別docstring）
    # ==========================================
    @staticmethod
    def add_patterns(df: pd.DataFrame, deviation: float = 0.04,
                      exhaustion_window: int = 3, confirm_window: int = 5):
        """一次執行頭肩型態、缺口分類、M頭/W底、旗形/三角旗形偵測，
        回傳補齊全部欄位後的 df。"""
        df = PatternEngine.detect_head_shoulders(df, deviation=deviation)
        df = PatternEngine.classify_gaps(df, exhaustion_window=exhaustion_window,
                                          confirm_window=confirm_window)
        df = PatternEngine.detect_double_top_bottom(df, deviation=deviation)
        df = PatternEngine.detect_flag_pennant(df)
        return df