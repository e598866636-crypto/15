import numpy as np
import pandas as pd


class MomentumEngine:
    """
    🚀 飆股動能引擎 (Momentum / Hot-Stock Engine) - TQAI Pro v2.9.6

    「飆股九層過濾 + 100分評分系統（含年線濾網 + 跨股票相對強度）」

    設計動機：
    StrategyEngine.ai_score 回答的是「當下多空優劣勢＋風控之後，這檔股票值不
    值得操作」，是給一般波段決策用的主審裁決。MomentumEngine 回答的是更窄、
    更嚴格的問題 ——「這檔股票現在的技術結構，像不像一檔正要噴出的飆股？」，
    門檻設計得比 ai_score 更嚴苛，並且明確把「假突破／背離失效」的懲罰算進
    總分，而不是像 ai_score 那樣做連續型風控懲罰。

    ⚠️ v2.9.6 重大修正（把 RS Rank / 相對成交量真正併入評分公式，而非只是
    並列指標顯示）：
    v2.9.5 之前，這個引擎完全只用「股票自己的歷史資料」評分——包括第⑤層
    的 RSI「相對強度」，其實是跟自己過去比，不是跟其他股票比。Minervini/
    O'Neil 系統一致強調 RS Rating（跨股票排名）是單一最重要的篩選因子，
    先前版本雖然新增了獨立的 RSRatingEngine，但只是在 UI 上「並列顯示」，
    沒有真正影響 momentum_score 這個核心評分數字——這代表兩檔 RS Rating
    分別是 30 分跟 90 分的股票，如果其他層數據相似，飆股評分可能一樣高，
    這在方法論上是不一致的。

    現在的九層過濾（100分配分，權重已重新分配，非簡單新增）：
      1. 年線多頭濾網 (15分，硬性關卡)：不變。
      2. 均線多頭排列 (12分，原15分)：權重下修，讓出空間給跨股票層。
      3. 價量齊揚-自身歷史 (8分，原15分)：RVOL 用自己過去的量能基準，
         保留但降權——跨股票的量能排名（第⑧層）是更嚴謹的版本。
      4. MACD 動能強度 (12分，原15分)：權重下修。
      5. 相對強度且未過熱 (8分，原10分)：RSI，權重下修，並更名強調這是
         「自身歷史」相對強度，跟第⑦層的跨股票 RS Rank 明確區分。
      6. 籌碼慣性代理指標 (5分，原10分)：OBV代理，權重下修。
      7. 【新增】RS Rank 跨股票相對強度 (15分)：直接採用 RSRatingEngine
         算出的跨股票百分位排名。⚠️ 這一層需要外部傳入 rs_rating 參數
         （由 ScannerEngine 完成全市場掃描、算出跨股票排名後才能得知），
         單獨對一檔股票呼叫 add_momentum_score() 而不傳入 rs_rating 時，
         這一層無法計分（見下方「不完整評分」說明）。
      8. 【新增】相對成交量排名 (10分)：跟第⑦層一樣，需要外部傳入
         relative_volume_percentile（該股當前 RVOL 在整個掃描池裡的
         百分位排名），衡量「這波量能擴張，在同一批股票裡算不算突出」，
         跟第③層「自己歷史量能」是互補但不同的兩個問題。
      9. 誘多/背離防禦 (15分，原20分，來自 DivergenceEngine)：權重小幅
         下修以平衡新增的兩層，但仍是所有層級中權重最高的單一防禦機制，
         反映「假訊號防禦」優先於「多加一個買進理由」的設計哲學。

    ⚠️「不完整評分」的誠實揭露（非常重要，請務必理解這個限制）：
    第⑦、⑧兩層需要「跨股票排名」，這種排名只有在同一批股票被一起掃描
    過後才存在，對單一股票歷史上每一天分別計算「當天在全市場的排名」
    需要每天的全市場快照資料，本專案沒有這個資料（yfinance 只給得到
    目前的股價歷史，不會告訴你三個月前『那一天』全市場的相對排名）。
    因此：
      - 若呼叫時沒有提供 rs_rating / relative_volume_percentile，這兩層
        一律計 0 分（不是用「猜測的中性分數」蒙混），代表「這是還沒
        納入跨股票比較的不完整評分」，回傳的 df 會有一個新欄位
        `momentum_score_complete`（bool）明確標示這件事，避免使用者誤把
        不完整評分當完整評分使用。
      - 這代表在沒有 rs_rating 的情況下，理論最高分只有 85 分（100 -
        15 - 10 的兩層跨股票分數），也就是「不提供跨股票排名的評分，
        結構上就拿不到 A 級（>=85）的滿分區間，只能拿到 B 級上緣」——
        這是刻意的設計，不是 bug：沒有相對強度確認的飆股，本來就不該被
        評為最高等級，這跟 Minervini 系統「RS Rating 不過標準直接淘汰」
        的精神一致。
      - 只有歷史上「最新一筆」這一天，才有意義去補上這兩層分數（因為
        跨股票排名只在「當下這次掃描」有效，不是歷史每一天都有）；
        呼叫端（ScannerEngine / app.py 個股頁面）取得 rs_rating 後，
        會針對最新一筆重新呼叫本函式一次，用完整資訊覆蓋掉先前的
        「不完整評分」，兩次呼叫用的是同一套公式，純粹差在有沒有
        第⑦⑧層的輸入值，不是兩套邏輯。

    ⚠️ 與 ai_score 的關係：
    momentum_score 刻意「不」跟 ai_score 加權平均在一起，避免重蹈
    strategy_engine.py 修正說明中提到的雙重計分問題（同一組原始事實被
    不同 Agent 各自算一次分、再疊加放大）。Dashboard 上兩者並列顯示，
    讓使用者自行比對「AI主審怎麼看」跟「這檔像不像飆股」兩個不同角度。

    ⚠️ 資料相依性與容錯：
    使用前應先跑過 IndicatorEngine（均線/RSI/MACD/RVOL/OBV）與
    DivergenceEngine.add_defense_signals（背離、誘多/誘空防禦旗標）。
    若相依欄位缺漏，該層一律以「中性、不加分」處理並繼續往下算，不會
    拋出例外中斷整條 pipeline（沿用本專案一貫的防禦性設計風格）。
    """

    _DIVERGENCE_LOOKBACK = 10  # 檢查最近幾天內是否有背離/誘多誘空訊號

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(default, index=df.index)

    @staticmethod
    def _score_rs_rank(rs_rating) -> float:
        """RS Rank 分層計分（15分滿分），門檻與 CanslimEngine._score_l 一致，
        避免同一個 RS Rating 數字在兩個引擎裡被翻譯成不一致的評分標準。"""
        if rs_rating is None or pd.isna(rs_rating):
            return 0.0
        rs_rating = float(rs_rating)
        if rs_rating >= 90:
            return 15.0
        if rs_rating >= 80:
            return 12.0
        if rs_rating >= 70:
            return 7.0
        if rs_rating >= 50:
            return 3.0
        return 0.0

    @staticmethod
    def _score_relative_volume(percentile) -> float:
        """相對成交量排名分層計分（10分滿分）。percentile 為 0~100。"""
        if percentile is None or pd.isna(percentile):
            return 0.0
        percentile = float(percentile)
        if percentile >= 90:
            return 10.0
        if percentile >= 75:
            return 7.0
        if percentile >= 50:
            return 3.0
        return 0.0

    @staticmethod
    def add_momentum_score(df: pd.DataFrame, rs_rating=None, relative_volume_percentile=None):
        """
        參數：
            rs_rating                     該股票在本次掃描池裡的 RS Rating
                                          (1~99，來自 RSRatingEngine)，僅
                                          套用在「最新一筆」，None 代表尚未
                                          取得跨股票排名（第⑦層計0分）。
            relative_volume_percentile     該股票當前 RVOL 在掃描池裡的百分位
                                          排名 (0~100)，僅套用在「最新一筆」，
                                          None 代表尚未取得（第⑧層計0分）。
        """
        df = df.copy()
        n = len(df)
        if n == 0 or 'close' not in df.columns:
            df['momentum_score'] = pd.Series(dtype=float)
            df['momentum_grade'] = pd.Series(dtype=object)
            df['is_a_grade_candidate'] = pd.Series(dtype=bool)
            df['momentum_penalty_alert'] = pd.Series(dtype=bool)
            df['reversal_watch'] = pd.Series(dtype=bool)
            df['trap_alert'] = pd.Series(dtype=bool)
            df['momentum_score_complete'] = pd.Series(dtype=bool)
            return df

        close = df['close']
        sma20 = MomentumEngine._col(df, 'sma_20')
        sma60 = MomentumEngine._col(df, 'sma_60')
        sma120 = MomentumEngine._col(df, 'sma_120')
        sma200 = MomentumEngine._col(df, 'sma_200')
        rvol = MomentumEngine._col(df, 'rvol', 1.0).fillna(1.0)
        macd_hist = MomentumEngine._col(df, 'macd_hist', 0.0).fillna(0.0)
        rsi = MomentumEngine._col(df, 'rsi_14', 50.0).fillna(50.0)
        obv = MomentumEngine._col(df, 'obv', 0.0)
        obv_sma = MomentumEngine._col(df, 'obv_sma', 0.0)

        # ---- 第①層：年線多頭濾網（硬性關卡，15分，不變）----
        sma200_slope = sma200.diff(10)
        yearline_pass = (close > sma200) & (sma200_slope >= 0)
        yearline_above_only = (close > sma200) & ~yearline_pass
        yearline_score = np.select([yearline_pass, yearline_above_only], [15, 7], default=0)

        # ---- 第②層：均線多頭排列（12分，原15分）----
        alignment_full = (close > sma20) & (sma20 > sma60) & (sma60 > sma120)
        alignment_partial = (close > sma20) & (sma20 > sma60) & ~alignment_full
        alignment_score = np.select([alignment_full, alignment_partial], [12, 6], default=0)

        # ---- 第③層：價量齊揚-自身歷史（8分，原15分）----
        price_up = close > close.shift(1)
        volume_score = np.select(
            [(rvol >= 2.0) & price_up, (rvol >= 1.3) & price_up],
            [8, 4], default=0
        )

        # ---- 第④層：MACD動能強度（12分，原15分）----
        macd_growing = macd_hist > macd_hist.shift(1)
        macd_score = np.select(
            [(macd_hist > 0) & macd_growing, macd_hist > 0],
            [12, 6], default=0
        )

        # ---- 第⑤層：相對強度(自身歷史)且未過熱（8分，原10分）----
        rsi_sweet_spot = (rsi >= 55) & (rsi <= 80)
        rsi_overheated = rsi > 80
        rsi_score = np.select([rsi_sweet_spot, rsi_overheated], [8, 2], default=0)

        # ---- 第⑥層：籌碼慣性代理指標（OBV，5分，原10分）----
        obv_score = np.where(obv > obv_sma, 5, 0)

        # ---- 第⑦層：RS Rank 跨股票相對強度（15分，NEW，只套用在最新一筆）----
        rs_rank_score = np.zeros(n)
        rs_rank_score[-1] = MomentumEngine._score_rs_rank(rs_rating)

        # ---- 第⑧層：相對成交量排名（10分，NEW，只套用在最新一筆）----
        relvol_score = np.zeros(n)
        relvol_score[-1] = MomentumEngine._score_relative_volume(relative_volume_percentile)

        raw_score = (yearline_score + alignment_score + volume_score + macd_score
                     + rsi_score + obv_score + rs_rank_score + relvol_score)

        # ---- 第⑨層：誘多/背離防禦（15分，原20分）----
        bearish_div_recent = MomentumEngine._col(df, 'bearish_divergence', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bull_trap_recent = MomentumEngine._col(df, 'bull_trap_confirmed', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bullish_div_recent = MomentumEngine._col(df, 'bullish_divergence', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)
        bear_trap_recent = MomentumEngine._col(df, 'bear_trap_confirmed', False).fillna(False).astype(bool) \
            .rolling(MomentumEngine._DIVERGENCE_LOOKBACK, min_periods=1).max().astype(bool)

        defense_score = np.select(
            [bull_trap_recent, bearish_div_recent],
            [0, 6],
            default=15
        )

        total_score = np.clip(raw_score + defense_score, 0, 100)
        momentum_score = pd.Series(total_score, index=df.index).astype(float)

        # 年線未過關（yearline_pass 為 False）屬於結構性缺陷：即使其他層分數
        # 很高，也不應該被歸類為「飆股」，因此對總分做一次硬性封頂修正，
        # 確保「跌破年線或年線走弱」的標的永遠評不到 A 級（>=85）。
        momentum_score = momentum_score.where(yearline_pass, np.minimum(momentum_score, 65))

        df['momentum_score'] = momentum_score
        df['momentum_grade'] = np.select(
            [momentum_score >= 85, momentum_score >= 70, momentum_score >= 55],
            ["A", "B", "C"], default="D"
        )
        df['is_a_grade_candidate'] = df['momentum_grade'] == "A"

        # ⚠️ v2.9.6 新增：明確標示最新一筆的評分是否納入了跨股票排名層
        # （第⑦⑧層）。False 代表這是「不完整評分」，理論上限只有85分，
        # 不該被拿來跟「完整評分」的其他股票直接比較。
        complete_flag = pd.Series(False, index=df.index)
        complete_flag.iloc[-1] = (rs_rating is not None) and (relative_volume_percentile is not None)
        df['momentum_score_complete'] = complete_flag

        # ⚠️ 修正說明：原本 trap_alert 把「會扣分的誘多警報」(bull_trap_recent /
        # bearish_div_recent) 跟「不影響本評分的底部反轉訊號」(bullish_div_recent /
        # bear_trap_recent，這兩個是看漲反轉訊號，對『飆股動能』評分而言是中性/
        # 甚至偏正面的資訊，defense_score 完全沒有把它們算進扣分) 全部 OR 在一起，
        # 導致只要單純出現底背離或誘空反轉這種「跟這次扣分無關」的訊號，
        # trap_alert 也會被設成 True。這樣一來：
        #   (a) get_momentum_breakdown 的「⑨ 誘多/背離防禦」會顯示「未通過」，
        #       但 defense_score 實際上是滿分，兩者互相矛盾；
        #   (b) app.py 顯示的「本層評分已扣分」文字，在這種情況下是錯的
        #       （分數根本沒被扣）。
        # 現在拆成兩個獨立欄位：
        #   momentum_penalty_alert：只反映「真的造成本層扣分」的訊號，
        #       用這個欄位判斷「⑨ 誘多/背離防禦」是否通過、以及是否該顯示
        #       「評分已扣分」。
        #   reversal_watch：純粹資訊性質的底部反轉觀察旗標（底背離／誘空
        #       確認），不影響飆股評分，但保留給想額外關注反轉訊號的使用者。
        #   trap_alert：維持原本「廣義警報雷達」用途（掃描戰情室的誘盤警報
        #       雷達本來就設計成不分方向、任何背離/假突破都想顯示），語意
        #       改為「近期市場結構出現值得留意的背離/假突破事件」，不再
        #       暗示「一定會扣分」。
        df['momentum_penalty_alert'] = bull_trap_recent | bearish_div_recent
        df['reversal_watch'] = bullish_div_recent | bear_trap_recent
        df['trap_alert'] = df['momentum_penalty_alert'] | df['reversal_watch']

        return df

    # ==========================================
    # 取得單一時間點（預設最新一筆）的七層評分明細
    # ==========================================
    @staticmethod
    def get_momentum_breakdown(df: pd.DataFrame, idx: int = -1, rs_rating=None, relative_volume_percentile=None):
        """回傳最新一筆的九層評分明細，供 Dashboard 顯示清單/雷達。
        rs_rating / relative_volume_percentile 若不提供，第⑦⑧層會顯示
        「尚未計算（需先跑全台股掃描或個股頁面才有跨股票排名）」。"""
        if df is None or df.empty:
            return []

        row = df.iloc[idx]

        def gf(col, default=np.nan):
            try:
                val = row[col] if col in row.index else default
                return float(val)
            except (TypeError, ValueError):
                return default

        close = gf('close')
        sma200 = gf('sma_200')
        sma20, sma60, sma120 = gf('sma_20'), gf('sma_60'), gf('sma_120')
        rvol = gf('rvol')
        macd_hist = gf('macd_hist')
        rsi = gf('rsi_14')
        obv, obv_sma = gf('obv'), gf('obv_sma')

        yearline_ok = pd.notna(close) and pd.notna(sma200) and close > sma200
        alignment_ok = (pd.notna(close) and pd.notna(sma20) and pd.notna(sma60) and pd.notna(sma120)
                        and close > sma20 > sma60 > sma120)
        obv_ok = pd.notna(obv) and pd.notna(obv_sma) and obv > obv_sma

        # ⚠️ 修正：第⑨層的「通過與否」改用 momentum_penalty_alert（只反映真的
        # 會扣分的誘多假突破／頂背離），不再用籠統的 trap_alert（那個還包含
        # 不影響本層分數的底部反轉觀察訊號），避免顯示「未通過」但實際上
        # defense_score 是滿分的矛盾。
        penalty_active = bool(row.get('momentum_penalty_alert', row.get('trap_alert', False)))
        reversal_watch_active = bool(row.get('reversal_watch', False))
        note = row.get('trap_note', '') if 'trap_note' in row.index else ''
        if not note and penalty_active:
            note = row.get('divergence_note', '') if 'divergence_note' in row.index else ''
        if not note and penalty_active:
            # trap_alert 是「近 N 天內曾觸發」的滾動旗標，觸發當天才會寫入
            # trap_note/divergence_note 文字，往後幾天旗標仍為 True 但當天
            # 沒有新事件文字，這裡補一句通用說明，避免顯示「近期無警報」
            # 卻同時判定本層未過關的矛盾訊息。
            note = f"近期（{MomentumEngine._DIVERGENCE_LOOKBACK}天內）曾觸發誘多假突破/頂背離警報，本層評分已扣分"
        elif not note and reversal_watch_active:
            note = f"近期（{MomentumEngine._DIVERGENCE_LOOKBACK}天內）出現底背離/誘空反轉觀察訊號，不影響本層分數，僅供留意"
        elif not note:
            note = "近期無誘多/背離警報"

        rs_available = rs_rating is not None and pd.notna(rs_rating)
        relvol_available = relative_volume_percentile is not None and pd.notna(relative_volume_percentile)

        breakdown = [
            {"layer": "① 年線多頭濾網", "passed": bool(yearline_ok),
             "detail": f"收盤 {close:.2f} vs 年線 {sma200:.2f}" if pd.notna(close) and pd.notna(sma200) else "資料不足"},
            {"layer": "② 均線多頭排列", "passed": bool(alignment_ok),
             "detail": "close > 20MA > 60MA > 120MA" if alignment_ok else "均線排列尚未完全多頭"},
            {"layer": "③ 價量齊揚(自身歷史)", "passed": pd.notna(rvol) and rvol >= 1.3,
             "detail": f"RVOL={rvol:.2f}" if pd.notna(rvol) else "資料不足"},
            {"layer": "④ MACD動能強度", "passed": pd.notna(macd_hist) and macd_hist > 0,
             "detail": f"柱狀體={macd_hist:.3f}" if pd.notna(macd_hist) else "資料不足"},
            {"layer": "⑤ 相對強度(自身歷史)未過熱", "passed": pd.notna(rsi) and 55 <= rsi <= 80,
             "detail": f"RSI={rsi:.1f}" if pd.notna(rsi) else "資料不足"},
            {"layer": "⑥ 籌碼慣性(OBV代理)", "passed": bool(obv_ok),
             "detail": "OBV在均線之上" if obv_ok else "OBV在均線之下或資料不足"},
            {"layer": "⑦ RS Rank(跨股票相對強度)", "passed": rs_available and rs_rating >= 70,
             "detail": f"RS Rating={rs_rating:.0f}/99" if rs_available else "⚠️ 尚未計算（需先跑全台股掃描或個股頁面才有跨股票排名）"},
            {"layer": "⑧ 相對成交量排名(跨股票)", "passed": relvol_available and relative_volume_percentile >= 75,
             "detail": f"百分位={relative_volume_percentile:.0f}%" if relvol_available else "⚠️ 尚未計算（需先跑全台股掃描或個股頁面才有跨股票排名）"},
            {"layer": "⑨ 誘多/背離防禦", "passed": not penalty_active,
             "detail": note},
        ]
        return breakdown