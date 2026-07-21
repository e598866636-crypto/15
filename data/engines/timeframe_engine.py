import numpy as np
import pandas as pd


class TimeframeEngine:
    """
    ⏳ 多週期策略與未來走向引擎 (Timeframe & Outlook Engine) - TQAI Pro v2.7

    背景與動機：
    StrategyEngine.ai_score 是「單一時間尺度」的綜合裁決，回答的是波段
    (2週~2個月) 這個尺度上「值不值得操作」。但使用者實務上依照自己的
    操作週期（當沖/短線、波段、長線存股）需要不同的判斷角度 —— 同一檔
    股票很常出現「短線過熱要小心、但長線結構仍偏多」這種分歧狀態，如果
    只看單一 ai_score 容易誤判操作時機。

    設計原則：
    - 本引擎完全「不重新計算」任何技術指標，只讀取 IndicatorEngine /
      StrategyEngine / DivergenceEngine 已經算好的欄位做二次判斷與整合，
      避免又冒出一套獨立、可能互相矛盾的指標定義。
    - 三個週期分別對應不同的既有欄位組合：
        短線 (約1~5個交易日)：EMA8 / RSI / KD / MACD 這類反應快的短週期
            指標，並納入近5日內的誘多/背離防禦訊號。
        波段 (約2週~2個月)：直接沿用 StrategyEngine 的 ai_score /
            market_regime，這正是 ai_score 原本設計要回答的問題，並額外
            納入近10日的防禦訊號滾動視窗（比短線視窗更寬）。
        長線 (半年以上)：偏重 sma_120 / sma_200 年線結構與其斜率，不受
            短期雜訊影響。
    - 三週期結果會再彙整成一個「未來走向 (outlook)」情境推演。

    ⚠️ 關於「未來走向 (Outlook)」的誠實揭露（務必保留此區塊）：
    這裡的 outlook 純粹是「如果目前技術結構延續，可能出現的情境」，是
    條件式的技術面推論 (if-then)，**不是**對未來價格的預測或保證，也不
    構成投資建議。技術分析無法保證未來走勢：總體經濟、產業消息、法人
    動向、公司基本面變化、國際情勢等因素都可能在任何時間點推翻目前的
    技術結構。使用者應自行判斷風險、獨立做出投資決策並自負盈虧，本引擎
    與其呼叫端 (Dashboard) 都不應把這個區塊呈現成「保證會發生的走勢」。

    ⚠️ 使用前提與容錯：
    使用前應先跑過 IndicatorEngine + StrategyEngine（提供 ai_score /
    market_regime 等欄位）；若同時跑過 DivergenceEngine，短線/波段的
    風險判斷會更準確（會納入誘多/背離防禦訊號），沒有的話則以中性、
    不加分/不扣分處理，不會拋出例外中斷呼叫端。

    ⚠️ 進場價位／出場價位／時機（v2.9 新增，務必先讀完這段再使用）：
    每個週期新增 trade_plan，提供「進場參考價」「出場參考價（停利/停損）」
    與「觸發條件」。這裡刻意不提供「日曆時間」的進出場時機——沒有任何
    技術分析方法能誠實預測「未來某個具體日期」股價會到哪裡，凡是宣稱
    能精準預測日期的工具都值得高度懷疑。這裡的「時機」指的是「進場/
    出場的技術面觸發條件」（例如「站上OO且爆量」「跌破OO」），價格何時
    觸及這些條件取決於市場本身的節奏，本引擎不猜測、也不宣稱知道。
    所有價位都只是「依目前技術結構與波動率(ATR)推算出的參考價位」，不是
    保證會被觸及的價位，市場可能直接跳空穿越、或永遠不回測到這個價位，
    不構成投資建議。波段週期的價位直接重用 StrategyEngine 已經算好的
    stop_loss/target_1/target_2/entry_signal/exit_signal，不重新定義一套
    獨立公式，避免同一件事被兩個地方各自算一次、產生互相矛盾的數字。
    """

    _SHORT_LOOKBACK = 5    # 短線防禦訊號視窗（交易日）
    _SWING_LOOKBACK = 10   # 波段防禦訊號視窗（交易日）
    _LONG_SLOPE_WINDOW = 20  # 長線年線斜率觀察視窗（交易日）

    # ==========================================
    # 工具方法
    # ==========================================
    @staticmethod
    def _get(row, col, default=np.nan):
        try:
            if col not in row.index:
                return default
            val = row[col]
            if pd.isna(val):
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _recent_flag(history: pd.DataFrame, col: str, lookback: int) -> bool:
        """檢查歷史資料近 lookback 天內，某個布林欄位是否曾經為 True。
        欄位不存在時視為「無資料、不觸發」，維持防禦性設計風格。"""
        if col not in history.columns or history.empty:
            return False
        window = history[col].tail(lookback)
        return bool(window.fillna(False).astype(bool).any())

    # ==========================================
    # 1. 短線判斷 (約1~5個交易日)
    # ==========================================
    @staticmethod
    def _assess_short_term(history: pd.DataFrame, row: pd.Series) -> dict:
        close = TimeframeEngine._get(row, 'close')
        ema8 = TimeframeEngine._get(row, 'ema_8')
        rsi = TimeframeEngine._get(row, 'rsi_14')
        k9, d9 = TimeframeEngine._get(row, 'k_9'), TimeframeEngine._get(row, 'd_9')
        macd_hist = TimeframeEngine._get(row, 'macd_hist')

        bull_pts, bear_pts, notes = 0, 0, []

        if pd.notna(close) and pd.notna(ema8):
            if close > ema8:
                bull_pts += 1
                notes.append("站上EMA8短均線")
            else:
                bear_pts += 1
                notes.append("跌破EMA8短均線")

        if pd.notna(macd_hist):
            if macd_hist > 0:
                bull_pts += 1
                notes.append("MACD柱狀體翻正")
            else:
                bear_pts += 1
                notes.append("MACD柱狀體翻負")

        if pd.notna(k9) and pd.notna(d9):
            if k9 > d9:
                bull_pts += 1
                notes.append("KD呈多頭排列")
            else:
                bear_pts += 1
                notes.append("KD呈空頭排列")

        if pd.notna(rsi):
            if rsi >= 80:
                bear_pts += 1
                notes.append(f"RSI={rsi:.0f}過熱，短線拉回風險升高")
            elif rsi <= 20:
                bull_pts += 1
                notes.append(f"RSI={rsi:.0f}超賣，短線可能出現反彈")

        defense_triggered = (
            TimeframeEngine._recent_flag(history, 'bull_trap_confirmed', TimeframeEngine._SHORT_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bearish_divergence', TimeframeEngine._SHORT_LOOKBACK)
        )
        if defense_triggered:
            bear_pts += 1
            notes.append(f"近{TimeframeEngine._SHORT_LOOKBACK}日內出現誘多假突破/頂背離警報")

        if bull_pts - bear_pts >= 2:
            view = "偏多"
        elif bear_pts - bull_pts >= 2:
            view = "偏空"
        else:
            view = "中性/觀望"

        action_map = {
            "偏多": "短線動能偏多，已進場者可續抱並嚴設短線停利/停損，未進場者不建議追高，可等待回測支撐再評估",
            "偏空": "短線動能偏弱，不建議短線搶反彈，留意是否持續跌破近期支撐",
            "中性/觀望": "短線多空訊號不明確，建議觀望，避免頻繁進出增加交易成本",
        }

        return {
            "view": view,
            "horizon": f"約1~{TimeframeEngine._SHORT_LOOKBACK}個交易日",
            "reason": "、".join(notes) if notes else "資料不足，暫無法判斷短線動能",
            "action": action_map[view],
        }

    # ==========================================
    # 2. 波段判斷 (約2週~2個月) — 沿用 ai_score / market_regime
    # ==========================================
    @staticmethod
    def _assess_swing(history: pd.DataFrame, row: pd.Series) -> dict:
        ai_score = TimeframeEngine._get(row, 'ai_score', 50.0)
        regime = row['market_regime'] if 'market_regime' in row.index else 'N/A'
        action_guide = row['action_guide'] if 'action_guide' in row.index else ''

        defense_bear = (
            TimeframeEngine._recent_flag(history, 'bull_trap_confirmed', TimeframeEngine._SWING_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bearish_divergence', TimeframeEngine._SWING_LOOKBACK)
        )
        reversal_watch = (
            TimeframeEngine._recent_flag(history, 'bear_trap_confirmed', TimeframeEngine._SWING_LOOKBACK)
            or TimeframeEngine._recent_flag(history, 'bullish_divergence', TimeframeEngine._SWING_LOOKBACK)
        )

        if ai_score >= 70 and not defense_bear:
            view = "偏多"
        elif ai_score <= 45 or defense_bear:
            view = "偏空"
        else:
            view = "中性/觀望"

        notes = [f"AI Score={ai_score:.1f}", f"市場狀態：{regime}"]
        if defense_bear:
            notes.append(f"近{TimeframeEngine._SWING_LOOKBACK}日內有背離/誘多警報，波段風險升高")
        if reversal_watch:
            notes.append(f"近{TimeframeEngine._SWING_LOOKBACK}日內有底背離/誘空反轉觀察訊號，可留意是否醞釀反彈（僅供觀察，非確認訊號）")

        return {
            "view": view,
            "horizon": "約2週~2個月",
            "reason": "、".join(notes),
            "action": action_guide if action_guide else "請參考AI主審裁決 (ai_score)",
        }

    # ==========================================
    # 3. 長線判斷 (半年以上) — 年線與長期均線結構
    # ==========================================
    @staticmethod
    def _assess_long_term(history: pd.DataFrame, row: pd.Series) -> dict:
        close = TimeframeEngine._get(row, 'close')
        sma200 = TimeframeEngine._get(row, 'sma_200')

        if pd.isna(sma200) or 'sma_200' not in history.columns:
            return {
                "view": "資料不足",
                "horizon": "半年以上",
                "reason": "尚無足夠交易日計算年線(SMA200)，暫不評估長線結構",
                "action": "資料不足，暫不提供長線建議",
            }

        sma200_series = history['sma_200'].dropna()
        sma200_slope = np.nan
        if len(sma200_series) > TimeframeEngine._LONG_SLOPE_WINDOW:
            sma200_slope = sma200_series.iloc[-1] - sma200_series.iloc[-TimeframeEngine._LONG_SLOPE_WINDOW]

        above_year_line = pd.notna(close) and close > sma200
        year_line_rising = pd.notna(sma200_slope) and sma200_slope > 0
        year_line_falling = pd.notna(sma200_slope) and sma200_slope < 0

        if above_year_line and year_line_rising:
            view = "偏多"
            reason = "股價站上年線且年線向上，長期趨勢結構健康"
        elif above_year_line and not year_line_rising:
            view = "中性偏多(轉弱觀察)"
            reason = "股價仍在年線之上，但年線走平或斜率轉弱，留意長期動能是否減弱"
        elif (not above_year_line) and (year_line_falling or pd.isna(sma200_slope)):
            view = "偏空"
            reason = "股價跌破年線，且年線走平或向下，長期趨勢結構偏弱"
        else:
            view = "中性/觀望"
            reason = "股價與年線關係尚不明確，長期結構仍在轉換中"

        action_map = {
            "偏多": "長線結構偏多，可列入長期存股/波段布局觀察名單，仍建議分批進場以平滑成本",
            "中性偏多(轉弱觀察)": "長線結構仍偏多但動能可能減弱，不建議大幅加碼，留意是否跌破年線",
            "偏空": "長線結構偏弱，現階段不建議長線佈局，待股價站回年線且年線翻揚後再重新評估",
            "中性/觀望": "長線結構尚不明確，建議持續觀察，暫緩大幅加減碼",
        }

        return {
            "view": view,
            "horizon": "半年以上",
            "reason": reason,
            "action": action_map.get(view, "建議持續觀察"),
        }

    # ==========================================
    # 3.5 進場價位／出場價位／觸發條件（v2.9 新增）
    # ==========================================
    @staticmethod
    def _build_short_term_plan(row: pd.Series) -> dict:
        """
        短線交易計畫（約1~5個交易日）。出場用較小的ATR倍數（短線波動
        本來就抓比較窄的區間，跟波段/長線的倍數不同，避免同一套倍數
        套用到所有週期）。

        ⚠️ 修正說明：進場參考價、出場目標價、停損價統一以「現價(close)」
        為計算基準。原本進場參考價曾經用EMA8拉回價計算，但EMA8是落後
        指標，在某些情況下EMA8拉回價會低於「現價-1倍ATR」算出來的停損
        價，出現「進場參考價比停損價還低」這種自相矛盾的畫面（等於還沒
        進場就已經跌破停損）。EMA8拉回區間改成放在進場條件的文字說明裡
        當作輔助參考，不再拿來當計算基準，確保進場價／出場目標價／停損
        價三者的相對關係永遠合理（停損 < 進場參考 < 出場目標）。
        """
        close = TimeframeEngine._get(row, 'close')
        atr = TimeframeEngine._get(row, 'atr_14')
        ema8 = TimeframeEngine._get(row, 'ema_8')

        if pd.isna(close) or pd.isna(atr):
            return {"available": False, "note": "資料不足（缺收盤價或ATR），暫無法產生短線價位參考。"}

        target = close + 1.5 * atr
        stop = close - 1.0 * atr
        ema8_note = f"（若拉回EMA8約 {ema8:.2f} 附近不破，可視為加分的分批進場輔助參考）" if pd.notna(ema8) else ""

        return {
            "available": True,
            "entry_ref_price": round(float(close), 2),
            "entry_condition": f"現價 {close:.2f} 附近，動能未破壞前可分批進場，不追高{ema8_note}",
            "exit_target_price": round(float(target), 2),
            "exit_target_condition": f"觸及約 {target:.2f}（現價+1.5倍ATR）可分批獲利了結",
            "exit_stop_price": round(float(stop), 2),
            "exit_stop_condition": f"跌破約 {stop:.2f}（現價-1.0倍ATR）則停損離場，不心存僥倖",
            "timing_note": "以上為技術面觸發條件，不是日曆日期；價格何時觸及取決於市場節奏，本引擎不做日期預測。",
        }

    @staticmethod
    def _build_swing_plan(row: pd.Series) -> dict:
        """
        波段交易計畫（約2週~2個月）。直接重用 StrategyEngine 已經算好的
        stop_loss / target_1 / target_2（2.0x/2.5x/5.0x ATR）與
        entry_signal / exit_signal，不重新定義一套獨立公式。
        """
        close = TimeframeEngine._get(row, 'close')
        stop_loss = TimeframeEngine._get(row, 'stop_loss')
        target_1 = TimeframeEngine._get(row, 'target_1')
        target_2 = TimeframeEngine._get(row, 'target_2')
        entry_signal = row['entry_signal'] if 'entry_signal' in row.index else None
        exit_signal = row['exit_signal'] if 'exit_signal' in row.index else None

        if pd.isna(close) or pd.isna(stop_loss) or pd.isna(target_1):
            return {"available": False, "note": "資料不足（缺 StrategyEngine 停損/停利欄位），暫無法產生波段價位參考。"}

        return {
            "available": True,
            "entry_ref_price": round(float(close), 2),
            "entry_condition": entry_signal or "參考 StrategyEngine 的 entry_signal 欄位（AI Score達70分以上且近期無背離/誘多警報）",
            "exit_target_price": round(float(target_1), 2),
            "exit_target_price_extended": None if pd.isna(target_2) else round(float(target_2), 2),
            "exit_target_condition": f"第一目標約 {target_1:.2f}（2.5倍ATR）可分批獲利了結"
                + (f"，若動能延續可續抱看第二目標約 {target_2:.2f}（5倍ATR）" if pd.notna(target_2) else ""),
            "exit_stop_price": round(float(stop_loss), 2),
            "exit_stop_condition": exit_signal or f"跌破約 {stop_loss:.2f}（2倍ATR停損）則停損離場",
            "timing_note": "以上為技術面觸發條件，不是日曆日期；價格何時觸及取決於市場節奏，本引擎不做日期預測。",
        }

    @staticmethod
    def _build_long_term_plan(row: pd.Series) -> dict:
        """
        長線交易計畫（半年以上）。進場參考拉回季線/年線的區間，出場改用
        「結構破壞條件」而非單純固定價位——長線操作更看重「趨勢結構是否
        還在」，用比短線/波段更寬的ATR倍數當價位參考僅供輔助，主要出場
        依據仍是年線結構是否破壞。
        """
        close = TimeframeEngine._get(row, 'close')
        atr = TimeframeEngine._get(row, 'atr_14')
        sma60 = TimeframeEngine._get(row, 'sma_60')
        sma200 = TimeframeEngine._get(row, 'sma_200')

        if pd.isna(close) or pd.isna(sma200):
            return {"available": False, "note": "資料不足（缺年線SMA200），暫無法產生長線價位參考。"}

        entry_pullback = sma60 if pd.notna(sma60) else sma200
        target = close + 8.0 * atr if pd.notna(atr) else None

        return {
            "available": True,
            "entry_ref_price": round(float(entry_pullback), 2),
            "entry_condition": (
                f"股價站上年線({sma200:.2f})的前提下，拉回季線約 {entry_pullback:.2f} 附近可分批布局，"
                f"建議分批而非單筆重壓以平滑成本"
            ),
            "exit_target_price": None if target is None else round(float(target), 2),
            "exit_target_condition": (
                f"長線操作建議以「結構是否還在」判斷去留，而非單純固定停利價；"
                + (f"若仍要有價位參考，約 {target:.2f}（現價+8倍ATR，長期粗略估算，準確度低於短線/波段）可作為分批獲利了結的參考。" if target else "")
            ),
            "exit_stop_price": round(float(sma200), 2),
            "exit_stop_condition": f"收盤跌破年線({sma200:.2f})且年線同時走平或向下，視為長線結構轉弱，應重新評估是否出場",
            "timing_note": "以上為技術面觸發條件，不是日曆日期；價格何時觸及取決於市場節奏，本引擎不做日期預測。",
        }

    # ==========================================
    # 3.6 短線進出場公式的簡化版歷史命中率檢查（v2.9.1 新增）
    # ==========================================
    @staticmethod
    def backtest_short_term_hit_rate(df: pd.DataFrame, lookback_horizon: int = 5, sample_window: int = 252) -> dict:
        """
        ⚠️ 這是簡化版的歷史命中率統計，**不是**跟 BacktestEngine 同等嚴謹的
        回測，目的是讓使用者知道「短線進出場公式」（現價±1.5x/1.0x ATR）
        過去在這檔股票的歷史資料上大致的命中率，幫助判斷這組固定倍數是否
        合理，不是保證未來表現、不構成投資建議。

        ⚠️ 簡化之處（務必先讀完再解讀數字）：
          1. 用「當天」的收盤價/ATR當進場基準，不像 BacktestEngine 那樣延遲
             到隔日開盤才成交——刻意簡化以降低計算複雜度，因此會比實際
             交易情境更樂觀，兩者的勝率數字不能直接比較。
          2. 完全沒有計算手續費、證交稅。
          3. 用 High/Low 判斷是否觸及目標/停損，若同一天內兩者理論上都
             可能被觸及，日線資料無法得知真實的觸價順序，這裡保守地優先
             判定為「停損觸發」（寧可低估勝率，不要高估）。
          4. 預設樣本視窗約252個交易日（近1年），不是嚴謹的統計顯著性
             檢定，只是一個粗略的參考基準。
        """
        if df is None or df.empty or len(df) < 30:
            return {"status": "unavailable", "message": "歷史資料不足（需要至少30個交易日），無法計算命中率。"}

        required = {"close", "high", "low", "atr_14"}
        if not required.issubset(df.columns):
            return {"status": "unavailable", "message": "缺少必要欄位(close/high/low/atr_14)，無法計算命中率。"}

        sub = df.tail(sample_window + lookback_horizon).reset_index(drop=True)
        n = len(sub)

        target_hit, stop_hit, neither, total = 0, 0, 0, 0

        for i in range(max(0, n - lookback_horizon)):
            close = sub["close"].iloc[i]
            atr = sub["atr_14"].iloc[i]
            if pd.isna(close) or pd.isna(atr) or atr <= 0:
                continue

            target = close + 1.5 * atr
            stop = close - 1.0 * atr

            future = sub.iloc[i + 1: i + 1 + lookback_horizon]
            if future.empty:
                continue

            hit_stop = bool((future["low"] <= stop).any())
            hit_target = bool((future["high"] >= target).any())

            total += 1
            if hit_stop:
                stop_hit += 1
            elif hit_target:
                target_hit += 1
            else:
                neither += 1

        if total == 0:
            return {"status": "unavailable", "message": "有效樣本數為0（可能是ATR欄位缺值過多），無法計算命中率。"}

        win_rate = target_hit / total * 100

        return {
            "status": "ok",
            "total_samples": total,
            "target_hit": target_hit,
            "stop_hit": stop_hit,
            "neither": neither,
            "win_rate_pct": round(win_rate, 1),
            "note": (
                f"近{total}個交易日樣本：目標優先命中 {target_hit} 次（{win_rate:.1f}%）、"
                f"停損優先觸發 {stop_hit} 次、{lookback_horizon}日內兩者皆未觸及 {neither} 次。"
                f"⚠️ 簡化統計，用當天收盤價/ATR當基準（比實際交易更樂觀），未計手續費，"
                f"同日內兩者都觸及時保守判定為停損。僅供參考，不代表未來表現，不構成投資建議。"
            ),
        }

    # ==========================================
    # 4. 未來走向 (Outlook) — 條件式情境推演，非預測保證
    # ==========================================
    @staticmethod
    def _build_outlook(row: pd.Series, short_term: dict, swing: dict, long_term: dict) -> dict:
        close = TimeframeEngine._get(row, 'close')
        atr = TimeframeEngine._get(row, 'atr_14')
        sma60 = TimeframeEngine._get(row, 'sma_60')

        views = [short_term['view'], swing['view'], long_term['view']]
        bull_votes = sum(1 for v in views if v.startswith('偏多'))
        bear_votes = sum(1 for v in views if v.startswith('偏空'))

        if bull_votes >= 2 and bear_votes == 0:
            bias = "偏多"
            bias_note = "短、波、長三個週期多數呈現偏多訊號，趨勢一致性較高"
        elif bear_votes >= 2 and bull_votes == 0:
            bias = "偏空"
            bias_note = "短、波、長三個週期多數呈現偏空訊號，趨勢一致性較高"
        elif short_term['view'] == '偏空' and long_term['view'].startswith('偏多'):
            bias = "短空長多(分歧)"
            bias_note = "短線出現拉回訊號，但長線結構仍偏多，較可能是波段回檔而非長期趨勢反轉，惟仍需觀察是否進一步破壞長線結構"
        elif short_term['view'] == '偏多' and long_term['view'] == '偏空':
            bias = "短多長空(分歧)"
            bias_note = "短線出現反彈動能，但長線結構偏空，反彈力道與延續性宜保守看待，慎防僅為逃命波或反彈出貨"
        else:
            bias = "中性/多空拉鋸"
            bias_note = "三個週期訊號不一致或均不明確，趨勢方向尚未明朗，建議耐心等待訊號收斂"

        resistance = close + 2.5 * atr if pd.notna(close) and pd.notna(atr) else np.nan
        support = close - 2.0 * atr if pd.notna(close) and pd.notna(atr) else np.nan

        scenarios = []
        if pd.notna(resistance):
            scenarios.append({
                "condition": f"若價格站穩並突破約 {resistance:.2f}（依近期ATR推算的壓力區）",
                "implication": "動能可能延續，短線偏多情境獲得驗證，惟仍應留意量能是否同步放大",
            })
        if pd.notna(support):
            scenarios.append({
                "condition": f"若價格跌破約 {support:.2f}（依近期ATR推算的支撐區）",
                "implication": "偏多結構可能轉弱，建議依原訂停損計畫執行，避免心存僥倖",
            })
        if pd.notna(sma60):
            scenarios.append({
                "condition": f"季線(60MA)目前約在 {sma60:.2f} 附近",
                "implication": "為觀察波段多空轉折的重要參考位置，跌破常伴隨波段轉弱訊號",
            })

        return {
            "bias": bias,
            "bias_note": bias_note,
            "key_levels": {
                "support_est": None if pd.isna(support) else round(float(support), 2),
                "resistance_est": None if pd.isna(resistance) else round(float(resistance), 2),
                "sma_60": None if pd.isna(sma60) else round(float(sma60), 2),
            },
            "scenarios": scenarios,
            "disclaimer": (
                "⚠️ 以上「未來走向」為根據目前技術結構所做的條件式情境推演（若...則...），"
                "並非對未來價格的預測或保證，也不構成投資建議。技術面隨時可能因基本面消息、"
                "法人動向、總體經濟或市場情緒轉變而失效，請勿單獨依賴本區塊做出投資決策，"
                "並自行評估風險與部位控管。"
            ),
        }

    # ==========================================
    # 主要進入點
    # ==========================================
    @staticmethod
    def build_report(df: pd.DataFrame, idx: int = -1) -> dict:
        """
        對單一時間點（預設最新一筆）產出「短線 / 波段 / 長線」三週期判斷，
        以及彙整後的「未來走向」情境推演報告，供 Dashboard 顯示。

        使用前提：df 應已跑過 IndicatorEngine + StrategyEngine（至少要有
        ai_score / market_regime 欄位）；若同時跑過 DivergenceEngine，
        短線/波段的風險判斷會納入誘多/背離防禦訊號、更為準確。

        Parameters
        ----------
        df  : 完整 pipeline 處理後的 DataFrame
        idx : 要評估的資料列位置（預設 -1 = 最新一筆），主要保留給未來
              想針對歷史某一天做回顧分析時使用
        """
        if df is None or df.empty:
            return {"error": "資料不足，無法產生多週期策略報告"}

        n = len(df)
        pos = idx if idx >= 0 else n + idx
        if pos < 0 or pos >= n:
            return {"error": "指定的資料位置超出範圍"}

        row = df.iloc[pos]
        history = df.iloc[: pos + 1]

        short_term = TimeframeEngine._assess_short_term(history, row)
        swing = TimeframeEngine._assess_swing(history, row)
        long_term = TimeframeEngine._assess_long_term(history, row)

        short_term["trade_plan"] = TimeframeEngine._build_short_term_plan(row)
        swing["trade_plan"] = TimeframeEngine._build_swing_plan(row)
        long_term["trade_plan"] = TimeframeEngine._build_long_term_plan(row)

        outlook = TimeframeEngine._build_outlook(row, short_term, swing, long_term)

        return {
            "short_term": short_term,
            "swing": swing,
            "long_term": long_term,
            "outlook": outlook,
        }