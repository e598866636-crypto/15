import pandas as pd
import numpy as np

class StrategyEngine:
    """
    🧠 多智能體決策引擎 (Strategy Engine) - TQAI Pro 前沿成長股特化版 v2.7.0

    ⚠️ 修正說明（相對 v2.5.2）：
    1. 【Bull/Bear 分數重疊】原本 bear_score 用 `close < ema_8`，這跟 bull_score
       用的 `close > ema_8` 是同一件事的邏輯反面 —— 等於同一個原始事實被
       bull/bear 兩個「Agent」各自計分一次，最終 ai_score 又把兩者加權平均，
       造成同一訊號被隱性放大（double counting）。現在把 bear_score 改用
       `close < sma_20`（短期結構）取代，跟 bull_score 的 ema_8（更短週期趨勢）
       是不同的均線／不同的時間尺度，降低兩個分數之間的機械式鏡射關係。
       （註：完全消除相關性不可能，因為兩者都源自同一組價格序列，這裡做的是
       降低「同一條件被算兩次」的明顯重複，而非宣稱兩個分數已經統計獨立。）
    2. 【風控懲罰斷崖】原本 risk_penalty 用 0.6 / 0.8 / 1.0 三段階梯，risk_score
       從 74 跳到 75 會讓 ai_score 瞬間乘上不同係數，造成訊號不穩定、容易在
       門檻邊緣來回閃爍。改為以 risk_score 30~90 為區間的連續線性插值
       （1.0 → 0.5），消除斷崖式跳動，同時維持風險越高、懲罰越重的方向性。

    ⚠️ 修正說明（v2.7.0 新增，屬於「資訊依據正確性」修正）：
    3. 【背離/誘多防禦訊號被算出卻從未被使用】DivergenceEngine 的 docstring
       明確說明它輸出的 bearish_divergence / bullish_divergence /
       bull_trap_confirmed / bear_trap_confirmed 都是「確認當下才標記」的
       因果安全欄位，可以放心餵給 StrategyEngine 當特徵；ScannerEngine 的
       pipeline 也確實把 DivergenceEngine 排在 StrategyEngine 之前執行。
       但舊版 generate_signals() 完全沒有讀取這些欄位 —— 等於這組算出來的
       防禦訊號從未真正影響過 ai_score 或買賣建議。實務後果：一檔股票即使
       近期才剛確認「誘多假突破」或「頂背離」，只要均線/MACD/量能等其他
       條件仍然亮眼，ai_score 依然可能高達 80~90 並顯示「🎯 積極作多」，
       跟 MomentumEngine 那邊「誘多/背離防禦扣分」的結論互相矛盾，構成
       系統內部對同一組事實的不一致判斷。現在把這些防禦訊號正式導入
       risk_score（見 defense_risk_add），讓風控審查真的會對近期的假突破/
       背離訊號提高警覺，方向與飆股評分的誘多示警一致。
    4. 【買賣點訊號不夠明確】原本只有連續型的 ai_score 與文字描述的
       action_guide，新增離散化的 entry_signal / exit_signal 欄位，把
       ai_score 門檻與防禦訊號結合成明確的買賣點提示，同時保留 ai_score
       本身供排序與回測使用。
    """
    @staticmethod
    def generate_signals(df: pd.DataFrame):
        df = df.copy()
        if df.empty or 'atr_14' not in df.columns:
            return df
            
        c, atr = df['close'], df['atr_14']
        
        # ==========================================
        # 1. 市場狀態識別層 (Market Regime Engine)
        # ==========================================
        cond_bull_trend = (df['sma_60_slope'] > 0.4) & (c > df['sma_60'])
        cond_bear_trend = (df['sma_60_slope'] < -0.4) & (c < df['sma_60'])
        
        vol_mean = df['volatility_ratio'].rolling(60, min_periods=5).mean()
        cond_high_vol = df['volatility_ratio'] > (vol_mean * 1.3)
        
        df['market_regime'] = np.select(
            [cond_bull_trend & ~cond_high_vol, cond_bear_trend, cond_high_vol],
            ["📈 穩健多頭趨勢", "📉 弱勢空頭格局", "⚠️ 高波動震盪區"],
            default="🔄 低波盤整區"
        )

        # ==========================================
        # 2. 多智能體辯論層 (Multi-Agent Layer)
        # ==========================================
        macd_growing = df['macd_hist'] > df['macd_hist'].shift(1)
        obv_bullish = df['obv'] > df['obv_sma']
        
        bull_score = np.clip(
            35 + 
            np.where(c > df['ema_8'], 15, 0) + 
            np.where((df['macd_hist'] > 0) & macd_growing, 15, np.where(df['macd_hist'] > 0, 10, 0)) + 
            np.where((df['rvol'] > 1.3) & (c > df['sma_20']), 15, 0) +
            np.where(obv_bullish, 10, 0) +  
            np.where(c > df['sma_60'], 10, 0), 0, 100
        )
        df['bull_score'] = bull_score
        df['bull_reason'] = np.select(
            [bull_score >= 80, bull_score >= 60],
            ["強烈看多：價量齊揚，MACD動能加速擴張且OBV籌碼湧入，多方掌握絕對優勢。", 
             "偏多看待：維持在關鍵均線之上，具備基礎上漲動能與量能支撐。"],
            default="動能平庸：缺乏關鍵性突破，多方量能不足。"
        )

        # 修正：close < sma_20（短期結構，20日均線）取代原本 close < ema_8
        # （原本跟 bull_score 的 close > ema_8 是同一件事的邏輯反面，造成重複計分）
        bear_score = np.clip(
            40 +
            np.where(c < df['sma_20'], 20, 0) +
            np.where(df['rsi_14'] < 45, 20, 0) +
            np.where(df['k_9'] < df['d_9'], 20, 0), 0, 100
        )
        df['bear_score'] = bear_score
        df['bear_reason'] = np.select(
            [bear_score >= 80, bear_score >= 60],
            ["強烈警告：跌破月線結構，指標高檔死叉或呈現嚴重弱勢，空方掌控局勢。",
             "潛在風險：出現部分動能衰退跡象，留意拉回風險。"],
            default="未見異常：目前無明顯做空結構破壞現象。"
        )

        bias_60 = (c - df['sma_60']) / (df['sma_60'] + 1e-9) * 100
        cond_overextended = (bias_60 > 25) | (bias_60 < -15)

        # ---- 修正 v2.7.0：導入 DivergenceEngine 的因果安全防禦訊號 ----
        # 只使用「確認當下」才會是 True 的欄位（見 divergence_engine.py 說明），
        # 不會有前視偏誤問題。用 5 天滾動視窗判斷「近期是否曾觸發」，
        # 避免訊號只在確認當天生效、隔天立刻被忽略。
        def _recent_defense_flag(col_name, lookback=5):
            if col_name in df.columns:
                return df[col_name].fillna(False).astype(bool) \
                    .rolling(lookback, min_periods=1).max().astype(bool)
            return pd.Series(False, index=df.index)

        recent_bearish_div = _recent_defense_flag('bearish_divergence')
        recent_bull_trap = _recent_defense_flag('bull_trap_confirmed')
        cond_defense_bear = recent_bearish_div | recent_bull_trap

        # 誘多假突破/頂背離剛確認時，加重風控懲罰（+30），確保 ai_score
        # 不會在防禦訊號亮起的當下仍顯示「積極作多」。
        defense_risk_add = np.where(cond_defense_bear, 30, 0)

        risk_score = np.clip(
            20 + 
            np.where(cond_overextended, 30, 0) + 
            np.where(cond_high_vol, 25, 0) + 
            np.where(c < df['sma_200'], 25, 0) +
            defense_risk_add, 0, 100
        )
        df['risk_score'] = risk_score
        df['defense_risk_flag'] = cond_defense_bear
        df['risk_reason'] = np.select(
            [cond_defense_bear, risk_score >= 70, risk_score >= 45],
            ["🚨 防禦示警：近期確認誘多假突破或頂背離，動能可能已經或即將失效，風控分數已加重懲罰。",
             "🚨 系統性風險高：結構性超買/超賣，或處於年線之下，隨時有重整或重挫風險。",
             "⚠️ 波動放大：市場情緒較為激動，高波動族群建議嚴守風險預算點位。"],
            default="✅ 風險可控：波動率與乖離率處於健康成長區，安全邊際合理。"
        )

        # ==========================================
        # 3. Judge Agent (主審裁決系統)
        # ==========================================
        raw_ai_score = (bull_score * 0.6) + ((100 - bear_score) * 0.4)

        # 修正：風控懲罰改為連續線性插值，避免階梯式斷崖跳動
        # risk_score <= 30 → 不懲罰 (×1.0)；risk_score >= 90 → 最大懲罰 (×0.5)；中間線性過渡
        risk_score_clipped = np.clip(risk_score, 30, 90)
        risk_penalty = 1.0 - (risk_score_clipped - 30) / (90 - 30) * 0.5
        df['risk_penalty'] = risk_penalty

        df['ai_score'] = np.clip(raw_ai_score * risk_penalty, 0, 100)
        
        df['confidence'] = np.select(
            [df['ai_score'] >= 75, df['ai_score'] <= 40],
            ["High (高信心)", "Low (低信心)"],
            default="Medium (中性)"
        )
        
        df['action_guide'] = np.select(
            [df['ai_score'] >= 70, (df['ai_score'] >= 45) & (df['ai_score'] < 70), df['ai_score'] < 45],
            ["🎯 積極作多 (Agent 共識偏多，通過前沿波段風控審查)", 
             "👀 震盪觀望 (多空分歧，建議控制倉位或等待拉回低接)", 
             "✂️ 嚴格保守 (空頭論點勝出或風險過高，建議減碼防禦)"],
            default="未知狀態"
        )
        
        # ==========================================
        # 4. 動態風險預算 (ATR 進攻與防守)
        # ==========================================
        df['stop_loss'] = c - (2.0 * atr)
        df['target_1'] = c + (2.5 * atr)
        df['target_2'] = c + (5.0 * atr)

        # ==========================================
        # 5. 買賣點訊號增強 (Enhanced Entry/Exit Signal)
        # ==========================================
        # ⚠️ 說明：entry_signal / exit_signal 是把連續型 ai_score 離散化成
        # 明確的買賣點提示，並且已經把上方修正的防禦訊號 (defense_risk_flag)
        # 考慮進去，避免出現「文字建議積極作多，但近期其實剛觸發誘多警報」
        # 這種矛盾情境。ai_score 本身仍保留給排序/回測使用，這裡新增的欄位
        # 是給使用者做「現在算不算一個進出場點」的快速判讀，不取代原本的
        # action_guide，兩者可以並列顯示。
        cond_entry_clean = (df['ai_score'] >= 70) & (~df['defense_risk_flag'])
        cond_entry_degraded = (df['ai_score'] >= 70) & (df['defense_risk_flag'])
        cond_exit_weak = df['ai_score'] <= 40
        cond_exit_defense = df['defense_risk_flag'] & (~cond_exit_weak)

        df['entry_signal'] = np.select(
            [cond_entry_clean, cond_entry_degraded],
            ["🟢 買進訊號：AI Score達標且近期無背離/誘多警報",
             "🟡 訊號降級：AI Score達標，但近期有背離/誘多警報，建議觀察或減碼分批進場"],
            default="⚪ 無明確買進訊號"
        )

        df['exit_signal'] = np.select(
            [cond_exit_weak, cond_exit_defense],
            ["🔴 賣出/減碼訊號：AI Score過低，多方論點轉弱",
             "🟠 風險示警：近期背離/誘多警報，建議嚴設停損或減碼防禦"],
            default="⚪ 無明確賣出訊號"
        )

        return df