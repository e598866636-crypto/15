import numpy as np
import pandas as pd


class StockAcademyEngine:
    """
    🎓 選股大師引擎 (Stock Academy Engine) - TQAI Pro 選股學院整合版 v1.1

    核心設計：
    將「選股學院──選股策略的哲學」五大維度系統化融入決策流程。每檔股票
    在市場面、基本面、技術面、籌碼面、財務面各自評分（各佔20分），
    合計100分，對應A+到F的學術等級評等。

    ⚠️ 修正說明（v1.1，相對 v1.0）：
    v1.0 版本有以下問題，這裡一併修正：

    1.【基本面／財務面曾是假評分】v1.0 的 _score_fundamental_dimension /
       _score_financial_dimension 不管股票體質好壞，永遠回傳固定 10/20 分，
       只是預留介面、實際上沒有真的評分。現在改為接受 fundamental_report
       參數（由 engines/fundamental_engine.py 的 FundamentalEngine.
       build_fundamental_report() 產出），有真實資料時才會依 EPS/本益比/
       營收成長/淨值（基本面）與 ROA/ROE/毛利率/營業利益率/負債權益比
       （財務面）做真正的評分；沒有資料時才誠實標記「資料不足」，並給
       中性 10 分（不加分也不扣分），不再假裝這是「已完整實作」。

    2.【eval() 反序列化字串很脆弱】v1.0 把每個維度的評語清單用 str(list)
       存進 DataFrame 欄位，再用 eval() 轉回來，只要文字裡出現特殊字元
       組合就可能解析失敗，也有非必要的安全疑慮。現在改成用「單一物件
       包一層 list 再指派」的方式安全地把 Python list 物件存進 DataFrame
       欄位（沿用 pandas 對 object dtype 欄位的原生支援），build_report()
       直接讀出原始 list，完全不需要 eval()。

    3.【權重正規化 bug】v1.0 若只傳部分維度的自訂權重，正規化只會處理
       使用者有給的 key，但計算總分時漏掉的維度會 fallback 用寫死的 0.2，
       這個 0.2 沒有被一起正規化，導致總權重悄悄超過或低於 1.0、總分失真。
       現在改為：先把五個維度都補齊（缺的 key 視為 0），再一起正規化，
       確保無論使用者傳入幾個維度的權重，加總後永遠是 1.0。

    4.【全市場批次掃描效能】基本面/財務面評分需要呼叫
       FundamentalEngine（背後是 yfinance `.info`，這是較慢、易有速率限制
       的端點，跟 K 線 `.history()` 不是同一組後端）。ScannerEngine 全市場
       批次掃描時，預設**不會**替每一檔股票額外呼叫 FundamentalEngine／
       ChipEngine（沿用 momentum_engine.py 對 ChipEngine 的同樣考量），
       這兩個維度在批次掃描模式下會顯示為「資料不足（快速模式）」的中性
       分數；只有在單檔深度分析頁面才會帶入完整的 chip_report /
       fundamental_report 算出真實分數。呼叫端可自行選擇是否要付出額外
       時間成本，為少量自訂清單開啟完整基本面/財務面評分。
    """

    # ============================================================
    # 維度定義與評分框架
    # ============================================================
    DIMENSION_NAMES = {
        "market": "📈 市場面（價量結構）",
        "fundamental": "📊 基本面（企業體質）",
        "technical": "📉 技術面（動能訊號）",
        "chip": "🏦 籌碼面（資金動向）",
        "financial": "💰 財務面（獲利能力）",
    }

    GRADE_SCALE = [
        (95, "A+", "🌟 超級優秀"),
        (90, "A", "⭐ 優秀"),
        (85, "B+", "👍 很好"),
        (80, "B", "✅ 良好"),
        (75, "B-", "👌 中上"),
        (70, "C+", "📌 及格"),
        (65, "C", "⚠️ 略顯不足"),
        (60, "C-", "❌ 不及格"),
        (0, "F", "🚫 劣質"),
    ]

    _DEFAULT_WEIGHTS = {
        "market": 0.20,
        "fundamental": 0.20,
        "technical": 0.20,
        "chip": 0.20,
        "financial": 0.20,
    }

    # ==========================================
    # 工具方法：安全地把 Python 物件存進 df 欄位 / 取回
    # ==========================================
    @staticmethod
    def _store_object_column(df: pd.DataFrame, col: str, obj):
        """把任意 Python 物件（例如 list）安全地存成整欄的常數值，
        不透過 str()/eval() 往返，避免特殊字元造成解析失敗。"""
        df[col] = pd.Series([obj] * len(df), index=df.index, dtype=object)

    # ============================================================
    # 1. 市場面評分（20分）：股價、漲跌幅、成交量、波動率、相對強度
    # ============================================================
    @staticmethod
    def _score_market_dimension(df: pd.DataFrame):
        """
        市場面評分原則（滿20分）：
          - 價格創新高（近半年內）       +4分
          - 漲幅在合理範圍（3~10%）     +4分
          - 成交量有所放大（RVOL>1.2）  +4分
          - 波動率正常（ATR相對穩健）   +4分
          - 相對強度不過熱               +4分
        """
        if df.empty:
            return 0, ["⚠️ 無K線資料，無法評分"]

        latest = df.iloc[-1]
        close = latest.get("close", np.nan)
        high_252 = df["high"].rolling(252, min_periods=1).max().iloc[-1]
        rvol = latest.get("rvol", 1.0)
        atr_ratio = (latest.get("atr_14", 0) / (close + 1e-9)) * 100
        rsi = latest.get("rsi_14", 50)

        score = 0
        details = []

        if close > 0 and high_252 > 0 and close >= high_252 * 0.95:
            score += 4
            details.append("✅ 股價接近或創新高（強勢訊號）")
        else:
            pct_from_high = (close / high_252 - 1) * 100 if high_252 > 0 else np.nan
            details.append(f"📉 股價距高點 {pct_from_high:.1f}%" if pd.notna(pct_from_high) else "⚠️ 高點資料不足")

        if len(df) >= 5:
            pct_5d = (close - df.iloc[-5]["close"]) / df.iloc[-5]["close"] * 100
            if 1 <= pct_5d <= 8:
                score += 4
                details.append(f"✅ 近5日漲幅 {pct_5d:.1f}%（適度上漲）")
            else:
                details.append(f"🔍 近5日漲幅 {pct_5d:.1f}%（評估中）")
        else:
            details.append("⚠️ 資料不足，無法評估漲幅")

        if rvol > 1.3:
            score += 4
            details.append(f"✅ 成交量放大（RVOL={rvol:.2f}）")
        elif rvol > 1.0:
            score += 2
            details.append(f"👌 成交量正常（RVOL={rvol:.2f}）")
        else:
            details.append(f"⚠️ 成交量萎縮（RVOL={rvol:.2f}）")

        if 0.8 <= atr_ratio <= 3.5:
            score += 4
            details.append(f"✅ 波動率適中（{atr_ratio:.1f}%）")
        elif atr_ratio < 0.5:
            score += 2
            details.append(f"📌 波動率過低（{atr_ratio:.1f}%，可能缺乏流動性）")
        else:
            details.append(f"⚠️ 波動率過高（{atr_ratio:.1f}%）")

        if 45 <= rsi <= 75:
            score += 4
            details.append(f"✅ RSI {rsi:.1f}（強勢但未過熱）")
        elif rsi < 45:
            score += 2
            details.append(f"👌 RSI {rsi:.1f}（偏弱，反彈機會）")
        else:
            details.append(f"⚠️ RSI {rsi:.1f}（超買風險）")

        return int(np.clip(score, 0, 20)), details

    # ============================================================
    # 2. 基本面評分（20分）：EPS、本益比、營收成長、淨值、規模
    # ============================================================
    @staticmethod
    def _score_fundamental_dimension(fundamental_report: dict = None):
        """
        基本面評分原則（滿20分，需要 fundamental_report 才會真正評分）：
          - EPS 為正（近四季獲利）                +4分
          - 本益比在合理區間（10~30）              +4分
          - 營收年增率為正                         +4分
          - 股價淨值比 (P/B) 合理（<3）            +4分
          - 存貨+應收帳款成長未明顯超過營收成長     +4分

        沒有 fundamental_report（或狀態非 ok）時，回傳中性 10 分並誠實
        標記「資料不足」，不假裝已完整評分（見 v1.1 修正說明第1點）。
        """
        details = []

        if not fundamental_report or fundamental_report.get("status") != "ok":
            msg = fundamental_report.get("message") if fundamental_report else None
            details.append(f"⚠️ 基本面資料不足，暫給中性分數（{msg or '未提供 fundamental_report'}）")
            return 10, details

        snap = fundamental_report.get("snapshot", {})
        score = 0
        checks_available = 0

        eps = snap.get("eps_ttm", np.nan)
        if pd.notna(eps):
            checks_available += 1
            if eps > 0:
                score += 4
                details.append(f"✅ EPS(TTM) {eps:.2f}，近四季獲利為正")
            else:
                details.append(f"⚠️ EPS(TTM) {eps:.2f}，近四季虧損")
        else:
            details.append("⚠️ EPS 資料不足")

        pe = snap.get("pe_ttm", np.nan)
        if pd.notna(pe):
            checks_available += 1
            if pe < 0:
                details.append(f"⚠️ 本益比為負（虧損狀態，不具評價意義）")
            elif 10 <= pe <= 30:
                score += 4
                details.append(f"✅ 本益比 {pe:.1f}，落在合理區間 (10~30)")
            elif pe < 10:
                score += 3
                details.append(f"👌 本益比 {pe:.1f}，偏低，留意是否有基本面隱憂")
            else:
                details.append(f"⚠️ 本益比 {pe:.1f}，偏高（>30），追高風險較高")
        else:
            details.append("⚠️ 本益比資料不足")

        rev_g = snap.get("revenue_growth_yoy", np.nan)
        if pd.notna(rev_g):
            checks_available += 1
            if rev_g > 0.1:
                score += 4
                details.append(f"✅ 營收年增率 {rev_g*100:.1f}%，明顯成長")
            elif rev_g > 0:
                score += 2
                details.append(f"👌 營收年增率 {rev_g*100:.1f}%，小幅成長")
            else:
                details.append(f"⚠️ 營收年增率 {rev_g*100:.1f}%，衰退")
        else:
            details.append("⚠️ 營收成長率資料不足")

        pb = snap.get("price_to_book", np.nan)
        if pd.notna(pb):
            checks_available += 1
            if 0 < pb <= 3:
                score += 4
                details.append(f"✅ 股價淨值比 {pb:.2f}，評價相對合理")
            elif pb > 3:
                score += 1
                details.append(f"⚠️ 股價淨值比 {pb:.2f}，評價偏高")
            else:
                details.append(f"📌 股價淨值比 {pb:.2f}")
        else:
            details.append("⚠️ 股價淨值比資料不足")

        inv_rec = fundamental_report.get("inventory_receivables_check", {})
        if inv_rec.get("status") == "ok":
            checks_available += 1
            if not inv_rec.get("risk_flag"):
                score += 4
                details.append("✅ 存貨+應收帳款成長未明顯超過營收成長")
            else:
                details.append("⚠️ 存貨+應收帳款成長明顯高於營收成長，留意滯銷/收款風險")
        else:
            details.append("⚠️ 存貨/應收帳款資料不足（yfinance 季報覆蓋率有限）")

        if checks_available == 0:
            details.insert(0, "⚠️ 所有基本面欄位皆缺值，暫給中性分數")
            return 10, details

        return int(np.clip(score, 0, 20)), details

    # ============================================================
    # 3. 技術面評分（20分）：均線、MACD、KD、背離檢測
    # ============================================================
    @staticmethod
    def _score_technical_dimension(df: pd.DataFrame):
        """
        技術面評分原則（滿20分）：
          - 均線多頭排列 (close>20MA>60MA>200MA)  +5分
          - MACD 動能轉強（DIF>DEA 且柱狀體>0） +5分
          - KD 黃金交叉或在強勢區           +5分
          - 無頂背離或誘多假突破警報       +5分
        """
        if df.empty:
            return 0, ["⚠️ 無K線資料，無法評分"]

        latest = df.iloc[-1]
        close = latest.get("close", np.nan)
        sma20 = latest.get("sma_20", np.nan)
        sma60 = latest.get("sma_60", np.nan)
        sma200 = latest.get("sma_200", np.nan)
        macd_hist = latest.get("macd_hist", 0)
        dif = latest.get("macd_dif", np.nan)
        dea = latest.get("macd_dea", np.nan)
        k9 = latest.get("k_9", 50)
        d9 = latest.get("d_9", 50)

        score = 0
        details = []

        if all(pd.notna(x) for x in [close, sma20, sma60, sma200]):
            if close > sma20 > sma60 > sma200:
                score += 5
                details.append("✅ 均線完全多頭排列（短中長線一致看多）")
            elif close > sma20 > sma60:
                score += 3
                details.append("👌 短期均線多頭排列（長線尚需確認）")
            else:
                details.append("❌ 均線排列不佳（結構性弱勢）")
        else:
            details.append("⚠️ 均線數據不足")

        if pd.notna(macd_hist) and pd.notna(dif) and pd.notna(dea):
            if macd_hist > 0 and dif > dea:
                score += 5
                details.append(f"✅ MACD 動能強勢（柱狀體={macd_hist:.3f}）")
            elif macd_hist > 0:
                score += 2
                details.append(f"👌 MACD 柱狀體翻正（{macd_hist:.3f}）")
            else:
                details.append(f"⚠️ MACD 動能轉弱（{macd_hist:.3f}）")
        else:
            details.append("⚠️ MACD 數據不足")

        if pd.notna(k9) and pd.notna(d9):
            if k9 > d9 and k9 > 50:
                score += 5
                details.append(f"✅ KD 黃金交叉且在強勢區（K={k9:.1f}）")
            elif k9 > d9:
                score += 2
                details.append(f"👌 KD 黃金交叉（K={k9:.1f}）")
            elif k9 < 30:
                score += 1
                details.append(f"⚠️ KD 超賣區（反彈機會，K={k9:.1f}）")
            else:
                details.append(f"📌 KD 中性區（K={k9:.1f}）")
        else:
            details.append("⚠️ KD 數據不足")

        bearish_div = bool(latest.get("bearish_divergence", False))
        bull_trap = bool(latest.get("bull_trap_confirmed", False))

        if bearish_div or bull_trap:
            details.append("⚠️ 近期確認頂背離或誘多假突破（動能可能減弱）")
        else:
            score += 5
            details.append("✅ 無背離或誘多警報（動能延續機率高）")

        return int(np.clip(score, 0, 20)), details

    # ============================================================
    # 4. 籌碼面評分（20分）：融資融券、外資、自營商、投信、OBV代理
    # ============================================================
    @staticmethod
    def _score_chip_dimension(df: pd.DataFrame, chip_report: dict = None):
        """
        籌碼面評分原則（滿20分）：
        若有 ChipEngine 報告則優先使用；沒有的話（例如全市場批次掃描時
        刻意不呼叫外部籌碼 API，見本檔案 v1.1 修正說明第4點），改用
        既有 K 線資料（OBV、RVOL）做代理評估，不會拋出例外或給零分。
        """
        score = 0
        details = []

        if chip_report and chip_report.get("status") == "ok":
            inst = chip_report.get("institutional")
            margin = chip_report.get("margin")

            if inst:
                f_net, t_net = inst.get("foreign_net", 0), inst.get("trust_net", 0)
                if f_net > 0 and t_net > 0:
                    score += 5
                    details.append(f"✅ 外資+投信同步買超（外資={f_net}, 投信={t_net}）")
                elif f_net > 0 or t_net > 0:
                    score += 2
                    details.append(f"👌 法人買超訊號（外資={f_net}, 投信={t_net}）")
                else:
                    details.append(f"⚠️ 法人賣超（外資={f_net}, 投信={t_net}）")

            if margin:
                m_chg = margin.get("margin_change", 0)
                s_chg = margin.get("short_change", 0)
                if m_chg < 0 and s_chg > 0:
                    score += 4
                    details.append("✅ 融資減融券增（散戶轉保守/看空）")
                elif m_chg > 0 and s_chg < 0:
                    score += 2
                    details.append("👌 融資增融券減（散戶轉樂觀）")
        else:
            details.append("📡 籌碼資料來源未連接（批次掃描快速模式，或該股籌碼資料暫不可用）；以下改用OBV/量能代理評估")

        if "obv" in df.columns and "obv_sma" in df.columns and not df.empty:
            obv_latest = df.iloc[-1].get("obv", 0)
            obv_sma = df.iloc[-1].get("obv_sma", 0)
            if obv_latest > obv_sma:
                score += 5
                details.append("✅ OBV > 均線（資金持續流入）")
            else:
                details.append("⚠️ OBV < 均線（資金流出跡象）")

        if "rvol" in df.columns and not df.empty:
            rvol = df.iloc[-1].get("rvol", 1.0)
            if rvol > 1.5:
                score += 3
                details.append(f"✅ 成交量明顯放大（RVOL={rvol:.2f}）")
            elif rvol > 1.0:
                score += 1
                details.append(f"👌 成交量小幅放大（RVOL={rvol:.2f}）")

        if not details:
            details.append("📊 籌碼面暫無異常訊號")

        return int(np.clip(score, 0, 20)), details

    # ============================================================
    # 5. 財務面評分（20分）：毛利率、營業利益率、ROA、ROE、負債權益比
    # ============================================================
    @staticmethod
    def _score_financial_dimension(fundamental_report: dict = None):
        """
        財務面評分原則（滿20分，需要 fundamental_report 才會真正評分）：
          - 毛利率 > 30%                    +4分
          - 營業利益率 > 10%（本業獲利能力） +4分
          - ROA（資產報酬率）> 5%           +4分
          - ROE（股東權益報酬率）> 10%      +4分
          - 負債比例 < 50%（精算值；抓不到資產負債表時退用負債權益比近似） +4分

        沒有 fundamental_report（或狀態非 ok）時，回傳中性 10 分並誠實
        標記「資料不足」（見 v1.1 修正說明第1點）。負債比例優先使用
        FundamentalEngine.get_precise_debt_ratio() 算出的精算值（負債/
        資產），抓不到資產負債表時才退而求其次用「負債權益比」近似值，
        兩者計算基礎不同，沿用 fundamental_engine.py 的誠實揭露，不
        混淆兩個名詞。
        """
        details = []

        if not fundamental_report or fundamental_report.get("status") != "ok":
            msg = fundamental_report.get("message") if fundamental_report else None
            details.append(f"⚠️ 財務面資料不足，暫給中性分數（{msg or '未提供 fundamental_report'}）")
            return 10, details

        snap = fundamental_report.get("snapshot", {})
        score = 0
        checks_available = 0

        gm = snap.get("gross_margin", np.nan)
        if pd.notna(gm):
            checks_available += 1
            if gm * 100 > 30:
                score += 4
                details.append(f"✅ 毛利率 {gm*100:.1f}%，具備獲利優勢")
            elif gm * 100 > 15:
                score += 2
                details.append(f"👌 毛利率 {gm*100:.1f}%，中等水準")
            else:
                details.append(f"⚠️ 毛利率 {gm*100:.1f}%，偏低")
        else:
            details.append("⚠️ 毛利率資料不足")

        om = snap.get("operating_margin", np.nan)
        if pd.notna(om):
            checks_available += 1
            if om * 100 > 10:
                score += 4
                details.append(f"✅ 營業利益率 {om*100:.1f}%，本業獲利能力佳")
            elif om > 0:
                score += 2
                details.append(f"👌 營業利益率 {om*100:.1f}%，本業小幅獲利")
            else:
                details.append(f"⚠️ 營業利益率 {om*100:.1f}%，本業虧損")
        else:
            details.append("⚠️ 營業利益率資料不足")

        roa = snap.get("roa", np.nan)
        if pd.notna(roa):
            checks_available += 1
            if roa * 100 > 5:
                score += 4
                details.append(f"✅ ROA {roa*100:.1f}%，資產運用效率佳")
            elif roa > 0:
                score += 2
                details.append(f"👌 ROA {roa*100:.1f}%，資產運用效率一般")
            else:
                details.append(f"⚠️ ROA {roa*100:.1f}%，資產報酬為負")
        else:
            details.append("⚠️ ROA 資料不足")

        roe = snap.get("roe", np.nan)
        if pd.notna(roe):
            checks_available += 1
            if roe * 100 > 10:
                score += 4
                details.append(f"✅ ROE {roe*100:.1f}%，股東權益報酬佳")
            elif roe > 0:
                score += 2
                details.append(f"👌 ROE {roe*100:.1f}%，股東權益報酬一般")
            else:
                details.append(f"⚠️ ROE {roe*100:.1f}%，股東權益報酬為負")
        else:
            details.append("⚠️ ROE 資料不足")

        dte = snap.get("debt_to_equity", np.nan)
        debt_ratio = snap.get("debt_ratio_pct", np.nan)
        if pd.notna(debt_ratio):
            checks_available += 1
            if debt_ratio < 50:
                score += 4
                details.append(f"✅ 負債比例 {debt_ratio:.1f}%（精算值：負債/資產），槓桿穩健")
            elif debt_ratio < 70:
                score += 2
                details.append(f"👌 負債比例 {debt_ratio:.1f}%（精算值），槓桿中等偏高")
            else:
                details.append(f"⚠️ 負債比例 {debt_ratio:.1f}%（精算值），槓桿偏高")
        elif pd.notna(dte):
            checks_available += 1
            if dte < 100:
                score += 4
                details.append(f"✅ 負債權益比(近似) {dte:.0f}%，槓桿穩健")
            elif dte < 200:
                score += 2
                details.append(f"👌 負債權益比(近似) {dte:.0f}%，槓桿中等偏高")
            else:
                details.append(f"⚠️ 負債權益比(近似) {dte:.0f}%，槓桿偏高")
        else:
            details.append("⚠️ 負債比例／負債權益比資料皆不足")

        if checks_available == 0:
            details.insert(0, "⚠️ 所有財務面欄位皆缺值，暫給中性分數")
            return 10, details

        return int(np.clip(score, 0, 20)), details

    # ============================================================
    # 綜合評分與學級轉換
    # ============================================================
    @staticmethod
    def _convert_to_grade(score: float) -> tuple:
        for threshold, grade, label in StockAcademyEngine.GRADE_SCALE:
            if score >= threshold:
                return grade, label
        return "F", "🚫 劣質"

    @staticmethod
    def _normalize_weights(weights: dict = None) -> dict:
        """把五個維度都補齊（缺的 key 視為 0）再一起正規化，確保無論
        使用者傳入幾個維度的自訂權重，加總後永遠等於 1.0（修正 v1.0
        的權重 bug，見上方 docstring 說明第3點）。"""
        base = dict(StockAcademyEngine._DEFAULT_WEIGHTS)
        if weights:
            base = {k: weights.get(k, 0.0) for k in base}

        weight_sum = sum(base.values())
        if weight_sum <= 0:
            return dict(StockAcademyEngine._DEFAULT_WEIGHTS)
        return {k: v / weight_sum for k, v in base.items()}

    @staticmethod
    def add_academy_score(df: pd.DataFrame,
                           chip_report: dict = None,
                           fundamental_report: dict = None,
                           weights: dict = None):
        """
        主入口：為整張 df 計算五維度評分與綜合評級。

        參數：
            df                 : 已完整跑過前面 pipeline 的 DataFrame
            chip_report        : ChipEngine.build_chip_report() 的輸出（可選；
                                  全市場批次掃描時通常不傳，改用OBV代理）
            fundamental_report : FundamentalEngine.build_fundamental_report()
                                  的輸出（可選；不傳則基本面/財務面顯示中性
                                  分數＋「資料不足」說明，見上方 docstring）
            weights            : 自訂權重 {market: 0.2, fundamental: 0.2, ...}
                                  預設均權 (各20%)，缺的維度視為0後一起正規化

        輸出：補充以下欄位（明細欄位存的是原始 list 物件，不是字串，讀取
        時直接用，不需要 eval()）
            academy_market_score, academy_market_details
            academy_fundamental_score, academy_fundamental_details
            academy_technical_score, academy_technical_details
            academy_chip_score, academy_chip_details
            academy_financial_score, academy_financial_details
            academy_total_score : 加權合計（0~100）
            academy_grade       : 學級 (A+/A/B+ ...)
            academy_label        : 綜合評語
        """
        weights = StockAcademyEngine._normalize_weights(weights)

        df = df.copy()

        market_score, market_details = StockAcademyEngine._score_market_dimension(df)
        technical_score, technical_details = StockAcademyEngine._score_technical_dimension(df)
        chip_score, chip_details = StockAcademyEngine._score_chip_dimension(df, chip_report)
        fundamental_score, fundamental_details = StockAcademyEngine._score_fundamental_dimension(fundamental_report)
        financial_score, financial_details = StockAcademyEngine._score_financial_dimension(fundamental_report)

        StockAcademyEngine._store_object_column(df, 'academy_market_details', market_details)
        StockAcademyEngine._store_object_column(df, 'academy_technical_details', technical_details)
        StockAcademyEngine._store_object_column(df, 'academy_chip_details', chip_details)
        StockAcademyEngine._store_object_column(df, 'academy_fundamental_details', fundamental_details)
        StockAcademyEngine._store_object_column(df, 'academy_financial_details', financial_details)

        df['academy_market_score'] = market_score
        df['academy_technical_score'] = technical_score
        df['academy_chip_score'] = chip_score
        df['academy_fundamental_score'] = fundamental_score
        df['academy_financial_score'] = financial_score

        total_score = (
            market_score * weights["market"] +
            technical_score * weights["technical"] +
            chip_score * weights["chip"] +
            fundamental_score * weights["fundamental"] +
            financial_score * weights["financial"]
        )
        # 上面五個維度原始滿分皆為20分，加權平均後還原成百分制（× 5）
        total_score_pct = np.clip(total_score * 5, 0, 100)

        df['academy_total_score'] = int(round(total_score_pct))
        grade, label = StockAcademyEngine._convert_to_grade(total_score_pct)
        df['academy_grade'] = grade
        df['academy_label'] = label

        return df

    # ============================================================
    # 診斷報告（供 Dashboard / 詳細分析使用）
    # ============================================================
    @staticmethod
    def build_report(df: pd.DataFrame, chip_report: dict = None, fundamental_report: dict = None) -> dict:
        """生成單檔股票的完整選股大師診斷報告。"""
        if df is None or df.empty:
            return {}

        df_with_scores = StockAcademyEngine.add_academy_score(
            df, chip_report=chip_report, fundamental_report=fundamental_report
        )
        latest = df_with_scores.iloc[-1]

        report = {
            "綜合評級": {
                "總分": int(latest.get('academy_total_score', 0)),
                "學級": latest.get('academy_grade', 'F'),
                "評語": latest.get('academy_label', ''),
            },
            "五維度明細": {
                "市場面": {
                    "分數": int(latest.get('academy_market_score', 0)),
                    "分析": latest.get('academy_market_details', []),
                },
                "基本面": {
                    "分數": int(latest.get('academy_fundamental_score', 0)),
                    "分析": latest.get('academy_fundamental_details', []),
                },
                "技術面": {
                    "分數": int(latest.get('academy_technical_score', 0)),
                    "分析": latest.get('academy_technical_details', []),
                },
                "籌碼面": {
                    "分數": int(latest.get('academy_chip_score', 0)),
                    "分析": latest.get('academy_chip_details', []),
                },
                "財務面": {
                    "分數": int(latest.get('academy_financial_score', 0)),
                    "分析": latest.get('academy_financial_details', []),
                },
            },
            "關鍵提示": StockAcademyEngine._build_insights(df_with_scores),
            "資料完整度提示": StockAcademyEngine._build_completeness_note(chip_report, fundamental_report),
        }

        return report

    @staticmethod
    def _build_completeness_note(chip_report: dict, fundamental_report: dict) -> str:
        """告知使用者本次評分是否使用了真實的籌碼/基本面外部資料，避免
        『批次快速模式的中性分數』被誤讀成『這檔股票體質真的不好』。"""
        chip_ok = bool(chip_report and chip_report.get("status") == "ok")
        fund_ok = bool(fundamental_report and fundamental_report.get("status") == "ok")

        if chip_ok and fund_ok:
            return "✅ 本次評分已納入真實籌碼與基本面/財務面資料，五維度皆為完整評分。"
        missing = []
        if not chip_ok:
            missing.append("籌碼面（改用OBV/量能代理）")
        if not fund_ok:
            missing.append("基本面／財務面（暫給中性10分）")
        return (
            f"⚠️ 本次評分缺少外部資料：{'、'.join(missing)}，總分可能被中性分數拉低或拉高，"
            "不完全反映真實體質。如需完整五維度評分，請在單檔深度分析頁面查看（會自動帶入"
            "ChipEngine／FundamentalEngine 的真實資料）。"
        )

    @staticmethod
    def _build_insights(df: pd.DataFrame) -> list:
        """萃取五維度中最值得注意的重點（最弱2項＋最強1項）。"""
        insights = []
        latest = df.iloc[-1]

        scores = {
            "市場面": latest.get('academy_market_score', 10),
            "技術面": latest.get('academy_technical_score', 10),
            "籌碼面": latest.get('academy_chip_score', 10),
            "基本面": latest.get('academy_fundamental_score', 10),
            "財務面": latest.get('academy_financial_score', 10),
        }

        weakest = sorted(scores.items(), key=lambda x: x[1])[:2]
        for dim, score in weakest:
            if score <= 10:
                insights.append(f"⚠️ {dim}評分偏低（{score}分），為主要考量點")

        strongest = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1]
        for dim, score in strongest:
            if score >= 15:
                insights.append(f"✅ {dim}表現突出（{score}分），為主要優勢")

        if not insights:
            insights.append("📊 各維度均衡發展，無明顯強弱")

        return insights