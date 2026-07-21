import numpy as np
import pandas as pd
import yfinance as yf


class FundamentalEngine:
    """
    📑 基本面與財務面中心 (Fundamental & Financial Center) - TQAI Pro v2.9 新增

    背景與動機：
    現有系統（IndicatorEngine／StrategyEngine／MomentumEngine／ChipEngine）已經
    涵蓋「選股學院」五大選股面向中的「技術面」與「籌碼面」，但完全沒有觸及
    「基本面」（EPS、本益比、營收、淨值、總市值、股本）與「財務面」（ROA、
    ROE、毛利率、營業利益率、負債比例、存貨與應收帳款）。這兩塊是評估公司
    「體質」與「估值是否合理」的核心依據，跟技術面回答的「現在該不該買賣」
    是互補而非取代的關係——技術面轉多但基本面地雷未爆、或基本面優秀但技術面
    尚未轉強，都是常見情境，因此獨立成一個引擎，不與 ai_score / momentum_score
    加權混合，維持本專案一貫「各 Agent 各自表述，使用者自行綜合判斷」的設計
    哲學（見 momentum_engine.py 對這點的說明）。

    ⚠️ 資料來源與誠實揭露（務必詳讀，這是本引擎最大的限制）：
    1. 資料完全來自 yfinance 的 `Ticker.info`（Yahoo Finance 財務摘要）。
       Yahoo Finance 對台股（尤其上市中小型股、上櫃、興櫃）的財務欄位覆蓋率
       與更新頻率，都遠不如美股，很多欄位可能是 None／缺值／甚至是落後好幾季
       的舊資料。任何一個欄位缺值都只會顯示「資料不足」，不會用預設值頂替、
       更不會用其他公司的資料去猜測，避免產生「看起來有數字、其實是假的」
       的誤導。
    2. 「成立時間」「主力買賣超」「集保庫存」「董監持股比例」「存貨與應收
       帳款明細」這幾項「選股學院」文件提到的指標，yfinance 完全沒有對應
       欄位（或只能透過近似的資產負債表科目粗略推算，可信度低），本引擎
       刻意不假造這些欄位，而是在對應區塊直接標註「yfinance 無此資料，
       如需要請改查詢公開資訊觀測站 (MOPS) 或券商看盤軟體」。
    3. 「股本」欄位 yfinance 沒有直接提供，這裡用「流通在外股數 × 台股常見
       面額 10 元」概略估算，跟公司實際財報上的股本數字可能有出入（尤其若
       公司有特別股或面額非 10 元），僅供規模概略參考，不是精確財報數字。
    4. 【v2.10 已改善】「負債比例」嚴謹定義是「負債總額 / 資產總額」。先前版本
       只能用 yfinance `info` 裡的 `debtToEquity`（負債權益比，即「負債 /
       股東權益」，分母不同）近似，容易與嚴謹定義混淆。v2.10 新增
       `get_precise_debt_ratio()`，直接抓資產負債表的「負債總額」與
       「資產總額」精算出真正的負債比例（欄位 `debt_ratio_pct`），
       `build_fundamental_report()` 會優先使用這個精算值；只有在抓不到
       資產負債表（例如興櫃/小型股覆蓋率不足）時，才會退而求其次改用
       `debt_to_equity` 近似值，且會在提示文字中明確標註「近似值」，
       不混淆兩者。
    5. ETF 沒有「基本面」可言（一籃子股票沒有單一 EPS/ROE），呼叫端應先用
       NameEngine.is_etf() 判斷，是 ETF 就不應該顯示本報告（或顯示明確提示），
       本引擎本身也會在偵測到常見 ETF 代碼時於 build_fundamental_report()
       回傳 status='not_applicable'。
    6. 本引擎不接受呼叫 TWSE/MOPS 官方財報 API（本開發環境沙盒無對外網路
       權限可實測），僅先完成語法與邏輯設計；正式上線前請在具備網路權限的
       環境驗證 yfinance 對目標股票清單的實際欄位覆蓋率。

    設計原則（沿用本專案一貫的防禦性風格）：
    - 任何單一欄位或整支股票的抓取失敗，都只會讓該欄位/該股票顯示「資料
      不足」或 status='unavailable'，不會拋出例外中斷呼叫端（ScannerEngine
      批次掃描、app.py 個股頁面）。
    - 刻意不將本引擎接入 ScannerEngine.scan()：全市場掃描時若每檔股票都
      額外呼叫一次 yfinance `.info`（這是一個相對昂貴、且常有速率限制的
      端點，跟 K 線的 `.history()` 不是同一組後端），會顯著拖慢掃描速度、
      提高被限速的風險。這跟 momentum_engine.py 刻意不呼叫 ChipEngine 的
      外部 API 是同樣的考量。本引擎設計給「個股深度分析」頁面單獨呼叫。
    """

    # 常見 ETF 代碼（沿用 name_engine.py / industry_engine.py 的清單），
    # ETF 沒有基本面可言，偵測到時直接標示 not_applicable，不產生誤導性的
    # 空白或零值財務數字。
    _ETF_CODES = {"0050", "0056", "00878", "006208", "00631L", "00713"}

    # 台股常見面額（新台幣元），用於「股本」概略估算
    _COMMON_PAR_VALUE = 10.0

    # ==========================================
    # 工具方法
    # ==========================================
    @staticmethod
    def _clean_ticker(ticker: str) -> str:
        return str(ticker).split(".")[0].strip()

    @staticmethod
    def _to_float(value):
        """安全轉型為 float，缺值/非數值一律回傳 np.nan，不用 0 或其他預設值頂替
        （0 在本益比、ROE 等欄位上有實際意義，不能拿來當「缺值」的替代品）。"""
        if value is None:
            return np.nan
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return np.nan
            return val
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _resolve_yf_ticker(stock_code: str):
        """依序嘗試 .TW / .TWO，沿用 data_engine.py 的候選後綴邏輯，
        確保基本面查詢跟股價查詢使用同一檔股票（不會查到不同市場別的同代碼）。"""
        code = FundamentalEngine._clean_ticker(stock_code)
        if not code.isdigit():
            return yf.Ticker(code), code

        for suffix in (".TW", ".TWO"):
            candidate = code + suffix
            tkr = yf.Ticker(candidate)
            try:
                info = tkr.info if hasattr(tkr, "info") else {}
            except Exception:
                info = {}
            # info 抓得到且至少有一個關鍵欄位有值，視為找到正確市場別
            if info and (info.get("trailingEps") is not None or info.get("marketCap") is not None):
                return tkr, candidate

        # 兩種後綴都查不到有效財務欄位：回傳最後一個候選，讓上層統一走
        # 「資料不足」的容錯路徑，而不是在這裡拋例外中斷。
        return yf.Ticker(code + ".TW"), code + ".TW"

    # ==========================================
    # 2. 精算負債比例（v2.10 新增，取代原本 debt_to_equity 的近似值）
    # ==========================================
    @staticmethod
    def get_precise_debt_ratio(stock_code: str) -> dict:
        """
        用真實資產負債表科目精算嚴謹定義的「負債比例 = 負債總額 / 資產總額」，
        取代原本只能用 debt_to_equity（負債 / 股東權益，分母不同）做近似的
        限制（見 class docstring 限制第4點）。

        優先用年報資產負債表（`Ticker.balance_sheet`），覆蓋率通常比季報好；
        年報缺值才退而求其次改用季報（`Ticker.quarterly_balance_sheet`）。
        任何一步缺值都直接回傳 status='insufficient_data'，不做估計填補。
        """
        code = FundamentalEngine._clean_ticker(stock_code)
        try:
            tkr, _ = FundamentalEngine._resolve_yf_ticker(code)
            bs = tkr.balance_sheet
            if bs is None or bs.empty:
                bs = tkr.quarterly_balance_sheet
        except Exception:
            return {"status": "insufficient_data"}

        if bs is None or bs.empty:
            return {"status": "insufficient_data"}

        def _row(df, keywords):
            for idx in df.index:
                if any(kw.lower() in str(idx).lower() for kw in keywords):
                    return df.loc[idx]
            return None

        total_assets_row = _row(bs, ["total assets"])
        total_liab_row = _row(bs, ["total liabilities net minority interest", "total liab"])

        if total_assets_row is None or total_liab_row is None:
            return {"status": "insufficient_data"}

        try:
            total_assets = float(total_assets_row.iloc[0])
            total_liab = float(total_liab_row.iloc[0])
        except (IndexError, ValueError, TypeError):
            return {"status": "insufficient_data"}

        if total_assets == 0 or np.isnan(total_assets) or np.isnan(total_liab):
            return {"status": "insufficient_data"}

        return {
            "status": "ok",
            "debt_ratio_pct": total_liab / total_assets * 100,
            "total_assets": total_assets,
            "total_liabilities": total_liab,
            "period": str(bs.columns[0]) if len(bs.columns) else None,
        }

    # ==========================================
    # 1. 原始欄位抓取
    # ==========================================
    @staticmethod
    def get_fundamental_snapshot(stock_code: str) -> dict:
        """
        回傳單一股票的基本面/財務面原始欄位快照。

        欄位缺值一律回傳 np.nan（數值欄位）或 None（文字欄位），呼叫端
        請用 pd.notna() 判斷是否要顯示，不要假設一定有值。
        """
        code = FundamentalEngine._clean_ticker(stock_code)

        try:
            tkr, resolved = FundamentalEngine._resolve_yf_ticker(code)
            info = tkr.info if hasattr(tkr, "info") else {}
        except Exception:
            info = {}
            resolved = None

        if not info:
            return {"status": "unavailable", "resolved_symbol": resolved}

        f = FundamentalEngine._to_float

        eps = f(info.get("trailingEps"))
        pe_ttm = f(info.get("trailingPE"))
        pe_forward = f(info.get("forwardPE"))
        pb = f(info.get("priceToBook"))
        book_value = f(info.get("bookValue"))
        market_cap = f(info.get("marketCap"))
        shares_out = f(info.get("sharesOutstanding"))
        revenue_ttm = f(info.get("totalRevenue"))
        revenue_growth = f(info.get("revenueGrowth"))  # 小數 (yoy)，非百分比
        roa = f(info.get("returnOnAssets"))             # 小數
        roe = f(info.get("returnOnEquity"))              # 小數
        gross_margin = f(info.get("grossMargins"))       # 小數
        operating_margin = f(info.get("operatingMargins"))  # 小數
        debt_to_equity = f(info.get("debtToEquity"))     # 通常已是百分比數字（例如 45.2 代表 45.2%）

        # 股本概略估算：流通在外股數 × 常見面額 10 元（見上方 docstring 限制 3）
        estimated_capital = shares_out * FundamentalEngine._COMMON_PAR_VALUE if pd.notna(shares_out) else np.nan

        # 精算負債比例（見 get_precise_debt_ratio 說明，取代 debt_to_equity 的近似值）
        try:
            debt_ratio_report = FundamentalEngine.get_precise_debt_ratio(code)
        except Exception:
            debt_ratio_report = {"status": "insufficient_data"}
        debt_ratio_pct = debt_ratio_report.get("debt_ratio_pct", np.nan) if debt_ratio_report.get("status") == "ok" else np.nan

        return {
            "status": "ok",
            "resolved_symbol": resolved,
            "eps_ttm": eps,
            "pe_ttm": pe_ttm,
            "pe_forward": pe_forward,
            "price_to_book": pb,
            "book_value_per_share": book_value,
            "market_cap": market_cap,
            "shares_outstanding": shares_out,
            "estimated_capital": estimated_capital,
            "revenue_ttm": revenue_ttm,
            "revenue_growth_yoy": revenue_growth,
            "roa": roa,
            "roe": roe,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "debt_to_equity": debt_to_equity,
            "debt_ratio_pct": debt_ratio_pct,
        }

    # ==========================================
    # 3. 存貨＋應收帳款 vs 營收成長 檢查（盡力而為，資料常缺）
    # ==========================================
    @staticmethod
    def check_inventory_receivables_risk(stock_code: str) -> dict:
        """
        「選股學院」財務面提到：若存貨＋應收帳款成長速度「大於」營收成長
        速度，可能是產品滯銷或帳款難以回收的警訊。

        ⚠️ 這裡用最近兩期「季」資產負債表與「季」損益表概略估算年增率，
        yfinance 的季報欄位命名常隨時間調整、且台股覆蓋率低，任何一個
        欄位取不到就直接回傳 status='insufficient_data'，不做任何插值
        或估計，避免產生看似精確、實則無據的財務健檢結論。
        """
        code = FundamentalEngine._clean_ticker(stock_code)
        try:
            tkr, _ = FundamentalEngine._resolve_yf_ticker(code)
            bs = tkr.quarterly_balance_sheet
            inc = tkr.quarterly_financials
        except Exception:
            return {"status": "insufficient_data"}

        if bs is None or bs.empty or inc is None or inc.empty or bs.shape[1] < 5 or inc.shape[1] < 5:
            return {"status": "insufficient_data"}

        def _row(df, keywords):
            for idx in df.index:
                if any(kw.lower() in str(idx).lower() for kw in keywords):
                    return df.loc[idx]
            return None

        inventory = _row(bs, ["inventory"])
        receivables = _row(bs, ["receivable"])
        revenue = _row(inc, ["total revenue", "revenue"])

        if inventory is None or receivables is None or revenue is None:
            return {"status": "insufficient_data"}

        try:
            # 最新一季 vs 去年同季（往前推 4 個季度欄位），欄位皆為時間新到舊排序
            inv_now, inv_yoy = float(inventory.iloc[0]), float(inventory.iloc[4])
            rec_now, rec_yoy = float(receivables.iloc[0]), float(receivables.iloc[4])
            rev_now, rev_yoy = float(revenue.iloc[0]), float(revenue.iloc[4])
        except (IndexError, ValueError, TypeError):
            return {"status": "insufficient_data"}

        if rev_yoy == 0 or inv_yoy == 0 or rec_yoy == 0:
            return {"status": "insufficient_data"}

        inv_rec_growth = ((inv_now + rec_now) - (inv_yoy + rec_yoy)) / (inv_yoy + rec_yoy) * 100
        revenue_growth = (rev_now - rev_yoy) / abs(rev_yoy) * 100

        flag_risk = inv_rec_growth > revenue_growth + 10  # 給 10 個百分點緩衝，避免雜訊觸發

        return {
            "status": "ok",
            "inventory_receivables_growth_pct": inv_rec_growth,
            "revenue_growth_pct": revenue_growth,
            "risk_flag": flag_risk,
        }

    # ==========================================
    # 4. 整合報告（供 Dashboard 顯示）
    # ==========================================
    @staticmethod
    def build_fundamental_report(stock_code: str) -> dict:
        code = FundamentalEngine._clean_ticker(stock_code)

        # ⚠️ 修正說明（v2.9.4）：原本只查一份寫死的6檔ETF清單(_ETF_CODES)，
        # 台股實際上有數百檔ETF，查其他檔（例如00919、00929）原本會被當成
        # 一般個股去跑基本面分析，得到一堆N/A卻顯示「中性狀態」，容易誤導
        # 成「這檔ETF基本面普通」而不是「這根本不適用」。改用
        # NameEngine.is_etf()（內建清單+通用的「00開頭」代碼型態判斷雙重
        # 保險），涵蓋範圍更廣，不再只靠一份容易過時的寫死清單。
        from engines.name_engine import NameEngine
        if code in FundamentalEngine._ETF_CODES or NameEngine.is_etf(code):
            return {
                "status": "not_applicable",
                "message": "ℹ️ ETF 為一籃子股票組合，沒有單一公司的 EPS/本益比/ROE 等基本面數字可言，本區塊不適用。請改看「ETF 專屬分析」區塊。",
            }

        snapshot = FundamentalEngine.get_fundamental_snapshot(code)
        if snapshot.get("status") != "ok":
            return {
                "status": "unavailable",
                "message": "⚠️ 暫時無法取得基本面資料（可能是 yfinance 對此股票的財務欄位覆蓋不足，興櫃/小型股尤其常見，或近期服務異常）。",
            }

        try:
            inv_rec = FundamentalEngine.check_inventory_receivables_risk(code)
        except Exception:
            inv_rec = {"status": "insufficient_data"}

        flags = []

        pe = snapshot["pe_ttm"]
        if pd.notna(pe):
            if pe < 0:
                flags.append("🔴 本益比為負值（近四季虧損），本益比在此情況下不具評價意義，須另行檢視虧損原因")
            elif pe < 10:
                flags.append(f"🟢 本益比 {pe:.1f}，屬偏低區間（<10），若基本面無惡化，長期投資可能有吸引力")
            elif pe < 15:
                flags.append(f"🟢 本益比 {pe:.1f}，屬相對偏低區間（<15）")
            elif pe > 40:
                flags.append(f"🔴 本益比 {pe:.1f}，屬偏貴區間（>40），追高風險較高")
            elif pe > 30:
                flags.append(f"🟡 本益比 {pe:.1f}，偏高（>30），不宜追高，建議留意成長性是否能支撐評價")

        roe = snapshot["roe"]
        if pd.notna(roe):
            roe_pct = roe * 100
            if roe_pct >= 20:
                flags.append(f"🟢 ROE {roe_pct:.1f}%，獲利能力優異（不依賴舉債仍能高效創造股東回報）")
            elif roe_pct < 0:
                flags.append(f"🔴 ROE {roe_pct:.1f}%，股東權益報酬為負，公司近期處於虧損侵蝕淨值狀態")
            elif roe_pct < 5:
                flags.append(f"🟡 ROE {roe_pct:.1f}%，獲利效率偏弱")

        gm = snapshot["gross_margin"]
        om = snapshot["operating_margin"]
        if pd.notna(gm) and pd.notna(om):
            if gm * 100 > 40:
                flags.append(f"🟢 毛利率 {gm*100:.1f}%，具備一定的技術/品牌或規模經濟優勢")
            if om < 0:
                flags.append(f"🔴 營業利益率 {om*100:.1f}%，本業虧損（尚未計入業外損益）")

        dte = snapshot["debt_to_equity"]
        debt_ratio = snapshot.get("debt_ratio_pct", np.nan)
        if pd.notna(debt_ratio):
            # 精算負債比例（負債總額/資產總額）可用時，優先採用這個嚴謹定義
            if debt_ratio > 70:
                flags.append(f"🔴 負債比例 {debt_ratio:.1f}%（負債/資產，精算值），財務槓桿偏高，需留意舉債風險")
            elif debt_ratio > 50:
                flags.append(f"🟡 負債比例 {debt_ratio:.1f}%（精算值），槓桿中等偏高")
            else:
                flags.append(f"🟢 負債比例 {debt_ratio:.1f}%（精算值），財務結構相對穩健")
        elif pd.notna(dte):
            # 抓不到資產負債表時，退而求其次用 debt_to_equity 近似值（分母不同，見引擎說明）
            if dte > 200:
                flags.append(f"🔴 負債權益比 {dte:.0f}%（近似值，非精算負債比例），財務槓桿偏高，需留意舉債風險與利息負擔")
            elif dte > 100:
                flags.append(f"🟡 負債權益比 {dte:.0f}%（近似值），槓桿中等偏高")

        rev_growth = snapshot["revenue_growth_yoy"]
        if pd.notna(rev_growth):
            if rev_growth > 0.2:
                flags.append(f"🟢 營收年增率 {rev_growth*100:.1f}%，公司處於明顯擴張期")
            elif rev_growth < -0.1:
                flags.append(f"🔴 營收年增率 {rev_growth*100:.1f}%，營收明顯衰退，須留意產業景氣或競爭力變化")

        if inv_rec.get("status") == "ok" and inv_rec.get("risk_flag"):
            flags.append(
                f"⚠️ 存貨+應收帳款成長率（{inv_rec['inventory_receivables_growth_pct']:.1f}%）"
                f"明顯高於營收成長率（{inv_rec['revenue_growth_pct']:.1f}%），"
                f"可能有產品滯銷或應收帳款難以回收的流動性風險，建議進一步檢視財報附註"
            )

        if not flags:
            flags.append("ℹ️ 各項基本面/財務面指標未觸發特別警示或亮點，屬中性狀態，或相關欄位資料不足無法判斷")

        return {
            "status": "ok",
            "snapshot": snapshot,
            "inventory_receivables_check": inv_rec,
            "flags": flags,
        }