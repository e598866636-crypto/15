"""
test_scan_dependent_features.py

🧪 離線煙霧測試（Smoke Test）：驗證 ScannerEngine.scan() 的欄位契約

用途：
    這次開發過程中發現一個 bug ——PortfolioEngine.build_portfolio() 假設
    result_df 有「產業」欄位，但 ScannerEngine.scan() 實際上不會產生這個
    欄位，導致要實際點過「建立建議投資組合」按鈕才會發現。這種「下游功能
    假設某欄位存在，但上游沒有產生」的問題，只有真的跑過那個特定操作
    順序才會現形。

    這支測試腳本用「跟 ScannerEngine.scan() 真實輸出結構完全一致」的合成
    資料，把所有依賴 result_df 的下游功能都跑一遍，檢查欄位契約有沒有對上
    ——不需要網路、不需要真實資料，任何時候改動 scanner_engine.py 或任何
    消費 result_df 的引擎後，都可以先跑這支測試，及早抓到欄位不對應的問題。

用法：
    把這個檔案放在專案根目錄（跟 app.py 同一層），確保 engines/ 資料夾在
    同一層可以被 import，然後執行：

        python test_scan_dependent_features.py

    全部通過會印出 "✅ 全部測試通過"；任何一項失敗會印出詳細錯誤原因，
    並且整支腳本會以非 0 狀態碼結束（方便串進 CI 或簡單的 pre-commit check）。

⚠️ 這不是取代真實環境驗證的萬能藥——這裡只驗證「函式之間的欄位契約」跟
「基本邏輯不會拋例外」，不驗證外部資料源(TWSE/TDCC/TAIFEX/yfinance)本身
會不會回傳正確格式的真實資料（那類問題只能在真實環境、真實網路連線下
才會出現，例如這次遇到的 TWSE 欄位改名，這支腳本測不出來）。
"""

import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def _build_synthetic_scan_result() -> pd.DataFrame:
    """
    產生一份欄位結構跟 ScannerEngine.scan() 真實輸出完全一致的合成資料。

    ⚠️ 這份欄位清單是從 scanner_engine.py 的 scan() 方法裡，results.append({...})
    那段程式碼逐一比對抄下來的權威清單。如果之後 scanner_engine.py 的輸出欄位
    有異動（新增/刪除/改名），這裡也要同步更新，否則這支測試會失去意義
    （測的是舊契約，不是新契約）。
    """
    return pd.DataFrame([
        {
            "排名": 1, "代碼": "2330", "名稱": "台積電", "標的": "[2330] 台積電",
            "市場別": "上市/上櫃", "收盤價": 950.0, "市場狀態": "📈 穩健多頭趨勢",
            "AI Score": 92.0, "飆股評分": 88.0, "飆股等級": "A",
            "買進訊號": "🟢 買進訊號：AI Score達標且近期無背離/誘多警報",
            "賣出訊號": "⚪ 無明確賣出訊號", "誘盤警報": "—",
            "信心度": 85.0, "資料品質": 98.0, "操作建議": "🎯 積極作多",
            "短線建議": "偏多操作", "波段建議": "偏多操作", "長線建議": "偏多操作",
            "未來走向": "偏多", "年化波動率": 25.0, "60日回撤": -8.5,
            "選股評級": "A", "選股評分": 88, "評級標籤": "優選",
            "市場面評分": 18, "基本面評分": 16, "技術面評分": 17,
            "籌碼面評分": 14, "財務面評分": 15,
        },
        {
            "排名": 2, "代碼": "2317", "名稱": "鴻海", "標的": "[2317] 鴻海",
            "市場別": "上市/上櫃", "收盤價": 105.0, "市場狀態": "🔄 低波盤整區",
            "AI Score": 78.0, "飆股評分": 65.0, "飆股等級": "B",
            "買進訊號": "⚪ 無明確買進訊號", "賣出訊號": "⚪ 無明確賣出訊號",
            "誘盤警報": "—", "信心度": 60.0, "資料品質": 95.0, "操作建議": "👀 震盪觀望",
            "短線建議": "中性觀望", "波段建議": "中性觀望", "長線建議": "中性觀望",
            "未來走向": "中性", "年化波動率": 20.0, "60日回撤": -5.2,
            "選股評級": "B", "選股評分": 70, "評級標籤": "中性",
            "市場面評分": 14, "基本面評分": 15, "技術面評分": 13,
            "籌碼面評分": 12, "財務面評分": 16,
        },
        {
            "排名": 3, "代碼": "2454", "名稱": "聯發科", "標的": "[2454] 聯發科",
            "市場別": "上市/上櫃", "收盤價": 1200.0, "市場狀態": "📈 穩健多頭趨勢",
            "AI Score": 95.0, "飆股評分": 90.0, "飆股等級": "A",
            "買進訊號": "🟢 買進訊號：AI Score達標且近期無背離/誘多警報",
            "賣出訊號": "⚪ 無明確賣出訊號", "誘盤警報": "—",
            "信心度": 90.0, "資料品質": 99.0, "操作建議": "🎯 積極作多",
            "短線建議": "偏多操作", "波段建議": "偏多操作", "長線建議": "偏多操作",
            "未來走向": "偏多", "年化波動率": 30.0, "60日回撤": -6.1,
            "選股評級": "A", "選股評分": 92, "評級標籤": "優選",
            "市場面評分": 19, "基本面評分": 18, "技術面評分": 18,
            "籌碼面評分": 15, "財務面評分": 17,
        },
    ])


def _check(name, fn):
    try:
        fn()
        print(f"✅ {name}")
        return True
    except Exception:
        print(f"❌ {name}")
        traceback.print_exc()
        return False


def main():
    result_df = _build_synthetic_scan_result()
    results = []

    # --- IndustryEngine ---
    def t_industry():
        from engines.industry_engine import IndustryEngine
        industry_df = IndustryEngine.rank_industries(result_df)
        assert not industry_df.empty, "產業排名結果不應為空"
        constituents = IndustryEngine.get_industry_constituents(result_df, industry_df.iloc[0]["產業"])
        assert isinstance(constituents, pd.DataFrame)

    results.append(_check("IndustryEngine.rank_industries / get_industry_constituents", t_industry))

    # --- PortfolioEngine：這次真正出問題的地方 ---
    portfolio_result = {}

    def t_portfolio():
        from engines.portfolio_engine import PortfolioEngine
        # 比照 app.py 目前的修法：呼叫前手動確保「產業」欄位存在
        df = result_df.copy()
        if "產業" not in df.columns:
            from engines.industry_engine import IndustryEngine
            df["產業"] = df["代碼"].apply(lambda c: IndustryEngine.get_industry(c))

        r = PortfolioEngine.build_portfolio(df, top_n=3, min_ai_score=70,
                                             max_industry_weight_pct=30, capital=1_000_000)
        assert r["status"] == "ok", f"預期成功，實際狀態：{r['status']}，訊息：{r.get('note')}"
        assert not r["weights_table"].empty
        portfolio_result["value"] = r

    results.append(_check("PortfolioEngine.build_portfolio（含產業欄位）", t_portfolio))

    # --- PortfolioEngine：刻意不補「產業」欄位，驗證會得到清楚的錯誤訊息而非崩潰 ---
    def t_portfolio_missing_industry():
        from engines.portfolio_engine import PortfolioEngine
        r = PortfolioEngine.build_portfolio(result_df, top_n=3)  # 原始 result_df，沒有「產業」欄位
        assert r["status"] != "ok"
        assert "產業" in r["note"], "缺欄位時應該要在錯誤訊息裡明確指出是哪個欄位"

    results.append(_check("PortfolioEngine.build_portfolio（刻意缺產業欄位，應優雅回報而非崩潰）", t_portfolio_missing_industry))

    # --- PortfolioEngine.build_rebalance_plan ---
    def t_rebalance():
        from engines.portfolio_engine import PortfolioEngine
        if "value" not in portfolio_result:
            raise RuntimeError("前面的 build_portfolio 測試沒成功，無法接續測試再平衡")
        target_table = portfolio_result["value"]["weights_table"]
        current_holdings = {"2330": 2, "2317": 15, "9999": 3}  # 9999 是刻意不在名單裡的代碼
        r = PortfolioEngine.build_rebalance_plan(current_holdings, target_table, min_adjust_lots=1)
        assert r["status"] == "ok"
        assert "9999" in r["rebalance_table"]["代碼"].values, "跌出目標名單的持股應該仍出現在再平衡表裡"

    results.append(_check("PortfolioEngine.build_rebalance_plan", t_rebalance))

    # --- ScannerEngine 的輔助方法 ---
    def t_scanner_helpers():
        from engines.scanner_engine import ScannerEngine
        assert not ScannerEngine.get_academy_top_n(result_df, n=3, min_grade="B").empty
        assert not ScannerEngine.get_a_grade_candidates(result_df).empty
        _ = ScannerEngine.get_trap_alerts(result_df)  # 允許為空（本測試資料沒有誘盤警報）
        weakest = ScannerEngine.get_dimension_weakest(result_df)
        assert "最弱維度" in weakest
        consensus = ScannerEngine.get_multi_signal_consensus(result_df)
        assert isinstance(consensus, pd.DataFrame)

    results.append(_check("ScannerEngine 輔助方法（Top-N / A級候選 / 誘盤警報 / 維度分析 / 三信號共識）", t_scanner_helpers))

    # --- app.py 的 ticker-scoped session_state helper（防止切換股票時顯示舊資料） ---
    def t_ticker_scoped_state():
        # 直接複製 app.py 裡兩個 helper 的邏輯做隔離測試，不需要真的啟動 Streamlit。
        class FakeSessionState(dict):
            pass

        def _set_ticker_scoped_state(state, key, ticker, value):
            state[key] = {"_ticker": ticker, "_value": value}

        def _get_ticker_scoped_state(state, key, ticker):
            wrapper = state.get(key)
            if wrapper and wrapper.get("_ticker") == ticker:
                return wrapper.get("_value")
            return None

        state = FakeSessionState()
        _set_ticker_scoped_state(state, "shareholding_report", "2330", {"status": "ok", "large_holder_pct": 68.5})
        assert _get_ticker_scoped_state(state, "shareholding_report", "2330") is not None, "查詢過的股票應該要拿得到資料"
        assert _get_ticker_scoped_state(state, "shareholding_report", "2317") is None, \
            "切換到沒查詢過的股票，不應該顯示上一檔股票的舊資料（這是實際發生過的 bug）"

    results.append(_check("app.py ticker-scoped session_state（防止切換股票顯示舊資料）", t_ticker_scoped_state))

    # --- ChipEngine：董監持股資料在關鍵欄位全部缺值時，應優雅處理而非崩潰或誤導 ---
    def t_chip_engine_nan_safety():
        import sys
        import types

        # 暫時把 requests 換成假的，模擬 TWSE 回傳一列「持股/設質欄位全部缺值」的資料
        # （真實世界可能發生在新上市或資料揭露不完整的公司），測完後還原，
        # 避免影響其他測試或污染全域狀態。
        real_requests = sys.modules.get("requests")
        fake_requests = types.ModuleType("requests")

        csv_text = (
            "出表日期,資料年月,公司代號,公司名稱,職稱,姓名,選任時持股,目前持股,設質股數,"
            "設質股數佔持股比例,內部人關係人目前持股合計,內部人關係人設質股數,內部人關係人設質比例\n"
            "1150515,11504,8888,測試公司,董事長本人,某某,1000,1000,0,N/A,N/A,0,N/A\n"
        )

        class FakeResp:
            text = csv_text
            encoding = None

            def raise_for_status(self):
                pass

        fake_requests.get = lambda *a, **k: FakeResp()
        sys.modules["requests"] = fake_requests

        try:
            for m in list(sys.modules):
                if m.startswith("engines.chip_engine"):
                    del sys.modules[m]
            from engines.chip_engine import ChipEngine

            result = ChipEngine.get_insider_holdings("8888", use_cache=False)
            assert result["status"] == "ok", "即使關鍵欄位全部缺值，也應該優雅回傳結果而不是拋例外"
            assert result["total_insider_holding"] == 0, "缺值時應該明確給0，不是讓 int(NaN) 拋例外"
            assert result["max_pledge_pct"] is None, "設質比例全部缺值時應該回傳 None，不是誤判成健康水位"
        finally:
            if real_requests is not None:
                sys.modules["requests"] = real_requests
            else:
                sys.modules.pop("requests", None)
            for m in list(sys.modules):
                if m.startswith("engines.chip_engine"):
                    del sys.modules[m]

    results.append(_check("ChipEngine 董監持股：關鍵欄位全部缺值時應優雅處理（曾經會崩潰或誤導）", t_chip_engine_nan_safety))

    # --- PortfolioEngine：相關性集中風險檢查（專業風控觀點新增） ---
    def t_correlation_check():
        import sys
        import types

        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        rng = np.random.default_rng(7)
        common_factor = rng.normal(0, 0.02, 100)

        def make_df(corr_with_factor, scale=0.01):
            noise = rng.normal(0, scale, 100)
            ret = corr_with_factor * common_factor + (1 - corr_with_factor) * noise
            price = 100 * np.cumprod(1 + ret)
            return pd.DataFrame({"date": dates, "close": price})

        price_data = {"2330": make_df(0.95), "3374": make_df(0.9), "1101": make_df(0.05)}

        real_data_engine_mod = sys.modules.get("engines.data_engine")
        fake_mod = types.ModuleType("engines.data_engine")

        class FakeDataEngine:
            @staticmethod
            def get_stock_data(code, use_cache=True, max_age_hours=6):
                return price_data.get(code, pd.DataFrame())

            @staticmethod
            def is_tw_code(code):
                return str(code).strip().isdigit()

        fake_mod.DataEngine = FakeDataEngine
        sys.modules["engines.data_engine"] = fake_mod

        try:
            for m in list(sys.modules):
                if m.startswith("engines.portfolio_engine"):
                    del sys.modules[m]
            from engines.portfolio_engine import PortfolioEngine

            r = PortfolioEngine._check_correlation_concentration(["2330", "3374", "1101"])
            assert r["status"] == "ok"
            assert set(r["max_correlation_pair"]) == {"2330", "3374"}, \
                "應該正確抓出相關性最高的一對（合成資料裡故意讓這兩檔高度相關）"
        finally:
            if real_data_engine_mod is not None:
                sys.modules["engines.data_engine"] = real_data_engine_mod
            else:
                sys.modules.pop("engines.data_engine", None)
            for m in list(sys.modules):
                if m.startswith("engines.portfolio_engine"):
                    del sys.modules[m]

    results.append(_check("PortfolioEngine 相關性集中風險檢查（專業風控觀點新增）", t_correlation_check))

    # --- NameEngine 通用ETF判斷 + ETFEngine 槓桿/反向分類 ---
    def t_etf_detection():
        from engines.name_engine import NameEngine
        from engines.etf_engine import ETFEngine

        # 這幾檔不在原本寫死的6檔清單裡，但代碼型態符合台股ETF慣例（00開頭）
        for code in ["00919", "00929", "00646"]:
            assert NameEngine.is_etf(code), f"{code} 應該要被判斷為ETF（通用型態偵測）"

        for code in ["2330", "1101", "2317"]:
            assert not NameEngine.is_etf(code), f"{code} 是一般股票，不應該被誤判為ETF"

        assert ETFEngine.classify_etf_type("00631L", "元大台灣50正2")["type"] == "leveraged"
        assert ETFEngine.classify_etf_type("00632R", "元大台灣50反1")["type"] == "inverse"
        assert ETFEngine.classify_etf_type("0050", "元大台灣50")["type"] == "plain"

    results.append(_check("NameEngine 通用ETF判斷 + ETFEngine 槓桿/反向分類", t_etf_detection))

    print()
    if all(results):
        print("✅ 全部測試通過")
        sys.exit(0)
    else:
        failed = len([r for r in results if not r])
        print(f"❌ 有 {failed} 項測試失敗，請往上翻看詳細錯誤訊息")
        sys.exit(1)


if __name__ == "__main__":
    main()
