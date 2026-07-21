import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as fgo
from engines.data_engine import DataEngine
from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.divergence_engine import DivergenceEngine
from engines.strategy_engine import StrategyEngine
from engines.momentum_engine import MomentumEngine
from engines.evidence_engine import EvidenceEngine
from engines.risk_engine import RiskEngine
from engines.backtest_engine import BacktestEngine
from engines.db_engine import DatabaseEngine
from engines.scanner_engine import ScannerEngine
from engines.chip_engine import ChipEngine
from engines.industry_engine import IndustryEngine
from engines.name_engine import NameEngine
from engines.timeframe_engine import TimeframeEngine
from engines.fundamental_engine import FundamentalEngine
from engines.etf_engine import ETFEngine
from engines.stock_academy_engine import StockAcademyEngine
from engines.stock_directory_engine import StockDirectoryEngine
from engines.macro_engine import MacroEngine
from engines.options_engine import OptionsEngine
from engines.sector_rotation_engine import SectorRotationEngine
from engines.portfolio_engine import PortfolioEngine
from engines.seasonality_engine import SeasonalityEngine
from engines.pattern_engine import PatternEngine
from engines.breakout_engine import BreakoutEngine
from engines.rs_rating_engine import RSRatingEngine
from engines.canslim_engine import CanslimEngine
from engines.stage_engine import StageEngine
from engines.decision_engine import DecisionEngine
from engines.news_engine import NewsEngine
from research.event_inventory import run_inventory

# 設置企業級寬螢幕版面
st.set_page_config(layout="wide", page_title="TQAI Pro Enterprise v2.9.11", page_icon="🏦")


def _resolve_stock_search(raw_input: str):
    """
    ⚠️ 新增功能：查找股票支援輸入中文名稱（不再只能輸入代碼）。

    解析規則：
      1. 輸入本身若符合台股代碼格式（含槓桿/反向ETF字母尾碼，見
         DataEngine.is_tw_code）→ 直接當代碼使用，不做名稱搜尋。
      2. 否則視為名稱搜尋：先查 NameEngine 內建觀察名單（快、不需要
         資料庫或網路），再查 StockDirectoryEngine 的全市場快取（需要
         先執行過 refresh_all() 才有資料，涵蓋範圍更廣，包含觀察名單
         以外的股票）。兩邊結果依代碼去重合併。

    回傳：(resolved_code, candidates, error_message)
      - 剛好一筆符合 → resolved_code 有值，其餘為 None
      - 找到多筆 → candidates 為 list，需要使用者從下拉選單挑選
      - 一筆都找不到 → error_message 說明原因
    """
    raw = str(raw_input).strip()
    if not raw:
        return None, None, "請輸入股票代碼或名稱"

    if DataEngine.is_tw_code(raw):
        return raw, None, None

    matches = list(NameEngine.search_by_name(raw))
    seen_codes = {m["code"] for m in matches}

    try:
        dir_df = StockDirectoryEngine.search_by_name(raw)
        for _, row in dir_df.iterrows():
            code = str(row["code"])
            if code not in seen_codes:
                matches.append({
                    "code": code,
                    "name": row.get("name", ""),
                    "market": row.get("market", ""),
                })
                seen_codes.add(code)
    except Exception:
        pass  # 全市場目錄尚未建立或查詢失敗，不影響內建觀察名單的搜尋結果

    if len(matches) == 1:
        return matches[0]["code"], None, None
    elif len(matches) > 1:
        return None, matches, None
    else:
        return None, None, (
            f"⚠️ 查無名稱包含「{raw}」的股票。可能是：(1) 全市場名稱目錄尚未建立"
            f"（需先執行過 StockDirectoryEngine.refresh_all()），只能搜尋到內建觀察"
            f"名單裡的股票；或 (2) 名稱打錯字。請確認名稱，或改用代碼查詢。"
        )


def _resolve_custom_scan_list(raw_list_str: str):
    """
    ⚠️ 新增功能：「全台股掃描」的自訂清單原本只吃逗號分隔的代碼
    （custom_list.split(",")），輸入中文名稱一律查無資料。現在跟個股搜尋框
    共用同一套解析邏輯（見上方 _resolve_stock_search），逗號分隔的每一項都
    可以是代碼或中文名稱，兩者也可以混用（例如 "2330,鴻海,00631L"）。

    批次輸入的情境跟單一搜尋框不同——沒有版面可以讓使用者針對每個模糊的
    項目跳出下拉選單挑選，所以規則是：
      1. 符合台股代碼格式 → 直接當代碼使用。
      2. 名稱剛好解析出唯一一檔 → 自動轉成代碼。
      3. 名稱解析出多筆候選 → 不猜測，直接跳過這一項，並回傳警告訊息列出
         候選清單，讓使用者自己改成更精確的名稱或直接填代碼，避免掃描到
         使用者沒有意圖要看的股票。
      4. 完全查無 → 跳過這一項並回傳警告。

    回傳 (resolved_tickers: list[str], warnings: list[str])
    """
    items = [x.strip() for x in str(raw_list_str).split(",") if x.strip()]
    resolved = []
    warnings = []
    for item in items:
        if DataEngine.is_tw_code(item):
            resolved.append(item)
            continue

        code, candidates, _err = _resolve_stock_search(item)
        if code:
            resolved.append(code)
        elif candidates:
            shown = candidates[:5]
            option_str = "、".join(f"[{c['code']}]{c['name']}" for c in shown)
            more = f" 等共 {len(candidates)} 筆" if len(candidates) > 5 else ""
            warnings.append(f"⚠️「{item}」找到多筆符合的股票（{option_str}{more}），為避免誤判已跳過此項，請改用代碼或更精確的名稱。")
        else:
            warnings.append(f"⚠️ 查無「{item}」對應的股票，已跳過此項。")

    return resolved, warnings


def _set_ticker_scoped_state(key: str, ticker: str, value):
    """
    ⚠️ 修正說明：大戶持股／董監持股／季節循環分析／短線命中率這幾個子功能的
    查詢結果，原本直接存在跟股票代碼無關的固定 session_state key 底下。
    情境重現：分析 2330 時查了大戶持股，接著換成分析 2317 並重新啟動分析，
    這個子功能區塊會「繼續顯示 2330 的舊資料」（因為沒有被清除，也不會
    自動重新查詢），而且更容易造成混淆的是：畫面上其他有用到當下 `ticker`
    變數的部分（例如大戶持股歷史趨勢圖）會正確顯示 2317 的資料，變成同一
    個畫面混雜兩檔股票的資訊。

    修法：把查詢當下的股票代碼一起存進去，讀取時比對是否跟目前分析中的
    股票一致，不一致就視為「這檔股票還沒查詢過」，不會顯示舊資料。
    """
    st.session_state[key] = {"_ticker": ticker, "_value": value}


def _get_ticker_scoped_state(key: str, ticker: str):
    wrapper = st.session_state.get(key)
    if wrapper and wrapper.get("_ticker") == ticker:
        return wrapper.get("_value")
    return None



with st.sidebar:
    st.header("⚙️ TQAI 決策中樞")
    mode = st.radio("功能選擇", ["🔍 個股深度分析", "📡 全台股掃描", "🏆 全市場排行榜",
                                 "🌍 總經戰情室", "📰 自選股新聞中心", "🔬 事件研究實驗室"])
    st.markdown("---")
    
    if mode == "🔍 個股深度分析":
        search_input = st.text_input(
            "輸入股票代碼或中文名稱 (例: 2330 或 台積電)", value="2330"
        )

        # ⚠️ 新增：中文名稱搜尋解析（見上方 _resolve_stock_search 說明）。
        # 這裡在每次腳本重跑時都重新解析（Streamlit 的標準模式），不需要
        # 額外的 session_state 狀態機，行為簡單且不會有過期快取的問題。
        ticker, name_candidates, resolve_error = _resolve_stock_search(search_input)

        if name_candidates:
            option_labels = [
                f"[{c['code']}] {c['name']}" + (f"（{c['market']}）" if c.get('market') else "")
                for c in name_candidates
            ]
            chosen_label = st.selectbox(
                f"🔎「{search_input.strip()}」找到 {len(name_candidates)} 筆符合的股票，請選擇：",
                option_labels,
            )
            ticker = name_candidates[option_labels.index(chosen_label)]["code"]
        elif resolve_error:
            st.warning(resolve_error)
        elif ticker and not str(search_input).strip().isdigit():
            # 純數字輸入不需要提示（使用者本來就是打代碼），但用中文名稱
            # 解析出唯一結果時，顯示解析結果讓使用者確認打的是對的股票。
            resolved_name = NameEngine.get_name(ticker)
            st.caption(f"🔎 已將「{search_input.strip()}」解析為 [{ticker}] {resolved_name}")

        run_btn = st.button(
            "🚀 啟動 AI 多智能體分析", use_container_width=True, type="primary",
            disabled=(not ticker),
        )
        use_cache = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True)
        force_refresh = st.button("🔄 強制重新抓取最新資料", use_container_width=True, disabled=(not ticker))
        scan_btn = False
        rank_btn = False
        macro_btn = False
        news_center_btn = False
        research_btn = False
    elif mode == "📡 全台股掃描":
        st.caption("預設使用台股常見權值股／熱門股觀察名單，也可以自行輸入想掃描的股票代碼。")
        custom_list = st.text_area(
            "自訂股票清單（逗號分隔，可混用代碼與中文名稱，留空則使用預設清單）",
            value="", placeholder="例如：2330,鴻海,00631L"
        )
        top_n = st.slider("顯示前 N 名", min_value=5, max_value=30, value=10)
        use_cache_scan = st.checkbox("🗄️ 使用資料庫快取 (建議開啟)", value=True, key="scan_cache")
        scan_btn = st.button("📡 啟動全台股掃描", use_container_width=True, type="primary")
        run_btn = False
        force_refresh = False
        rank_btn = False
        macro_btn = False
        news_center_btn = False
        research_btn = False
    elif mode == "🏆 全市場排行榜":
        st.caption("抓取當日（或最近交易日）全市場三大法人買賣超排行榜，僅涵蓋上市股票。")
        rank_top_n = st.slider("每類別顯示前 N 名", min_value=5, max_value=50, value=20, key="rank_top_n")
        rank_btn = st.button("🏆 抓取最新排行榜", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        force_refresh = False
        macro_btn = False
        news_center_btn = False
        research_btn = False
    elif mode == "🌍 總經戰情室":
        st.caption(
            "🌍 總經戰情室：抓取美元指數／美元台幣／VIX／黃金／原油／美國十年期公債殖利率／"
            "那斯達克／標普500／費半(SOX)／台灣加權指數，作為判斷大盤環境的背景參考。"
        )
        st.caption("⚠️ 這裡的訊號是常見經驗法則的方向性參考，不是嚴謹統計檢定，也不會用來計算任何個股的 AI Score。")
        macro_btn = st.button("🌍 抓取最新總經數據", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        rank_btn = False
        force_refresh = False
        news_center_btn = False
        research_btn = False
    elif mode == "📰 自選股新聞中心":
        st.caption(
            "📰 自選股新聞中心：彙總你在「交易日誌」功能裡標記過狀態的自選股"
            "（讀取既有的持股狀態紀錄），逐檔查新聞篇數與情緒，做成每日摘要。"
        )
        st.caption("⚠️ 只彙總你自己的自選股清單，不是大盤或美股/Fed等國際總經新聞，範圍界定見 NewsEngine 說明。")
        news_center_btn = st.button("📰 更新自選股新聞總覽", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        rank_btn = False
        macro_btn = False
        force_refresh = False
        research_btn = False
    else:
        st.caption(
            "🔬 事件研究實驗室（Research Lab）：對現有 BreakoutEngine 的箱型突破訊號"
            "做歷史事件盤點（Phase A Event Inventory）——只統計「發生過幾次、分布在哪、"
            "有量/無量確認各占多少」，不計算報酬率、不做回測，也不會影響 AI Score 或任何"
            "既有的個股分析/決策邏輯。"
        )
        st.caption(
            "⚠️ 全市場掃描（約1000~1700檔）保守估計要數十分鐘以上（純程式運算成本，"
            "非網路延遲），建議先用「觀察名單」跑過一次確認邏輯與資料格式正常，"
            "再考慮全市場。中途可以關閉分頁，之後勾選「續跑」從上次進度接著跑。"
        )
        ei_universe_label = st.radio(
            "掃描範圍", ["觀察名單（快，建議先跑）", "全市場（慢，需先建立代碼目錄）"],
            key="ei_universe",
        )
        ei_limit = st.number_input(
            "限制處理檔數（0 = 不限制，測試用）", min_value=0, value=0, step=5, key="ei_limit",
        )
        ei_resume = st.checkbox(
            "續跑（讀取上次進度的 checkpoint，中斷後不用整批重跑）", value=True, key="ei_resume",
        )
        ei_sleep = st.slider(
            "每檔股票間隔秒數（避免短時間內對 yfinance/TWSE 打太快）",
            min_value=0.0, max_value=2.0, value=0.3, step=0.1, key="ei_sleep",
        )
        research_btn = st.button("🔬 開始事件盤點 (Phase A)", use_container_width=True, type="primary")
        run_btn = False
        scan_btn = False
        rank_btn = False
        macro_btn = False
        news_center_btn = False
    
    st.markdown("---")
    try:
        db_stats = DatabaseEngine.get_db_stats()
        st.caption(f"📦 快取股票數: {db_stats['cached_tickers']} 檔 ／ {db_stats['total_rows']} 筆K線")
    except Exception:
        st.caption("📦 資料庫尚未初始化（首次查詢後會自動建立）")
    
    st.markdown("---")
    st.caption("架構版本: TQAI Pro v2.5 (Bridge)")
    st.caption("核心引擎: 13-Layer Multi-Agent")

if mode == "🔍 個股深度分析" and force_refresh and ticker:
    try:
        DatabaseEngine.clear_cache(str(ticker).strip())
        st.sidebar.success(f"已清除 {ticker} 的快取，請重新點擊分析按鈕")
    except Exception:
        pass

# ==========================================
# 📡 全台股掃描模式
# ==========================================
# ⚠️ 修正說明：原本整個結果顯示區塊都包在 `if scan_btn:` 底下。st.button() 的
# True 狀態只維持「按下當下觸發的那一次 rerun」，使用者接下來只要跟頁面上
# 任何其他元件互動（例如下面的產業下拉選單），Streamlit 就會重新執行整個
# script，這時候 scan_btn 已經變回 False，整個 if 區塊（包含下拉選單本身
# 依賴的資料）就不會被渲染，導致「切換產業看不到結果」。
# 修正做法：按下按鈕時把掃描結果寫進 st.session_state，顯示邏輯改成
# 「只要 session_state 裡有資料就渲染」，不再綁定按鈕的瞬間狀態。
if mode == "📡 全台股掃描":
    if scan_btn:
        if custom_list.strip():
            scan_tickers, resolve_warnings = _resolve_custom_scan_list(custom_list)
            for w in resolve_warnings:
                st.warning(w)
            if not scan_tickers:
                st.error("⚠️ 自訂清單解析後沒有任何有效的股票代碼，請確認輸入內容，或留空使用預設清單。")
        else:
            scan_tickers = ScannerEngine.DEFAULT_WATCHLIST

        if scan_tickers:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _update_progress(done, total, current_ticker):
                progress_bar.progress(done / total)
                status_text.caption(f"掃描中：{current_ticker} ({done}/{total})")

            result_df, error_df = ScannerEngine.scan(
                tickers=scan_tickers, use_cache=use_cache_scan, progress_callback=_update_progress
            )

            progress_bar.empty()
            status_text.empty()

            st.session_state["scan_result_df"] = result_df
            st.session_state["scan_error_df"] = error_df
            st.session_state["scan_ticker_count"] = len(scan_tickers)

    if "scan_result_df" in st.session_state:
        result_df = st.session_state["scan_result_df"]
        error_df = st.session_state["scan_error_df"]

        # ⚠️ 修正說明：ScannerEngine.scan() 的原始回傳結果並沒有「產業」欄位——
        # 這欄位原本只在 IndustryEngine.rank_industries() 內部算給畫面顯示用，
        # 沒有寫回 result_df 本身。PortfolioEngine.build_portfolio() 需要這個
        # 欄位做產業集中度上限計算，原本呼叫時才發現缺欄位而失敗
        # （錯誤訊息:「缺少必要欄位：產業」）。這裡在 result_df 一取出來就
        # 統一補上，讓後面所有用到它的地方（產業排名／產業輪動／投資組合
        # 建構）都能一致地拿到這個欄位，不用各自重複計算。
        if not result_df.empty and "產業" not in result_df.columns:
            result_df = result_df.copy()
            result_df["產業"] = result_df["代碼"].apply(lambda code: IndustryEngine.get_industry(code))

        st.markdown(f"## 📡 全台股掃描戰情室 (共 {st.session_state['scan_ticker_count']} 檔)")

        if result_df.empty:
            st.error("掃描失敗，所有股票皆無法取得資料，請確認代碼是否正確或網路連線。")
        else:
            st.success(f"✅ 掃描完成，成功 {len(result_df)} 檔，失敗 {len(error_df)} 檔。")

            st.markdown(f"### 🏆 Top {top_n} 排行榜（依 AI Score 排序）")
            top_df = ScannerEngine.get_top_n(result_df, top_n)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🚦 Hard Gate 篩選漏斗 (v2.9.10 新增)")
            st.caption(
                "⚠️ 這是對本次掃描結果做「事後依序篩選」，不是重跑計算——每一關淘汰"
                "掉的股票就不會進入下一關，讓你清楚看到候選名單是怎麼從全部標的"
                "一路篩到最後幾檔的，缺值的股票在該關一律放行（資料不足不等於不合格）。"
            )
            gate_col1, gate_col2, gate_col3, gate_col4, gate_col5 = st.columns(5)
            with gate_col1:
                gate_bullish = st.checkbox("Gate1 排除空頭", value=True, key="gate_bullish")
            with gate_col2:
                gate_liquid = st.checkbox("Gate2 排除極低流動性", value=True, key="gate_liquid")
            with gate_col3:
                gate_canslim = st.number_input("Gate3 CAN SLIM≥", min_value=0, max_value=100, value=60, key="gate_canslim")
            with gate_col4:
                gate_rs = st.number_input("Gate4 RS Rating≥", min_value=0, max_value=99, value=70, key="gate_rs")
            with gate_col5:
                gate_grade = st.selectbox("Gate5 飆股等級≥", ["D", "C", "B", "A"], index=2, key="gate_grade")

            gate_result = ScannerEngine.apply_hard_gates(
                result_df, require_bullish_market=gate_bullish, exclude_illiquid=gate_liquid,
                min_canslim_score=gate_canslim, min_rs_rating=gate_rs, min_momentum_grade=gate_grade,
            )
            funnel_df = pd.DataFrame(gate_result["funnel"])
            if not funnel_df.empty:
                fcol1, fcol2 = st.columns([2, 1])
                with fcol1:
                    st.dataframe(funnel_df.rename(columns={"gate": "關卡", "count": "剩餘檔數"}),
                                 use_container_width=True, hide_index=True)
                with fcol2:
                    st.metric("最終通過候選數", f"{len(gate_result['passed'])} / {len(result_df)}")
            if not gate_result["passed"].empty:
                gate_show_cols = [c for c in ["排名", "代碼", "標的", "產業", "AI Score", "市場狀態",
                                               "流動性", "CANSLIM評分", "RS Rating", "飆股等級"]
                                   if c in gate_result["passed"].columns]
                st.dataframe(gate_result["passed"][gate_show_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("目前設定的關卡門檻下，沒有標的能通過全部五關。")
            with st.expander(f"🔍 查看被淘汰的 {len(gate_result['rejected'])} 檔標的與淘汰原因"):
                if not gate_result["rejected"].empty:
                    reject_show_cols = [c for c in ["代碼", "標的", "淘汰關卡"] if c in gate_result["rejected"].columns]
                    st.dataframe(gate_result["rejected"][reject_show_cols], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🚀 A級飆股候選 (Momentum A-Grade Candidates)")
            st.caption("依「飆股七層過濾＋100分評分系統（含年線濾網）」篩選，總分需 ≥ 85 分且已站穩年線才會列為 A 級。")
            a_grade_df = ScannerEngine.get_a_grade_candidates(result_df)
            if a_grade_df.empty:
                st.caption("目前掃描名單中沒有符合 A 級門檻的標的。")
            else:
                # 版本相容性保護：只顯示目前 result_df 實際存在的欄位，避免
                # scanner_engine.py 版本沒對齊（例如缺少新版才有的短線/波段/
                # 長線/未來走向欄位）時整頁直接 KeyError 崩潰。
                preferred_cols = ["排名", "標的", "市場別", "收盤價", "飆股評分", "飆股等級", "AI Score",
                                  "買進訊號", "誘盤警報", "操作建議", "短線建議", "波段建議", "長線建議", "未來走向"]
                show_cols = [c for c in preferred_cols if c in a_grade_df.columns]
                missing_cols = [c for c in preferred_cols if c not in a_grade_df.columns]
                if missing_cols:
                    st.caption(f"⚠️ 偵測到 scanner_engine.py 版本較舊，缺少欄位：{', '.join(missing_cols)}（請確認已更新 engines/scanner_engine.py 與 engines/timeframe_engine.py 並重啟服務）")
                st.dataframe(a_grade_df[show_cols], use_container_width=True, hide_index=True)
                st.caption("⚠️ A級代表『當下技術結構符合飆股七層過濾條件』，不代表保證持續噴出，仍請留意下方誘盤警報雷達與個股風險。")

            st.markdown("---")
            st.markdown("### 🛡️ 誘盤警報雷達 (Fake-Signal / Trap Radar)")
            st.caption("彙整近期觸發 MACD 背離或假突破/假跌破（誘多/誘空）警報的標的，依 AI Score 排序。")
            trap_df = ScannerEngine.get_trap_alerts(result_df)
            if trap_df.empty:
                st.caption("✅ 掃描名單中目前無標的觸發背離或假突破/假跌破警報。")
            else:
                preferred_cols_trap = ["排名", "標的", "市場別", "收盤價", "AI Score", "飆股評分",
                                       "誘盤警報", "賣出訊號", "操作建議", "短線建議", "波段建議", "未來走向"]
                show_cols_trap = [c for c in preferred_cols_trap if c in trap_df.columns]
                missing_cols_trap = [c for c in preferred_cols_trap if c not in trap_df.columns]
                if missing_cols_trap:
                    st.caption(f"⚠️ 偵測到 scanner_engine.py 版本較舊，缺少欄位：{', '.join(missing_cols_trap)}（請確認已更新 engines/scanner_engine.py 與 engines/timeframe_engine.py 並重啟服務）")
                st.dataframe(trap_df[show_cols_trap], use_container_width=True, hide_index=True)
                st.caption("⚠️ 誘盤警報代表近期偵測到MACD背離或價格假突破/假跌破收回，動能可能已經或即將失效，請留意追高/殺低風險。")

            st.markdown("---")
            st.markdown("### 🏭 產業中心 (Industry Center) — 產業強弱排名")
            industry_df = IndustryEngine.rank_industries(result_df)
            if industry_df.empty:
                st.caption("⚠️ 尚無法產生產業排名（掃描結果為空，或股票代碼皆不在預設產業對照表中）。")
            else:
                st.dataframe(industry_df, use_container_width=True, hide_index=True)

                industry_options = industry_df["產業"].tolist()
                selected_industry = st.selectbox(
                    "🔍 點選查看該產業內的成分股表現", industry_options, key="industry_select"
                )
                if selected_industry:
                    constituents = IndustryEngine.get_industry_constituents(result_df, selected_industry)
                    st.dataframe(constituents, use_container_width=True, hide_index=True)

                st.markdown("#### 🔁 產業輪動觀察（時間序列版）")
                st.caption(
                    "⚠️ 上面的產業排名是「這次掃描當下」的單一快照；這裡改用本次掃描名單裡每檔股票"
                    "的歷史股價，計算「等權重」平均報酬曲線，觀察最近5日/20日/60日哪個產業動能"
                    "轉強或轉弱。這不是市值加權的真正產業指數，樣本也僅限本次掃描名單，僅供參考。"
                )
                rotation_btn = st.button("🔁 計算產業輪動（讀取歷史股價）", key="rotation_btn")
                if rotation_btn:
                    with st.spinner("計算產業輪動中..."):
                        try:
                            scan_codes = result_df["代碼"].astype(str).tolist()
                            st.session_state["rotation_report"] = SectorRotationEngine.build_rotation_report(
                                tickers=scan_codes, use_cache=use_cache_scan
                            )
                        except Exception as e:
                            st.session_state["rotation_report"] = {"status": "unavailable", "error": str(e)}

                rotation_report = st.session_state.get("rotation_report")
                if rotation_report and rotation_report.get("status") == "ok":
                    st.dataframe(rotation_report["rotation_table"], use_container_width=True, hide_index=True)
                    st.line_chart(rotation_report["return_curves"])
                    if rotation_report.get("failed_tickers"):
                        st.caption(f"⚠️ 以下股票歷史股價抓取失敗，未列入產業輪動計算：{', '.join(rotation_report['failed_tickers'])}")
                elif rotation_report and rotation_report.get("status") == "unavailable":
                    st.caption("⚠️ 目前資料不足以計算產業輪動（可能是掃描名單股票太少、都是ETF/未分類，或歷史資料不足）。")

            st.markdown("---")
            st.markdown("### 🎓 選股大師 Top 候選 (五維度評分，快速模式)")
            st.caption("⚠️ 批次掃描為了效能，刻意不對每檔股票額外呼叫籌碼/基本面外部資料源，籌碼面改用OBV/量能代理、基本面與財務面顯示中性分數，僅供初步篩選；決定下單前建議點進「個股深度分析」查看該股完整的五維度評分（會自動帶入真實籌碼與基本面資料）。")
            academy_top_df = ScannerEngine.get_academy_top_n(result_df, n=top_n, min_grade="B")
            if academy_top_df.empty:
                st.caption("目前掃描名單中沒有符合 B 級以上門檻的標的（快速模式下基本面/財務面為中性分數，門檻天生較難達到）。")
            else:
                preferred_cols_ac = ["排名", "標的", "市場別", "收盤價", "選股評級", "選股評分",
                                     "市場面評分", "基本面評分", "技術面評分", "籌碼面評分", "財務面評分", "AI Score", "飆股等級"]
                show_cols_ac = [c for c in preferred_cols_ac if c in academy_top_df.columns]
                st.dataframe(academy_top_df[show_cols_ac], use_container_width=True, hide_index=True)

            dim_weakest = ScannerEngine.get_dimension_weakest(result_df)
            if dim_weakest:
                st.caption(f"📊 本次掃描名單維度分析：最弱維度為「{dim_weakest['最弱維度']}」(平均 {dim_weakest['最弱維度平均分']} 分)，最強維度為「{dim_weakest['最強維度']}」(平均 {dim_weakest['最強維度平均分']} 分)。")

            st.markdown("---")
            st.markdown("### ⚡ 三信號共識 (AI Score + 飆股評分 + 選股評級 三者皆看好)")
            st.caption("三套評分系統分別回答短期/動能/中長期三個不同時間尺度的問題，三者同時共識代表短中長期角度一致，非保證獲利。")
            consensus_df = ScannerEngine.get_multi_signal_consensus(result_df)
            if consensus_df.empty:
                st.caption("目前掃描名單中沒有三個信號同時共識看好的標的。")
            else:
                preferred_cols_cs = ["排名", "標的", "市場別", "收盤價", "AI Score", "飆股評分", "選股評分", "共識強度", "選股評級", "飆股等級"]
                show_cols_cs = [c for c in preferred_cols_cs if c in consensus_df.columns]
                st.dataframe(consensus_df[show_cols_cs], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 💼 投資組合建構 (Portfolio Construction)")
            st.caption(
                "⚠️ 規則式（反波動度加權＋產業集中度上限）資金配置建議，不是嚴謹的最佳化結果，"
                "也不是下單指令，僅供參考起點，下單前請自行覆核。"
            )
            pf_col1, pf_col2, pf_col3, pf_col4 = st.columns(4)
            with pf_col1:
                pf_top_n = st.number_input("最多持股檔數", min_value=2, max_value=30, value=10, key="pf_top_n")
            with pf_col2:
                pf_min_score = st.number_input("AI Score 門檻", min_value=0, max_value=100, value=70, key="pf_min_score")
            with pf_col3:
                pf_industry_cap = st.number_input("單一產業權重上限(%)", min_value=10, max_value=100, value=30, key="pf_industry_cap")
            with pf_col4:
                pf_capital = st.number_input("預計投入資金(元)", min_value=10000, value=1000000, step=10000, key="pf_capital")
            pf_exclude_illiquid = st.checkbox(
                "排除極低流動性標的 (🔴，v2.9.7 新增)", value=True, key="pf_exclude_illiquid",
                help="日均成交值過低的股票，建議權重算得再精確也很難照建議金額進出，預設排除。"
            )

            if st.button("💼 建立建議投資組合", key="pf_build_btn"):
                try:
                    st.session_state["portfolio_result"] = PortfolioEngine.build_portfolio(
                        result_df, top_n=pf_top_n, min_ai_score=pf_min_score,
                        max_industry_weight_pct=pf_industry_cap, capital=pf_capital,
                        exclude_illiquid=pf_exclude_illiquid,
                    )
                except Exception as e:
                    st.session_state["portfolio_result"] = {"status": "empty", "note": f"⚠️ 建立投資組合時發生錯誤：{e}"}

            portfolio_result = st.session_state.get("portfolio_result")
            if portfolio_result:
                if portfolio_result["status"] != "ok":
                    st.warning(portfolio_result["note"])
                else:
                    st.dataframe(portfolio_result["weights_table"], use_container_width=True, hide_index=True)
                    pf_dcol1, pf_dcol2 = st.columns(2)
                    with pf_dcol1:
                        st.markdown("**產業集中度分布**")
                        st.dataframe(portfolio_result["industry_breakdown"], use_container_width=True, hide_index=True)
                    with pf_dcol2:
                        st.metric("實際配置金額", f"NT$ {portfolio_result['total_allocated']:,.0f}")
                        st.metric("零股/未配置餘額", f"NT$ {portfolio_result['cash_remaining']:,.0f}")
                    st.caption(portfolio_result["note"])

                    # v2.9.7 新增：極低流動性排除清單
                    _illiquid_excl = portfolio_result.get("illiquid_excluded")
                    if _illiquid_excl is not None and not _illiquid_excl.empty:
                        with st.expander(f"🔴 已排除 {len(_illiquid_excl)} 檔極低流動性標的"):
                            st.dataframe(_illiquid_excl, use_container_width=True, hide_index=True)

                    corr_check = portfolio_result.get("correlation_check", {})
                    if corr_check.get("status") == "ok":
                        with st.expander("🔬 相關性集中風險檢查（專業風控觀點，補足產業上限防不住的風險）"):
                            st.caption(
                                "⚠️ 產業集中度上限只防得住「掛在同一個官方產業分類」的風險，防不住「不同產業"
                                "但實際上齊漲齊跌」的相關性風險。這裡額外計算候選股歷史報酬的相關性。"
                            )
                            cc1, cc2 = st.columns(2)
                            cc1.metric("平均兩兩相關係數", corr_check["avg_correlation"])
                            cc2.metric("最高相關的一對", f"{corr_check['max_correlation_pair'][0]}-{corr_check['max_correlation_pair'][1]}",
                                       delta=f"{corr_check['max_correlation']}")
                            for f in corr_check["flags"]:
                                st.markdown(f"- {f}")

                    # v2.9.6 新增：Portfolio Beta / VaR 匯總
                    with st.expander("📐 投資組合層級 Beta / VaR 匯總（v2.9.6 新增）"):
                        if st.button("計算組合 Beta / VaR", key="pf_risk_btn"):
                            with st.spinner("計算中，需個別抓取每檔候選股與大盤基準資料..."):
                                st.session_state["portfolio_risk_result"] = PortfolioEngine.compute_portfolio_risk(
                                    portfolio_result["weights_table"]
                                )
                        pf_risk = st.session_state.get("portfolio_risk_result")
                        if pf_risk:
                            if pf_risk.get("status") == "ok":
                                prc1, prc2 = st.columns(2)
                                prc1.metric("Portfolio Beta（加權平均，精確值）", f"{pf_risk['portfolio_beta']:.2f}")
                                prc2.metric("Portfolio VaR 95%（保守上界近似）", f"{pf_risk['portfolio_var_95_pct']:.2f}%")
                                st.dataframe(pf_risk["per_stock"], use_container_width=True, hide_index=True)
                            st.caption(pf_risk.get("note", ""))

                    st.markdown("#### 🔄 投資組合再平衡")
                    st.caption(
                        "⚠️ 輸入你目前實際持有的股票與張數，跟上面的目標配置比對，算出需要加碼/減碼的標的。"
                        "這是靜態快照比較，不考慮交易成本或稅務影響，不構成投資建議。"
                    )
                    holdings_input = st.text_area(
                        "目前持股（代碼:張數，逗號分隔）", value="",
                        placeholder="例如：2330:5,2317:10,3711:3", key="rebalance_holdings_input",
                    )
                    min_adjust_input = st.number_input(
                        "調整門檻（張數差距小於此不建議調整）", min_value=1, max_value=50, value=1,
                        key="rebalance_min_lots",
                    )

                    if st.button("🔄 計算再平衡建議", key="rebalance_btn"):
                        try:
                            holdings = {}
                            for item in holdings_input.split(","):
                                item = item.strip()
                                if not item or ":" not in item:
                                    continue
                                code, lots = item.split(":", 1)
                                code = code.strip()
                                if code:
                                    holdings[code] = int(float(lots.strip()))
                            st.session_state["rebalance_result"] = PortfolioEngine.build_rebalance_plan(
                                holdings, portfolio_result["weights_table"], min_adjust_lots=min_adjust_input,
                            )
                        except Exception as e:
                            st.session_state["rebalance_result"] = {"status": "empty", "note": f"⚠️ 計算再平衡時發生錯誤：{e}"}

                    rebalance_result = st.session_state.get("rebalance_result")
                    if rebalance_result:
                        if rebalance_result.get("status") != "ok":
                            st.warning(rebalance_result.get("note", "⚠️ 無法計算再平衡。"))
                        else:
                            st.dataframe(rebalance_result["rebalance_table"], use_container_width=True, hide_index=True)
                            st.caption(rebalance_result["note"])

            with st.expander("📋 查看完整掃描結果"):
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            if not error_df.empty:
                with st.expander(f"⚠️ 掃描失敗清單 ({len(error_df)} 檔)"):
                    st.dataframe(error_df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 請在左側設定掃描清單，然後點擊「📡 啟動全台股掃描」。")

# ==========================================
# 🏆 全市場排行榜模式（v2.8 新增，對應選股學院「排行榜選股法」）
# ==========================================
if mode == "🏆 全市場排行榜":
    if rank_btn:
        with st.spinner("抓取 TWSE 全市場三大法人買賣超資料中..."):
            try:
                st.session_state["rank_result"] = ChipEngine.get_market_wide_institutional_ranking(top_n=rank_top_n)
            except Exception as e:
                st.session_state["rank_result"] = {"status": "unavailable", "message": f"⚠️ 排行榜抓取失敗：{e}"}

    rank_result = st.session_state.get("rank_result")

    if rank_result:
        if rank_result.get("status") != "ok":
            st.warning(rank_result.get("message", "⚠️ 暫時無法取得排行榜資料。"))
        else:
            st.success(f"📅 資料日期：{rank_result['date']}（共 {rank_result['total_stocks']} 檔上市股票有三大法人買賣超資料）")
            st.caption("⚠️ 僅涵蓋「上市」股票，上櫃（TPEx）因資料來源不同，暫不支援此排行榜。資料來源：TWSE 三大法人買賣超日報 (T86)。")

            category = st.selectbox("選擇法人類別", ["三大法人合計", "外資", "投信", "自營商"])
            rankings = rank_result["rankings"][category]

            # 把「標的」（[代碼] 名稱，跟其他頁面格式一致）排到最前面顯示，
            # 原始的「代碼」「名稱」欄位保留在 DataFrame 裡（CSV 下載會完整
            # 匯出全部欄位），只是顯示順序上讓「標的」優先。
            display_cols = ["標的", "外資買賣超", "投信買賣超", "自營商買賣超", "三大法人合計買賣超"]

            def _reorder(d):
                cols = [c for c in display_cols if c in d.columns]
                rest = [c for c in d.columns if c not in cols]
                return d[cols + rest]

            buy_df = _reorder(rankings["買超前N名"])
            sell_df = _reorder(rankings["賣超前N名"])

            rk_col1, rk_col2 = st.columns(2)
            with rk_col1:
                st.markdown(f"#### 🟢 {category} 買超前 N 名")
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ 下載買超名單 (CSV)",
                    data=buy_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"institutional_ranking_{category}_buy_{rank_result['date']}.csv",
                    mime="text/csv",
                    key="dl_buy_csv",
                )
            with rk_col2:
                st.markdown(f"#### 🔴 {category} 賣超前 N 名")
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ 下載賣超名單 (CSV)",
                    data=sell_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"institutional_ranking_{category}_sell_{rank_result['date']}.csv",
                    mime="text/csv",
                    key="dl_sell_csv",
                )

            st.caption("💡 提示：連續多日出現在同一類別買超前段班的個股，依選股學院文件經驗，較具備波段上漲潛力；建議搭配「個股深度分析」頁面查看該股的技術面/基本面是否同步支持。")
    else:
        st.info("👈 請在左側點擊「🏆 抓取最新排行榜」。")

# ==========================================
# 🌍 總經戰情室模式（v2.9 新增）
# ==========================================
# ⚠️ 範圍說明：這裡只涵蓋能用 yfinance 真實取得的總經代理指標子集合，
# 詳見 macro_engine.py 的 class docstring。這是「背景參考儀表板」，
# 不會影響任何個股的 ai_score/momentum_score 計算。
if mode == "🌍 總經戰情室":
    if macro_btn:
        with st.spinner("抓取總經與跨市場報價中..."):
            try:
                snapshot = MacroEngine.get_snapshot()
                flags = MacroEngine.build_macro_flags(snapshot)
                st.session_state["macro_snapshot"] = snapshot
                st.session_state["macro_flags"] = flags
            except Exception as e:
                st.session_state["macro_snapshot"] = None
                st.session_state["macro_flags"] = [f"⚠️ 總經資料抓取失敗：{e}"]

    snapshot = st.session_state.get("macro_snapshot")
    flags = st.session_state.get("macro_flags")

    if snapshot:
        st.markdown("## 🌍 總經戰情室")

        rows = []
        for name, item in snapshot.items():
            if item.get("status") != "ok":
                rows.append({"指標": name, "代碼": item.get("symbol", ""), "最新值": "N/A",
                             "日變動%": None, "5日變動%": None, "20日變動%": None})
            else:
                rows.append({
                    "指標": name, "代碼": item["symbol"], "最新值": item["latest"],
                    "日變動%": item["chg_1d_pct"], "5日變動%": item["chg_5d_pct"],
                    "20日變動%": item["chg_20d_pct"],
                })
        macro_df = pd.DataFrame(rows)
        st.dataframe(macro_df, use_container_width=True, hide_index=True)

        failed_items = {name: item for name, item in snapshot.items() if item.get("status") != "ok"}
        if failed_items:
            with st.expander(f"⚠️ 有 {len(failed_items)} 項指標抓取失敗，點擊查看技術細節"):
                for name, item in failed_items.items():
                    st.caption(f"**{name}**（{item.get('symbol', '')}）：{item.get('message', '未知原因')}")

        st.markdown("#### 📋 方向性訊號（經驗法則，僅供參考）")
        for f in (flags or []):
            st.markdown(f"- {f}")

        st.caption("⚠️ 以上規則為市場常見經驗法則，不是嚴謹統計檢定結果，不構成投資建議；資料來源為 Yahoo Finance 即時/近期報價。")

        st.markdown("---")
        st.markdown("#### 📐 臺指選擇權 Put/Call Ratio")
        st.caption(
            "⚠️ 資料源：TAIFEX官方公開網頁，僅涵蓋臺指選擇權(TXO)，不含個股選擇權/Greeks/IV；"
            "PCR偏高/偏低在市場上有兩種相反的解讀角度（避險需求 vs 情緒過度反轉訊號），僅供參考。"
        )
        if st.button("📐 抓取最新PCR資料", key="pcr_btn"):
            with st.spinner("抓取TAIFEX臺指選擇權Put/Call Ratio中..."):
                try:
                    st.session_state["pcr_report"] = OptionsEngine.build_pcr_report()
                except Exception as e:
                    st.session_state["pcr_report"] = {"status": "unavailable", "message": f"⚠️ PCR資料抓取時發生錯誤：{e}"}

        pcr_report = st.session_state.get("pcr_report")
        if pcr_report:
            if pcr_report.get("status") != "ok":
                st.warning(pcr_report.get("message", "⚠️ PCR資料暫時無法使用。"))
            else:
                pcr_col1, pcr_col2 = st.columns(2)
                pcr_col1.metric("未平倉量PCR%", f"{pcr_report['pcr_oi_pct']}%")
                pcr_col2.metric("成交量PCR%", f"{pcr_report['pcr_volume_pct']}%")
                st.line_chart(pcr_report["history"].set_index("date")[["pcr_oi_pct", "pcr_volume_pct"]])
                for f in pcr_report["flags"]:
                    st.markdown(f"- {f}")
    elif flags:
        for f in flags:
            st.warning(f)
    else:
        st.info("👈 請在左側點擊「🌍 抓取最新總經數據」。")

# ==========================================
# 📰 自選股新聞中心（Phase 2，v2.9.12 新增）
# ==========================================
# ⚠️ 說明：彙總 watchlist_status 表（既有的自選股/持股狀態紀錄，沒有另
# 開新表）裡每一檔股票的新聞，逐檔查詢篇數與情緒，並用 NewsEngine.
# build_daily_market_summary() 做成當日重點主題彙總。範圍限定在使用者
# 自己的自選股，不是大盤或美股/Fed等國際總經新聞（見 NewsEngine 說明）。
if mode == "📰 自選股新聞中心":
    if news_center_btn:
        with st.spinner("查詢自選股清單並逐檔抓取新聞中（股數多時較久，已使用快取加速）..."):
            try:
                overview = NewsEngine.get_watchlist_news_overview()
                codes = [row["code"] for row in overview]
                daily_summary = NewsEngine.build_daily_market_summary(codes) if codes else None
                st.session_state["watchlist_news_overview"] = overview
                st.session_state["watchlist_daily_summary"] = daily_summary
            except Exception as e:
                st.session_state["watchlist_news_overview"] = None
                st.session_state["watchlist_daily_summary"] = None
                st.error(f"⚠️ 自選股新聞查詢時發生錯誤：{e}")

    overview = st.session_state.get("watchlist_news_overview")
    daily_summary = st.session_state.get("watchlist_daily_summary")

    if overview is None:
        st.info("👈 請在左側點擊「📰 更新自選股新聞總覽」。")
    elif not overview:
        st.info("目前資料庫裡沒有任何自選股狀態紀錄（在「🔍 個股深度分析」的交易日誌功能標記過的股票才會出現在這裡）。")
    else:
        st.markdown("## 📰 自選股新聞中心")

        if daily_summary:
            d_col1, d_col2 = st.columns(2)
            d_col1.metric("今日市場情緒（彙總自選股）", daily_summary["market_bias"],
                          f"加權分數 {daily_summary['market_weighted_bias_score']} · {'★' * daily_summary['market_stars']}")
            d_col2.metric("涵蓋股票數", len(daily_summary["per_stock"]))

            st.markdown("**今日重點主題**")
            for b in daily_summary["bullets"]:
                st.markdown(f"- {b}")
            st.caption("⚠️ 這是彙總你自選股清單新聞的統計結果，不是大盤或國際總經新聞摘要，範圍請見上方說明。")
            st.markdown("---")

        st.markdown("**各股新聞篇數與情緒**")
        overview_df = pd.DataFrame(overview)
        if not overview_df.empty:
            overview_df["stars"] = overview_df["stars"].apply(lambda s: "★" * s + "☆" * (5 - s))
            overview_df = overview_df.rename(columns={
                "code": "代碼", "name": "名稱", "total": "新聞篇數",
                "overall_bias": "整體偏向", "stars": "星等",
            })
            st.dataframe(overview_df, use_container_width=True, hide_index=True)



# ==========================================
# 🔬 事件研究實驗室（Research Lab / Breakout Event Inventory, Phase A）
# ==========================================
# ⚠️ 這裡只負責觸發 research/event_inventory.py 的 Phase A 事件盤點、並把
# 結果用 Streamlit 呈現出來——統計邏輯、Stop Rule 門檻、Registry 一致性
# 驗證全部都在 research/ 模組裡，這裡不重複實作，也不修改任何既有的
# AI Score / Decision Engine 邏輯，兩邊完全獨立（見上傳的 research/ 說明）。
if mode == "🔬 事件研究實驗室":
    if research_btn:
        universe_scope = "full" if ei_universe_label.startswith("全市場") else "watchlist"

        # 全市場模式如果 stock_directory 尚未建立，run_inventory() 內部
        # 會自動退回觀察名單並只寫進 log 檔案，使用者在 UI 上看不到這件事；
        # 這裡額外檢查一次，讓使用者在畫面上也能看到同樣的提醒。
        if universe_scope == "full":
            try:
                _dir_df = StockDirectoryEngine.list_universe(exclude_etf=True)
                if _dir_df.empty:
                    st.warning(
                        "⚠️ 全市場代碼目錄（stock_directory）尚未建立，"
                        "已自動退回使用「觀察名單」。若要真正掃描全市場，"
                        "請先執行 StockDirectoryEngine.refresh_all() 建立代碼清單。"
                    )
            except Exception:
                st.warning("⚠️ 無法確認全市場代碼目錄狀態，若清單為空將自動退回觀察名單。")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def _update_ei_progress(done, total, current_ticker):
            if total:
                progress_bar.progress(min(done / total, 1.0))
            status_text.caption(f"事件盤點中：{current_ticker} ({done}/{total})")

        with st.spinner("執行 Breakout Event Inventory 中（依範圍與檔數，可能需要一段時間）..."):
            try:
                ei_summary = run_inventory(
                    scope=universe_scope,
                    limit=(ei_limit if ei_limit and ei_limit > 0 else None),
                    resume=ei_resume,
                    sleep_sec=ei_sleep,
                    progress_callback=_update_ei_progress,
                )
                st.session_state["event_inventory_summary"] = ei_summary
            except Exception as e:
                st.session_state["event_inventory_summary"] = None
                st.error(f"⚠️ 事件盤點執行時發生錯誤：{e}")

        progress_bar.empty()
        status_text.empty()

    ei_summary = st.session_state.get("event_inventory_summary")

    if ei_summary is None:
        st.info("👈 請在左側設定掃描範圍後，點擊「🔬 開始事件盤點 (Phase A)」。")
    elif ei_summary.get("status") != "ok":
        st.warning(f"目前沒有可顯示的事件資料。狀態：{ei_summary.get('status')}")
        st.markdown(f"**Stop Rule 判定：** {ei_summary.get('stop_rule_verdict', 'N/A')}")
    else:
        st.markdown("## 🔬 Breakout Event Inventory — Phase A Summary")

        verdict = ei_summary["stop_rule_verdict"]
        if verdict.startswith("GO"):
            st.success(f"✅ {verdict}")
        elif verdict.startswith("CAUTION"):
            st.warning(f"⚠️ {verdict}")
        else:
            st.error(f"🛑 {verdict}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("事件總數", ei_summary["total_events"])
        m2.metric("有事件的股票數", ei_summary["distinct_tickers_with_events"])
        m3.metric("有量確認占比", f"{ei_summary['volume_confirmed_pct']}%")
        m4.metric("前10檔集中度", f"{ei_summary['top10_share_pct']}%")

        st.caption(
            f"資料期間：{ei_summary['date_range'][0]} ~ {ei_summary['date_range'][1]}　｜　"
            f"每年平均事件數：{ei_summary['events_per_year'] if ei_summary['events_per_year'] is not None else 'N/A'}"
        )
        if ei_summary.get("events_per_year_note"):
            st.caption(f"⚠️ {ei_summary['events_per_year_note']}")

        if ei_summary.get("concentration_warning"):
            st.warning(ei_summary["concentration_warning"])
        if ei_summary.get("temporal_concentration_warning"):
            st.warning(ei_summary["temporal_concentration_warning"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**每檔股票事件數分布**")
            st.caption(
                f"中位數 {ei_summary['events_per_ticker_median']} ／ "
                f"P25 {ei_summary['events_per_ticker_p25']} ／ "
                f"P75 {ei_summary['events_per_ticker_p75']}"
            )
            top10_df = pd.DataFrame(
                list(ei_summary["top10_tickers"].items()), columns=["代碼", "事件數"]
            )
            st.dataframe(top10_df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**量能 / ETF / 漲停 / 重複突破 占比**")
            st.caption(
                f"有量確認：{ei_summary['volume_confirmed_count']} 筆　｜　"
                f"無量確認：{ei_summary['volume_not_confirmed_count']} 筆　｜　"
                f"無法判斷：{ei_summary['volume_unknown_count']} 筆"
            )
            if ei_summary.get("etf_event_count") is not None:
                st.caption(f"ETF 事件：{ei_summary['etf_event_count']} 筆 ({ei_summary['etf_event_pct']}%)")
            if ei_summary.get("limit_up_event_count") is not None:
                st.caption(
                    f"漲停突破：{ei_summary['limit_up_event_count']} 筆 "
                    f"({ei_summary['limit_up_event_pct']}%) — 這些事件當天很可能無法實際成交進場"
                )
            if ei_summary.get("repeat_event_pct") is not None:
                st.caption(f"重複突破(同股票第2次以上)占比：{ei_summary['repeat_event_pct']}%")
            if ei_summary.get("liquidity_breakdown"):
                st.caption(f"流動性分布：{ei_summary['liquidity_breakdown']}"
                           f"（低流動性占比 {ei_summary.get('low_liquidity_pct', 'N/A')}%）")

        st.markdown("---")
        st.markdown("#### 📄 原始事件明細（Source of Truth）")
        try:
            from research.event_inventory import EVENTS_CSV
            events_raw_df = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
            st.caption(f"共 {len(events_raw_df)} 筆原始事件，Summary 只是從這份資料算出來的統計摘要。")
            st.dataframe(events_raw_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ 下載原始事件 CSV", data=events_raw_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="breakout_events_raw.csv", mime="text/csv",
            )
        except Exception:
            st.caption("（尚未找到原始事件 CSV，可能是這次盤點沒有偵測到任何事件。）")

        try:
            from research.event_inventory import ERRORS_CSV
            errors_df = pd.read_csv(ERRORS_CSV, encoding="utf-8-sig")
            if not errors_df.empty:
                with st.expander(f"⚠️ 有 {len(errors_df)} 檔股票處理失敗，點擊查看"):
                    st.dataframe(errors_df, use_container_width=True, hide_index=True)
        except Exception:
            pass

# ==========================================
# 🔍 個股深度分析模式
# ==========================================
# ⚠️ 修正說明：比照上面「📡 全台股掃描模式」已經修過的同一種問題。
# run_btn 的 True 只在按下當下觸發的那一次 rerun 有效；這個分析頁面底下
# 的大戶持股／董監持股／季節循環分析都是額外的按鈕，點擊後會觸發新的
# rerun，這時候 run_btn 已經變回 False。如果直接用 run_btn 當作整個
# 分析結果區塊的顯示條件，會導致「點擊這些子按鈕之後，上面整份分析
# 結果都消失、看起來像沒東西」。改用 session_state 記住「目前正在分析
# 哪一檔」，顯示條件改成檢查這個持久化的狀態，而不是按鈕的瞬間狀態。
if run_btn and ticker:
    st.session_state["analysis_ticker"] = ticker

analysis_ticker = st.session_state.get("analysis_ticker")

if mode == "🔍 個股深度分析" and analysis_ticker:
    ticker = analysis_ticker  # 讓下面既有的渲染邏輯繼續沿用 ticker 這個變數名稱
    with st.spinner("啟動底層特徵引擎與 Agent 辯論中..."):
        # AI Decision Pipeline 流水線執行
        # ⚠️ 修正說明：這段原本完全沒有 try/except。DataEngine.get_stock_data()
        # 依設計會在查無資料時主動 raise Exception（見 data_engine.py），
        # 例如代碼打錯、興櫃股票資料覆蓋率低、網路不穩或yfinance暫時異常
        # 都會觸發——原本這個例外會直接讓整個 Streamlit 頁面顯示一坨原始
        # Python traceback崩潰，而不是友善的錯誤訊息。現在包起來，失敗時
        # 顯示清楚的錯誤原因並用 st.stop() 乾淨地中止這次執行，不影響
        # 使用者下次操作。
        try:
            df = DataEngine.get_stock_data(ticker, use_cache=use_cache)
            df = IndicatorEngine.add_indicators(df)
            df = StructureEngine.add_swing_points(df)
            df = PatternEngine.add_patterns(df)  # v2.9.5：接入原本孤兒的頭肩/缺口/雙頂雙底/旗形偵測
            df = StageEngine.add_stage_analysis(df)  # v2.9.6：Weinstein 四階段分析
            df = RiskEngine.add_risk_metrics(df)
            df = RiskEngine.add_liquidity_metrics(df)  # v2.9.7 新增：流動性風險（見 risk_engine.py 說明）
            df = DivergenceEngine.add_defense_signals(df)
            df = StrategyEngine.generate_signals(df)
            df = MomentumEngine.add_momentum_score(df)
            df = EvidenceEngine.add_evidence(df)
        except Exception as e:
            st.error(f"⚠️ 無法取得或處理「{ticker}」的股價資料：{e}")
            st.stop()

        # ==========================================
        # 終極防禦工程：欄位對齊與安全檢查 (避免 KeyError)
        # ==========================================
        # ⚠️ 修正說明：這個欄位補齊區塊原本放在 BacktestEngine.run_backtest(df)
        # 之後才執行。若上游引擎因為資料不足（例如 atr_14 缺失）而沒有算出
        # stop_loss/target_1，代表 BacktestEngine 執行當下這兩欄根本不存在
        # （ATR 停損停利機制被靜默關閉，只能靠 ai_score 出場），但畫面上
        # 事後才補上的「假停損價」卻會顯示出來，兩者完全對不上。現在把這個
        # 區塊移到 BacktestEngine 之前，確保回測引擎跟畫面顯示看到的是
        # 同一組欄位/數值。
        #
        # 同時修正：原本 fallback 用的停損/停利倍數 (1.5x / 2.0x / 4.0x) 跟
        # StrategyEngine 實際公式 (2.0x / 2.5x / 5.0x) 不一致，但下方 UI 卡片
        # 的標籤是寫死的「2.0x / 2.5x / 5.0x ATR」——一旦 fallback 觸發，數字
        # 與標籤會互相矛盾，對風控工具而言是會誤導下單的顯示錯誤。現在統一
        # 成跟 StrategyEngine 相同的倍數，維持全系統公式一致。
        fallback_atr = df['atr_14'] if 'atr_14' in df.columns else df['close'] * 0.02

        required_columns_with_defaults = {
            'market_regime': '🔄 盤整或未知狀態 (請更新 strategy_engine.py)',
            'ai_score': 50.0,
            'confidence': 'Medium (中性)',
            'action_guide': '👀 震盪觀望 (多空拉鋸，建議控制倉位)',
            'bull_reason': '⚠️ 多頭辯護因子未完全載入，請確認底層引擎版本。',
            'bear_reason': '⚠️ 空頭辯護因子未完全載入，請確認底層引擎版本。',
            'risk_reason': '⚠️ 風控審查因子未完全載入，請確認底層引擎版本。',
            'stop_loss': df['close'] - (2.0 * fallback_atr),
            'target_1': df['close'] + (2.5 * fallback_atr),
            'target_2': df['close'] + (5.0 * fallback_atr),
            'zigzag': np.nan,
            'confidence_pct': 50.0,
            'data_quality_pct': 100.0,
            'confidence_label': '🟡 中性信心（請更新 evidence_engine.py）',
            'volatility_annualized': np.nan,
            'rolling_mdd_60d': np.nan,
            'var_95_pct': np.nan,
            'var_99_pct': np.nan,
            'bearish_divergence': False,
            'bullish_divergence': False,
            'divergence_note': '',
            'breakout_up': False,
            'breakout_down': False,
            'bull_trap_confirmed': False,
            'bear_trap_confirmed': False,
            'trap_note': '',
            'momentum_score': 50.0,
            'momentum_grade': 'C (請更新 momentum_engine.py)',
            'momentum_score_complete': False,
            'is_a_grade_candidate': False,
            'momentum_penalty_alert': False,
            'reversal_watch': False,
            'trap_alert': False,
            'defense_risk_flag': False,
            'entry_signal': '⚪ 無明確買進訊號 (請更新 strategy_engine.py)',
            'exit_signal': '⚪ 無明確賣出訊號 (請更新 strategy_engine.py)',
            'stage': np.nan,
            'stage_label': '⚠️ 資料不足（請更新 stage_engine.py）',
            'stage_note': '',
        }

        for col, default_val in required_columns_with_defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # ==========================================
        # 回測資料準備
        # ==========================================
        # ⚠️ 修正說明：df 的最後一筆可能是 DataEngine 用「盤中即時報價」撐開
        # OHLC 產生的估計值（is_intraday_estimate=True），本質上是尚未收盤
        # 定案的資料。RiskEngine.compute_beta() 已經會排除這一筆再計算，
        # 但原本 BacktestEngine 卻直接把它當一般歷史K棒跑進全樣本回測（含
        # 停損/停利觸價判斷與報酬率計算），等於用未定案資料做決策，跟
        # RiskEngine 的處理方式不一致。這裡讓回測輸入資料也排除掉這一筆，
        # 並且刻意不覆蓋主要的 df（Dashboard 顯示仍然要看得到最新的即時估計）。
        if 'is_intraday_estimate' in df.columns and bool(df['is_intraday_estimate'].iloc[-1]):
            backtest_input_df = df.iloc[:-1].copy()
        else:
            backtest_input_df = df

        if len(backtest_input_df) >= 30:
            _, report = BacktestEngine.run_backtest(backtest_input_df)
        else:
            report = {
                'total_return': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0,
                'total_fees_paid': 0.0,
                'exit_breakdown': {'停損出場': 0, '停利出場': 0, 'AI Score轉弱出場': 0},
                'note': "⚠️ 排除盤中即時估計值後，可用歷史資料不足，暫不執行回測。",
            }

        # 大盤基準資料（用於 Beta 計算），抓取失敗不影響主流程
        try:
            benchmark_df = DataEngine.get_benchmark_data()
        except Exception:
            benchmark_df = None
        
        try:
            risk_report = RiskEngine.build_risk_report(df, benchmark_df)
        except Exception:
            risk_report = None
        
        try:
            chip_report = ChipEngine.build_chip_report(ticker)
        except Exception:
            chip_report = {"status": "unavailable", "message": "⚠️ 籌碼中心初始化失敗。"}

        try:
            fundamental_report = FundamentalEngine.build_fundamental_report(ticker)
        except Exception:
            fundamental_report = {"status": "unavailable", "message": "⚠️ 基本面與財務面中心初始化失敗。"}

        # ==========================================
        # v2.9.5 新增：RS Rating（相對本次觀察名單的排名）
        # ==========================================
        # ⚠️ BreakoutEngine 的介面是「dict報告」模式（跟 chip_report/
        # fundamental_report 同一種模式），跟其他「df進df出」引擎不同，
        # 所以刻意只在個股深度分析頁面呼叫，不塞進批次 scan() 的迴圈
        # （原因見對話中的說明：批次模式沒有 theme_score/chip_report可即時
        # 提供，硬塞會產生大量看起來很差、但其實只是「資料不足」的假分數）。
        try:
            _rs_universe_raw = {}
            _rvol_universe_raw = {}  # v2.9.6 新增：跟 RS 用同一輪迴圈收集相對成交量
            for _t in ScannerEngine.DEFAULT_WATCHLIST:
                try:
                    _udf = DataEngine.get_stock_data(_t, use_cache=True, max_age_hours=6)
                    _rs_raw = RSRatingEngine.compute_raw_score(_udf)
                    if _rs_raw.get('status') == 'ok':
                        _rs_universe_raw[_t] = _rs_raw['raw_score']
                    _udf_ind = IndicatorEngine.add_indicators(_udf)
                    if 'rvol' in _udf_ind.columns and pd.notna(_udf_ind['rvol'].iloc[-1]):
                        _rvol_universe_raw[_t] = float(_udf_ind['rvol'].iloc[-1])
                except Exception:
                    continue
            _self_raw = RSRatingEngine.compute_raw_score(df)
            if _self_raw.get('status') == 'ok':
                _rs_universe_raw[ticker] = _self_raw['raw_score']
            if 'rvol' in df.columns and pd.notna(df['rvol'].iloc[-1]):
                _rvol_universe_raw[ticker] = float(df['rvol'].iloc[-1])
            _rs_rankings = RSRatingEngine.rank_universe(_rs_universe_raw)
            _rvol_rankings = RSRatingEngine.rank_universe(_rvol_universe_raw)
            rs_info = _rs_rankings.get(ticker, {'rs_rating': None, 'universe_size': 0})
            rvol_info = _rvol_rankings.get(ticker, {'percentile': None})
        except Exception:
            rs_info = {'rs_rating': None, 'universe_size': 0}
            rvol_info = {'percentile': None}

        # v2.9.6：現在才知道跨股票 RS Rating 與相對成交量排名，重新計算
        # momentum_score，取代第 715 行算出的「不完整評分」（見
        # MomentumEngine docstring）。只重算這一層，不用重跑整條 pipeline。
        try:
            df = MomentumEngine.add_momentum_score(
                df, rs_rating=rs_info.get('rs_rating'), relative_volume_percentile=rvol_info.get('percentile')
            )
            latest = df.iloc[-1]
        except Exception:
            pass

        try:
            breakout_report = BreakoutEngine.analyze(df, theme_score=0, fundamental_score=0, chip_report=chip_report)
        except Exception as e:
            breakout_report = {'error': f'⚠️ 飆股評分引擎執行失敗：{e}'}

        try:
            canslim_report = CanslimEngine.analyze(
                df, fundamental_report=fundamental_report, chip_report=chip_report,
                market_regime=df['market_regime'].iloc[-1] if 'market_regime' in df.columns else None,
                rs_rating=rs_info.get('rs_rating'),
            )
        except Exception as e:
            canslim_report = None

        # 個股深度分析頁面：帶入真實的 chip_report / fundamental_report，
        # 算出完整的選股大師五維度評分（跟批次掃描的「快速模式」不同，
        # 那邊為了效能刻意不呼叫這兩個外部資料源，見 scanner_engine.py 說明）。
        try:
            academy_report = StockAcademyEngine.build_report(
                df, chip_report=chip_report, fundamental_report=fundamental_report
            )
        except Exception:
            academy_report = {}

        try:
            tf_report = TimeframeEngine.build_report(df)
        except Exception:
            tf_report = {}

        latest = df.iloc[-1]
        
        try:
            evidence_list = EvidenceEngine.get_evidence_list(df, -1)
        except Exception:
            evidence_list = []
        
        # ==========================================
        # Dashboard Layer: 戰情室視覺化
        # ==========================================
        st.markdown(f"## 📊 標的 `{NameEngine.get_tag(ticker)}` 企業級分析戰情室")
        st.caption(f"市場別：{NameEngine.get_market_type(ticker)}")

        # ==========================================
        # 📋 Watchlist 狀態機 (v2.9.10 新增，v2.9.11 擴充部位欄位)
        # ==========================================
        # ⚠️ 這是「使用者手動標記」的追蹤工具，不會自動幫你判斷該不該換狀態，
        # 也不會自動幫你算/改進場價、股數、停損價，見 db_engine.py 的誠實
        # 範圍界定說明。狀態與部位資訊會存進本機 SQLite，跨 session 持久保存。
        _wl_current = DatabaseEngine.get_watchlist_status(ticker)
        wl_col1, wl_col2, wl_col3 = st.columns([1, 2, 1])
        with wl_col1:
            _wl_status_text = _wl_current['status'] if _wl_current else "（尚未追蹤）"
            st.metric("📋 追蹤狀態", _wl_status_text)
            if _wl_current and _wl_current.get('entry_price'):
                st.caption(
                    f"進場價 {_wl_current['entry_price']:.2f} × {_wl_current.get('shares') or 0:.0f}股"
                    + (f"　目前停損 {_wl_current['current_stop']:.2f}" if _wl_current.get('current_stop') else "")
                )
        with wl_col2:
            _wl_options = DatabaseEngine.WATCHLIST_STATES
            _wl_default_idx = _wl_options.index(_wl_current['status']) if _wl_current and _wl_current['status'] in _wl_options else 0
            wl_new_status = st.selectbox("切換狀態", _wl_options, index=_wl_default_idx, key=f"wl_select_{ticker}")
            wl_note = st.text_input("備註（選填）", value=_wl_current.get('note', '') if _wl_current else '', key=f"wl_note_{ticker}")
            with st.expander("部位資訊（選填，未填則沿用原本紀錄）"):
                wl_pos_col1, wl_pos_col2, wl_pos_col3 = st.columns(3)
                wl_entry_price = wl_pos_col1.number_input(
                    "進場價", min_value=0.0, value=float(_wl_current.get('entry_price') or 0.0) if _wl_current else 0.0,
                    key=f"wl_entry_{ticker}")
                wl_shares = wl_pos_col2.number_input(
                    "股數", min_value=0.0, value=float(_wl_current.get('shares') or 0.0) if _wl_current else 0.0,
                    key=f"wl_shares_{ticker}")
                wl_stop = wl_pos_col3.number_input(
                    "目前停損價", min_value=0.0, value=float(_wl_current.get('current_stop') or 0.0) if _wl_current else 0.0,
                    key=f"wl_stop_{ticker}")
        with wl_col3:
            st.markdown("&nbsp;")
            if st.button("更新狀態", key=f"wl_update_{ticker}"):
                _wl_result = DatabaseEngine.set_watchlist_status(
                    ticker, wl_new_status, note=wl_note or None,
                    entry_price=wl_entry_price or None, shares=wl_shares or None, current_stop=wl_stop or None,
                )
                if _wl_result["status"] == "ok":
                    st.success(f"已更新為「{wl_new_status}」")
                    st.rerun()
                else:
                    st.error(_wl_result["message"])
        with st.expander("📜 狀態變化歷史"):
            _wl_hist = DatabaseEngine.get_watchlist_history(ticker)
            if _wl_hist.empty:
                st.caption("尚無歷史紀錄。")
            else:
                st.dataframe(_wl_hist, use_container_width=True, hide_index=True)

        # ==========================================
        # 📔 交易日誌 (Trade Journal) — v2.9.11 新增
        # ==========================================
        # ⚠️ 這裡記錄的是「使用者自己輸入的實際成交」，用來計算真實勝率/
        # 期望值——跟下方 BacktestEngine 的歷史模擬是兩回事，刻意分開
        # 呈現，不會混為一談，見 db_engine.py 的說明。
        with st.expander("📔 交易日誌（記錄實際成交，計算真實績效）"):
            st.caption("⚠️ 這裡的統計只反映你自己輸入的實際交易，不是回測模擬；請只填寫真的成交過的價格與股數。")
            tj_tab1, tj_tab2 = st.tabs(["新增交易", "歷史紀錄與統計"])
            with tj_tab1:
                tj_mode = st.radio("動作", ["記錄新進場", "登記出場"], horizontal=True, key=f"tj_mode_{ticker}")
                if tj_mode == "記錄新進場":
                    tj_c1, tj_c2, tj_c3 = st.columns(3)
                    tj_entry_date = tj_c1.date_input("進場日期", key=f"tj_entry_date_{ticker}")
                    tj_entry_price = tj_c2.number_input("進場價", min_value=0.0, key=f"tj_entry_price_{ticker}")
                    tj_shares = tj_c3.number_input("股數", min_value=0.0, key=f"tj_shares_{ticker}")
                    tj_strategy = st.text_input("策略標籤（選填，例如 VCP / CANSLIM）", key=f"tj_strategy_{ticker}")
                    tj_note = st.text_input("備註（選填）", key=f"tj_journal_note_{ticker}")
                    if st.button("送出進場紀錄", key=f"tj_submit_entry_{ticker}"):
                        if tj_entry_price <= 0 or tj_shares <= 0:
                            st.error("⚠️ 進場價與股數需大於0。")
                        else:
                            _tj_res = DatabaseEngine.log_trade_entry(
                                ticker, str(tj_entry_date), tj_entry_price, tj_shares,
                                strategy_tag=tj_strategy or None, note=tj_note or None,
                            )
                            st.success(f"已記錄，交易編號 #{_tj_res['trade_id']}")
                            st.rerun()
                else:
                    _tj_open = DatabaseEngine.get_trade_journal(ticker=ticker)
                    _tj_open = _tj_open[_tj_open["exit_price"].isna()] if not _tj_open.empty else _tj_open
                    if _tj_open.empty:
                        st.caption("目前沒有進行中（尚未登記出場）的交易。")
                    else:
                        _tj_id_options = {
                            f"#{r['id']} - {r['entry_date']} 進場 {r['entry_price']:.2f} × {r['shares']:.0f}股": r['id']
                            for _, r in _tj_open.iterrows()
                        }
                        tj_pick = st.selectbox("選擇要登記出場的交易", list(_tj_id_options.keys()), key=f"tj_pick_{ticker}")
                        tj_ec1, tj_ec2 = st.columns(2)
                        tj_exit_date = tj_ec1.date_input("出場日期", key=f"tj_exit_date_{ticker}")
                        tj_exit_price = tj_ec2.number_input("出場價", min_value=0.0, key=f"tj_exit_price_{ticker}")
                        if st.button("送出出場紀錄", key=f"tj_submit_exit_{ticker}"):
                            if tj_exit_price <= 0:
                                st.error("⚠️ 出場價需大於0。")
                            else:
                                DatabaseEngine.log_trade_exit(_tj_id_options[tj_pick], str(tj_exit_date), tj_exit_price)
                                st.success("已登記出場")
                                st.rerun()
            with tj_tab2:
                _tj_hist = DatabaseEngine.get_trade_journal(ticker=ticker)
                if _tj_hist.empty:
                    st.caption("尚無交易紀錄。")
                else:
                    st.dataframe(_tj_hist, use_container_width=True, hide_index=True)
                    _tj_stats = DatabaseEngine.compute_journal_stats(ticker=ticker)
                    if _tj_stats.get("status") == "ok":
                        tjs1, tjs2, tjs3, tjs4 = st.columns(4)
                        tjs1.metric("真實勝率", f"{_tj_stats['win_rate_pct']:.1f}%")
                        tjs2.metric("平均獲利", f"{_tj_stats['avg_win_pct']:.2f}%")
                        tjs3.metric("平均虧損", f"{_tj_stats['avg_loss_pct']:.2f}%")
                        tjs4.metric("期望值", f"{_tj_stats['expectancy_pct']:.2f}%")
                        if "low_sample_warning" in _tj_stats:
                            st.warning(_tj_stats["low_sample_warning"])

        st.markdown("---")

        st.markdown("---")

        # --- 頂部：宏觀狀態與最終裁決 ---
        r1, r2, r3, r4 = st.columns(4)
        
        # 兼容處理單值或 Series 形式的 close
        raw_close = latest['close']
        latest_close = float(raw_close.iloc[0] if isinstance(raw_close, (np.ndarray, list, pd.Series)) else raw_close)
        
        r1.metric("即時收盤價", f"{latest_close:.2f}")
        r2.metric("🌍 市場狀態 (Market Regime)", f"{latest['market_regime']}")
        r3.metric("🤖 綜合 AI 評分 (AI Score)", f"{latest['ai_score']:.1f} / 100")
        r4.metric("🛡️ 決策信心 (Confidence)", f"{latest['confidence']}")
        
        st.info(f"**⚖️ 主審裁決 (Judge Action Guide)：** {latest['action_guide']}")

        sig_col1, sig_col2 = st.columns(2)
        sig_col1.markdown(f"**🎯 買進訊號：** {latest.get('entry_signal', 'N/A')}")
        sig_col2.markdown(f"**🚪 賣出訊號：** {latest.get('exit_signal', 'N/A')}")
        st.caption("ℹ️ 買賣訊號已納入近期背離/誘多假突破防禦訊號（見下方風險中心與誘盤警報），與 AI Score 為互補資訊，仍請自行評估風險。")

        if bool(latest.get('trap_alert', False)):
            trap_msg = latest.get('trap_note', '') or latest.get('divergence_note', '') or "近期偵測到背離或假突破/假跌破警報。"
            st.warning(f"🛡️ **誘盤警報：** {trap_msg}")

        st.markdown("---")
        
        # --- 證據與信心模型 (Evidence & Confidence Engine) ---
        st.markdown("### 🧾 證據與信心模型 (Evidence & Confidence Engine)")
        
        e1, e2, e3 = st.columns([1, 1, 2])
        e1.metric("🤖 AI Score", f"{latest['ai_score']:.1f}")
        e2.metric("📊 資料品質 (Data Quality)", f"{latest['data_quality_pct']:.0f}%")
        e3.metric("🧠 信心程度 (Confidence)", f"{latest['confidence_pct']:.0f}%  ·  {latest['confidence_label']}")
        
        if evidence_list:
            bull_ev = [e for e in evidence_list if e['polarity'] == 'bull']
            bear_ev = [e for e in evidence_list if e['polarity'] == 'bear']
            risk_ev = [e for e in evidence_list if e['polarity'] == 'risk']
            
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            
            def render_evidence(col, title, items, empty_msg):
                with col:
                    st.markdown(f"**{title}**")
                    if not items:
                        st.caption(empty_msg)
                    for item in items:
                        stars = "★" * item['stars'] + "☆" * (5 - item['stars'])
                        detail = f"　_{item['detail']}_" if item.get('detail') else ""
                        st.markdown(f"- {item['label']}　`{stars}`{detail}")
            
            render_evidence(ev_col1, "🟢 多頭證據", bull_ev, "目前無明顯多頭證據")
            render_evidence(ev_col2, "🔴 空頭證據", bear_ev, "目前無明顯空頭證據")
            render_evidence(ev_col3, "🛡️ 風險證據", risk_ev, "目前無異常風險證據")
        else:
            st.caption("⚠️ 尚無足夠資料生成證據清單。")
        
        st.markdown("---")

        # --- 經典技術指標：MTM 動量指標／寶塔線 (v2.8 新增) ---
        st.markdown("### 📐 經典技術指標 (MTM 動量指標／寶塔線)")
        mt1, mt2 = st.columns(2)

        with mt1:
            mtm_val = latest.get('mtm', np.nan)
            mtm_val = float(mtm_val.iloc[0] if isinstance(mtm_val, (np.ndarray, list, pd.Series)) else mtm_val)
            if pd.notna(mtm_val):
                direction = "🟢 動能為正（近期股價高於n日前）" if mtm_val > 0 else ("🔴 動能為負（近期股價低於n日前）" if mtm_val < 0 else "🟡 動能持平")
                st.metric("📊 MTM 動量指標", f"{mtm_val:.2f}", help="MTM = 今日收盤價 - n日前收盤價，衡量股價變動的速度")
                st.caption(direction)
            else:
                st.caption("⚠️ MTM 資料不足（K線筆數不足n天）")
            st.caption("⚠️ MTM 訊號容易在0上下反覆穿越、雜訊較多，建議搭配均線/MACD等趨勢指標一起判讀，不宜單獨作為買賣依據。")

        with mt2:
            pagoda_trend = latest.get('pagoda_trend', None)
            if pagoda_trend == "red":
                st.success("🔴➡️🟩 寶塔線：目前為「翻紅」狀態（趨勢向上已確立）")
            elif pagoda_trend == "black":
                st.error("⬛ 寶塔線：目前為「翻黑」狀態（趨勢向下已確立）")
            else:
                st.caption("⚠️ 寶塔線資料不足，尚未產生翻轉訊號")
            st.caption("⚠️ 寶塔線屬於落後指標，翻紅/翻黑代表趨勢「已經」確立，而非預測轉折，訊號通常會落後真正的高低點數天，盤整格局容易出現進出頻繁但無獲利的情況。")

        st.markdown("---")
        
        # --- 風險中心 (Risk Center) ---
        st.markdown("### 🛡️ 風險中心 (Risk Center)")
        
        if risk_report:
            rk1, rk2, rk3, rk4, rk5, rk6 = st.columns(6)
            
            vol = risk_report['volatility_annualized']
            mdd = risk_report['max_drawdown_60d']
            beta = risk_report['beta']
            var95 = risk_report['var_95_pct']
            rr = risk_report['reward_risk_ratio']
            avg_tv = risk_report.get('avg_trading_value_20d', float('nan'))
            
            rk1.metric("📈 年化波動率", f"{vol:.1f}%" if pd.notna(vol) else "N/A")
            rk2.metric("📉 60日最大回撤", f"{mdd:.1f}%" if pd.notna(mdd) else "N/A")
            rk3.metric("β Beta (相對大盤)", f"{beta:.2f}" if pd.notna(beta) else "N/A")
            rk4.metric("🎲 VaR (95%, 單日)", f"{var95:.1f}%" if pd.notna(var95) else "N/A")
            rk5.metric("⚖️ 報酬風險比 (RR)", f"{rr:.2f}" if pd.notna(rr) else "N/A")
            # v2.9.7 新增：流動性風險（近20日日均成交值），見 risk_engine.py 說明
            rk6.metric("💧 流動性", risk_report.get('liquidity_level', 'N/A'),
                       help=f"近20日日均成交值：約 {avg_tv/1e6:.0f} 百萬元" if pd.notna(avg_tv) else "資料不足")
            
            st.markdown(f"**綜合風險等級：{risk_report['risk_level']}**")
            for flag in risk_report['risk_flags']:
                st.caption(flag)
        else:
            st.caption("⚠️ 風險中心計算失敗，請確認 risk_engine.py 是否已正確放入 engines 資料夾。")
        
        st.markdown("---")
        
        # --- 籌碼中心 (Chip Center) ---
        st.markdown("### 🏦 籌碼中心 (Chip Center)")
        
        if chip_report and chip_report.get("status") == "ok":
            inst = chip_report.get("institutional")
            margin = chip_report.get("margin")
            
            ch1, ch2, ch3, ch4 = st.columns(4)
            if inst:
                ch1.metric("🌍 外資買賣超 (張)", f"{inst['foreign_net']/1000:,.0f}", help=f"資料日期：{inst['date']}")
                ch2.metric("🏛️ 投信買賣超 (張)", f"{inst['trust_net']/1000:,.0f}")
                ch3.metric("💼 自營商買賣超 (張)", f"{inst['dealer_net']/1000:,.0f}")
                ch4.metric("📊 三大法人合計 (張)", f"{inst['total_net']/1000:,.0f}")
            else:
                st.caption("ℹ️ 暫無三大法人買賣超資料（可能為上櫃股票或近期非交易日）。")
            
            if margin:
                cm1, cm2 = st.columns(2)
                cm1.metric("💳 融資餘額變化 (張)", f"{margin['margin_change']/1000:,.1f}", help=f"目前餘額 {margin['margin_balance']/1000:,.0f} 張")
                cm2.metric("📉 融券餘額變化 (張)", f"{margin['short_change']/1000:,.1f}", help=f"目前餘額 {margin['short_balance']/1000:,.0f} 張")
            
            for flag in chip_report.get("flags", []):
                st.caption(flag)

            # v2.9.7 新增：融資餘額趨勢（見 chip_engine.py get_margin_trend 說明）。
            # 刻意做成按鈕、不隨頁面自動觸發——這個查詢要逐日打 TWSE API，
            # 比其他籌碼資料慢很多，不適合每次進頁面就自動打。
            with st.expander("💳 融資餘額趨勢（近20個交易日，需額外查詢）"):
                st.caption("⚠️ 這個查詢要逐日呼叫 TWSE 融資融券頁面，約需20次請求，比其他籌碼資料慢，僅在按下按鈕時才查詢。")
                if st.button("查詢融資餘額趨勢", key="margin_trend_btn"):
                    with st.spinner("逐日查詢融資餘額中（約需數秒至數十秒）..."):
                        try:
                            st.session_state["margin_trend_report"] = ChipEngine.get_margin_trend(ticker)
                        except Exception as e:
                            st.session_state["margin_trend_report"] = {"status": "unavailable", "message": f"⚠️ 查詢失敗：{e}"}
                margin_trend = st.session_state.get("margin_trend_report")
                if margin_trend and margin_trend.get("status") == "ok":
                    mt1, mt2 = st.columns(2)
                    mt1.metric("融資餘額變化率", f"{margin_trend['change_pct']:+.1f}%",
                               help=f"取樣 {margin_trend['days_used']} 個交易日")
                    mt2.metric("最新融資餘額 (張)", f"{margin_trend['latest_balance']/1000:,.0f}")
                    st.caption(margin_trend["flag"])
                elif margin_trend:
                    st.caption(margin_trend.get("message", "⚠️ 查詢失敗或資料不足。"))
        else:
            msg = chip_report.get("message", "⚠️ 暫時無法取得籌碼資料。") if chip_report else "⚠️ 暫時無法取得籌碼資料。"
            st.caption(msg)
            st.caption("ℹ️ 籌碼中心僅支援上市股票，資料來源為 TWSE 公開資訊，需要對外網路連線。")
        
        st.markdown("---")

        # --- 基本面與財務面中心 (Fundamental & Financial Center) ---
        st.markdown("### 📑 基本面與財務面中心 (Fundamental & Financial Center)")
        st.caption("資料來源：yfinance 財務摘要（Yahoo Finance）。台股（尤其中小型股/上櫃/興櫃）覆蓋率與更新頻率有限，缺值一律顯示「資料不足」，不做估計填補。與上方技術面/籌碼面為互補視角，非取代關係。")

        if fundamental_report.get("status") == "not_applicable":
            st.info(fundamental_report.get("message"))
            st.markdown("#### 📊 ETF 專屬分析")
            st.caption(
                "⚠️ 沒有淨值(NAV)資料，無法計算折溢價；也沒有追蹤誤差資料——這兩項是機構評估ETF的重要指標，"
                "但受限於免費資料源，本引擎老實承認做不到。技術面/籌碼面分析請參考本頁面其他區塊，ETF本身"
                "跟一般股票用同一套市場交易，既有分析同樣適用。"
            )
            try:
                etf_report = ETFEngine.build_etf_report(ticker, etf_name=NameEngine.get_name(ticker))
            except Exception as e:
                etf_report = {"status": "unavailable", "message": f"⚠️ ETF分析時發生錯誤：{e}"}

            if etf_report.get("status") == "ok":
                info = etf_report["info"]
                etf_col1, etf_col2, etf_col3 = st.columns(3)
                etf_col1.metric("ETF類型", etf_report["etf_type"]["label"])
                etf_col2.metric("資產規模(原幣別,來源:yfinance)", f"{info['total_assets']:,.0f}" if info.get("total_assets") else "資料不足")
                etf_col3.metric("配息殖利率", f"{info['yield_pct']:.2f}%" if info.get("yield_pct") is not None else "資料不足")
                for f in etf_report["flags"]:
                    st.markdown(f"- {f}")
            else:
                st.warning(etf_report.get("message", "⚠️ ETF分析暫時無法使用。"))
        elif fundamental_report.get("status") != "ok":
            st.caption(fundamental_report.get("message", "⚠️ 暫時無法取得基本面資料。"))
        else:
            snap = fundamental_report["snapshot"]

            def _fmt(val, suffix="", multiplier=1, decimals=1):
                if pd.isna(val):
                    return "資料不足"
                return f"{val * multiplier:.{decimals}f}{suffix}"

            fd1, fd2, fd3, fd4 = st.columns(4)
            fd1.metric("💰 EPS (近四季, TTM)", _fmt(snap["eps_ttm"], decimals=2))
            fd2.metric("📊 本益比 (TTM)", _fmt(snap["pe_ttm"], decimals=1))
            fd3.metric("📈 本益比 (預估, Forward)", _fmt(snap["pe_forward"], decimals=1))
            fd4.metric("📘 股價淨值比 (P/B)", _fmt(snap["price_to_book"], decimals=2))

            fd5, fd6, fd7, fd8 = st.columns(4)
            mc = snap["market_cap"]
            fd5.metric("🏢 總市值", f"{mc/1e8:,.1f} 億" if pd.notna(mc) else "資料不足")
            ec = snap["estimated_capital"]
            fd6.metric("🧮 股本 (概略估算)", f"{ec/1e8:,.1f} 億" if pd.notna(ec) else "資料不足",
                       help="以流通在外股數 × 台股常見面額10元概略估算，非精確財報數字，詳見引擎說明")
            fd7.metric("📖 每股淨值", _fmt(snap["book_value_per_share"], decimals=2))
            rev = snap["revenue_ttm"]
            fd8.metric("💵 營收 (近四季)", f"{rev/1e8:,.1f} 億" if pd.notna(rev) else "資料不足")

            fd9, fd10, fd11, fd12 = st.columns(4)
            fd9.metric("🚀 營收年增率", _fmt(snap["revenue_growth_yoy"], suffix="%", multiplier=100))
            fd10.metric("🏦 ROA (資產報酬率)", _fmt(snap["roa"], suffix="%", multiplier=100))
            fd11.metric("💎 ROE (股東權益報酬率)", _fmt(snap["roe"], suffix="%", multiplier=100))
            debt_ratio = snap.get("debt_ratio_pct", np.nan)
            if pd.notna(debt_ratio):
                fd12.metric("⚖️ 負債比例 (精算)", _fmt(debt_ratio, suffix="%", decimals=1),
                            help="以真實資產負債表精算：負債總額 / 資產總額（嚴謹定義）")
            else:
                fd12.metric("⚖️ 負債權益比 (近似)", _fmt(snap["debt_to_equity"], suffix="%", decimals=0),
                            help="抓不到資產負債表精算值，退而求其次改用 yfinance 的負債/股東權益比近似，兩者分母不同")

            fd13, fd14 = st.columns(2)
            fd13.metric("🏭 毛利率", _fmt(snap["gross_margin"], suffix="%", multiplier=100))
            fd14.metric("⚙️ 營業利益率", _fmt(snap["operating_margin"], suffix="%", multiplier=100))

            st.markdown("**綜合觀察：**")
            for flag in fundamental_report.get("flags", []):
                st.caption(flag)

            st.caption("ℹ️ 「成立時間」「主力買賣超」「集保庫存」「董監持股比例」等選股學院提及的指標，yfinance 無對應資料來源，本區塊不假造這些欄位；如需要請改查詢公開資訊觀測站 (MOPS) 或券商看盤軟體。")

        st.markdown("---")

        # --- 選股大師五維度評分 (Stock Academy Engine) ---
        st.markdown("### 🎓 選股大師 (五維度評分：市場面／基本面／技術面／籌碼面／財務面)")

        if academy_report:
            comp = academy_report.get("綜合評級", {})
            ac1, ac2, ac3 = st.columns([1, 1, 2])
            ac1.metric("🎓 綜合評分", f"{comp.get('總分', 0)} / 100")
            ac2.metric("🏅 學級", f"{comp.get('學級', 'F')}")
            ac3.metric("💬 評語", f"{comp.get('評語', '')}")

            st.caption(academy_report.get("資料完整度提示", ""))

            dims = academy_report.get("五維度明細", {})
            dim_cols = st.columns(5)
            dim_order = ["市場面", "基本面", "技術面", "籌碼面", "財務面"]
            for col, dim_name in zip(dim_cols, dim_order):
                data = dims.get(dim_name, {"分數": 0, "分析": []})
                with col:
                    st.markdown(f"**{dim_name}**　`{data['分數']}/20`")
                    for point in data.get("分析", []):
                        st.caption(point)

            st.markdown("**🎯 關鍵提示：**")
            for tip in academy_report.get("關鍵提示", []):
                st.markdown(f"- {tip}")

            st.caption("⚠️ 選股大師評級是「中長期體質」視角，跟上方 AI Score（短期波段）、飆股評分（動能結構）是三套獨立、互不加權混合的評分系統，刻意分開呈現，避免同一組事實被重複計分。三者請並參，不應只依賴其中一項下單。")
        else:
            st.caption("⚠️ 選股大師評分計算失敗，請確認 stock_academy_engine.py 是否已正確放入 engines 資料夾。")

        st.markdown("---")

        # ==========================================
        # v2.9.5 新增：波段交易核心 (VCP/Breakout + RS Rating + CAN SLIM)
        # ==========================================
        # ⚠️ 這三個區塊組合起來，對應原本專案評估報告指出的最大缺口——
        # 「有很多分析能力，但沒有真正的波段交易核心」。BreakoutEngine 其實
        # 早就寫好（七層飆股評分，邏輯上等同 VCP/Darvas Box 的箱型收斂+突破
        # 確認），只是從未被 app.py import；PatternEngine 同樣如此。這裡正式
        # 把兩者接上，並新增 RS Rating（跨股票排名）與 CAN SLIM 量化評分
        # （原本完全沒有的兩塊）。
        st.markdown("### 🚀 波段交易核心 (VCP/Breakout · RS Rating · CAN SLIM · Stage Analysis)")

        # ==========================================
        # v2.9.6 新增：決策共識儀表板 (Decision Consensus)
        # v2.9.7 擴充：接入籌碼面/產業輪動/流動性/總經背景，見 decision_engine.py
        # v2.9.9 擴充：接入公司治理（董監設質比例），見 decision_engine.py
        # ==========================================
        # ⚠️ 規則式共識彙整，不是機器學習/LLM決策系統，詳見 decision_engine.py 說明。
        try:
            # 產業輪動訊號：需要使用者先在「🏭 產業分析」頁面按下計算按鈕，
            # 這裡才讀得到 st.session_state["rotation_report"]；沒有計算過時
            # sector_signal 保持 None，DecisionEngine 會如實顯示「資料不足」，
            # 不會為了「看起來完整」而硬湊一個假訊號。
            sector_signal = None
            _rotation_report = st.session_state.get("rotation_report")
            if _rotation_report and _rotation_report.get("status") == "ok":
                try:
                    _stock_industry = IndustryEngine.get_industry(ticker)
                    _rtable = _rotation_report.get("rotation_table")
                    if _rtable is not None and not _rtable.empty and _stock_industry in _rtable["產業"].values:
                        _row = _rtable[_rtable["產業"] == _stock_industry].iloc[0]
                        sector_signal = {"status": "ok", "signal": _row["輪動訊號"], "industry": _stock_industry}
                except Exception:
                    sector_signal = None

            # 總經背景：需要使用者先在「🌍 總經戰情室」頁面按下抓取按鈕，
            # 同樣沒有值時就原樣傳 None，不計入投票（見 decision_engine.py 說明）。
            _macro_flags = st.session_state.get("macro_flags")

            # 公司治理（董監設質）：需要使用者先在下方「董監事持股與設質分析」
            # 展開區按下查詢按鈕，才會有 ticker-scoped 的 insider_report；
            # 沒有值時原樣傳 None，DecisionEngine 會顯示「資料不足」。
            _insider_report = _get_ticker_scoped_state("insider_report", ticker)

            # 新聞面：需要使用者已在下方「📰 新聞情緒中心」展開區按過
            # 「抓取最新相關新聞並分析情緒」，才會有 ticker-scoped 的
            # news_report；沒有值時原樣傳 None，DecisionEngine 會顯示
            # 「資料不足」。
            _news_report_for_consensus = _get_ticker_scoped_state("news_report", ticker)

            consensus = DecisionEngine.build_consensus(
                latest, canslim_report=canslim_report, breakout_report=breakout_report,
                rs_rating=rs_info.get('rs_rating'), risk_level=risk_report.get('risk_level'),
                chip_report=chip_report, sector_signal=sector_signal, macro_flags=_macro_flags,
                liquidity_level=risk_report.get('liquidity_level'), insider_report=_insider_report,
                news_report=_news_report_for_consensus,
            )
            st.markdown(f"#### 🧭 共識儀表板：{consensus['decision']}")
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("偏多維度", f"{consensus['bullish_count']} / {consensus['total_dims']}")
            cd2.metric("偏空維度", f"{consensus['bearish_count']} / {consensus['total_dims']}")
            cd3.metric("一致度", f"{consensus['agreement_pct']:.0f}%")
            vote_cols = st.columns(len(consensus['votes']))
            for col, (dim, (label, _)) in zip(vote_cols, consensus['votes'].items()):
                col.markdown(f"**{dim}**\n\n{label}")
            if consensus.get('macro_context'):
                with st.expander("🌍 總經背景（僅供參考，不計入偏多/偏空票數）"):
                    for m in consensus['macro_context']:
                        st.caption(m)
            st.caption(consensus['disclosure'])
        except Exception as e:
            st.caption(f"⚠️ 共識儀表板計算失敗：{e}")

        st.markdown("---")

        # v2.9.6 新增：Weinstein 四階段分析
        stage_summary = StageEngine.get_stage_summary(df)
        st.markdown("**📊 Stage Analysis (Weinstein 四階段)**")
        st_col1, st_col2 = st.columns([1, 3])
        st_col1.metric("目前階段", stage_summary['stage_label'])
        with st_col2:
            st.caption(stage_summary['stage_note'] or "資料不足，無法判斷階段。")
            st.caption("⚠️ 簡化版判斷（僅用均線位置與斜率），未納入成交量型態與相對大盤強弱，"
                       "適合當第一層濾網，不建議單獨依賴。")

        bo_col, rs_col = st.columns([2, 1])

        with bo_col:
            st.markdown("**🚀 飆股七層評分 (BreakoutEngine — 概念對應 VCP/Darvas Box)**")
            if breakout_report and 'error' not in breakout_report:
                bo1, bo2 = st.columns(2)
                bo1.metric("總分", f"{breakout_report['total_score']:.1f} / {breakout_report['max_possible_score']}")
                bo2.metric("等級", breakout_report['grade'])
                if breakout_report.get('baiting_alert'):
                    st.warning(breakout_report['baiting_alert'])
                yl = breakout_report.get('year_line_filter', {})
                st.caption(f"年線濾網：{yl.get('reason', 'N/A')}")
                with st.expander("查看七層評分明細"):
                    for name, item in breakout_report.get('score_breakdown', {}).items():
                        st.markdown(f"- **{name}**：{item['score']} / {item['max']}"
                                    + (f"　_{item.get('source', '')}_" if item.get('source') else ""))
                st.caption("⚠️ 「產業題材」與「基本面3新2益」兩項需要人工/新聞判定，本頁面預設輸入0分，"
                           "代表這兩項『資料不足』而非『真的沒題材』，總分會因此系統性偏低，解讀時請把這點納入考量。")
            else:
                st.caption(breakout_report.get('error', '⚠️ 飆股評分資料不足（建議至少30個交易日）。') if breakout_report else "⚠️ 資料不足")

        with rs_col:
            st.markdown("**🏆 RS Rating (相對強度排名)**")
            rs_rating_val = rs_info.get('rs_rating')
            if rs_rating_val is not None:
                st.metric("RS Rating", f"{rs_rating_val} / 99")
                st.caption(RSRatingEngine.grade_from_rating(rs_rating_val))
            else:
                st.metric("RS Rating", "N/A")
                st.caption("⚠️ 排名母體不足，無法計算")
            st.caption(f"ℹ️ 排名母體：本次觀察名單共 {rs_info.get('universe_size', 0)} 檔股票"
                       "（**不是**全市場排名，母體越小統計意義越弱，僅供本觀察名單內部比較）。")

        st.markdown("**📋 CAN SLIM 量化評分**")
        if canslim_report:
            cs1, cs2 = st.columns([1, 2])
            cs1.metric("CAN SLIM 總分", f"{canslim_report['total_score']:.1f} / {canslim_report['max_score']} ({canslim_report['pct']:.0f}%)")
            cs2.metric("評級", canslim_report['grade'])
            letter_cols = st.columns(7)
            for col, (letter, item) in zip(letter_cols, canslim_report['letters'].items()):
                with col:
                    st.markdown(f"**{letter}**　`{item['score']}/{item['max']}`")
            with st.expander("查看七項評分細節"):
                letter_names = {'C': 'Current Earnings（代理：營收年增率）', 'A': 'Annual Earnings（代理：ROE）',
                                'N': 'New High（新高）', 'S': 'Supply/Demand（量能）',
                                'L': 'Leader（RS Rating）', 'I': 'Institutional（法人認養）',
                                'M': 'Market Direction（大盤方向）'}
                for letter, item in canslim_report['letters'].items():
                    st.markdown(f"- **{letter} — {letter_names.get(letter, '')}**：{item['note']}")
            st.warning(canslim_report['disclosure'])
        else:
            st.caption("⚠️ CAN SLIM 評分計算失敗。")

        st.markdown("---")

        # --- 動態部位配置 (Kelly / ATR Position Sizing) ---
        st.markdown("### 🎯 動態部位配置 (Kelly Criterion / ATR Position Sizing)")
        st.caption("⚠️ 以下兩種部位配置皆為輔助工具，非下單建議；請務必先閱讀各自的警告說明再參考使用。")

        pos_col1, pos_col2 = st.columns(2)
        with pos_col1:
            st.markdown("**📐 ATR 部位配置**")
            account_equity_input = st.number_input("假設帳戶淨值（元）", min_value=10000, value=1000000, step=10000, key=f"acct_{ticker}")
            risk_pct_input = st.slider("單筆願意承受的最大虧損（% 帳戶淨值）", 0.5, 5.0, 1.0, 0.5, key=f"riskpct_{ticker}")
            atr_val = latest.get('atr_14', np.nan)
            atr_val = float(atr_val.iloc[0] if isinstance(atr_val, (np.ndarray, list, pd.Series)) else atr_val) if pd.notna(atr_val) else np.nan
            atr_pos = RiskEngine.compute_atr_position_size(account_equity_input, latest_close, atr_val, risk_pct_per_trade=risk_pct_input)
            if atr_pos.get('status') == 'ok':
                st.metric("建議股數", f"{atr_pos['shares']:,}")
                st.caption(f"部位金額：{atr_pos['position_value']:,.0f} 元（佔淨值 {atr_pos['position_pct_of_equity']:.1f}%）")
                st.caption(atr_pos['note'])
            else:
                st.caption(atr_pos.get('note', '⚠️ 資料不足'))

        with pos_col2:
            st.markdown("**🎲 Kelly 準則部位配置**")
            kelly_result = RiskEngine.compute_kelly_fraction(
                report.get('win_rate', 0), report.get('avg_win_pct', 0), report.get('avg_loss_pct', 0)
            )
            if kelly_result.get('status') == 'ok':
                st.metric("建議部位比例（半凱利封頂）", f"{kelly_result['kelly_pct']:.1f}%")
                st.caption(f"全凱利原始值：{kelly_result['full_kelly_pct']:.1f}%（僅供參考，實務不建議直接使用全凱利）")
                st.caption(kelly_result['note'])
            else:
                st.caption(kelly_result.get('note', '⚠️ 資料不足'))
            st.caption("⚠️ Kelly 計算基於本頁面上方全樣本內回測的歷史勝率/賺賠比，交易筆數過少時可信度低，"
                       "且历史績效不代表未來表現。")

        st.markdown("---")

        # --- 核心：多智能體辯論面板 (Multi-Agent Debate) ---
        st.markdown("### 🗣️ 多智能體對抗辯論 (Multi-Agent Debate)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(f"**🟢 Bull Agent (多頭辯護)**\n\n💡 觀點：\n\n{latest['bull_reason']}")
        with c2:
            st.error(f"**🔴 Bear Agent (空頭辯護)**\n\n💡 觀點：\n\n{latest['bear_reason']}")
        with c3:
            st.warning(f"**🛡️ Risk Agent (風控審查)**\n\n💡 觀點：\n\n{latest['risk_reason']}")
            
        st.markdown("---")

        # --- 多週期策略與未來走向 (Timeframe & Outlook Engine) ---
        st.markdown("### ⏳ 多週期策略判斷 (短線／波段／長線)")
        if tf_report and "error" not in tf_report:
            tf_col1, tf_col2, tf_col3 = st.columns(3)

            def _render_tf_card(col, title, info):
                with col:
                    st.markdown(f"**{title}**　`{info.get('view', 'N/A')}`")
                    st.caption(f"操作週期：{info.get('horizon', 'N/A')}")
                    st.markdown(f"依據：{info.get('reason', '')}")
                    st.markdown(f"建議：{info.get('action', '')}")

                    plan = info.get("trade_plan", {})
                    if plan.get("available"):
                        st.markdown("**📍 進場／出場價位參考**")
                        st.caption(f"進場參考價：約 {plan['entry_ref_price']}｜{plan['entry_condition']}")
                        target_line = f"出場目標：約 {plan['exit_target_price']}｜{plan['exit_target_condition']}"
                        st.caption(target_line)
                        st.caption(f"停損價：約 {plan['exit_stop_price']}｜{plan['exit_stop_condition']}")
                        st.caption(f"ℹ️ {plan['timing_note']}")
                    elif plan.get("note"):
                        st.caption(f"ℹ️ {plan['note']}")

            _render_tf_card(tf_col1, "⚡ 短線", tf_report.get("short_term", {}))
            _render_tf_card(tf_col2, "🌊 波段", tf_report.get("swing", {}))
            _render_tf_card(tf_col3, "🏔️ 長線", tf_report.get("long_term", {}))

            with st.expander("📊 短線進出場公式的簡化版歷史命中率檢查"):
                st.caption(
                    "⚠️ 這是簡化統計，不是跟 BacktestEngine 同等嚴謹的回測——用當天收盤價/ATR"
                    "當基準（比實際交易更樂觀），沒有計算手續費。僅供參考，不代表未來表現，不構成投資建議。"
                )
                if st.button("📊 執行短線命中率檢查", key="short_term_hitrate_btn"):
                    try:
                        _set_ticker_scoped_state("short_term_hitrate", ticker, TimeframeEngine.backtest_short_term_hit_rate(df))
                    except Exception as e:
                        _set_ticker_scoped_state("short_term_hitrate", ticker, {"status": "unavailable", "message": f"檢查時發生錯誤：{e}"})

                hitrate = _get_ticker_scoped_state("short_term_hitrate", ticker)
                if hitrate:
                    if hitrate.get("status") != "ok":
                        st.warning(hitrate.get("message", "⚠️ 無法計算命中率。"))
                    else:
                        hr1, hr2, hr3 = st.columns(3)
                        hr1.metric("目標優先命中", f"{hitrate['target_hit']} 次")
                        hr2.metric("停損優先觸發", f"{hitrate['stop_hit']} 次")
                        hr3.metric("目標命中率", f"{hitrate['win_rate_pct']}%")
                        st.caption(hitrate["note"])

            st.warning(
                "⚠️ 以上進場／出場價位都是「依目前技術結構與波動率(ATR)推算出的參考價位」，"
                "不是保證會被觸及的價位，也不是對未來日期的預測，市場可能直接跳空穿越、"
                "或永遠不回測到這個價位，不構成投資建議，請自行評估風險。"
            )

            st.markdown("#### 🔮 未來走向 (情境推演)")
            outlook = tf_report.get("outlook", {})
            st.markdown(f"**綜合傾向：`{outlook.get('bias', 'N/A')}`** — {outlook.get('bias_note', '')}")

            levels = outlook.get("key_levels", {})
            lv1, lv2, lv3 = st.columns(3)
            lv1.metric("推算支撐區", f"{levels.get('support_est')}" if levels.get('support_est') is not None else "N/A")
            lv2.metric("推算壓力區", f"{levels.get('resistance_est')}" if levels.get('resistance_est') is not None else "N/A")
            lv3.metric("季線(60MA)", f"{levels.get('sma_60')}" if levels.get('sma_60') is not None else "N/A")

            for scenario in outlook.get("scenarios", []):
                st.markdown(f"- **{scenario['condition']}** → {scenario['implication']}")

            st.warning(outlook.get("disclaimer", "⚠️ 本區塊僅供技術面情境參考，不構成投資建議。"))
        else:
            st.caption("⚠️ 尚無足夠資料生成多週期策略報告（請確認 timeframe_engine.py 是否已正確放入 engines 資料夾）。")

        st.markdown("---")

        # --- 飆股動能引擎 (Momentum Engine) ---
        st.markdown("### 🚀 飆股動能引擎 (九層過濾 + 100分評分系統，含跨股票 RS Rank)")
        mo1, mo2, mo3 = st.columns([1, 1, 2])
        mo1.metric("🚀 飆股評分 (Momentum Score)", f"{float(latest['momentum_score']):.1f} / 100")
        mo2.metric("🏅 飆股等級", f"{latest['momentum_grade']}")
        if bool(latest.get('momentum_score_complete', False)):
            st.caption("✅ 本評分已納入跨股票 RS Rank 與相對成交量排名（第⑦⑧層），為完整評分。")
        else:
            st.caption("⚠️ 本評分尚未納入跨股票 RS Rank/相對成交量排名（第⑦⑧層計0分），"
                       "理論上限僅85分，非完整評分，請參考下方九層明細。")
        with mo3:
            # 修正：原本只要 trap_alert 為 True 就顯示「本層評分已扣分」，但
            # trap_alert 也包含不影響本層分數的底部反轉觀察訊號（底背離/誘空
            # 確認），會出現「顯示扣分警告，但分數其實是滿分」的矛盾。改用
            # momentum_penalty_alert（真正造成扣分的誘多假突破/頂背離）判斷。
            penalty_active = bool(latest.get('momentum_penalty_alert', latest.get('trap_alert', False)))
            reversal_watch_active = bool(latest.get('reversal_watch', False))
            if penalty_active:
                st.error("🛡️ 誘多/背離防禦層：近期觸發誘多假突破/頂背離警報，本層評分已扣分")
            elif reversal_watch_active:
                st.info("👀 觀察：近期出現底背離/誘空反轉訊號（不影響飆股評分，僅供留意可能的底部訊號）")
            else:
                st.success("🛡️ 誘多/背離防禦層：近期無警報，本層滿分")

        try:
            breakdown = MomentumEngine.get_momentum_breakdown(
                df, -1, rs_rating=rs_info.get('rs_rating'), relative_volume_percentile=rvol_info.get('percentile')
            )
        except Exception:
            breakdown = []

        if breakdown:
            for layer in breakdown:
                icon = "✅" if layer["passed"] else "⬜"
                st.markdown(f"{icon} **{layer['layer']}**　_{layer['detail']}_")
        else:
            st.caption("⚠️ 尚無足夠資料生成飆股評分明細。")

        st.caption("⚠️ 飆股評分與 AI Score 是兩套獨立邏輯，刻意不互相加權平均：AI Score 回答『值不值得操作』，飆股評分回答『技術結構像不像飆股』，請兩者並參，不應單獨依賴其中一項下單。")

        st.markdown("---")
        
        # --- 數據面板：回測與風險預算 ---
        # ⚠️ 修正說明：原標題「Walk-Forward 樣本內回測」用詞矛盾且誤導 ——
        # Walk-Forward 指的是滾動式樣本外驗證，這裡實際上是全樣本內回測，
        # 進出場門檻與策略顯示邏輯相同，不構成獨立的樣本外驗證。已改用
        # 正確名稱，並把 BacktestEngine 回傳的 note／出場原因拆解／已扣費用
        # 一併顯示，避免使用者誤信這是嚴謹的策略驗證結果。
        col_bt1, col_bt2 = st.columns([1, 1])
        with col_bt1:
            st.markdown("#### 📈 全樣本內回測 (近兩年，已扣手續費/證交稅)")
            m1, m2, m3 = st.columns(3)
            m1.metric("策略總報酬率", f"{report['total_return']:.2f}%")
            m2.metric("交易勝率", f"{report['win_rate']:.1f}%")
            m3.metric("最大歷史回撤 (MDD)", f"{report['max_drawdown']:.2f}%")

            # v2.9.5 新增：CAGR / Sharpe / Sortino / 期望值 / Profit Factor
            # ——原本只有總報酬率跟MDD，沒辦法判斷這個報酬是用多大風險換來的，
            # 也沒辦法跟其他策略比較「風險調整後報酬」。
            if 'cagr_pct' in report:
                m4, m5, m6 = st.columns(3)
                cagr = report.get('cagr_pct')
                m4.metric("CAGR (年化報酬率)", f"{cagr:.2f}%" if cagr is not None else "N/A")
                sharpe = report.get('sharpe_ratio')
                m5.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A",
                          help="無風險利率簡化為0；交易筆數過少時可信度低")
                sortino = report.get('sortino_ratio')
                m6.metric("Sortino Ratio", f"{sortino:.2f}" if sortino is not None else "N/A")

                m7, m8 = st.columns(2)
                pf = report.get('profit_factor')
                m7.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "N/A",
                          help=">1 代表總獲利大於總虧損")
                m8.metric("期望值 (每筆交易平均報酬)", f"{report.get('expectancy_pct', 0):.2f}%")

            if 'total_fees_paid' in report:
                st.caption(f"💸 已扣總交易成本：約 {report['total_fees_paid']:,.0f} 元")
            if 'exit_breakdown' in report:
                breakdown = report['exit_breakdown']
                st.caption(
                    f"出場原因分布 — 停損: {breakdown.get('停損出場', 0)} 次 / "
                    f"停利: {breakdown.get('停利出場', 0)} 次 / "
                    f"AI Score轉弱: {breakdown.get('AI Score轉弱出場', 0)} 次"
                )
            # v2.9.7 新增：流動性/參與率真實性檢查（見 backtest_engine.py 說明）
            liq_check = report.get('liquidity_realism')
            if liq_check and liq_check.get('checked'):
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("平均參與率", f"{liq_check['avg_participation_pct']:.1f}%",
                           help="買進股數 ÷ 當天真實成交量")
                lc2.metric("最高單筆參與率", f"{liq_check['max_participation_pct']:.1f}%")
                lc3.metric("參與率過高筆數", f"{liq_check['high_participation_trades']}/{report['total_trades']}")
                if liq_check['high_participation_trades'] > 0:
                    st.warning(liq_check['note'])
                else:
                    st.caption(liq_check['note'])
            elif liq_check:
                st.caption(liq_check['note'])
            if 'note' in report:
                st.caption(report['note'])

        with col_bt2:
            st.markdown("#### 🎯 動態風險預算 (基於 ATR 動能)")
            n1, n2, n3 = st.columns(3)
            
            # 兼容處理停損停利值
            raw_sl = latest['stop_loss']
            raw_t1 = latest['target_1']
            raw_t2 = latest['target_2']
            val_sl = float(raw_sl.iloc[0] if isinstance(raw_sl, (np.ndarray, list, pd.Series)) else raw_sl)
            val_t1 = float(raw_t1.iloc[0] if isinstance(raw_t1, (np.ndarray, list, pd.Series)) else raw_t1)
            val_t2 = float(raw_t2.iloc[0] if isinstance(raw_t2, (np.ndarray, list, pd.Series)) else raw_t2)

            # 修正：標籤倍數原本與 StrategyEngine 實際公式（2.0x / 2.5x / 5.0x ATR）不符，
            # 已修正為正確倍數，避免使用者依錯誤標示的風險預算下單。
            n1.metric("🛡️ 建議停損點 (2.0x ATR)", f"{val_sl:.2f}")
            n2.metric("🎯 目標獲利 1 (2.5x ATR)", f"{val_t1:.2f}")
            n3.metric("🏆 目標獲利 2 (5.0x ATR)", f"{val_t2:.2f}")

        st.markdown("---")

        # ==========================================
        # v2.9.5 新增：多策略比較 (Strategy Comparison)
        # ==========================================
        # ⚠️ 原本只能一次跑一組參數，沒辦法直接比較「積極型」vs「保守型」
        # 進出場門檻何者風險調整後報酬較好。這裡用 BacktestEngine.compare_strategies
        # 內建三組常見的門檻組合，並依 Sharpe Ratio 排序，取代「憑感覺選門檻」。
        st.markdown("#### ⚔️ 多策略比較 (相同資料，不同進出場門檻)")
        try:
            strategy_variants = [
                {'name': '🔥 積極型 (70/45)', 'entry_threshold': 70, 'exit_score_threshold': 45},
                {'name': '⚖️ 均衡型 (75/50)', 'entry_threshold': 75, 'exit_score_threshold': 50},
                {'name': '🛡️ 保守型 (80/55)', 'entry_threshold': 80, 'exit_score_threshold': 55},
            ]
            compare_df = BacktestEngine.compare_strategies(backtest_input_df, strategy_variants)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.caption("⚠️ 三組門檻皆為全樣本內回測，彼此之間互相比較仍然只是同一份歷史資料的不同切法，"
                       "不構成樣本外驗證；Sharpe/Sortino 在交易筆數很少時（例如個位數）統計上不可靠，"
                       "請一併參考「交易次數」欄位再判斷。")
        except Exception as e:
            st.caption(f"⚠️ 多策略比較執行失敗：{e}")

        # ==========================================
        # v2.9.6 新增：樣本外分段穩健性檢定 (Segment Validation)
        # ==========================================
        st.markdown("#### 🧪 分段穩健性檢定（非參數優化式 Walk-Forward）")
        try:
            seg_result = BacktestEngine.run_segment_validation(backtest_input_df, n_segments=4)
            if seg_result['status'] == 'ok':
                sc1, sc2 = st.columns(2)
                sc1.metric("穩健度 (獲利段數比例)", f"{seg_result['consistency_ratio']*100:.0f}%")
                sc2.metric("獲利段數 / 總段數", f"{seg_result['consistent_positive_segments']} / {seg_result['n_segments']}")
                seg_table = pd.DataFrame(seg_result['segments'])
                st.dataframe(seg_table, use_container_width=True, hide_index=True)
                st.warning(seg_result['note'])
            else:
                st.caption(seg_result.get('message', '⚠️ 資料不足，無法進行分段檢定。'))
        except Exception as e:
            st.caption(f"⚠️ 分段穩健性檢定執行失敗：{e}")

        # ==========================================
        # 互動式 K 線圖與斐波那契矩陣
        # ==========================================
        st.markdown("---")
        recent_df = df.tail(150)
        
        fig = fgo.Figure()
        
        # 繪製 K 線
        fig.add_trace(fgo.Candlestick(
            x=recent_df['date'], open=recent_df['open'], 
            high=recent_df['high'], low=recent_df['low'], close=recent_df['close'], name="K線"
        ))
        
        # 繪製 ZigZag 市場結構線 (過濾掉 NaN)
        if 'zigzag' in recent_df.columns:
            zigzag_df = recent_df.dropna(subset=['zigzag'])
            if not zigzag_df.empty:
                fig.add_trace(fgo.Scatter(
                    x=zigzag_df['date'], y=zigzag_df['zigzag'],
                    mode='lines+markers', name='市場結構 (ZigZag)',
                    line=dict(color='cyan', width=1, dash='dot'),
                    marker=dict(size=6, color='cyan')
                ))

        # 標註 MACD 背離與誘多/誘空防禦警報（均為因果安全的「確認當下」標記，
        # 不是回填到轉折發生的那一天，詳見 divergence_engine.py 說明）
        marker_specs = [
            ('bearish_divergence', '🔴 頂背離', 'high', 12, 'triangle-down', '#ff4d4d'),
            ('bullish_divergence', '🟢 底背離', 'low', -12, 'triangle-up', '#33cc33'),
            ('bull_trap_confirmed', '⚠️ 誘多假突破', 'high', 20, 'x', '#ffcc00'),
            ('bear_trap_confirmed', '⚠️ 誘空假跌破', 'low', -20, 'x', '#ff9933'),
        ]
        for col, label, ref_col, offset, symbol, color in marker_specs:
            if col in recent_df.columns:
                hits = recent_df[recent_df[col] == True]
                if not hits.empty:
                    y_vals = hits[ref_col] + offset if offset > 0 else hits[ref_col] + offset
                    fig.add_trace(fgo.Scatter(
                        x=hits['date'], y=y_vals, mode='markers', name=label,
                        marker=dict(size=11, symbol=symbol, color=color, line=dict(width=1, color='white'))
                    ))
        
        # 自動繪製近期斐波那契回撤線
        max_h = recent_df['high'].max()
        min_l = recent_df['low'].min()
        diff = max_h - min_l
        
        fib_levels = {
            "100% (High)": max_h,
            "61.8% (Golden)": min_l + 0.618 * diff,
            "50.0% (Mid)": min_l + 0.500 * diff,
            "38.2% (Support)": min_l + 0.382 * diff,
            "0% (Low)": min_l
        }
        
        colors = ['#ff4d4d', '#ffcc00', '#33cc33', '#3399ff', '#cccccc']
        for (label, price), color in zip(fib_levels.items(), colors):
            # 確保價格為純數值類型
            val_price = float(price.iloc[0] if isinstance(price, (np.ndarray, list, pd.Series)) else price)
            fig.add_hline(
                y=val_price, line_dash="dash", line_color=color, opacity=0.4,
                annotation_text=f"Fib {label}  {val_price:.1f}", 
                annotation_position="top left",
                annotation_font=dict(color=color, size=10)
            )

        fig.update_layout(
            template="plotly_dark", 
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        with st.expander("🏦 大戶持股／千張大戶分析 — 資料源：TDCC集保戶股權分散表，點擊展開後執行"):
            st.caption(
                "⚠️ 每週更新一次（反映集保帳戶而非實質受益人，非即時資料），"
                "「大戶」採業界慣例定義：持股1,000張以上；歷史趨勢是本系統自己逐週累積的記錄，"
                "剛開始使用時筆數會很少，屬正常現象。"
            )
            if st.button("🏦 查詢大戶持股分布（抓取TDCC開放資料）", key="shareholding_btn"):
                with st.spinner("下載並解析集保戶股權分散表中（全市場單一檔案，第一次查詢較慢）..."):
                    try:
                        _set_ticker_scoped_state("shareholding_report", ticker, ChipEngine.get_shareholding_distribution(ticker))
                    except Exception as e:
                        _set_ticker_scoped_state("shareholding_report", ticker, {"status": "unavailable", "message": f"⚠️ 大戶持股資料查詢時發生錯誤：{e}"})

            shareholding_report = _get_ticker_scoped_state("shareholding_report", ticker)
            if shareholding_report:
                if shareholding_report.get("status") != "ok":
                    st.warning(shareholding_report.get("message", "⚠️ 大戶持股資料暫時無法使用。"))
                else:
                    sh_col1, sh_col2, sh_col3 = st.columns(3)
                    sh_col1.metric("資料日期", shareholding_report["date"])
                    sh_col2.metric("千張大戶持股佔比", f"{shareholding_report['large_holder_pct']}%")
                    sh_col3.metric("千張大戶人數", f"{shareholding_report['large_holder_count']:,}")

                    st.dataframe(shareholding_report["tiers"], use_container_width=True, hide_index=True)
                    for f in shareholding_report["flags"]:
                        st.markdown(f"- {f}")

                    trend = ChipEngine.get_shareholding_trend(ticker)
                    if len(trend) >= 2:
                        st.markdown("**大戶持股佔比歷史趨勢（本系統累積記錄）**")
                        st.line_chart(trend.set_index("date")["large_holder_pct"])
                    else:
                        st.caption("ℹ️ 目前只有本次查詢的單一筆記錄，趨勢圖需要多次（跨週）查詢後才會累積出來。")

        with st.expander("🏛️ 董監事／大股東持股與設質分析 — 資料源：證交所董監持股開放資料，僅支援上市股票"):
            st.caption(
                "⚠️ 只支援上市股票（上櫃另有不同資料檔案，尚未驗證，暫不支援），每月更新一次，"
                "反映申報當下持股，不是即時資料；設質比例高不代表一定有問題，需搭配其他資訊綜合判斷，不構成投資建議。"
            )
            if st.button("🏛️ 查詢董監事持股與設質狀況（抓取證交所開放資料）", key="insider_btn"):
                with st.spinner("下載並解析董監事持股資料中（全市場單一檔案，第一次查詢較慢）..."):
                    try:
                        _set_ticker_scoped_state("insider_report", ticker, ChipEngine.get_insider_holdings(ticker))
                    except Exception as e:
                        _set_ticker_scoped_state("insider_report", ticker, {"status": "unavailable", "message": f"⚠️ 董監事持股資料查詢時發生錯誤：{e}"})

            insider_report = _get_ticker_scoped_state("insider_report", ticker)
            if insider_report:
                if insider_report.get("status") != "ok":
                    st.warning(insider_report.get("message", "⚠️ 董監事持股資料暫時無法使用。"))
                else:
                    in_col1, in_col2 = st.columns(2)
                    in_col1.metric("資料年月", insider_report["data_month"])
                    pledge_display = "N/A" if insider_report['max_pledge_pct'] is None else f"{insider_report['max_pledge_pct']}%"
                    in_col2.metric("設質比例（個人/關係人較高者）", pledge_display)

                    st.dataframe(insider_report["detail"], use_container_width=True, hide_index=True)
                    for f in insider_report["flags"]:
                        st.markdown(f"- {f}")

                    if not insider_report["high_pledge_table"].empty:
                        st.markdown("**⚠️ 個人設質比例超過50%的內部人**")
                        st.dataframe(insider_report["high_pledge_table"], use_container_width=True, hide_index=True)

        with st.expander("📅 季節循環分析 (Seasonality) — 需額外抓取10年歷史資料，點擊展開後執行"):
            st.caption(
                "⚠️ 這裡只統計「月份別歷史報酬」，不做除權息行情/選舉行情等需要額外事件資料源的分析；"
                "樣本數很小（10年資料每個月份也只有約10個獨立觀察值），任何規律都可能只是雜訊，"
                "不是穩定可複製的效應，僅供參考，不能單獨依此做進出場決策。"
            )
            if st.button("📅 執行季節循環分析（抓取10年歷史資料）", key="seasonality_btn"):
                with st.spinner("抓取長天期歷史資料並計算月份統計中..."):
                    try:
                        _set_ticker_scoped_state("seasonality_report", ticker, SeasonalityEngine.build_seasonality_report(ticker))
                    except Exception as e:
                        _set_ticker_scoped_state("seasonality_report", ticker, {"status": "unavailable", "message": f"⚠️ 季節循環分析時發生錯誤：{e}"})

            seasonality_report = _get_ticker_scoped_state("seasonality_report", ticker)
            if seasonality_report:
                if seasonality_report.get("status") != "ok":
                    st.warning(seasonality_report.get("message", "⚠️ 季節性分析暫時無法使用。"))
                else:
                    st.caption(f"共涵蓋約 {seasonality_report['years_covered']} 個曆年的歷史資料。")
                    st.dataframe(seasonality_report["monthly_table"], use_container_width=True, hide_index=True)
                    for f in seasonality_report["flags"]:
                        st.markdown(f"- {f}")

        # ==========================================
        # 📰 新聞情緒中心（Phase 1+2，v2.9.12 新增）
        # ==========================================
        # ⚠️ 說明：來源是 Google 新聞 RSS（免費/免金鑰），情緒判斷是分層
        # 加權關鍵字（Tier1硬資訊權重2／Tier2軟敘述權重1）+ 時間衰減，
        # 不是語意理解，詳細範圍與限制見 NewsEngine class docstring。
        # 這裡刻意「不」把新聞情緒併入七層評分或AI總分，只單獨呈現，
        # 由使用者自行跟技術面/籌碼面/基本面交叉判斷，理由見達人視角提醒。
        # Phase 2：AI摘要（規則式baseline，可選接LLM）、概念股關聯、
        # 與技術面共振檢查（用DecisionEngine既有的評分換算成星等）。
        with st.expander("📰 新聞情緒中心 — 資料源：Google新聞RSS，規則式情緒分析（免費，非LLM）"):
            st.caption(
                "⚠️ 新聞情緒是落後或同步指標，不是獨立的進出場訊號，"
                "規則式關鍵字判讀無法理解反諷、條件句或「利多出盡」這類語境，"
                "僅供輔助參考，不構成投資建議。"
            )
            if st.button("📰 抓取最新相關新聞並分析情緒", key="news_btn"):
                with st.spinner("抓取新聞並進行情緒分析中..."):
                    try:
                        _news_stock_name = NameEngine.get_name(ticker)
                        _set_ticker_scoped_state(
                            "news_report", ticker,
                            NewsEngine.get_news_with_sentiment(ticker, name=_news_stock_name),
                        )
                    except Exception as e:
                        _set_ticker_scoped_state(
                            "news_report", ticker,
                            {"status": "error", "error": f"⚠️ 新聞抓取時發生錯誤：{e}", "items": []},
                        )

            news_report = _get_ticker_scoped_state("news_report", ticker)
            if news_report:
                if news_report.get("status") == "error":
                    st.error(news_report.get("error", "⚠️ 新聞暫時無法使用。"))
                elif news_report.get("status") == "empty" or not news_report.get("items"):
                    st.info("目前查無相關新聞（可能是搜尋關鍵字沒有命中，或近期確實沒有報導）。")
                else:
                    stats = news_report["summary_stats"]
                    vol = news_report.get("volume_change", {})

                    st.caption(
                        f"共抓到 {stats['total']} 篇原始報導，事件去重後為 {stats.get('unique_events', stats['total'])} 則獨立事件"
                        "（同一事件被多家媒體轉載只算一次，避免情緒分數被灌水）。"
                    )
                    n_col1, n_col2, n_col3 = st.columns(3)
                    n_col1.metric("整體新聞偏向（加權，去重後）", stats["overall_bias"],
                                  f"加權分數 {stats['weighted_bias_score']}")
                    n_col2.metric("利多／利空／中性（去重後事件數）",
                                  f"{stats['bullish']} / {stats['bearish']} / {stats['neutral']}")
                    trend_label = {"increase": "📈 較上次增加", "decrease": "📉 較上次減少",
                                    "flat": "→ 與上次持平", "unknown": "（首次查詢，無比較基準）"}
                    n_col3.metric("新聞量變化（原始篇數）", trend_label.get(vol.get("trend"), "—"))

                    ai_summary = news_report.get("ai_summary", {})
                    _stars = ai_summary.get("stars", 1)
                    st.markdown("**📋 AI摘要**")
                    st.write(f"{'★' * _stars}{'☆' * (5 - _stars)}　整體評價：{ai_summary.get('label', '')}")
                    for b in ai_summary.get("bullets", []):
                        st.markdown(f"- {b}")
                    st.caption(
                        f"摘要方式: {'LLM' if ai_summary.get('method') == 'llm' else '規則式（統計高頻事件標籤，非語意摘要）'}"
                    )

                    # 新聞 × 技術面 共振檢查：技術面星等用上方「共識儀表板」
                    # (consensus) 的偏多維度比例粗略換算，需要使用者已經跑過
                    # 主分析流程才會有 consensus 這個變數，沒有就跳過不硬湊。
                    try:
                        _tech_stars = max(1, min(5, round(1 + 4 * (consensus['bullish_count'] / consensus['total_dims']))))
                        _resonance = NewsEngine.compute_resonance(_stars, _tech_stars)
                        st.markdown("**🔗 新聞 × 技術面 共振檢查**")
                        st.caption(
                            f"新聞面 {'★' * _stars} ／ 技術面(共識儀表板換算) {'★' * _tech_stars}"
                        )
                        st.info(_resonance["message"])
                    except NameError:
                        st.caption("ℹ️ 尚未產生「共識儀表板」結果，暫時無法做新聞×技術面共振檢查。")
                    except Exception:
                        pass

                    st.markdown("**達人視角提醒**")
                    for note in news_report.get("investor_notes", []):
                        st.caption(f"• {note}")

                    st.markdown("---")
                    icon_map = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
                    for item in news_report["items"]:
                        if not item.get("is_representative", True):
                            continue  # 非代表新聞（同事件的重複報導）預設收起不重複列出
                        sentiment = item["sentiment"]
                        icon = icon_map.get(sentiment["label"], "🟡")
                        method_label = "LLM" if sentiment.get("method") == "llm" else "規則式"
                        st.markdown(f"{icon} **{item['title']}**")
                        tag_str = "、".join(item.get("event_tags", [])) or "無"
                        dup_note = f"，另有 {item['duplicate_count']} 篇媒體報導同一事件（已合併計分，不重複列出）" if item.get("duplicate_count") else ""
                        conf = item.get("event_confidence") or {}
                        conf_note = f" · 事件可信度: {conf.get('level','—')}（{conf.get('source_count','—')}家獨立來源)" if conf else ""
                        st.caption(
                            f"{item.get('source', '')}（可信度權重{item.get('source_weight', 1):.2f}） · {item.get('published', '')} · "
                            f"分析方式: {method_label} · 標籤: {tag_str}{dup_note}{conf_note}"
                        )
                        if item.get("related_concepts"):
                            concept_str = "、".join(
                                f"{c['name']}({c['code']}) {'★' * c['stars']}"
                                for c in item["related_concepts"][:5]
                            )
                            st.caption(f"📌 可能受影響個股（示範性對照表，非投資建議）: {concept_str}")
                        if item.get("link"):
                            st.caption(f"[原文連結]({item['link']})")