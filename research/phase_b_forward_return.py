# -*- coding: utf-8 -*-
"""
research/phase_b_forward_return.py

📈 Breakout Alpha Study — Phase B：Forward Return Event Study

⚠️ 這支程式回答的問題（跟外部 Research 建議對齊，每個 metadata 對應一個
    明確可驗證的假說，不是只丟一張統計表）：
    B.1 Volume  ：有量突破是否比無量突破具有更高的未來報酬？
    B.2 Liquidity：低流動性是否會降低突破成功率，或只是提高波動？
    B.3 Breakout Count：第一次突破是否比重複突破更具有 Alpha？

── 跟 event_inventory.py 的分工 ──
event_inventory.py（Phase A）只回答「有多少事件、分佈在哪裡」，刻意不算
Forward Return。這支程式接手 Phase A 的產出（research/output/
breakout_events_raw.csv），對每一筆事件計算 5D/10D/20D 前瞻報酬與最大
回落（MDD），再依三個維度分組比較——這就是 Phase B 的全部工作，不做
特徵工程、不做模型、不做決策。

── 為什麼三個維度共用同一支程式，而不是各寫一支 ──
三個研究問題的統計骨架完全一樣：「事件發生後，Forward Return 有沒有因為
某個 metadata 值不同而系統性不同」，差異只在分組欄位跟分組方式（二元/
三元/桶）。共用同一套 compute_forward_returns() + group_forward_return_study()
避免同一段「抓價格、算報酬、算MDD、做檢定」的邏輯被複製三次、之後要修
要改三次。

── 統計方法選擇：為什麼用 Mann-Whitney U / Kruskal-Wallis，不用 t-test ──
金融報酬率分佈通常右偏、有肥尾，不滿足 t-test 的常態假設，尤其在事件數
偏少（Phase A 目前是 CAUTION 等級，100~300 個事件）時更明顯。這裡改用
無母數檢定（不假設分佈形狀），檢定力雖然略低於母數方法，但在樣本違反
常態假設時更不容易得出偽陽性結論。

── 誠實揭露：多重比較問題 ──
三個維度 × 三個 horizon（5D/10D/20D）= 9 次檢定，如果每次都用 p<0.05
當作「顯著」的判斷標準，在虛無假設全部為真的情況下，預期仍會有將近
1-(0.95)^9 ≈ 37% 的機率至少出現一次「假陽性」顯著結果。這支程式的
summary 輸出會附上 Bonferroni 校正後的門檻（0.05/9 ≈ 0.0056）供參考，
但不會自動用校正後門檻去篩選/隱藏任何一次檢定結果——所有 9 次結果都會
完整輸出，校正門檻只是提醒「不要因為看到一個 p<0.05 就直接下結論」。

── 使用方式 ──
    python -m research.phase_b_forward_return --dimension volume
    python -m research.phase_b_forward_return --dimension liquidity
    python -m research.phase_b_forward_return --dimension breakout_count
    python -m research.phase_b_forward_return --dimension all

⚠️ 這支程式需要對外部網路（yfinance）逐檔抓歷史股價，本開發沙盒環境沒有
對外網路權限，只完成了語法檢查與邏輯設計，並用合成資料做過自我測試
（見 research/test_phase_b_forward_return.py），**尚未在真實資料上跑過**。
請在你自己有網路權限的環境跑一次。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engines.data_engine import DataEngine
from engines.logging_config import get_logger
from research.event_inventory import EVENTS_CSV, OUTPUT_DIR

logger = get_logger(__name__)

HORIZONS = (5, 10, 20)  # 交易日

# 9 次檢定（3 維度 × 3 horizon）的 Bonferroni 校正門檻，只供參考顯示，
# 不用來自動篩選/隱藏任何一次檢定結果（見檔案開頭說明）。
BONFERRONI_ALPHA = round(0.05 / (3 * len(HORIZONS)), 4)

RESULTS_CSV_TEMPLATE = os.path.join(OUTPUT_DIR, "phase_b_{dimension}_events_with_returns.csv")
SUMMARY_JSON_TEMPLATE = os.path.join(OUTPUT_DIR, "phase_b_{dimension}_summary.json")


# ==========================================
# 第一步：對每筆事件算 Forward Return / MDD
# ==========================================
def compute_forward_returns(events_df: pd.DataFrame, horizons: tuple = HORIZONS,
                             use_cache: bool = True, price_lookup: dict = None) -> pd.DataFrame:
    """
    對 events_df 的每一列（一筆突破事件），用該股票的歷史收盤價序列，
    計算事件發生後 horizons 天的前瞻報酬（%）與到 20D 為止的最大回落（MDD，%）。

    ⚠️ 避免 look-ahead bias 的關鍵：進場價用「事件當天收盤價」（跟
    event_inventory.py 的 edge-triggered 判斷基準一致——那天已經確認
    breakout_confirmed，收盤價是該天已知的最後一筆資料），forward return
    只往「事件日之後」看，不會用到事件日當天或之前還沒發生的資訊。

    price_lookup: 可選，{ticker: price_df} 的字典，測試時用合成資料注入，
    避免呼叫 DataEngine 對外發送真實請求。正式使用時留空，會自動用
    DataEngine.get_stock_data() 逐檔抓取（同一檔股票在同一次呼叫內只抓
    一次，不會每筆事件各抓一次）。

    新增欄位：
      forward_return_{h}d      事件後第 h 個交易日的報酬率（%），資料不足時為 NaN
      insufficient_future_data 事件日之後可用交易日數 < max(horizons)，代表
                                20D 報酬是用不完整資料算的，或根本算不出來
                                （不是「報酬剛好是0」，是「還沒到那天」，
                                 這兩者意義完全不同，必須分開標示，不能用
                                 NaN 之後又被下游誤當成0處理）
      mdd_20d                  事件後20個交易日內（不足20天則用實際可用天數）
                                的最大回落（%，負值），用「事件日收盤價」為基準，
                                不是用區間內最高點為基準——因為我們關心的是
                                「進場之後最慘會賠多少」，不是區間內部的波動
      win_20d                  forward_return_20d > 0 時為 True，資料不足
                                （NaN）時為 None，不會被誤當成 False
    """
    if events_df is None or events_df.empty:
        return events_df

    events_df = events_df.copy()
    max_h = max(horizons)
    price_lookup = price_lookup or {}

    for h in horizons:
        events_df[f"forward_return_{h}d"] = np.nan
    events_df["mdd_20d"] = np.nan
    events_df["win_20d"] = None
    events_df["insufficient_future_data"] = None

    tickers = events_df["ticker"].unique()
    for ticker in tickers:
        if ticker in price_lookup:
            price_df = price_lookup[ticker]
        else:
            try:
                price_df = DataEngine.get_stock_data(ticker, use_cache=use_cache)
            except Exception:
                logger.exception(f"[{ticker}] Phase B 抓取歷史股價失敗，此股票的所有事件 forward return 記為 NaN")
                price_df = None

        if price_df is None or price_df.empty or "close" not in price_df.columns:
            continue

        price_df = price_df.sort_values("date").reset_index(drop=True)
        price_df["_date_str"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
        date_to_idx = {d: i for i, d in enumerate(price_df["_date_str"])}
        closes = price_df["close"].values

        row_idx_list = events_df.index[events_df["ticker"] == ticker]
        for row_idx in row_idx_list:
            event_date = str(events_df.at[row_idx, "event_date"])
            entry_idx = date_to_idx.get(event_date)
            if entry_idx is None:
                # 事件日在目前的價格快取範圍之外（例如快取只保留近2年，
                # 但事件是更早以前的），此列所有 forward return 保持 NaN，
                # insufficient_future_data 標記為 True 讓下游知道不是「無反應」
                events_df.at[row_idx, "insufficient_future_data"] = True
                continue

            available_future_days = len(closes) - 1 - entry_idx
            entry_price = closes[entry_idx]
            if entry_price is None or entry_price <= 0 or np.isnan(entry_price):
                events_df.at[row_idx, "insufficient_future_data"] = True
                continue

            events_df.at[row_idx, "insufficient_future_data"] = bool(available_future_days < max_h)

            for h in horizons:
                if available_future_days >= h:
                    fwd_price = closes[entry_idx + h]
                    events_df.at[row_idx, f"forward_return_{h}d"] = round(
                        (float(fwd_price) / float(entry_price) - 1) * 100, 3)

            # MDD：事件日收盤價為基準，看未來最多20天內（或可用天數）最低點跌了多少
            window_end = min(entry_idx + 20, len(closes) - 1)
            if window_end > entry_idx:
                window_closes = closes[entry_idx: window_end + 1]
                min_price = float(np.nanmin(window_closes))
                mdd = round((min_price / float(entry_price) - 1) * 100, 3)
                events_df.at[row_idx, "mdd_20d"] = mdd

            fwd20 = events_df.at[row_idx, "forward_return_20d"]
            events_df.at[row_idx, "win_20d"] = (bool(fwd20 > 0) if pd.notna(fwd20) else None)

    return events_df


# ==========================================
# 第二步：依維度分組 + 無母數檢定
# ==========================================
def _bootstrap_ci_mean(values: pd.Series, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    """對 values 做 bootstrap resampling，回傳 mean 的 (lower, upper) 百分位信賴區間。
    ⚠️ n<5 時信賴區間本身沒有意義（重抽樣也抽不出有代表性的分佈），直接回傳 (None, None)，
    不會硬算一個看起來煞有其事、但其實毫無穩定性的區間。"""
    valid = values.dropna().values
    if len(valid) < 5:
        return None, None
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(valid, size=len(valid), replace=True).mean() for _ in range(n_boot)])
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 - (1 - ci) / 2) * 100
    return round(float(np.percentile(boot_means, lower_pct)), 3), round(float(np.percentile(boot_means, upper_pct)), 3)


def _group_stats(sub_df: pd.DataFrame, horizons: tuple = HORIZONS, n_boot: int = 1000) -> dict:
    """單一分組的敘述統計：樣本數、各 horizon 的平均/中位數報酬（含 bootstrap 95% CI）、
    勝率（含 bootstrap 95% CI）、平均MDD。

    ⚠️ 新增 bootstrap CI 的原因：單一平均值本身沒有不確定性資訊，樣本數少時尤其容易
    誤導——「平均 +3%」聽起來像是穩定的正報酬，但如果 95% CI 是 [-2%, +8%]，代表這個
    正數其實還在雜訊範圍內，不能拿來當結論。CI 用非參數 bootstrap（不假設分佈形狀），
    跟本檔案選用無母數檢定的理由一致。"""
    stats = {"n": int(len(sub_df))}
    if "ticker" in sub_df.columns:
        # ⚠️ 這個數字比 n 更重要：186個「事件」不是186個獨立樣本，如果同一組裡
        # n=114 但 n_unique_tickers 只有20出頭，代表這組結果高度依賴少數幾檔股票
        # 的走勢，Kruskal/Mann-Whitney 假設的「觀測值互相獨立」在事件層級上其實
        # 不成立（同一檔股票的多次事件，走勢/報酬有序列相關）。這裡只誠實揭露
        # 這個數字，不假裝用 cluster-robust 方法解決——要解決需要換成以「股票」
        # 為抽樣單位的統計方法，是比目前更進一步的研究設計問題。
        stats["n_unique_tickers"] = int(sub_df["ticker"].nunique())
    for h in horizons:
        col = f"forward_return_{h}d"
        valid = sub_df[col].dropna()
        ci_lo, ci_hi = _bootstrap_ci_mean(valid, n_boot=n_boot)
        stats[f"forward_return_{h}d"] = {
            "n_valid": int(len(valid)),
            "mean": round(float(valid.mean()), 3) if len(valid) else None,
            "median": round(float(valid.median()), 3) if len(valid) else None,
            "std": round(float(valid.std()), 3) if len(valid) > 1 else None,
            "mean_ci95": [ci_lo, ci_hi],
        }
    win_valid = sub_df["win_20d"].dropna()
    win_ci_lo, win_ci_hi = _bootstrap_ci_mean(win_valid.astype(float), n_boot=n_boot)
    stats["win_rate_20d_pct"] = round(float(win_valid.mean()) * 100, 1) if len(win_valid) else None
    stats["win_rate_20d_ci95_pct"] = ([round(win_ci_lo * 100, 1), round(win_ci_hi * 100, 1)]
                                       if win_ci_lo is not None else [None, None])
    mdd_valid = sub_df["mdd_20d"].dropna()
    stats["mean_mdd_20d_pct"] = round(float(mdd_valid.mean()), 3) if len(mdd_valid) else None
    return stats


def group_forward_return_study(events_with_returns: pd.DataFrame, group_col: str,
                                hypothesis: str, horizons: tuple = HORIZONS,
                                min_group_n: int = 20) -> dict:
    """
    依 group_col 分組比較 forward return，並對每個 horizon 做無母數檢定。

    ⚠️ min_group_n（預設20）：任一組樣本數低於這個門檻時，仍會輸出敘述統計，
    但檢定結果會標示 "insufficient_sample"，不會硬跑一個小樣本檢定又不
    提醒——小樣本下無母數檢定的檢定力本來就很低，跑出來的 p-value 很容易
    不穩定，不應該被當成有意義的結論。

    回傳格式：
        {
            "hypothesis": "...",
            "group_col": "volume_confirmed",
            "groups": {"True": {...}, "False": {...}},
            "tests": {
                "forward_return_5d": {"test": "mannwhitneyu", "statistic": ..,
                                       "p_value": .., "verdict": "..."},
                ...
            },
            "bonferroni_alpha_9_tests": 0.0056,
        }
    """
    from scipy import stats as scipy_stats

    df = events_with_returns.copy()
    df = df[df[group_col].notna()]

    group_values = sorted(df[group_col].unique().tolist(), key=str)
    groups = {str(v): _group_stats(df[df[group_col] == v], horizons) for v in group_values}

    tests = {}
    if len(group_values) == 2:
        g1 = df[df[group_col] == group_values[0]]
        g2 = df[df[group_col] == group_values[1]]
        for h in horizons:
            col = f"forward_return_{h}d"
            v1, v2 = g1[col].dropna(), g2[col].dropna()
            if len(v1) < min_group_n or len(v2) < min_group_n:
                tests[col] = {"test": "mannwhitneyu", "statistic": None, "p_value": None, "effect_size": None,
                              "verdict": (f"insufficient_sample（組{group_values[0]} n={len(v1)}、"
                                          f"組{group_values[1]} n={len(v2)}，至少一組 < min_group_n={min_group_n}，"
                                          f"不下檢定結論）")}
                continue
            try:
                stat, p = scipy_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                # 效果量：rank-biserial correlation（Wendt公式），跟 U 統計量對應同一組
                # v1/v2 順序，範圍 -1~1，|r|>=0.5 大致對應大效果、0.3左右中效果、0.1左右
                # 小效果（Cohen的經驗法則移植到無母數效果量的常見對應，非嚴格換算）。
                # p-value 小不代表效果量大——這是這次新增的目的：避免「n很大時，微小
                # 差異也能測出p<0.05」被誤讀成「有實際交易價值」。
                n1, n2 = len(v1), len(v2)
                rank_biserial = round(1 - (2 * float(stat)) / (n1 * n2), 3)
                tests[col] = {
                    "test": "mannwhitneyu", "statistic": round(float(stat), 3), "p_value": round(float(p), 4),
                    "effect_size": {"type": "rank_biserial_correlation", "value": rank_biserial},
                    "verdict": ("p < Bonferroni門檻，但仍只是初步觀察，不是正式因果結論"
                                if p < BONFERRONI_ALPHA else
                                "p >= Bonferroni門檻，未通過多重比較校正後的顯著性門檻"),
                }
            except Exception:
                logger.exception(f"{group_col} / {col} Mann-Whitney U 檢定失敗")
                tests[col] = {"test": "mannwhitneyu", "statistic": None, "p_value": None,
                              "effect_size": None, "verdict": "檢定執行失敗"}
    elif len(group_values) > 2:
        for h in horizons:
            col = f"forward_return_{h}d"
            samples = [df[df[group_col] == v][col].dropna() for v in group_values]
            if any(len(s) < min_group_n for s in samples):
                n_report = ", ".join(f"{v}: n={len(s)}" for v, s in zip(group_values, samples))
                tests[col] = {"test": "kruskal", "statistic": None, "p_value": None, "effect_size": None,
                              "verdict": f"insufficient_sample（{n_report}；至少一組 < min_group_n={min_group_n}，不下檢定結論）"}
                continue
            try:
                stat, p = scipy_stats.kruskal(*samples)
                # 效果量：epsilon-squared（H統計量的標準化版本，Kruskal-Wallis對應
                # eta-squared的無母數類比），範圍約0~1，跟rank-biserial一樣提醒
                # 「p很小不等於效果量大」。
                n_total = sum(len(s) for s in samples)
                k = len(samples)
                # ⚠️ epsilon-squared 公式在 H < k-1 時（組間差異比隨機期望還小）
                # 會算出負值，但「解釋變異量」的定義上不可能是負的——這是估計式
                # 在虛無假設幾乎成立時的已知瑕疵，不是真的「負效果」，統計上慣例
                # 是把它視為約等於0（沒有可偵測的組間差異），這裡直接floor在0，
                # 不回傳誤導性的負數。
                epsilon_sq = max(0.0, round(float(stat - k + 1) / (n_total - k), 3)) if n_total > k else None
                tests[col] = {
                    "test": "kruskal", "statistic": round(float(stat), 3), "p_value": round(float(p), 4),
                    "effect_size": {"type": "epsilon_squared", "value": epsilon_sq},
                    "verdict": ("p < Bonferroni門檻，但仍只是初步觀察，不是正式因果結論"
                                if p < BONFERRONI_ALPHA else
                                "p >= Bonferroni門檻，未通過多重比較校正後的顯著性門檻"),
                }
            except Exception:
                logger.exception(f"{group_col} / {col} Kruskal-Wallis 檢定失敗")
                tests[col] = {"test": "kruskal", "statistic": None, "p_value": None,
                              "effect_size": None, "verdict": "檢定執行失敗"}

    return {
        "hypothesis": hypothesis,
        "group_col": group_col,
        "groups": groups,
        "tests": tests,
        "bonferroni_alpha_9_tests": BONFERRONI_ALPHA,
    }


# ==========================================
# 三個維度的分組欄位準備
# ==========================================
def prepare_breakout_count_bucket(events_df: pd.DataFrame) -> pd.DataFrame:
    """把 event_seq_for_ticker（該股票第幾次突破）分桶成第1/2/3/第4次以上，
    對應 B.3「第一次突破是否比重複突破更具有Alpha」的假說。"""
    events_df = events_df.copy()

    def _bucket(seq):
        if pd.isna(seq):
            return None
        seq = int(seq)
        if seq >= 4:
            return "第4次以上"
        return f"第{seq}次"

    events_df["breakout_count_bucket"] = events_df["event_seq_for_ticker"].apply(_bucket)
    return events_df


DIMENSION_CONFIG = {
    "volume": {
        "group_col": "volume_confirmed",
        "prepare": lambda df: df,
        "hypothesis": "有量突破是否比無量突破具有更高的未來報酬？",
    },
    "liquidity": {
        "group_col": "liquidity_level_at_event",
        "prepare": lambda df: df,
        "hypothesis": "低流動性是否會降低突破成功率，或只是提高波動（MDD更深）？",
    },
    "breakout_count": {
        "group_col": "breakout_count_bucket",
        "prepare": prepare_breakout_count_bucket,
        "hypothesis": "第一次突破是否比重複突破更具有 Alpha？",
    },
}


# ==========================================
# Orchestration
# ==========================================
def run_phase_b(dimension: str, events_csv: str = None, use_cache: bool = True) -> dict:
    """跑完整個 Phase B 流程：讀事件 → 算 forward return → 分組檢定 → 存檔 → 回傳 summary。"""
    if dimension not in DIMENSION_CONFIG:
        raise ValueError(f"未知的 dimension：{dimension}，可用值：{list(DIMENSION_CONFIG.keys())}")

    path = events_csv or EVENTS_CSV
    if not os.path.exists(path):
        return {"status": "unavailable",
                "message": f"⚠️ 找不到事件檔案 {path}，請先跑過 Phase A（event_inventory.py）。"}

    # ⚠️ dtype={"ticker": str} 是必要的，不是保險做法：ticker 欄位裡的 "0050"
    # 這類 ETF 代碼開頭有 0，如果讓 pandas 自動推斷型別，會被當成數字讀成 50
    # （int64），前面的 0 悄悄消失。這不只是顯示問題——後面 compute_forward_returns()
    # 會拿這個被截斷的代碼去問 yfinance「50.TW」，查無此股票，直接抓取失敗，
    # 而且是那種「有印出錯誤但不會讓整支程式中斷」的失敗，很容易被忽略。
    events_df = pd.read_csv(path, dtype={"ticker": str})
    if events_df.empty:
        return {"status": "unavailable", "message": "⚠️ 事件檔案是空的，無法做 Phase B 研究。"}

    config = DIMENSION_CONFIG[dimension]
    events_df = config["prepare"](events_df)

    events_with_returns = compute_forward_returns(events_df, use_cache=use_cache)

    n_insufficient = int(events_with_returns["insufficient_future_data"].fillna(False).sum())
    if n_insufficient:
        logger.warning(f"[phase_b:{dimension}] {n_insufficient}/{len(events_with_returns)} 筆事件因為"
                        f"未來資料不足（太接近資料最後一天），forward return 不完整或無法計算。")

    summary = group_forward_return_study(events_with_returns, config["group_col"], config["hypothesis"])
    summary["dimension"] = dimension
    summary["total_events"] = int(len(events_with_returns))
    summary["events_with_insufficient_future_data"] = n_insufficient

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    events_with_returns.to_csv(RESULTS_CSV_TEMPLATE.format(dimension=dimension), index=False, encoding="utf-8-sig")
    with open(SUMMARY_JSON_TEMPLATE.format(dimension=dimension), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["status"] = "ok"
    return summary


def print_summary(summary: dict):
    if summary.get("status") != "ok":
        print(summary.get("message", "⚠️ Phase B 執行失敗。"))
        return
    print(f"\n=== Phase B：{summary['dimension']} ===")
    print(f"假說：{summary['hypothesis']}")
    print(f"事件總數：{summary['total_events']}（其中 {summary['events_with_insufficient_future_data']} 筆因未來資料不足而不完整）")
    print(f"Bonferroni 校正門檻（9次檢定）：{summary['bonferroni_alpha_9_tests']}")
    for group_name, stats in summary["groups"].items():
        win_ci = stats.get("win_rate_20d_ci95_pct", [None, None])
        ticker_note = f"，來自{stats['n_unique_tickers']}檔不同股票" if "n_unique_tickers" in stats else ""
        print(f"\n【{group_name}】n={stats['n']}{ticker_note}，win_rate_20d={stats['win_rate_20d_pct']}% "
              f"(95%CI {win_ci[0]}~{win_ci[1]}%)，mean_mdd_20d={stats['mean_mdd_20d_pct']}%")
        for h in HORIZONS:
            fr = stats[f"forward_return_{h}d"]
            ci = fr.get("mean_ci95", [None, None])
            print(f"  {h}D：mean={fr['mean']}% (95%CI {ci[0]}~{ci[1]}%), median={fr['median']}%, n_valid={fr['n_valid']}")
    print("\n檢定結果：")
    for col, t in summary["tests"].items():
        es = t.get("effect_size")
        es_str = f", effect_size={es['type']}={es['value']}" if es else ""
        print(f"  {col}：{t['test']}, statistic={t['statistic']}, p={t['p_value']}{es_str} → {t['verdict']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B：Forward Return Event Study")
    parser.add_argument("--dimension", choices=["volume", "liquidity", "breakout_count", "all"], default="all")
    parser.add_argument("--events-csv", default=None, help="覆蓋預設的 breakout_events_raw.csv 路徑")
    parser.add_argument("--no-cache", action="store_true", help="略過 DataEngine 的資料庫快取，強制重新抓取股價")
    args = parser.parse_args()

    dims = list(DIMENSION_CONFIG.keys()) if args.dimension == "all" else [args.dimension]
    for d in dims:
        result = run_phase_b(d, events_csv=args.events_csv, use_cache=not args.no_cache)
        print_summary(result)
