"""
tests/golden/verify_macd_migration.py

Phase 2D — MACD 遷移的獨立驗證腳本（跟 run_golden.py / run_mutations.py
互補，不是取代）。

這支腳本回答兩個問題，run_golden.py 只間接回答第一個：
  1. 「舊公式（遷移前，內聯計算）」跟「新公式（FeatureProvider.ensure_macd）」
     在同一份資料上，逐欄位 diff 是否真的是 0.0？
     （run_golden.py 驗證的是「跟凍結的 Golden 基準比」，這裡驗證的是
     「新舊兩個實作互相比」——概念上更直接對應「這是重構、不是改變
     行為」這句話本身。）
  2. 「macd_hist == macd_dif - macd_dea」這個定義上必須成立的內部一致性，
     是否真的沒有因為搬移順序或資料 alignment 而被破壞？

用法：
    python tests/golden/verify_macd_migration.py

⚠️ 誠實揭露：這裡用合成資料自我測試（沒有網路權限抓真實股價），
跟 research/ 模組先前的作法一致。合成資料只能驗證「數學邏輯本身
一致」，不能取代 run_golden.py 對真實市場資料的回歸測試——兩者
都需要跑，不是二選一。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.feature_provider import FeatureProvider  # noqa: E402


def _old_macd_formula(df: pd.DataFrame, source_col: str = "close") -> pd.DataFrame:
    """遷移前 indicator_engine.py 內聯計算的原始版本（照抄，不是重新設計），
    只用來當作比對基準，不會被其他程式碼呼叫。"""
    df = df.copy()
    c = df[source_col]
    df["macd_dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]
    return df


def _make_synthetic_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    # 模擬有趨勢也有雜訊的股價走勢，而不是純隨機亂走，這樣 EMA/MACD
    # 才會有真正的趨勢/背離特徵可以檢查，不是全部趨近於 0。
    trend = np.linspace(100, 140, n)
    noise = rng.normal(0, 1.5, n)
    closes = trend + np.cumsum(noise) * 0.3
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def check_formula_equivalence(df: pd.DataFrame) -> bool:
    old_df = _old_macd_formula(df)
    new_df = FeatureProvider.ensure_macd(df.copy())

    cols = ["macd_dif", "macd_dea", "macd_hist"]
    all_ok = True
    print("--- ① Formula Diff（舊公式 vs FeatureProvider.ensure_macd）---")
    for col in cols:
        diff = (old_df[col] - new_df[col]).abs().max()
        ok = diff == 0.0 or pd.isna(diff)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {col:12s} max diff = {diff:.2e}  [{status}]")
    return all_ok


def check_hist_consistency(df: pd.DataFrame) -> bool:
    result_df = FeatureProvider.ensure_macd(df.copy())
    expected_hist = result_df["macd_dif"] - result_df["macd_dea"]
    diff = (result_df["macd_hist"] - expected_hist).abs().max()
    ok = diff == 0.0 or pd.isna(diff)
    print("\n--- ② Internal Consistency（macd_hist == macd_dif - macd_dea）---")
    print(f"  max diff = {diff:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def check_indicator_engine_matches_feature_provider(df: pd.DataFrame) -> bool:
    """確認 indicator_engine.py 實際遷移後的呼叫路徑，跟直接呼叫
    FeatureProvider.ensure_macd() 產生完全一致的結果——這是防止「遷移
    時漏改、或改到別的地方」的最直接的一道檢查。"""
    from engines.indicator_engine import IndicatorEngine

    full_df = IndicatorEngine.add_indicators(df.copy())
    direct_df = FeatureProvider.ensure_macd(df.copy())

    cols = ["macd_dif", "macd_dea", "macd_hist"]
    all_ok = True
    print("\n--- ③ IndicatorEngine.add_indicators() vs 直接呼叫 FeatureProvider ---")
    for col in cols:
        diff = (full_df[col] - direct_df[col]).abs().max()
        ok = diff == 0.0 or pd.isna(diff)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {col:12s} max diff = {diff:.2e}  [{status}]")
    return all_ok


def main():
    df = _make_synthetic_df()

    r1 = check_formula_equivalence(df)
    r2 = check_hist_consistency(df)
    r3 = check_indicator_engine_matches_feature_provider(df)

    print("\n" + "=" * 60)
    if r1 and r2 and r3:
        print("Total: PASS（合成資料驗證通過，仍需另外執行 run_golden.py 對真實市場資料回歸測試）")
        sys.exit(0)
    else:
        print("Total: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
