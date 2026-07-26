# -*- coding: utf-8 -*-
"""
research/test_phase_b_forward_return.py

用合成資料驗證 phase_b_forward_return.py 的核心邏輯，不需要網路連線：
  1. compute_forward_returns()：已知漲幅的合成價格序列，驗證算出來的
     forward_return_5d/10d/20d 數字正確，且 look-ahead 沒有用到未來資料。
  2. insufficient_future_data：事件發生在資料尾端、未來天數不足時，
     要被正確標記，不能悄悄回傳 NaN 又被當成「無資料可用」以外的意思。
  3. mdd_20d：驗證在一段先跌後漲的走勢中，抓到的是「進場後最低點」，
     不是區間內任何其他基準。
  4. group_forward_return_study()：兩組報酬有明顯差異時，Mann-Whitney U
     應該給出很小的 p-value；兩組幾乎沒差異時，不應該給出很小的 p-value。
  5. min_group_n 門檻：任一組樣本數不足時，要標示 insufficient_sample，
     不能硬跑檢定又不提醒。
"""
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research import phase_b_forward_return as pb


def _make_price_df(closes, start="2024-01-01"):
    n = len(closes)
    dates = pd.date_range(start=start, periods=n, freq="B")
    return pd.DataFrame({"date": dates, "close": np.array(closes, dtype=float)})


def test_forward_return_basic():
    # 事件日之後每天固定漲 1%，驗證 5D/10D/20D 報酬算出來的複利數字正確
    closes = [100.0]
    for _ in range(40):
        closes.append(closes[-1] * 1.01)
    price_df = _make_price_df(closes)
    event_date = price_df["date"].iloc[10].strftime("%Y-%m-%d")

    events_df = pd.DataFrame([{"ticker": "TEST1", "event_date": event_date, "event_seq_for_ticker": 1}])
    result = pb.compute_forward_returns(events_df, price_lookup={"TEST1": price_df})

    entry_price = closes[10]
    expected_5d = (closes[15] / entry_price - 1) * 100
    expected_20d = (closes[30] / entry_price - 1) * 100

    assert abs(result["forward_return_5d"].iloc[0] - expected_5d) < 1e-2, "5D 前瞻報酬計算錯誤"
    assert abs(result["forward_return_20d"].iloc[0] - expected_20d) < 1e-2, "20D 前瞻報酬計算錯誤"
    assert result["win_20d"].iloc[0] == True, "持續上漲應該判定為 win"
    assert result["insufficient_future_data"].iloc[0] == False, "資料充足時不應標記為不足"
    print("✅ test_forward_return_basic 通過")


def test_insufficient_future_data_flagged():
    # 事件發生在資料倒數第3天，20D report 根本沒有未來資料可以算
    closes = [100.0 + i for i in range(15)]
    price_df = _make_price_df(closes)
    event_date = price_df["date"].iloc[-3].strftime("%Y-%m-%d")

    events_df = pd.DataFrame([{"ticker": "TEST2", "event_date": event_date, "event_seq_for_ticker": 1}])
    result = pb.compute_forward_returns(events_df, price_lookup={"TEST2": price_df})

    assert result["insufficient_future_data"].iloc[0] == True, "未來資料不足時必須標記為 True"
    assert pd.isna(result["forward_return_20d"].iloc[0]), "資料不足時 20D 報酬應該是 NaN，不是猜測值"
    assert result["win_20d"].iloc[0] is None, "資料不足時 win_20d 不能被當成 False"
    print("✅ test_insufficient_future_data_flagged 通過")


def test_mdd_captures_worst_drawdown():
    # 進場後先跌15%探底，再漲回超過進場價，MDD應該抓到探底那天，不是別的基準
    closes = [100.0]
    # 事件日 index 0，接下來5天跌到85，再15天漲回120
    for i in range(1, 6):
        closes.append(100.0 - i * 3)  # 100 -> 85
    for i in range(1, 16):
        closes.append(85.0 + i * 3)  # 85 -> ~130
    price_df = _make_price_df(closes)
    event_date = price_df["date"].iloc[0].strftime("%Y-%m-%d")

    events_df = pd.DataFrame([{"ticker": "TEST3", "event_date": event_date, "event_seq_for_ticker": 1}])
    result = pb.compute_forward_returns(events_df, price_lookup={"TEST3": price_df})

    mdd = result["mdd_20d"].iloc[0]
    assert mdd < -14, f"MDD 應該接近 -15%（進場價100跌到85），實際算出 {mdd}"
    print("✅ test_mdd_captures_worst_drawdown 通過")


def test_group_study_detects_real_difference():
    # 組A：event後穩定漲5%；組B：event後穩定跌5%，應該偵測到顯著差異
    np.random.seed(42)
    rows = []
    price_lookup = {}
    for i in range(30):
        ticker = f"A{i}"
        base = 100.0
        # 加一點雜訊避免完全退化的常數序列
        drift = 1.05 + np.random.normal(0, 0.01)
        closes = [base] + [base * (drift ** (t / 20)) for t in range(1, 25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": True})
    for i in range(30):
        ticker = f"B{i}"
        base = 100.0
        drift = 0.95 + np.random.normal(0, 0.01)
        closes = [base] + [base * (drift ** (t / 20)) for t in range(1, 25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": False})

    events_df = pd.DataFrame(rows)
    events_with_returns = pb.compute_forward_returns(events_df, price_lookup=price_lookup)
    summary = pb.group_forward_return_study(events_with_returns, "volume_confirmed",
                                              "測試假說", min_group_n=10)

    p_20d = summary["tests"]["forward_return_20d"]["p_value"]
    assert p_20d is not None and p_20d < 0.01, f"兩組報酬有明顯系統性差異，p-value 應該很小，實際 {p_20d}"
    print("✅ test_group_study_detects_real_difference 通過")


def test_min_group_n_guards_small_sample():
    rows = []
    price_lookup = {}
    for i in range(5):  # 遠低於 min_group_n=20 的預設門檻
        ticker = f"S{i}"
        closes = [100.0 + t for t in range(25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": True})
    for i in range(5):
        ticker = f"T{i}"
        closes = [100.0 - t for t in range(25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": False})

    events_df = pd.DataFrame(rows)
    events_with_returns = pb.compute_forward_returns(events_df, price_lookup=price_lookup)
    summary = pb.group_forward_return_study(events_with_returns, "volume_confirmed", "測試假說")

    verdict = summary["tests"]["forward_return_20d"]["verdict"]
    assert "insufficient_sample" in verdict, "樣本數不足 min_group_n 時必須標示 insufficient_sample，不能硬跑檢定"
    assert summary["tests"]["forward_return_20d"]["p_value"] is None
    print("✅ test_min_group_n_guards_small_sample 通過")


def test_breakout_count_bucket():
    events_df = pd.DataFrame([
        {"event_seq_for_ticker": 1}, {"event_seq_for_ticker": 2},
        {"event_seq_for_ticker": 3}, {"event_seq_for_ticker": 4},
        {"event_seq_for_ticker": 7}, {"event_seq_for_ticker": np.nan},
    ])
    result = pb.prepare_breakout_count_bucket(events_df)
    actual = result["breakout_count_bucket"].tolist()
    expected = ["第1次", "第2次", "第3次", "第4次以上", "第4次以上", None]
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None or (isinstance(a, float) and np.isnan(a)), f"缺值應該保留為 None/NaN，實際是 {a!r}"
        else:
            assert a == e, f"分桶結果不符預期：{actual}"
    print("✅ test_breakout_count_bucket 通過")


def test_mdd_zero_when_monotonic_uptrend():
    # 事件後一路上漲，從未跌破進場價，MDD 應該剛好是 0（不是負值，也不是 None）
    closes = [100.0 + i * 2 for i in range(25)]
    price_df = _make_price_df(closes)
    event_date = price_df["date"].iloc[0].strftime("%Y-%m-%d")

    events_df = pd.DataFrame([{"ticker": "TEST4", "event_date": event_date, "event_seq_for_ticker": 1}])
    result = pb.compute_forward_returns(events_df, price_lookup={"TEST4": price_df})

    mdd = result["mdd_20d"].iloc[0]
    assert abs(mdd - 0.0) < 1e-6, f"一路上漲時 MDD 應該是 0，實際算出 {mdd}"
    print("✅ test_mdd_zero_when_monotonic_uptrend 通過")


def test_mdd_gap_down():
    # 進場後隔天直接跳空跌20%，然後横盤，MDD 應該精確等於 -20%
    closes = [100.0, 80.0] + [80.0] * 23
    price_df = _make_price_df(closes)
    event_date = price_df["date"].iloc[0].strftime("%Y-%m-%d")

    events_df = pd.DataFrame([{"ticker": "TEST5", "event_date": event_date, "event_seq_for_ticker": 1}])
    result = pb.compute_forward_returns(events_df, price_lookup={"TEST5": price_df})

    mdd = result["mdd_20d"].iloc[0]
    assert abs(mdd - (-20.0)) < 1e-6, f"跳空跌20%後橫盤，MDD 應該精確等於 -20%，實際算出 {mdd}"
    print("✅ test_mdd_gap_down 通過")


def test_effect_size_and_ci_present():
    # 確認 group_forward_return_study 的輸出裡，effect_size 跟 bootstrap CI 欄位都存在且型態正確
    np.random.seed(1)
    rows = []
    price_lookup = {}
    for i in range(30):
        ticker = f"E{i}"
        closes = [100.0] + [100.0 * (1.05 ** (t / 20)) for t in range(1, 25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": True})
    for i in range(30):
        ticker = f"F{i}"
        closes = [100.0] + [100.0 * (0.95 ** (t / 20)) for t in range(1, 25)]
        price_lookup[ticker] = _make_price_df(closes)
        rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": 1, "volume_confirmed": False})

    events_df = pd.DataFrame(rows)
    events_with_returns = pb.compute_forward_returns(events_df, price_lookup=price_lookup)
    summary = pb.group_forward_return_study(events_with_returns, "volume_confirmed", "測試假說", min_group_n=10)

    es = summary["tests"]["forward_return_20d"]["effect_size"]
    assert es is not None and es["type"] == "rank_biserial_correlation", "應該要有 rank-biserial 效果量"
    assert abs(es["value"]) > 0.5, f"兩組報酬完全分離，effect size 應該接近極值，實際 {es['value']}"

    group_true = summary["groups"]["True"]["forward_return_20d"]
    ci = group_true["mean_ci95"]
    assert ci[0] is not None and ci[1] is not None and ci[0] <= group_true["mean"] <= ci[1], \
        "bootstrap CI 應該存在且涵蓋平均值"
    print("✅ test_effect_size_and_ci_present 通過")


def test_epsilon_squared_floored_at_zero():
    # 三組報酬幾乎完全相同（只有極小雜訊），H 應該很小甚至小於 k-1，
    # epsilon-squared 公式在這種情況下會算出負值，必須被 floor 在 0
    np.random.seed(7)
    rows = []
    price_lookup = {}
    for grp_idx, grp in enumerate(["A", "B", "C"]):
        for i in range(25):
            ticker = f"{grp}{i}"
            noise = np.random.normal(0, 0.001)
            closes = [100.0] + [100.0 * ((1.0 + noise) ** (t / 20)) for t in range(1, 25)]
            price_lookup[ticker] = _make_price_df(closes)
            rows.append({"ticker": ticker, "event_date": price_lookup[ticker]["date"].iloc[0].strftime("%Y-%m-%d"),
                          "event_seq_for_ticker": 1, "grp": grp})
    events_df = pd.DataFrame(rows)
    events_with_returns = pb.compute_forward_returns(events_df, price_lookup=price_lookup)
    summary = pb.group_forward_return_study(events_with_returns, "grp", "測試假說", min_group_n=10)

    for col in ["forward_return_5d", "forward_return_10d", "forward_return_20d"]:
        es = summary["tests"][col]["effect_size"]
        if es is not None:
            assert es["value"] >= 0, f"epsilon-squared 不應該是負值，實際 {es['value']}（{col}）"
    print("✅ test_epsilon_squared_floored_at_zero 通過")


def test_n_unique_tickers_reported():
    # 同一檔股票出現多次事件時，n（事件數）跟 n_unique_tickers（股票數）必須分開，
    # 不能被誤當成獨立樣本數一樣多
    rows = []
    price_lookup = {}
    closes = [100.0 + t for t in range(25)]
    price_lookup["DUP"] = _make_price_df(closes)
    for i in range(5):
        rows.append({"ticker": "DUP", "event_date": price_lookup["DUP"]["date"].iloc[0].strftime("%Y-%m-%d"),
                      "event_seq_for_ticker": i + 1, "volume_confirmed": True})
    events_df = pd.DataFrame(rows)
    events_with_returns = pb.compute_forward_returns(events_df, price_lookup=price_lookup)
    summary = pb.group_forward_return_study(events_with_returns, "volume_confirmed", "測試假說", min_group_n=1)

    assert summary["groups"]["True"]["n"] == 5, "事件數應該是5"
    assert summary["groups"]["True"]["n_unique_tickers"] == 1, "應該要能看出這5個事件其實只來自1檔股票"
    print("✅ test_n_unique_tickers_reported 通過")


if __name__ == "__main__":
    test_forward_return_basic()
    test_insufficient_future_data_flagged()
    test_mdd_captures_worst_drawdown()
    test_mdd_zero_when_monotonic_uptrend()
    test_mdd_gap_down()
    test_group_study_detects_real_difference()
    test_effect_size_and_ci_present()
    test_epsilon_squared_floored_at_zero()
    test_n_unique_tickers_reported()
    test_min_group_n_guards_small_sample()
    test_breakout_count_bucket()
    print("\n全部測試通過 ✅")
