"""
tests/test_chip_flow_engine.py

用合成資料驗證 ChipFlowEngine 的核心邏輯：
  1. _streak_from_series()：連買/連賣天數計算是否正確（含中斷、掛零、
     反轉的邊界情況）。
  2. _streak_from_balance_series()：融資融券餘額連續增減天數計算。
  3. compute_chip_score()：分數計算與 is_validated=False 是否正確附帶。

不需要網路連線，這支腳本只測試純函式邏輯，不測試 _collect_daily_snapshots()
（那部分依賴真實 TWSE 網路請求，無法在沙盒環境驗證，需要在有網路的環境
另外跑一次確認實際欄位格式與端點行為）。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.chip_flow_engine import ChipFlowEngine


def test_streak_from_series_simple_buy_streak():
    # 由舊到新：最新3天都是買超，之前是賣超
    values = [-100, -50, 200, 300, 150]
    result = ChipFlowEngine._streak_from_series(values)
    assert result["direction"] == "buy"
    assert result["days"] == 3
    assert result["cumulative_shares"] == 200 + 300 + 150
    print("PASS: 連買天數計算正確（中斷於前一段賣超）")


def test_streak_from_series_flat_breaks_streak():
    values = [100, 200, 0, 150]
    result = ChipFlowEngine._streak_from_series(values)
    # 最新一天是買超(150)，往回數只到掛零那天就中斷（掛零不算buy也不算sell）
    assert result["direction"] == "buy"
    assert result["days"] == 1
    print("PASS: 掛零會正確中斷連續天數")


def test_streak_from_series_latest_flat_returns_flat():
    values = [100, 200, 300, 0]
    result = ChipFlowEngine._streak_from_series(values)
    assert result["direction"] == "flat"
    assert result["days"] == 0
    print("PASS: 最新一天掛零，直接回傳 flat")


def test_streak_from_series_empty():
    result = ChipFlowEngine._streak_from_series([])
    assert result == {"direction": "flat", "days": 0, "cumulative_shares": 0}
    print("PASS: 空序列回傳安全預設值")


def test_streak_from_balance_series_decreasing():
    # 融資餘額連續下降
    values = [1000, 950, 900, 880, 880]  # 最後一天沒變化(880==880)
    result = ChipFlowEngine._streak_from_balance_series(values)
    assert result["direction"] == "flat"  # 最新一天差值是0
    print("PASS: 餘額序列最新一天無變化，回傳 flat")

    values2 = [1000, 950, 900, 850]
    result2 = ChipFlowEngine._streak_from_balance_series(values2)
    assert result2["direction"] == "decreasing"
    assert result2["days"] == 3
    assert result2["change"] == 850 - 1000
    print("PASS: 餘額連續下降天數與變化量計算正確")


def test_streak_from_balance_series_insufficient_data():
    result = ChipFlowEngine._streak_from_balance_series([100])
    assert result["direction"] == "flat"
    assert result["days"] == 0
    print("PASS: 資料點不足2個時安全回傳 flat")


def test_compute_chip_score_full_buy_streak_gets_max_points():
    institutional_streaks = {
        "streaks": {
            "2330": {
                "foreign": {"direction": "buy", "days": 15, "cumulative_shares": 1000},
                "trust": {"direction": "buy", "days": 15, "cumulative_shares": 500},
                "dealer": {"direction": "sell", "days": 2, "cumulative_shares": -100},
                "total": {"direction": "buy", "days": 15, "cumulative_shares": 1400},
            }
        }
    }
    margin_streaks = {
        "streaks": {
            "2330": {
                "margin_balance": {"direction": "decreasing", "days": 15, "change": -500},
                "short_balance": {"direction": "decreasing", "days": 15, "change": -200},
            }
        }
    }
    result = ChipFlowEngine.compute_chip_score(institutional_streaks, margin_streaks,
                                                 streak_days_for_full_score=10)
    score = result["2330"]
    assert score["chip_score"] == 100.0, f"預期滿分100，實際 {score['chip_score']}"
    assert score["is_validated"] is False, "is_validated 必須永遠是 False（未經驗證的權重）"
    for item in score["breakdown"].values():
        assert item["points"] == item["max_points"], "連續天數超過門檻應該封頂在滿分，不能超過"
    print("PASS: 全部滿足連續天數門檻時，總分正確封頂在100且 is_validated=False")


def test_compute_chip_score_partial_streak_linear_interpolation():
    institutional_streaks = {
        "streaks": {
            "1101": {
                "foreign": {"direction": "buy", "days": 5, "cumulative_shares": 100},
                "trust": {"direction": "sell", "days": 3, "cumulative_shares": -50},  # 方向不符，不給分
                "dealer": {"direction": "flat", "days": 0, "cumulative_shares": 0},
                "total": {"direction": "buy", "days": 5, "cumulative_shares": 50},
            }
        }
    }
    margin_streaks = {"streaks": {}}  # 1101 沒有融資融券資料

    result = ChipFlowEngine.compute_chip_score(institutional_streaks, margin_streaks,
                                                 streak_days_for_full_score=10)
    score = result["1101"]
    # foreign: 5/10 * 25 = 12.5
    assert score["breakdown"]["foreign_buy_streak"]["points"] == 12.5
    # trust方向是sell，不應該給分
    assert score["breakdown"]["trust_buy_streak"]["points"] == 0.0
    # margin/short streaks 缺失時應該安全回傳0分，不應該拋錯
    assert score["breakdown"]["margin_decreasing"]["points"] == 0.0
    assert score["breakdown"]["short_covering"]["points"] == 0.0
    print("PASS: 部分連續天數線性內插、方向不符不給分、缺失資料安全回傳0分")


def run_all():
    test_streak_from_series_simple_buy_streak()
    test_streak_from_series_flat_breaks_streak()
    test_streak_from_series_latest_flat_returns_flat()
    test_streak_from_series_empty()
    test_streak_from_balance_series_decreasing()
    test_streak_from_balance_series_insufficient_data()
    test_compute_chip_score_full_buy_streak_gets_max_points()
    test_compute_chip_score_partial_streak_linear_interpolation()
    print("\n所有自我測試通過（合成資料，_collect_daily_snapshots 依賴真實網路"
          "尚未驗證，需在有網路的環境另外執行確認真實 TWSE 欄位格式）")


if __name__ == "__main__":
    run_all()
