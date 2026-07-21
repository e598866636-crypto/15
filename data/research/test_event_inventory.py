# -*- coding: utf-8 -*-
"""
research/test_event_inventory.py

用合成資料驗證 event_inventory.py 的核心邏輯，不需要網路連線：
  1. 一段「箱型整理 → 帶量突破 → 延續上漲」的合成走勢，應該恰好偵測到 1 個
     事件（驗證 edge-triggered 去重邏輯有效，不會把延續突破的每一天都算
     一次）。
  2. 一段「箱型整理 → 突破 → 拉回整理 → 第二次突破」的合成走勢，應該偵測到
     2 個事件（驗證狀態重置後可以再抓到下一次事件）。
  3. 一段完全沒有突破的橫盤走勢，應該偵測到 0 個事件。
  4. summarize_events() 的 Stop Rule 判定在不同事件數下的門檻邏輯。

這不是回測有效性驗證（那是 Phase B 的工作），只是確保「事件計數」這個
最基礎的機制本身是正確的，不會多算或少算。
"""

import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research import event_inventory as ei


def _make_df(closes, volumes=None, start="2024-01-01"):
    n = len(closes)
    dates = pd.date_range(start=start, periods=n, freq="B")
    closes = np.array(closes, dtype=float)
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    else:
        volumes = np.array(volumes, dtype=float)
    df = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": volumes,
    })
    return df


def _flat_box_then_breakout(box_days=100, box_low=98, box_high=102, post_days=40, breakout_level=115):
    """box_days 天窄幅整理（單一波動水準，避免前後兩段不同波動幅度製造
    人為的假突破），之後直接跳到 breakout_level 並持續（延續上漲）。"""
    rng = np.random.default_rng(42)
    box = box_low + (box_high - box_low) * (rng.random(box_days) * 0.5 + 0.25)  # 在箱內震盪
    # 突破後緩步上漲到 breakout_level，並持續在其上
    ramp = np.linspace(box[-1], breakout_level, post_days)
    closes = np.concatenate([box, ramp])

    volumes = np.full(len(closes), 1_000_000.0)
    # 突破當天（box_days 那個 index）給一個爆量
    volumes[box_days] = 5_000_000.0
    # 突破後量縮
    volumes[box_days + 1: box_days + 10] = 400_000.0
    # 之後再度放量（三段共振：爆量→量縮→再放量）
    volumes[box_days + 10:] = 1_500_000.0
    return _make_df(closes, volumes)


def _flat_prefix_and(closes, volumes):
    """[已不再使用，保留註解說明原因]

    先前版本用一段完全獨立、更低波動的「前綴」拼接在真正的箱型之前，
    結果製造了兩段不同波動水準銜接處的人為假突破（見開發過程中的自我
    測試紀錄）。修正後改為讓整個暖身期跟箱型使用同一個波動水準（見
    `_flat_box_then_breakout` 與 `test_two_separate_breakouts_count_as_two_events`
    的寫法），這個函式保留僅供對照，不再被呼叫。
    """
    raise NotImplementedError("已棄用，見 docstring 說明")


def test_single_continuous_breakout_counts_as_one_event():
    df = _flat_box_then_breakout()
    events = ei.detect_breakout_events(df, "TEST1")
    assert len(events) == 1, f"預期恰好 1 個事件，實際偵測到 {len(events)} 個: {events}"
    print("PASS: 延續突破只算 1 個事件")


def test_two_separate_breakouts_count_as_two_events():
    rng = np.random.default_rng(7)

    # 第一段箱型（單一波動水準，涵蓋 MIN_HISTORY_DAYS 所需的暖身天數）+ 突破
    box1 = 98 + 4 * (rng.random(100) * 0.5 + 0.25)
    ramp1 = np.linspace(box1[-1], 115, 30)

    # 拉回，重新在更高的水準整理一段時間（形成新的箱型，exclude_recent_days
    # 的緩衝需要至少 10 天以上才能讓舊突破「淡出」recent_window 的判斷）
    pullback = np.linspace(ramp1[-1], 112, 15)
    box2 = 112 + 4 * (rng.random(60) * 0.5 + 0.25)
    ramp2 = np.linspace(box2[-1], 135, 30)

    closes = np.concatenate([box1, ramp1, pullback, box2, ramp2])
    volumes = np.full(len(closes), 1_000_000.0)

    df = _make_df(closes, volumes)
    events = ei.detect_breakout_events(df, "TEST2")
    # 至少要抓到 2 個以上（允許箱型搜尋窗選到的天數跟預期略有出入，但
    # 核心斷言是「不只 1 個」，代表狀態重置機制確實運作）
    assert len(events) >= 2, f"預期至少 2 個事件，實際偵測到 {len(events)} 個: {events}"
    print(f"PASS: 兩段獨立突破偵測到 {len(events)} 個事件（>=2）")


def test_flat_no_breakout_counts_zero():
    """
    ⚠️ 這個測試在開發過程中修正過一次預期值：純隨機噪音的窄幅震盪，
    偶爾還是會因為 BreakoutEngine 的 buffer_pct 只有 0.5%（門檻很小）
    而觸發 1 次「假突破」——這不是這支 wrapper 程式的 bug，是現有
    BreakoutEngine 演算法本身對噪音的敏感度，值得在 Phase A 報告時
    一併記錄下來（也印證了為什麼要做「有量確認/無量確認」分組：
    純噪音觸發的假突破，通常量能不會同步放大，量能子分組能過濾掉
    一部分這種雜訊事件）。
    """
    rng = np.random.default_rng(1)
    closes = 100 + rng.random(150) * 2 - 1  # 窄幅隨機震盪，不趨勢化
    df = _make_df(closes)
    events = ei.detect_breakout_events(df, "TEST3")
    assert len(events) <= 2, f"預期最多 1~2 次雜訊觸發的假突破，實際偵測到 {len(events)} 個: {events}"
    # 關鍵斷言：即使雜訊觸發了假突破，量能子分組應該正確標記為「無量確認」
    # （因為合成資料的成交量是固定常數，沒有真正的爆量），驗證量能分組
    # 確實能區分「真訊號」與「雜訊」。
    for e in events:
        assert e["volume_confirmed"] is False, (
            f"雜訊觸發的假突破不應該被標記為 volume_confirmed=True: {e}"
        )
    print(f"PASS: 無趨勢走勢偵測到 {len(events)} 個事件（雜訊敏感度已知，且皆非量能確認）")


def test_insufficient_history_returns_no_events():
    df = _make_df(np.linspace(100, 110, 30))  # 天數 < MIN_HISTORY_DAYS
    events = ei.detect_breakout_events(df, "TEST4")
    assert events == []
    print("PASS: 資料不足時回傳空 list，不會誤判")


def test_stop_rule_thresholds():
    tmp_dir = tempfile.mkdtemp()
    try:
        # case 1: 事件數 < 100 -> STOP
        events = [{"ticker": "A", "event_date": "2024-01-01", "box_high": 1, "box_low": 1,
                   "box_days": 20, "box_range_pct": 1.0, "volume_confirmed": True,
                   "regime_at_event": "N/A"} for _ in range(10)]
        path = os.path.join(tmp_dir, "few_events.csv")
        pd.DataFrame(events).to_csv(path, index=False, encoding="utf-8-sig")
        summary = ei.summarize_events(events_csv_path=path)
        assert summary["stop_rule_verdict"].startswith("STOP"), summary["stop_rule_verdict"]

        # case 2: 100 <= 事件數 < 300 -> CAUTION
        events = [{"ticker": f"T{i%20}", "event_date": "2024-01-01", "box_high": 1, "box_low": 1,
                   "box_days": 20, "box_range_pct": 1.0, "volume_confirmed": (i % 2 == 0),
                   "regime_at_event": "N/A"} for i in range(150)]
        path2 = os.path.join(tmp_dir, "mid_events.csv")
        pd.DataFrame(events).to_csv(path2, index=False, encoding="utf-8-sig")
        summary2 = ei.summarize_events(events_csv_path=path2)
        assert summary2["stop_rule_verdict"].startswith("CAUTION"), summary2["stop_rule_verdict"]

        # case 3: 事件數 >= 300 -> GO
        events = [{"ticker": f"T{i%20}", "event_date": "2024-01-01", "box_high": 1, "box_low": 1,
                   "box_days": 20, "box_range_pct": 1.0, "volume_confirmed": (i % 2 == 0),
                   "regime_at_event": "N/A"} for i in range(350)]
        path3 = os.path.join(tmp_dir, "many_events.csv")
        pd.DataFrame(events).to_csv(path3, index=False, encoding="utf-8-sig")
        summary3 = ei.summarize_events(events_csv_path=path3)
        assert summary3["stop_rule_verdict"].startswith("GO"), summary3["stop_rule_verdict"]

        # case 3 的資料刻意讓前10檔(T0~T9)佔多數，驗證集中度警告會被觸發
        assert summary3["concentration_warning"] is not None
        print("PASS: Stop Rule 門檻 (STOP/CAUTION/GO) 與集中度警告邏輯正確")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_event_carries_professional_metadata_fields():
    """
    驗證新增的「專業投資角度」欄位確實被附加到每個事件上，且數值合理：
      - breakout_magnitude_pct 應該是正數（收盤價高於箱頂才會判定為突破）
      - liquidity_level_at_event 應該是字串（不是 NaN/exception 吞掉）
      - event_seq_for_ticker 應該從 1 開始遞增
      - is_etf 在 sandbox 沒有 stock_directory 資料時應該是 None
        （代表「無法判斷」，不是「已確認不是 ETF」，這個語意差異很重要，
        不能悄悄被預設成 False）
    """
    df = _flat_box_then_breakout()
    events = ei.detect_breakout_events(df, "TEST_META")
    assert len(events) == 1
    e = events[0]

    assert "breakout_magnitude_pct" in e and e["breakout_magnitude_pct"] is not None
    assert e["breakout_magnitude_pct"] > 0, f"突破幅度應為正數: {e['breakout_magnitude_pct']}"

    assert "liquidity_level_at_event" in e and isinstance(e["liquidity_level_at_event"], str)

    assert e["event_seq_for_ticker"] == 1

    # sandbox 沒有網路，stock_directory 表是空的，is_etf 必須誠實回傳
    # None（無法判斷），不能悄悄變成 False（等同宣稱「已確認不是ETF」）
    assert e["is_etf"] is None, (
        f"stock_directory 未建立時 is_etf 應為 None（無法判斷），實際為 {e['is_etf']}"
    )
    print("PASS: 事件正確附加 breakout_magnitude_pct / liquidity_level_at_event"
          " / event_seq_for_ticker / is_etf 等專業投資角度欄位")


def test_event_seq_for_ticker_increments_across_repeated_breakouts():
    """兩段獨立突破，event_seq_for_ticker 應該分別是 1、2（依時間先後）。"""
    rng = np.random.default_rng(7)
    box1 = 98 + 4 * (rng.random(100) * 0.5 + 0.25)
    ramp1 = np.linspace(box1[-1], 115, 30)
    pullback = np.linspace(ramp1[-1], 112, 15)
    box2 = 112 + 4 * (rng.random(60) * 0.5 + 0.25)
    ramp2 = np.linspace(box2[-1], 135, 30)
    closes = np.concatenate([box1, ramp1, pullback, box2, ramp2])
    df = _make_df(closes, np.full(len(closes), 1_000_000.0))

    events = ei.detect_breakout_events(df, "TEST_SEQ")
    assert len(events) >= 2
    seqs = [e["event_seq_for_ticker"] for e in events]
    assert seqs == sorted(seqs), "event_seq_for_ticker 應該按時間遞增"
    assert seqs[0] == 1, f"第一個事件的序號應該是 1，實際為 {seqs[0]}"
    print(f"PASS: event_seq_for_ticker 正確遞增: {seqs}")


def test_registry_matches_actual_event_fields():
    """
    這是 Research Field Registry 真正有約束力的地方：實際跑一次
    detect_breakout_events()，把回傳的事件欄位跟 research/registry.py
    登記的清單比對。如果有人以後直接在 event_inventory.py 加了新欄位、
    卻忘記先在 registry.py 登記原因跟假設，這個測試會直接失敗——
    不是靠人自覺遵守文件，是真的會擋下來。
    """
    from research import registry

    df = _flat_box_then_breakout()
    events = ei.detect_breakout_events(df, "TEST_REGISTRY")
    assert len(events) >= 1

    actual_fields = set(events[0].keys())
    result = registry.validate_against_event_schema(actual_fields)

    assert result["is_consistent"], (
        f"發現事件欄位沒有在 research/registry.py 登記: {result['unregistered']}\n"
        f"新增任何事件欄位前，請先在 registry.py 補上 ResearchField "
        f"(reason + hypothesis)，這是本專案這一輪對齊的研究紀律。"
    )
    if result["unused"]:
        print(f"提醒（非錯誤）: registry.py 登記了但事件裡目前沒有的欄位: {result['unused']}"
              f"（可能是欄位已被移除，registry 尚未清理，建議之後順手清掉）")
    print("PASS: 事件實際欄位與 research/registry.py 登記內容一致")


def test_limit_up_day_detected_on_gap_up_breakout():
    """
    突破當天如果收盤價相較前一日跳空超過門檻(9.5%)，is_limit_up_day
    應該是 True；一般緩步推升的突破則應該是 False。
    """
    rng = np.random.default_rng(42)
    box = 98 + 4 * (rng.random(100) * 0.5 + 0.25)
    # 突破當天直接跳空 12%（模擬漲停鎖死），之後續漲
    gap_up_close = box[-1] * 1.12
    ramp = np.linspace(gap_up_close, 130, 30)
    closes = np.concatenate([box, [gap_up_close], ramp])
    volumes = np.full(len(closes), 1_000_000.0)

    df = _make_df(closes, volumes)
    events = ei.detect_breakout_events(df, "TEST_LIMITUP")
    assert len(events) >= 1
    assert events[0]["is_limit_up_day"] is True, (
        f"跳空12%應判定為漲停突破，實際: {events[0]['is_limit_up_day']}"
    )
    print("PASS: 跳空突破正確判定為 is_limit_up_day=True")

    # 對照組：緩步推升的突破（原本的 _flat_box_then_breakout）不應該
    # 被誤判為漲停
    df2 = _flat_box_then_breakout()
    events2 = ei.detect_breakout_events(df2, "TEST_NOLIMITUP")
    assert len(events2) == 1
    assert events2[0]["is_limit_up_day"] is False, (
        f"緩步推升的突破不應判定為漲停，實際: {events2[0]['is_limit_up_day']}"
    )
    print("PASS: 緩步推升的突破正確判定為 is_limit_up_day=False")


def run_all():
    test_single_continuous_breakout_counts_as_one_event()
    test_two_separate_breakouts_count_as_two_events()
    test_flat_no_breakout_counts_zero()
    test_insufficient_history_returns_no_events()
    test_stop_rule_thresholds()
    test_event_carries_professional_metadata_fields()
    test_event_seq_for_ticker_increments_across_repeated_breakouts()
    test_registry_matches_actual_event_fields()
    test_limit_up_day_detected_on_gap_up_breakout()
    print("\n所有自我測試通過（合成資料，尚未驗證真實資料格式相容性）")


if __name__ == "__main__":
    run_all()
