# -*- coding: utf-8 -*-
"""
research/registry.py

📋 Research Field Registry（研究欄位登記簿）

⚠️ 為什麼需要這個檔案
Event Inventory 已經從「一個突破訊號」擴充到「突破 + 突破幅度 + 流動性
+ ETF 標記 + 事件序號 + regime」，如果沒有紀律，下一版很容易變成：

    breakout → magnitude → liquidity → ETF → seq → ATR → RSI → Beta →
    Industry → PEG → Market Cap → Volatility → Gap → ...

最後 Research Dataset 有 300 個欄位，真正被拿來分析的只有 5 個——這是
量化研究專案最常見的失控模式，此檔案的目的就是防止它發生。

規則只有一條：
    **每一個出現在事件 CSV 裡的欄位，都必須先在這裡登記「為什麼加、
    驗證什麼假設」，才能加進 event_inventory.py。**

這不是靠自覺遵守，`validate_against_event_schema()` 會實際比對
`detect_breakout_events()` 產出的欄位跟這裡登記的清單，兩邊對不上就
會丟出錯誤（見 research/test_event_inventory.py 的
`test_registry_matches_actual_event_fields`），逼這個檔案跟實際程式碼
保持同步，而不是寫完就沒人再更新的「文件」。

── 目前的立場（跟本輪對齊）──
下一步先不要再增加欄位。先把 Phase A（Event Inventory）真的跑起來，
看過實際事件數、事件分布之後，再回頭決定這裡列的假設哪些值得驗證、
哪些其實沒資訊量可以拿掉。這個檔案本身也要接受同樣的紀律：新增一筆
登記，不代表這個欄位就該存在到永遠。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ResearchField:
    name: str                      # 對應事件 dict / CSV 的欄位名稱
    added_in: str                  # 哪一個研究階段/版本加的（例如 "Phase A v1"）
    reason: str                    # 為什麼要記錄這個欄位（工程或研究上的動機）
    hypothesis: str                # 這個欄位打算驗證的假設，若還沒有明確假設要寫「純描述性，尚無假設」
    owner: str                     # 這個欄位屬於哪個研究專案（目前都是 Breakout Alpha Study）
    is_structural: bool = False    # True＝事件的識別/定位欄位（ticker、日期等），不是研究變數本身


# ==========================================
# 結構性欄位（事件的身分識別，不是研究變數，但仍需登記，
# 讓 validate_against_event_schema() 能完整比對）
# ==========================================
_STRUCTURAL_FIELDS = [
    ResearchField(
        name="ticker",
        added_in="Phase A v1",
        reason="事件屬於哪一檔股票，用於後續 join 回財報/籌碼/新聞等其他研究維度。",
        hypothesis="純識別欄位，無假設。",
        owner="Breakout Alpha Study",
        is_structural=True,
    ),
    ResearchField(
        name="event_date",
        added_in="Phase A v1",
        reason="事件發生日，用於計算 Forward Return（Phase B）與跟 regime 時間序列對齊。",
        hypothesis="純識別欄位，無假設。",
        owner="Breakout Alpha Study",
        is_structural=True,
    ),
]

# ==========================================
# 研究變數欄位
# ==========================================
_RESEARCH_FIELDS = [
    ResearchField(
        name="box_high",
        added_in="Phase A v1",
        reason="突破的壓力價位本身，Phase B 計算 Forward Return 的基準點之一。",
        hypothesis="純描述性，尚無假設（是 breakout_magnitude_pct 的計算基礎，不重複驗證）。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="box_low",
        added_in="Phase A v1",
        reason="整理箱下緣，BreakoutEngine 現有邏輯拿來當建議停損價，一併記錄供後續參考。",
        hypothesis="純描述性，尚無假設。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="box_days",
        added_in="Phase A v1",
        reason="整理箱維持的天數——這就是「整理多久」，涵蓋了原本另外提議的"
               "「days_in_box」概念，兩者是同一件事，故不重複新增欄位。",
        hypothesis="整理時間越長，突破後的延續性可能越高（籌碼沉澱越久，"
                   "浮額換手越乾淨）；但也可能相反（整理太久代表買盤動能不足），"
                   "方向不預設，留給 Phase B 驗證。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="box_range_pct",
        added_in="Phase A v1",
        reason="整理箱的窄幅程度，用來事後檢查 12% 門檻是否設得太鬆"
               "（如果大量事件的 range_pct 都貼近 12% 上限，代表門檻可能"
               "需要收緊，選到的不是真正的「窄幅整理」）。",
        hypothesis="箱型越窄，突破的訊號品質可能越高。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="volume_confirmed",
        added_in="Phase A v1",
        reason="是否通過 BreakoutEngine 既有的三段量價共振驗證"
               "（爆量突破→量縮→再放量），現成資料，不用新寫邏輯。",
        hypothesis="有量確認的突破，後續延續率應該高於無量確認"
                   "（這是本研究的第一個、也是最基本的可驗證假設）。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="regime_at_event",
        added_in="Phase A v1",
        reason="事件發生當天的大盤 regime（只依賴 ^TWII，跟個股無關，"
               "逐日滾動算過一次即可）。目前只當 metadata 記錄，不當"
               "分組依據——事件數不夠多之前分組只會製造假象的精確度。",
        hypothesis="多頭 regime 下的突破延續率高於空頭/高波動 regime"
                   "（是否要驗證，等 Phase A 事件數確認足夠切分後再決定）。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="breakout_magnitude_pct",
        added_in="Phase A v1 補強",
        reason="收盤價超出箱頂的百分比——量化「突破的乾脆度」，"
               "區分壓線 0.5% 勉強過門檻 vs 站上箱頂 8% 的決心。",
        hypothesis="突破幅度越大，後續延續率越高（Event Strength 越強，"
                   "資金決心越明確）。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="liquidity_level_at_event",
        added_in="Phase A v1 補強",
        reason="複用既有 RiskEngine.add_liquidity_metrics()，把突破當天的"
               "流動性等級記錄下來。這不是預測因子，是「能不能真的執行」"
               "的現實限制（Execution Constraint），跟 Alpha 是兩件事，"
               "但兩者都要記錄才能判斷一個訊號有沒有實際交易價值。",
        hypothesis="非預測性假設：低流動性事件即使「統計上」延續率高，"
                   "也可能無法用有意義的部位規模執行，須跟其他欄位分開看，"
                   "不直接混入報酬率統計。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="is_etf",
        added_in="Phase A v1 補強",
        reason="ETF 的「突破」是追蹤指數的技術現象，可能沒有個股資金動能"
               "可以解釋。定位是 Dataset Filter（未來研究可選擇 exclude ETF"
               "或 ETF only），不是拿來加權評分的 Alpha 因子。"
               "⚠️ stock_directory 表未建立時此欄位為 None（無法判斷），"
               "不可誤讀為 False。",
        hypothesis="ETF 突破事件的延續率統計特性可能與個股不同"
                   "（應該分開看，不應該混在同一個母體裡下結論）。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="event_seq_for_ticker",
        added_in="Phase A v1 補強",
        reason="同一檔股票在樣本期間內的第幾次突破事件，成本極低"
               "（單純計數），但能讓 Phase B 檢驗「第一次突破」跟"
               "「重複突破」是否應該被當成不同母體。",
        hypothesis="方向不預設——可能第一次突破成功率最高（真正的"
                   "起漲點），也可能重複測試後才是真突破（壓力區被"
                   "反覆確認），由資料決定，不預設答案。",
        owner="Breakout Alpha Study",
    ),
    ResearchField(
        name="is_limit_up_day",
        added_in="Phase A v1 補強第二輪",
        reason="判斷突破當天是否為漲停(或貼近漲停，近似門檻9.5%)。"
               "這是 Execution Constraint，不是 Alpha 因子——漲停鎖死"
               "當天很可能根本無法用假設的進場價成交，Phase B 若要"
               "計算 Forward Return，這類事件的「進場可行性」需要"
               "另外處理，不能跟一般事件用同一套進場價假設。",
        hypothesis="漲停突破事件的後續延續率統計特性可能與非漲停突破"
                   "不同（可能是消息面直接跳空鎖死，而非技術面量價"
                   "堆疊出來的突破，驅動力不同），應分開看，不宜混入"
                   "同一母體下結論。",
        owner="Breakout Alpha Study",
    ),
]

ALL_FIELDS = _STRUCTURAL_FIELDS + _RESEARCH_FIELDS
FIELD_NAMES = {f.name for f in ALL_FIELDS}


def get_field(name: str) -> Optional[ResearchField]:
    for f in ALL_FIELDS:
        if f.name == name:
            return f
    return None


def validate_against_event_schema(actual_field_names) -> dict:
    """
    比對「事件 dict 實際擁有的欄位」跟「這份 Registry 登記的欄位」，
    回傳兩邊的差異，讓呼叫端（例如測試）可以斷言兩者一致。

    - unregistered: 事件裡有，但 Registry 沒登記過的欄位
      （代表有人加了新欄位卻沒有先在這裡登記原因與假設，違反本檔案
      開頭說明的規則）
    - unused: Registry 登記過，但事件裡已經沒有的欄位
      （代表欄位被拿掉了，但登記還留著，這是良性的，只是提醒該清理，
      不算違規——研究過程中砍掉沒資訊量的欄位是正常的，但保留紀錄
      可以避免未來重複造同一個欄位）
    """
    actual = set(actual_field_names)
    registered = FIELD_NAMES
    return {
        "unregistered": sorted(actual - registered),
        "unused": sorted(registered - actual),
        "is_consistent": (actual - registered) == set(),
    }


def print_registry():
    print("\n" + "=" * 70)
    print("Research Field Registry — Breakout Alpha Study")
    print("=" * 70)
    for f in ALL_FIELDS:
        kind = "[結構性]" if f.is_structural else "[研究變數]"
        print(f"\n{kind} {f.name}  (加入於: {f.added_in}, 負責研究: {f.owner})")
        print(f"  原因: {f.reason}")
        print(f"  假設: {f.hypothesis}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_registry()
