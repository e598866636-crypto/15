"""
test_chipflow.py

手動驗證用腳本：呼叫 ChipFlowEngine.compute_institutional_streaks，
確認連買/連賣天數是否合理。放在專案根目錄（跟 engines/ 同一層）執行：

    python test_chipflow.py
"""
from engines.chip_flow_engine import ChipFlowEngine

result = ChipFlowEngine.compute_institutional_streaks(lookback_days=30)
print("status:", result.get('status'))
print("days_used:", result.get('days_used'))

streaks = result.get('streaks', {})
print(f"共 {len(streaks)} 檔股票有連續天數資料，隨機挑 5 檔看看：")

sample = list(streaks.items())[:5]
for code, streak in sample:
    print(code, streak)
