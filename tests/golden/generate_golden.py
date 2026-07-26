"""
Golden Dataset — Step 1：抓取原始 OHLCV 資料

⚠️ 這支腳本必須在你本機（有網路、能連 yfinance）執行，我的沙箱環境連不到
finance.yahoo.com，沒辦法幫你跑這一步。跑完之後，把整個 tests/golden/raw/
資料夾（含 .parquet）跟這份程式一起放回你的 repo，我才能在下一步用同一份
凍結資料驗證 Engine 改動前後是否一致。

用法：
    cd TQAI_Pro
    python tests/golden/generate_golden.py

會依 GOLDEN_TICKERS 清單逐一呼叫 DataEngine.get_stock_data()，存成
tests/golden/raw/{ticker}.parquet。之後 Golden Regression 永遠讀這份
「凍結」的原始資料重新跑 pipeline，不會每次都重新抓即時報價——
這樣才能保證「跑兩次結果不同」一定是 Engine 邏輯改變造成的，
不是市場資料本身每天都在變動造成的雜訊。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.data_engine import DataEngine

# 依討論結果選定的 7 檔，涵蓋大型權值 / IC / EMS / AI Server 供應鏈 /
# 中型股 / ETF / 低成交量小型股，避免只驗證熱門股的偏誤。
GOLDEN_TICKERS = {
    "2330": "台積電（大型權值）",
    "2454": "聯發科（IC 設計）",
    "2317": "鴻海（EMS）",
    "3017": "奇鋐（AI Server 供應鏈）",
    "3661": "世芯-KY（中型股）",
    "0050": "元大台灣50（ETF）",
    "6485": "點序（低成交量小型股，示例代碼，請依實際需求替換）",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    results = {}
    for ticker, label in GOLDEN_TICKERS.items():
        try:
            df = DataEngine.get_stock_data(ticker, use_cache=False)
            out_path = os.path.join(RAW_DIR, f"{ticker}.parquet")
            df.to_parquet(out_path, index=False)
            results[ticker] = f"OK  ({len(df)} 筆, 存至 {out_path})"
        except Exception as e:
            results[ticker] = f"FAIL ({e})"

    print("=" * 50)
    print("Golden Dataset 原始資料抓取結果")
    print("=" * 50)
    for ticker, label in GOLDEN_TICKERS.items():
        print(f"{ticker} {label:20s} -> {results[ticker]}")
    print("=" * 50)
    print("完成後，把 tests/golden/raw/ 一起交給 Claude 做下一步（凍結 snapshot）。")


if __name__ == "__main__":
    main()
