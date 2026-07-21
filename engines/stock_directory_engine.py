import io
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests


class StockDirectoryEngine:
    """
    🏷️ 全市場代碼/名稱目錄 (Stock Directory Engine)

    提供「[代碼][名稱]」全系統標籤化所需的對照表，涵蓋：
      - 上市 (TWSE)
      - 上櫃 (TPEx)
      - 興櫃 (Emerging Stock Market)
      - ETF（實務上 ETF 本身就掛在上市或上櫃清單內，這裡額外用產業別欄位
             把它們獨立標記出來，方便「興櫃/ETF 異動專區」單獨篩選）

    資料來源：證交所 ISIN 公開查詢頁（isin.twse.com.tw），這是市場上取得
    「全市場代碼+名稱+市場別」最穩定的公開來源之一。

    ⚠️ 重要限制（誠實揭露）：
    1. 這個引擎需要對外部網站發送 HTTP 請求，本開發環境（沙盒）目前沒有
       對外網路權限，所以以下的 fetch_* 方法在這裡「無法實際執行測試」，
       只完成了語法檢查與程式邏輯設計。請在你自己有網路權限的環境（例如
       本機或伺服器）先跑一次 `StockDirectoryEngine.refresh_all()` 驗證：
         (a) ISIN 頁面的 HTML 表格結構是否與程式解析邏輯相符（證交所偶爾
             會微調頁面格式，若解析出來是空的，多半是這裡需要對應調整）
         (b) 網路逾時/被擋（User-Agent、速率限制）等狀況的實際處理
    2. 興櫃股票的資料完整度（尤其是 yfinance 是否有對應的歷史K線）普遍
       比上市櫃差很多，很多興櫃個股在 yfinance 根本查不到，DataEngine
       串接時務必做好「查無資料」的容錯（既有的 try/except 已具備基本保護，
       但使用者應該預期興櫃標的的技術分析覆蓋率會明顯偏低）。
    """

    ISIN_URLS = {
        "listed": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",   # 上市 (含上市ETF)
        "otc": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",      # 上櫃
        "emerging": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=5",  # 興櫃
    }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    }

    SUFFIX_MAP = {"listed": ".TW", "otc": ".TWO", "emerging": ".TWO"}

    # ==========================================
    # 1. 抓取單一市場別清單
    # ==========================================
    @staticmethod
    def fetch_market_list(market: str) -> pd.DataFrame:
        """
        market: 'listed' | 'otc' | 'emerging'
        回傳欄位: code, name, market, industry, is_etf, yf_suffix
        """
        if market not in StockDirectoryEngine.ISIN_URLS:
            raise ValueError(f"不支援的市場別: {market}，請用 listed/otc/emerging")

        url = StockDirectoryEngine.ISIN_URLS[market]
        try:
            resp = requests.get(url, headers=StockDirectoryEngine.HEADERS, timeout=15)
            resp.encoding = "big5"  # ISIN 頁面編碼慣例為 Big5
            tables = pd.read_html(io.StringIO(resp.text))
        except Exception as e:
            raise RuntimeError(f"抓取 {market} 清單失敗（可能是無網路權限或頁面格式變動）: {e}")

        if not tables:
            raise RuntimeError(f"{market} 頁面沒有解析到任何表格，請確認 ISIN 頁面結構是否變動")

        raw = tables[0]
        # ISIN 頁面第一列通常是欄位標題重複，實際資料從第二列開始，第一欄格式為 "代碼　名稱"
        raw.columns = raw.iloc[0]
        raw = raw.iloc[1:].reset_index(drop=True)

        code_name_col = raw.columns[0]
        industry_col = next((c for c in raw.columns if "產業別" in str(c)), None)

        records = []
        for _, row in raw.iterrows():
            cell = str(row[code_name_col]).strip()
            if "\u3000" in cell:  # 全形空白分隔代碼與名稱
                parts = cell.split("\u3000", 1)
            elif " " in cell:
                parts = cell.split(" ", 1)
            else:
                continue
            if len(parts) != 2:
                continue
            code, name = parts[0].strip(), parts[1].strip()
            if not code or not name:
                continue

            industry = str(row[industry_col]).strip() if industry_col else ""
            is_etf = ("ETF" in industry.upper()) or ("受益" in industry) or name.upper().startswith("ETF")

            records.append({
                "code": code,
                "name": name,
                "market": market,
                "industry": industry,
                "is_etf": bool(is_etf),
                "yf_suffix": StockDirectoryEngine.SUFFIX_MAP[market],
            })

        df = pd.DataFrame(records)
        # 過濾掉非個股/ETF的雜項列（例如權證、debenture 等代碼通常長度或格式不同，這裡先保留基本股票/ETF代碼格式）
        if not df.empty:
            df = df[df["code"].str.match(r"^[0-9A-Za-z]{4,6}$")].reset_index(drop=True)
        return df

    @staticmethod
    def refresh_all(db_path: str = None) -> dict:
        """抓取全部三個市場別並寫入本地快取，回傳各市場筆數統計。"""
        from engines.db_engine import DatabaseEngine
        stats = {}
        for market in StockDirectoryEngine.ISIN_URLS:
            try:
                df = StockDirectoryEngine.fetch_market_list(market)
                DatabaseEngine.save_stock_directory(df, db_path=db_path)
                stats[market] = len(df)
            except Exception as e:
                stats[market] = f"失敗: {e}"
        return stats

    # ==========================================
    # 2. 查詢介面（依賴本地快取，需先 refresh_all 過一次）
    # ==========================================
    @staticmethod
    def get_name(code: str, db_path: str = None) -> str:
        from engines.db_engine import DatabaseEngine
        code = str(code).split(".")[0].strip()
        name = DatabaseEngine.lookup_stock_name(code, db_path=db_path)
        return name or "未知代碼"

    @staticmethod
    def format_label(code: str, db_path: str = None) -> str:
        """回傳企劃書要求的全系統標籤格式：[代碼] 名稱"""
        code = str(code).split(".")[0].strip()
        name = StockDirectoryEngine.get_name(code, db_path=db_path)
        return f"[{code}] {name}" if name != "未知代碼" else f"[{code}]"

    @staticmethod
    def list_universe(markets=None, etf_only: bool = False, exclude_etf: bool = False, db_path: str = None) -> pd.DataFrame:
        """
        取得可掃描的全市場清單（供 ScannerEngine.scan_breakout 使用）。
        markets: None 表示全部（listed/otc/emerging），或傳入子集合的 list
        etf_only / exclude_etf: 互斥的篩選條件，用來單獨產出「興櫃/ETF 異動專區」
        """
        from engines.db_engine import DatabaseEngine
        df = DatabaseEngine.load_stock_directory(markets=markets, db_path=db_path)
        if df.empty:
            return df
        if etf_only:
            df = df[df["is_etf"]]
        elif exclude_etf:
            df = df[~df["is_etf"]]
        return df.reset_index(drop=True)

    # ==========================================
    # 3. 依名稱搜尋（新增：支援「查找股票輸入中文名稱」，涵蓋全市場）
    # ==========================================
    @staticmethod
    def search_by_name(keyword: str, markets=None, db_path: str = None) -> pd.DataFrame:
        """
        在本地快取的全市場代碼/名稱目錄（stock_directory 表）中，依名稱
        關鍵字做子字串搜尋，回傳 code/market/name/industry/is_etf/yf_suffix。

        ⚠️ 前提：這張快取表需要先執行過 StockDirectoryEngine.refresh_all()
        （需要對外網路權限抓 TWSE ISIN 頁面）才會有資料；若尚未執行過或
        沙盒無網路權限，回傳空 DataFrame，呼叫端應該要有 fallback（例如
        改用 NameEngine.search_by_name() 查內建觀察名單，或提示使用者改用
        代碼查詢），不應該假設這裡一定查得到資料。
        """
        from engines.db_engine import DatabaseEngine
        keyword = str(keyword).strip()
        if not keyword:
            return pd.DataFrame()

        try:
            df = DatabaseEngine.load_stock_directory(markets=markets, db_path=db_path)
        except Exception:
            return pd.DataFrame()

        if df.empty or "name" not in df.columns:
            return pd.DataFrame()

        matches = df[df["name"].astype(str).str.contains(keyword, regex=False, na=False)]
        return matches.reset_index(drop=True)