import re

import pandas as pd


class NameEngine:
    """
    🏷️ 名稱標籤引擎 (Name & Market Tagging Engine) - TQAI Pro v2.7

    背景：
    此前系統各處（掃描結果表、產業排名、個股戰情室標題...）只顯示股票代碼，
    使用者得自己記代碼對應公司名稱；跨市場（上市／上櫃／興櫃／ETF）掃描時
    尤其不直覺，也容易在跨引擎顯示時（Scanner／Industry／Dashboard）各自
    土法煉鋼維護一份對照表，彼此漸漸兜不起來。

    這個引擎提供「全系統統一」的 `[代碼] 名稱` 標籤化服務，並附帶市場別判斷，
    讓 ScannerEngine、IndustryEngine、Dashboard 都用同一套資料源顯示。

    ⚠️ 限制與設計取捨：
    1. NAME_MAP／MARKET_TYPE_OVERRIDES 是內建的靜態對照表，目前只涵蓋本專案
       觀察名單與常見 ETF 出現過的代碼。若代碼不在表內，get_name() 回傳
       「未知名稱」而不是拋例外，確保掃描全市場時單一未知代碼不會中斷整條
       pipeline（呼應 ScannerEngine.scan() 本來就有的 try/except 容錯設計）。
    2. 興櫃 (Emerging Stock Market) 目前沒有內建代碼清單 —— 主因是
       yfinance 對興櫃股票的資料覆蓋率非常低（見 data_engine.py 的說明），
       就算標了「興櫃」名稱，實際上多半還是抓不到K線，暫不無中生有假造
       一份看起來完整、但其實抓不到資料的清單，避免誤導使用者。
       EMERGING_TICKERS 保留為擴充位，未來若串接興櫃專屬資料源
       （例如證櫃買中心公開資訊）可以直接補進來。
    3. 如需完整全市場代碼對照，建議未來改串接 TWSE/TPEx 官方的證券基本資料
       API 動態查詢；靜態表是目前先滿足既有觀察名單需求的權宜作法。
    """

    # === 股票／ETF 代碼 → 名稱 對照表 ===
    NAME_MAP = {
        # --- 跨界生醫與前沿科技觀察名單 ---
        "2330": "台積電", "3374": "精材", "6223": "旺矽", "3711": "日月光投控",
        "6841": "長佳智能", "2382": "廣達", "3231": "緯創", "2356": "英業達",
        "6472": "保瑞", "6901": "鑽石生技", "4743": "合一", "6712": "長聖",
        # --- 大盤權值觀察組 ---
        "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2379": "瑞昱",
        "3034": "聯詠", "2412": "中華電", "2881": "富邦金", "2603": "長榮",
        "1515": "力山", "1101": "台泥",
        # --- 常見 ETF（供 ETF 擴充功能示範／預設清單使用）---
        "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
        "006208": "富邦台50", "00631L": "元大台灣50正2", "00713": "元大台灣高息低波",
    }

    # === 市場別對照：僅標記「非上市」的特例，預設 fallback 視為上市/上櫃 ===
    MARKET_TYPE_OVERRIDES = {
        "0050": "ETF", "0056": "ETF", "00878": "ETF",
        "006208": "ETF", "00631L": "ETF", "00713": "ETF",
    }

    # 興櫃代碼清單（見上方類別註解，目前刻意保留空白，避免假造覆蓋率）
    EMERGING_TICKERS = set()

    @staticmethod
    def _clean(code) -> str:
        return str(code).split(".")[0].strip()

    @staticmethod
    def get_name(code) -> str:
        return NameEngine.NAME_MAP.get(NameEngine._clean(code), "未知名稱")

    @staticmethod
    def get_market_type(code) -> str:
        c = NameEngine._clean(code)
        if c in NameEngine.MARKET_TYPE_OVERRIDES:
            return NameEngine.MARKET_TYPE_OVERRIDES[c]
        if c in NameEngine.EMERGING_TICKERS:
            return "興櫃"
        return "上市/上櫃"

    @staticmethod
    def is_etf(code) -> bool:
        c = NameEngine._clean(code)
        if NameEngine.get_market_type(c) == "ETF":
            return True
        # ⚠️ 修正說明：原本 is_etf() 只查 MARKET_TYPE_OVERRIDES 這份寫死的
        # 6檔清單（0050/0056/00878/006208/00631L/00713），台股實際上有數百檔
        # ETF，查其他檔（例如 00919、00929、00646...）原本會被判定為「不是
        # ETF」，導致下游（例如 FundamentalEngine）誤把 ETF 當一般個股去跑
        # 基本面分析，產生一堆N/A卻顯示「中性狀態」的誤導畫面。這裡補上
        # 通用的代碼型態判斷：台股 ETF 代碼慣例上以「00」開頭（例如 0050、
        # 006208、00919），一般個股代碼則不會以00開頭——這是市場慣例，
        # 不是100%保證沒有例外，但已經能涵蓋絕大多數ETF代碼。
        return bool(re.match(r"^00\d{2,4}[A-Za-z]{0,2}$", c))

    @staticmethod
    def is_likely_etf_code(code) -> bool:
        """單獨暴露這個通用型態判斷，供不想依賴 MARKET_TYPE_OVERRIDES 硬清單
        的呼叫端使用（例如 FundamentalEngine 判斷是否該跳過基本面分析）。"""
        return bool(re.match(r"^00\d{2,4}[A-Za-z]{0,2}$", NameEngine._clean(code)))

    @staticmethod
    def get_tag(code) -> str:
        """回傳全系統統一格式：'[代碼] 名稱'，用於表格、標題、圖表標籤。"""
        c = NameEngine._clean(code)
        return f"[{c}] {NameEngine.get_name(c)}"

    @staticmethod
    def tag_dataframe(df: pd.DataFrame, code_col: str = "代碼", name_col: str = "名稱",
                       tag_col: str = "標的", market_col: str = "市場別") -> pd.DataFrame:
        """
        為掃描結果 DataFrame 統一補上名稱／標籤／市場別欄位。
        不修改／覆蓋原始代碼欄位，確保後續 join、篩選仍可用純代碼比對，
        不會因為欄位被改成 '[2330] 台積電' 這種複合字串而找不到對應列。
        """
        if df is None or df.empty or code_col not in df.columns:
            return df
        df = df.copy()
        df[name_col] = df[code_col].apply(NameEngine.get_name)
        df[tag_col] = df[code_col].apply(NameEngine.get_tag)
        df[market_col] = df[code_col].apply(NameEngine.get_market_type)
        return df

    # ==========================================
    # 依中文/英文名稱反查代碼（新增：支援「查找股票輸入中文名稱」）
    # ==========================================
    @staticmethod
    def search_by_name(keyword: str) -> list:
        """
        依關鍵字在內建 NAME_MAP（本專案觀察名單）中做子字串搜尋，讓使用者
        可以直接輸入公司/ETF中文名稱（例如「台積電」、「元大台灣50」）
        查詢，而不是只能輸入代碼。

        ⚠️ 涵蓋範圍限制：這裡只涵蓋內建觀察名單（NAME_MAP），不是全市場
        名稱目錄。若要涵蓋全市場任意股票的名稱搜尋，請搭配
        StockDirectoryEngine.search_by_name()（需要先執行過 refresh_all()
        建立本地快取，見 stock_directory_engine.py 說明）。呼叫端建議兩者
        合併使用：先查這裡（快、不需要資料庫/網路），再查
        StockDirectoryEngine 補齊觀察名單以外的股票。

        回傳格式：[{'code': '2330', 'name': '台積電', 'market': '上市/上櫃'}, ...]
        找不到回傳空 list，不拋例外。
        """
        keyword = str(keyword).strip()
        if not keyword:
            return []
        results = []
        for code, name in NameEngine.NAME_MAP.items():
            if keyword in name:
                results.append({
                    "code": code,
                    "name": name,
                    "market": NameEngine.get_market_type(code),
                })
        return results