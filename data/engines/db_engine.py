import json
import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DatabaseEngine:
    """
    🗄️ Database 資料庫中心 (SQLite)

    避免每次都重新呼叫 yfinance 下載歷史資料：
      - 第一次查詢某檔股票時，下載完整歷史並寫入 SQLite
      - 之後在「新鮮期限」內再次查詢，直接從本地資料庫讀取，速度更快、也減少對 yfinance 的請求量
      - 過了新鮮期限（預設 6 小時）才會重新向 yfinance 抓取最新資料並覆寫快取

    資料庫檔案預設位置：<專案根目錄>/data/tqai.db

    ⚠️ 修正說明（v2.6）：
    新增 is_intraday_estimate 欄位（配合 DataEngine 的即時股價覆蓋修正）。
    這個欄位標記某一筆資料是否為「盤中即時估計值」而非正式收盤資料，
    讓 RiskEngine 在做 Beta 對齊等跨資料源比較時，可以選擇排除掉這種
    尚未定案的最後一筆，避免跟大盤基準（沒有即時覆蓋）的時間點錯位比較。
    使用 ALTER TABLE ... ADD COLUMN 做向後相容的 schema migration，
    舊資料庫升級時不會遺失既有快取資料。

    ⚠️ v2.9.10 新增：Watchlist 狀態機（回應「機構等級升級建議」文件的
    第六項建議）。這是本專案第一個「跨 session 持久化的使用者操作紀錄」
    功能——SQLite 早就存在（原本只拿來快取股價），這裡加了兩張新表：
    `watchlist_status`（每檔股票目前狀態）與 `watchlist_status_history`
    （狀態變化的完整歷史紀錄，只增不改）。狀態機本身只允許固定的
    狀態集合與「順序遞增或直接出場/歸檔」的合理轉換，不允許跳躍式
    的不合理轉換（例如從「觀察中」直接跳到「加碼」），細節見
    `WATCHLIST_STATES` 與 `is_valid_transition()`。

    ⚠️ 誠實範圍界定：這是「使用者手動標記狀態」的記錄工具，不是自動化
    交易系統——狀態轉換需要使用者自己在 UI 按下切換，系統不會自動幫你
    判斷「現在應該從觀察中變成準備中」，那需要額外的規則引擎（可以是
    未來的擴充方向，但這次沒有做，避免自動化判斷跟人工判斷混在一起
    搞不清楚是誰做的決定）。
    """

    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tqai.db"
    )

    # ==========================================
    # 連線與資料表初始化
    # ==========================================
    @staticmethod
    def get_connection(db_path: str = None):
        db_path = db_path or DatabaseEngine.DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        DatabaseEngine._init_schema(conn)
        return conn

    @staticmethod
    def _init_schema(conn):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                is_intraday_estimate INTEGER DEFAULT 0,
                PRIMARY KEY (ticker, date)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                ticker TEXT PRIMARY KEY,
                resolved_symbol TEXT,
                last_updated TEXT,
                row_count INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_directory (
                code TEXT NOT NULL,
                market TEXT NOT NULL,
                name TEXT,
                industry TEXT,
                is_etf INTEGER DEFAULT 0,
                yf_suffix TEXT,
                last_updated TEXT,
                PRIMARY KEY (code, market)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kv_cache (
                key TEXT PRIMARY KEY,
                updated_at TEXT,
                payload_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shareholding_snapshot_history (
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                large_holder_pct REAL,
                total_holders INTEGER,
                total_shares INTEGER,
                PRIMARY KEY (date, code)
            )
        """)
        # v2.9.10 新增：Watchlist 狀態機（見 class 說明的 v2.9.10 段落）。
        # current 只存「目前狀態」（單列/股，方便查詢與更新）；history 存
        # 每一次狀態變化的完整紀錄（可累加，不覆寫），兩張表分開是為了讓
        # 「查目前狀態」保持 O(1) 查詢，不用每次都去 history 表算最新一筆。
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_status (
                ticker TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                note TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                note TEXT,
                changed_at TEXT NOT NULL
            )
        """)
        # 向後相容：舊資料庫可能沒有這個欄位，補上去但不影響既有資料
        cur.execute("PRAGMA table_info(stock_prices)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "is_intraday_estimate" not in existing_cols:
            try:
                cur.execute("ALTER TABLE stock_prices ADD COLUMN is_intraday_estimate INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

        # v2.9.11 新增：watchlist_status 補上輕量部位欄位（進場價/股數/
        # 目前停損價），回應「Trade Manager／Position Engine」建議——但
        # 誠實範圍界定：這只是「使用者自己填的紀錄」，不是自動下單或自動
        # 移動停損，系統不會幫你算或幫你改，純粹是把「這檔股票我進場價/
        # 股數/目前停損設在哪」跟既有的狀態機存在同一個地方，方便查閱。
        cur.execute("PRAGMA table_info(watchlist_status)")
        wl_cols = {row[1] for row in cur.fetchall()}
        for col_name, col_type in [("entry_price", "REAL"), ("shares", "REAL"), ("current_stop", "REAL")]:
            if col_name not in wl_cols:
                try:
                    cur.execute(f"ALTER TABLE watchlist_status ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        # v2.9.11 新增：交易日誌 (Trade Journal)——使用者自己記錄「實際
        # 成交」的進出場，不是回測模擬、也不是自動產生。這是本專案唯一
        # 可以誠實計算「真實勝率/真實期望值」的資料來源（BacktestEngine
        # 算的是歷史模擬，Journal 算的是使用者真的做過的交易），兩者刻意
        # 分開呈現，不會混在一起變成假裝更精確的單一數字。
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                shares REAL NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                strategy_tag TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

    # ==========================================
    # 寫入快取
    # ==========================================
    @staticmethod
    def save_prices(ticker: str, df: pd.DataFrame, resolved_symbol: str = None, db_path: str = None):
        if df is None or df.empty:
            return

        conn = DatabaseEngine.get_connection(db_path)
        try:
            cols = ["date", "open", "high", "low", "close", "volume"]
            save_df = df[cols].copy()
            if "is_intraday_estimate" in df.columns:
                save_df["is_intraday_estimate"] = df["is_intraday_estimate"].astype(bool).astype(int)
            else:
                save_df["is_intraday_estimate"] = 0
            save_df["date"] = pd.to_datetime(save_df["date"]).dt.strftime("%Y-%m-%d")
            save_df.insert(0, "ticker", str(ticker))

            save_df.to_sql("stock_prices_staging", conn, if_exists="replace", index=False)

            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO stock_prices
                    (ticker, date, open, high, low, close, volume, is_intraday_estimate)
                SELECT ticker, date, open, high, low, close, volume, is_intraday_estimate
                FROM stock_prices_staging
            """)
            cur.execute("DROP TABLE stock_prices_staging")
            cur.execute("""
                INSERT OR REPLACE INTO sync_metadata (ticker, resolved_symbol, last_updated, row_count)
                VALUES (?, ?, ?, ?)
            """, (str(ticker), resolved_symbol, datetime.now().isoformat(), len(save_df)))
            conn.commit()
        finally:
            conn.close()

    # ==========================================
    # 讀取快取
    # ==========================================
    @staticmethod
    def load_prices(ticker: str, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume, is_intraday_estimate FROM stock_prices "
                "WHERE ticker = ? ORDER BY date",
                conn, params=(str(ticker),)
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            if "is_intraday_estimate" in df.columns:
                df["is_intraday_estimate"] = df["is_intraday_estimate"].fillna(0).astype(bool)
        return df

    @staticmethod
    def get_resolved_symbol(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT resolved_symbol FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    @staticmethod
    def get_last_updated(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT last_updated FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            row = cur.fetchone()
        finally:
            conn.close()

        if not row or not row[0]:
            return None
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            return None

    @staticmethod
    def is_fresh(ticker: str, max_age_hours: float = 6, db_path: str = None) -> bool:
        last_updated = DatabaseEngine.get_last_updated(ticker, db_path)
        if last_updated is None:
            return False
        return (datetime.now() - last_updated) < timedelta(hours=max_age_hours)

    # ==========================================
    # 工具方法：清除單一股票快取 / 查詢資料庫狀態
    # ==========================================
    @staticmethod
    def clear_cache(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM stock_prices WHERE ticker = ?", (str(ticker),))
            cur.execute("DELETE FROM sync_metadata WHERE ticker = ?", (str(ticker),))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def get_db_stats(db_path: str = None) -> dict:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT ticker) FROM stock_prices")
            ticker_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM stock_prices")
            row_count = cur.fetchone()[0]
        finally:
            conn.close()
        return {"cached_tickers": ticker_count, "total_rows": row_count}

    # ==========================================
    # 股票代碼/名稱目錄快取（配合 StockDirectoryEngine）
    # ==========================================
    @staticmethod
    def save_stock_directory(df: pd.DataFrame, db_path: str = None):
        """寫入/覆蓋某個市場別的代碼/名稱清單快取。"""
        if df is None or df.empty:
            return
        conn = DatabaseEngine.get_connection(db_path)
        try:
            save_df = df[["code", "market", "name", "industry", "is_etf", "yf_suffix"]].copy()
            save_df["is_etf"] = save_df["is_etf"].astype(bool).astype(int)
            save_df["last_updated"] = datetime.now().isoformat()

            markets_in_df = save_df["market"].unique().tolist()
            cur = conn.cursor()
            for m in markets_in_df:
                cur.execute("DELETE FROM stock_directory WHERE market = ?", (m,))

            save_df.to_sql("stock_directory_staging", conn, if_exists="replace", index=False)
            cur.execute("""
                INSERT OR REPLACE INTO stock_directory
                    (code, market, name, industry, is_etf, yf_suffix, last_updated)
                SELECT code, market, name, industry, is_etf, yf_suffix, last_updated
                FROM stock_directory_staging
            """)
            cur.execute("DROP TABLE stock_directory_staging")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def load_stock_directory(markets=None, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            if markets:
                placeholders = ",".join(["?"] * len(markets))
                query = f"SELECT * FROM stock_directory WHERE market IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=list(markets))
            else:
                df = pd.read_sql_query("SELECT * FROM stock_directory", conn)
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        if not df.empty and "is_etf" in df.columns:
            df["is_etf"] = df["is_etf"].fillna(0).astype(bool)
        return df

    @staticmethod
    def lookup_stock_name(code: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM stock_directory WHERE code = ? LIMIT 1", (str(code),))
            row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    # ==========================================
    # 通用小型快取 (Generic Key-Value Cache)
    # ==========================================
    # ⚠️ 用途：給「不是逐日K線、也不是股票代碼目錄」的其他小型、需要新鮮期限
    # 的資料使用（目前用途：ChipEngine.get_market_wide_institutional_ranking
    # 的全市場當日排行原始資料）。比照 get_stock_data() 系列的快取設計精神
    # （新鮮期限內直接吃快取、過期才重新抓取），但用一個通用的 key/JSON
    # payload 儲存方式，避免每多一種需要快取的資料就要新增一張專屬資料表。
    #
    # ⚠️ JSON 序列化陷阱：payload 內若含有 DataFrame.to_dict() 轉出來的數值，
    # 底層型別通常是 numpy.int64／numpy.float64／numpy.bool_，這些型別
    # `json.dumps` 預設不認得，會直接拋 TypeError。這裡用 default=
    # 參數統一轉型，呼叫端不需要自己在存入前手動轉換每一欄位。
    @staticmethod
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    @staticmethod
    def set_cache(key: str, payload, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, default=DatabaseEngine._json_default)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO kv_cache (key, updated_at, payload_json)
                VALUES (?, ?, ?)
            """, (key, datetime.now().isoformat(), payload_json))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def get_cache(key: str, max_age_hours: float = 6, db_path: str = None):
        """回傳 None 代表快取不存在或已過期（呼叫端應改為重新產生資料並呼叫
        set_cache 更新）。命中時回傳 {'updated_at':.., 'payload':..}。"""
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT updated_at, payload_json FROM kv_cache WHERE key = ?", (key,))
            row = cur.fetchone()
        except Exception:
            row = None
        finally:
            conn.close()

        if not row:
            return None

        updated_at, payload_json = row
        try:
            updated_dt = datetime.fromisoformat(updated_at)
        except Exception:
            return None
        if (datetime.now() - updated_dt) >= timedelta(hours=max_age_hours):
            return None

        try:
            payload = json.loads(payload_json)
        except Exception:
            return None

        return {"updated_at": updated_at, "payload": payload}

    # ==========================================
    # 大戶持股(千張大戶)歷史快照（配合 ChipEngine.get_shareholding_distribution）
    # ==========================================
    # ⚠️ 說明：TDCC 集保戶股權分散表開放資料每次抓到的都只是「當週最新
    # 快照」，官方沒有提供逐週歷史（TDCC自家查詢介面雖然有個股近1年歷史，
    # 但那是另一組不同的查詢方式，不在這次開放資料 CSV 範圍內）。這裡改成
    # 每次成功抓到新一週的資料時，自己把「大戶持股佔比」精簡摘要
    # （不是整份原始CSV）存進這張小表，隨著這個系統被持續使用，自然
    # 累積出屬於這個系統自己的歷史趨勢，讓使用者可以看到「大戶持股比例
    # 最近幾週的變化」，而不是每次都只看得到單一時間點的快照。
    @staticmethod
    def save_shareholding_snapshot(date: str, code: str, large_holder_pct: float,
                                    total_holders: int, total_shares: int, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO shareholding_snapshot_history
                    (date, code, large_holder_pct, total_holders, total_shares)
                VALUES (?, ?, ?, ?, ?)
            """, (date, str(code), large_holder_pct, total_holders, total_shares))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def load_shareholding_history(code: str, weeks: int = 12, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT date, large_holder_pct, total_holders, total_shares "
                "FROM shareholding_snapshot_history WHERE code = ? ORDER BY date DESC LIMIT ?",
                conn, params=(str(code), weeks)
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # ==========================================
    # 📋 Watchlist 狀態機 (v2.9.10 新增)
    # ==========================================
    # 狀態集合對應「機構等級升級建議」文件的第六項建議：
    #   觀察中 → 準備中 → 已建倉 → 持有中 → 加碼 / 減碼 → 出場 → 已歸檔
    # 「加碼」「減碼」不是終點狀態，做完動作後應該手動改回「持有中」，
    # 這裡不強制自動轉換（見 class docstring 的誠實範圍界定）。
    WATCHLIST_STATES = ["觀察中", "準備中", "已建倉", "持有中", "加碼", "減碼", "出場", "已歸檔"]

    # 允許的轉換：key 可以轉換到 value 集合內的任何狀態。刻意只允許
    # 「順著流程走」或「任何時候都能出場/歸檔」（畢竟真實世界不會照著
    # 教科書流程走，例如觀察中的股票可能直接被移除追蹤），但不允許
    # 「已歸檔」復活或「出場」倒退回持有中（出場後如果要重新追蹤，
    # 應該視為一次新的觀察，用 set_status 重新從「觀察中」開始，而不是
    # 讓歷史記錄出現「出場→持有中」這種不合理的倒退）。
    _WATCHLIST_TRANSITIONS = {
        "觀察中": {"準備中", "出場", "已歸檔"},
        "準備中": {"已建倉", "觀察中", "出場", "已歸檔"},
        "已建倉": {"持有中", "出場", "已歸檔"},
        "持有中": {"加碼", "減碼", "出場", "已歸檔"},
        "加碼": {"持有中", "出場", "已歸檔"},
        "減碼": {"持有中", "出場", "已歸檔"},
        "出場": {"已歸檔", "觀察中"},  # 出場後仍可回到「觀察中」重新開始追蹤（視為新一輪，非狀態倒退）
        "已歸檔": set(),  # 終點狀態，不允許復活；要重新追蹤請用新的 set_status 從「觀察中」開始
    }

    @staticmethod
    def is_valid_transition(from_status: str, to_status: str) -> bool:
        """
        from_status 為 None（該股票尚無任何狀態紀錄）時，允許直接設為
        任何合法狀態（通常會是「觀察中」，但不強制，尊重使用者的判斷）。
        """
        if from_status is None:
            return to_status in DatabaseEngine.WATCHLIST_STATES
        if from_status == to_status:
            return True  # 允許「原地更新備註」而不強制真的換狀態
        return to_status in DatabaseEngine._WATCHLIST_TRANSITIONS.get(from_status, set())

    @staticmethod
    def set_watchlist_status(ticker: str, new_status: str, note: str = None, db_path: str = None,
                              entry_price: float = None, shares: float = None, current_stop: float = None) -> dict:
        """
        設定股票的追蹤狀態，並寫入一筆歷史紀錄。若轉換不合法（見
        is_valid_transition），拒絕寫入並回傳原因，不會默默接受不合理
        的狀態跳躍。

        entry_price/shares/current_stop（v2.9.11 新增）為選填的輕量部位
        欄位，None 表示「這次更新不動這個欄位」，會沿用資料庫裡原本的值，
        不會被覆蓋成空值。
        """
        if new_status not in DatabaseEngine.WATCHLIST_STATES:
            return {"status": "error", "message": f"⚠️ 不是合法的狀態：{new_status}"}

        ticker = str(ticker).strip()
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, entry_price, shares, current_stop FROM watchlist_status WHERE ticker = ?", (ticker,))
            row = cur.fetchone()
            current_status = row[0] if row else None
            final_entry_price = entry_price if entry_price is not None else (row[1] if row else None)
            final_shares = shares if shares is not None else (row[2] if row else None)
            final_stop = current_stop if current_stop is not None else (row[3] if row else None)

            if not DatabaseEngine.is_valid_transition(current_status, new_status):
                return {
                    "status": "error",
                    "message": f"⚠️ 不允許的狀態轉換：{current_status or '（尚無紀錄）'} → {new_status}，"
                                f"請遵循流程順序，或先轉為「出場」再重新開始追蹤。",
                }

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("""
                INSERT INTO watchlist_status (ticker, status, note, updated_at, entry_price, shares, current_stop)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET status=excluded.status, note=excluded.note,
                    updated_at=excluded.updated_at, entry_price=excluded.entry_price,
                    shares=excluded.shares, current_stop=excluded.current_stop
            """, (ticker, new_status, note, now, final_entry_price, final_shares, final_stop))
            cur.execute("""
                INSERT INTO watchlist_status_history (ticker, from_status, to_status, note, changed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, current_status, new_status, note, now))
            conn.commit()
            return {"status": "ok", "ticker": ticker, "from": current_status, "to": new_status}
        finally:
            conn.close()

    @staticmethod
    def get_watchlist_status(ticker: str, db_path: str = None):
        conn = DatabaseEngine.get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT status, note, updated_at, entry_price, shares, current_stop "
                "FROM watchlist_status WHERE ticker = ?", (str(ticker).strip(),)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "status": row[0], "note": row[1], "updated_at": row[2],
                "entry_price": row[3], "shares": row[4], "current_stop": row[5],
            }
        finally:
            conn.close()

    @staticmethod
    def get_watchlist_history(ticker: str, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT from_status, to_status, note, changed_at FROM watchlist_status_history "
                "WHERE ticker = ? ORDER BY changed_at DESC",
                conn, params=(str(ticker).strip(),)
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    @staticmethod
    def list_watchlist(status_filter=None, db_path: str = None) -> pd.DataFrame:
        """列出所有有狀態紀錄的股票，可選擇只列出特定狀態（例如只看「持有中」）。"""
        conn = DatabaseEngine.get_connection(db_path)
        try:
            if status_filter:
                df = pd.read_sql_query(
                    "SELECT ticker, status, note, updated_at FROM watchlist_status WHERE status = ? ORDER BY updated_at DESC",
                    conn, params=(status_filter,)
                )
            else:
                df = pd.read_sql_query(
                    "SELECT ticker, status, note, updated_at FROM watchlist_status ORDER BY updated_at DESC", conn
                )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    # ==========================================
    # 📔 交易日誌 (Trade Journal) — v2.9.11 新增
    # ==========================================
    # ⚠️ 誠實範圍界定：這裡記錄的是使用者自己輸入的「實際成交」，系統不會
    # 自動幫你新增或修改這裡的紀錄（不像 watchlist_status 是純狀態標記，
    # 這裡的價格/股數會被拿去算真實損益，錯誤的資料會算出錯誤的績效，
    # 務必只填實際成交的內容）。這是本專案唯一可以誠實談「真實勝率」的
    # 地方——BacktestEngine 算的是歷史模擬（見 backtest_engine.py 的
    # in-sample 說明），此處算的是使用者真的做過的交易，兩者不應該被
    # 混為一談或互相取代。
    @staticmethod
    def log_trade_entry(ticker: str, entry_date: str, entry_price: float, shares: float,
                         strategy_tag: str = None, note: str = None, db_path: str = None) -> dict:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trade_journal (ticker, entry_date, entry_price, shares, strategy_tag, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(ticker).strip(), entry_date, entry_price, shares, strategy_tag, note, now, now))
            conn.commit()
            return {"status": "ok", "trade_id": cur.lastrowid}
        finally:
            conn.close()

    @staticmethod
    def log_trade_exit(trade_id: int, exit_date: str, exit_price: float, db_path: str = None) -> dict:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.cursor()
            cur.execute("""
                UPDATE trade_journal SET exit_date = ?, exit_price = ?, updated_at = ?
                WHERE id = ?
            """, (exit_date, exit_price, now, trade_id))
            conn.commit()
            if cur.rowcount == 0:
                return {"status": "error", "message": f"⚠️ 找不到交易紀錄 id={trade_id}"}
            return {"status": "ok", "trade_id": trade_id}
        finally:
            conn.close()

    @staticmethod
    def get_trade_journal(ticker: str = None, db_path: str = None) -> pd.DataFrame:
        conn = DatabaseEngine.get_connection(db_path)
        try:
            if ticker:
                df = pd.read_sql_query(
                    "SELECT * FROM trade_journal WHERE ticker = ? ORDER BY entry_date DESC",
                    conn, params=(str(ticker).strip(),)
                )
            else:
                df = pd.read_sql_query("SELECT * FROM trade_journal ORDER BY entry_date DESC", conn)
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    @staticmethod
    def compute_journal_stats(ticker: str = None, db_path: str = None) -> dict:
        """
        只用「已出場」（exit_price 不為空）的交易計算真實績效統計。
        進行中的交易不計入勝率/期望值（未實現損益尚未定案，混進來會扭曲
        統計）。交易筆數過少（<5筆）時明確提醒統計可信度低，不假裝一個
        小樣本的勝率有代表性。
        """
        df = DatabaseEngine.get_trade_journal(ticker=ticker, db_path=db_path)
        if df.empty:
            return {"status": "no_data", "message": "尚無交易紀錄。"}

        closed = df[df["exit_price"].notna()].copy()
        if closed.empty:
            return {"status": "no_closed_trades", "message": "尚無已出場的交易，無法計算真實績效統計。", "open_trades": len(df)}

        closed["pnl_pct"] = (closed["exit_price"] - closed["entry_price"]) / closed["entry_price"] * 100
        closed["pnl_amount"] = (closed["exit_price"] - closed["entry_price"]) * closed["shares"]

        wins = closed[closed["pnl_pct"] > 0]
        losses = closed[closed["pnl_pct"] <= 0]
        win_rate = len(wins) / len(closed) * 100
        avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0.0
        avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0.0
        expectancy_pct = (win_rate / 100 * avg_win_pct) + ((1 - win_rate / 100) * avg_loss_pct)

        result = {
            "status": "ok",
            "total_closed_trades": len(closed),
            "open_trades": len(df) - len(closed),
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "expectancy_pct": round(expectancy_pct, 2),
            "total_pnl_amount": round(closed["pnl_amount"].sum(), 0),
        }
        if len(closed) < 5:
            result["low_sample_warning"] = (
                f"⚠️ 只有 {len(closed)} 筆已出場交易，樣本數過少，這裡的勝率/期望值"
                f"統計可信度低，不足以代表真實策略優勢，僅供參考。"
            )
        return result