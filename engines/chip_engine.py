import io
from datetime import datetime, timedelta

import pandas as pd
import requests

from engines.logging_config import get_logger

logger = get_logger(__name__)


class ChipEngine:
    """
    🏦 籌碼中心 (Chip Center)

    抓取台灣證券交易所 (TWSE) 公開資訊：
      - 三大法人買賣超（外資及陸資 / 投信 / 自營商）
      - 融資融券餘額

    以及台灣集中保管結算所 (TDCC) 開放資料：
      - 集保戶股權分散表（大戶持股／千張大戶集中度，v2.9 新增）

    限制：
      - 三大法人/融資融券目前僅支援「上市」股票（.TW），上櫃（.TWO）因資料來源不同（TPEx），暫不支援
      - 僅在交易日有資料；若查詢日為假日或盤後資料尚未公布，會自動往前找最近的交易日
      - 依賴 TWSE OpenAPI，若官方介面改版或服務異常，會回傳 status='unavailable'，不影響主程式運作
      - 股權分散表（TDCC）涵蓋上市/上櫃/興櫃全市場，見下方 get_shareholding_distribution() 說明的額外限制
    """

    TWSE_INSTITUTIONAL_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
    TWSE_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

    # TDCC（臺灣集中保管結算所）集保戶股權分散表開放資料，全市場單一CSV，
    # 每週更新。來源：https://data.gov.tw/dataset/11452
    TDCC_SHAREHOLDING_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"

    # ⚠️ 級距對照表誠實揭露：TDCC 原始CSV的「持股分級」欄位只給數字代碼
    # (1~17)，不含級距說明文字。這裡依業界慣例（Goodinfo/CMoney/神秘金字塔
    # 等第三方籌碼網站都採用同一套）補上人類可讀的級距標籤，這是市場共識
    # 慣例，不是TDCC官方在這份CSV裡直接提供的欄位——若集保結算所未來調整
    # 級距定義，這裡需要同步更新。第16級是罕見的極端值調整列，第17級固定
    # 是全部級距（1~16）的合計列（人數/股數/佔比100%），皆不計入「大戶」
    # 級距判斷。
    SHAREHOLDING_TIER_LABELS = {
        1: "1～999股", 2: "1,000～5,000股", 3: "5,001～10,000股",
        4: "10,001～15,000股", 5: "15,001～20,000股", 6: "20,001～30,000股",
        7: "30,001～40,000股", 8: "40,001～50,000股", 9: "50,001～100,000股",
        10: "100,001～200,000股", 11: "200,001～400,000股", 12: "400,001～600,000股",
        13: "600,001～800,000股", 14: "800,001～1,000,000股", 15: "1,000,001股以上",
        16: "調整列", 17: "合計",
    }
    # 千張大戶／大戶：持股 1,000,001股以上（= 1,000張以上），對應第15級。
    LARGE_HOLDER_TIER = 15

    # 證交所「上市公司董監事持股餘額明細資料」開放資料，全市場單一CSV，
    # 每月更新。來源：https://data.gov.tw/dataset/22811
    # ⚠️ 只涵蓋上市（_L），上櫃另有不同檔案，尚未驗證，暫不支援。
    TWSE_INSIDER_HOLDINGS_URL = "https://mopsfin.twse.com.tw/opendata/t187ap11_L.csv"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    }

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """
        ⚠️ 統一的安全整數轉換：直接 int(value) 在 value 是 NaN 時會拋出
        ValueError（例如 TDCC/TWSE 原始資料裡剛好某一列的數字欄位是缺值），
        這裡統一攔截，缺值時明確回傳 default（預設0），而不是讓整個查詢
        因為單一筆缺值資料而崩潰。全 ChipEngine 內需要把 pandas 數值轉成
        int 顯示的地方都應該用這個，不要直接寫 int(...)。
        """
        try:
            if pd.isna(value):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    # ==========================================
    # 工具方法
    # ==========================================
    @staticmethod
    def _to_int(value):
        try:
            return int(str(value).replace(",", "").replace(" ", "").replace("+", ""))
        except Exception:
            return 0

    @staticmethod
    def _find_col(columns, keyword):
        return next((c for c in columns if keyword in c), None)

    # ==========================================
    # 三大法人買賣超
    # ==========================================
    @staticmethod
    def _record_fetch_error(key: str, message: str):
        """
        統一記錄「這次外部資料抓取失敗的原因」：同時寫進 _last_fetch_error
        （給 Dashboard 顯示技術細節用）跟本地 log 檔案（見 logging_config.py
        說明）。寫進 log 檔案的用意：之前的除錯模式一直是「使用者點擊按鈕→
        截圖畫面上的錯誤訊息→貼給開發者」，這個迴圈每次都要重現操作才能
        排查問題。有了 log 之後，之後排查可以直接翻 logs/tqai.log，不需要
        每次都靠使用者重新操作重現錯誤。
        """
        ChipEngine._last_fetch_error[key] = message
        logger.warning(f"[{key}] {message}")

    @staticmethod
    def _fetch_institutional_single_day(date_str: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                ChipEngine.TWSE_INSTITUTIONAL_URL,
                params={"date": date_str, "selectType": "ALL", "response": "json"},
                headers=ChipEngine.HEADERS, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            ChipEngine._record_fetch_error("institutional", f"{type(e).__name__}: {e}")
            return pd.DataFrame()

        if data.get("stat") != "OK" or not data.get("data"):
            # stat != OK 對這支 API 而言，很多時候單純代表「這天不是交易日／
            # 尚未公布」，是正常情況，不一定是連線問題，這裡如實記錄官方
            # 回應的 stat/文字說明，方便判斷是哪一種。
            ChipEngine._record_fetch_error("institutional", f"TWSE回應 stat={data.get('stat')}，說明：{data.get('stat')}")
            return pd.DataFrame()

        ChipEngine._last_fetch_error.pop("institutional", None)

        df = pd.DataFrame(data["data"], columns=data.get("fields", []))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def get_institutional_snapshot(stock_code: str, lookback_days: int = 10):
        """往前找最近 lookback_days 天內第一個有資料的交易日，回傳該股票的三大法人買賣超明細。"""
        stock_code = str(stock_code).split(".")[0].strip()

        for delta in range(lookback_days):
            d = datetime.now() - timedelta(days=delta)
            df = ChipEngine._fetch_institutional_single_day(d.strftime("%Y%m%d"))
            if df.empty:
                continue

            code_col = ChipEngine._find_col(df.columns, "證券代號")
            if code_col is None:
                continue

            df[code_col] = df[code_col].astype(str).str.strip()
            match = df[df[code_col] == stock_code]
            if match.empty:
                continue

            row = match.iloc[0]

            def g(keyword):
                col = ChipEngine._find_col(df.columns, keyword)
                return ChipEngine._to_int(row[col]) if col else 0

            # ⚠️ 修正說明：TWSE 已把欄位名稱從「外資及陸資買賣超股數」改成
            # 「外陸資買賣超股數(不含外資自營商)」（拿掉了「及」字），原本
            # 兩個關鍵字都對不上，導致 foreign_net 一直靜默拿到 0，个股籌碼
            # 報告裡的外資買賣超長期以來都是錯的（顯示成0）而不是抓取失敗
            # ——因為 g() 找不到欄位時回傳 0，不會被察覺是資料缺失。這裡
            # 依序嘗試新舊兩種命名，確保不管 TWSE 用哪一種都抓得到。
            foreign_net = g("外陸資買賣超股數") or g("外資及陸資買賣超股數") or g("外資買賣超股數")
            trust_net = g("投信買賣超股數")
            dealer_net = g("自營商買賣超股數(自行買賣)") or g("自營商買賣超股數")
            total_net = g("三大法人買賣超股數") or (foreign_net + trust_net + dealer_net)

            return {
                "date": d.strftime("%Y-%m-%d"),
                "foreign_net": foreign_net,
                "trust_net": trust_net,
                "dealer_net": dealer_net,
                "total_net": total_net,
            }

        return None

    # ==========================================
    # 融資融券
    # ==========================================
    @staticmethod
    def _fetch_margin_single_day(date_str: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                ChipEngine.TWSE_MARGIN_URL,
                params={"date": date_str, "selectType": "ALL", "response": "json"},
                headers=ChipEngine.HEADERS, timeout=10,
            )
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        if data.get("stat") != "OK" or not data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(data["data"], columns=data.get("fields", []))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def get_margin_snapshot(stock_code: str, lookback_days: int = 10):
        stock_code = str(stock_code).split(".")[0].strip()

        for delta in range(lookback_days):
            d = datetime.now() - timedelta(days=delta)
            df = ChipEngine._fetch_margin_single_day(d.strftime("%Y%m%d"))
            if df.empty:
                continue

            code_col = ChipEngine._find_col(df.columns, "代號")
            if code_col is None:
                continue

            df[code_col] = df[code_col].astype(str).str.strip()
            match = df[df[code_col] == stock_code]
            if match.empty:
                continue

            row = match.iloc[0]

            def g(keyword):
                col = ChipEngine._find_col(df.columns, keyword)
                return ChipEngine._to_int(row[col]) if col else 0

            margin_balance = g("融資今日餘額") or g("融資餘額")
            margin_change = g("融資增減")
            short_balance = g("融券今日餘額") or g("融券餘額")
            short_change = g("融券增減")

            return {
                "date": d.strftime("%Y-%m-%d"),
                "margin_balance": margin_balance,
                "margin_change": margin_change,
                "short_balance": short_balance,
                "short_change": short_change,
            }

        return None

    # ==========================================
    # 融資餘額趨勢 (Margin Balance Trend) — v2.9.7 新增
    # ==========================================
    # ⚠️ 動機（專業投資角度覆核既有引擎後補上的一塊）：get_margin_snapshot()
    # 只回傳「最新一天」的融資餘額與單日增減，沒有趨勢——但實務上，判斷
    # 「這波上漲是不是靠散戶融資堆出來的」，看的是一段期間的餘額變化率，
    # 不是單日增減（單日增減雜訊很大）。融資餘額短期內急速堆高，是台股
    # 實務上常見的「散戶追高、籌碼浮動、後續一旦股價拉回容易觸發融資斷頭
    # 連環砍」的風險訊號；融資餘額快速去化則可能是恐慌性斷頭（短期加速
    # 探底，需搭配其他訊號判斷是否為出清訊號）或籌碼沉澱（偏正面）。
    # 這裡只呈現「餘額變化率」與雙向可能解讀，不做單一方向的斷定。
    @staticmethod
    def get_margin_trend(stock_code: str, lookback_days: int = 20) -> dict:
        """
        逐日呼叫 TWSE 融資融券頁面（跟 get_margin_snapshot 同一資料源），
        收集近 lookback_days 個「有資料」的交易日融資餘額，計算變化率。

        ⚠️ 效能與資料源限制：TWSE 這個 API 沒有「單一股票區間查詢」的
        端點，只有「單日全市場」查詢，逐日呼叫是唯一取得序列資料的方式，
        呼叫次數等於 lookback_days（實際會因假日略多，見迴圈上限），
        比 get_margin_snapshot() 慢很多，只建議在個股深度分析頁面主動
        呼叫，不建議放進批次掃描（會讓每檔股票多花 lookback_days 次
        HTTP 請求，容易被視為異常流量或大幅拖慢整體掃描速度）。
        """
        stock_code = str(stock_code).split(".")[0].strip()
        records = []
        max_calendar_lookback = int(lookback_days * 1.6) + 5  # 涵蓋假日/非交易日的緩衝

        for delta in range(max_calendar_lookback):
            if len(records) >= lookback_days:
                break
            d = datetime.now() - timedelta(days=delta)
            df = ChipEngine._fetch_margin_single_day(d.strftime("%Y%m%d"))
            if df.empty:
                continue
            code_col = ChipEngine._find_col(df.columns, "代號")
            if code_col is None:
                continue
            df[code_col] = df[code_col].astype(str).str.strip()
            match = df[df[code_col] == stock_code]
            if match.empty:
                continue
            row = match.iloc[0]
            bal_col = ChipEngine._find_col(df.columns, "融資今日餘額") or ChipEngine._find_col(df.columns, "融資餘額")
            if bal_col is None:
                continue
            balance = ChipEngine._to_int(row[bal_col])
            records.append({"date": d.strftime("%Y-%m-%d"), "margin_balance": balance})

        if len(records) < 5:
            return {"status": "insufficient_data",
                    "message": f"⚠️ 僅取得 {len(records)} 個交易日的融資餘額資料，不足以判斷趨勢（建議至少5個交易日）。"}

        records = list(reversed(records))  # 由舊到新
        oldest_balance = records[0]["margin_balance"]
        latest_balance = records[-1]["margin_balance"]

        if oldest_balance == 0:
            return {"status": "insufficient_data", "message": "⚠️ 期初融資餘額為0，無法計算變化率。"}

        change_pct = (latest_balance - oldest_balance) / oldest_balance * 100

        if change_pct >= 20:
            flag = (f"🟡 近{len(records)}個交易日融資餘額增加 {change_pct:.1f}%，散戶槓桿明顯升溫，"
                    "若股價後續拉回，較容易觸發融資追繳/斷頭的連鎖賣壓，建議留意籌碼穩定度，"
                    "非單純利多或利空，需搭配價格結構一起判斷。")
        elif change_pct <= -20:
            flag = (f"🟡 近{len(records)}個交易日融資餘額減少 {abs(change_pct):.1f}%，可能是籌碼去化沉澱（偏正面），"
                    "也可能是股價下跌引發的恐慌性斷頭去化（偏負面），兩種情境都會讓餘額下降，"
                    "請對照同期間股價走勢判斷是哪一種。")
        else:
            flag = f"ℹ️ 近{len(records)}個交易日融資餘額變化 {change_pct:+.1f}%，尚未達顯著槓桿升溫/去化門檻（±20%）。"

        return {
            "status": "ok",
            "days_used": len(records),
            "oldest_balance": oldest_balance,
            "latest_balance": latest_balance,
            "change_pct": round(change_pct, 1),
            "flag": flag,
        }

    # ==========================================
    # 全市場三大法人買賣超排行榜（v2.8 新增，對應選股學院「排行榜選股法」）
    # ==========================================
    # ⚠️ 合併說明：這個方法原本被誤放在一個沒有 .py 副檔名、未被任何地方
    # import 的孤兒檔案（Chip_engine___PY）裡，導致 app.py 實際 import 的
    # 這份 chip_engine.py 完全沒有這個方法 —— 「🏆 全市場排行榜」頁面因此
    # 100% 會拋出 AttributeError。現在正式合併進來，並保留原檔的孤兒版本
    # 供刪除。
    @staticmethod
    def get_market_wide_institutional_ranking(lookback_days: int = 10, top_n: int = 20,
                                               use_cache: bool = True, max_age_hours: float = 4,
                                               db_path: str = None):
        """
        🏆 全市場三大法人買賣超排行榜 (Market-wide Institutional Ranking)

        對應選股學院文件「排行榜選股法」：外資買賣超／投信買賣超／自營商買賣超
        排行榜，直接追蹤三大法人的資金流向。

        跟 get_institutional_snapshot()（查「單一股票」的三大法人買賣超）不同，
        這裡是對 _fetch_institutional_single_day() 抓回來的「當日全市場」資料
        做排序，取得買超/賣超前 N 名——複用同一支 TWSE API，不會額外增加對外
        請求次數。

        ⚠️ 資料範圍限制（沿用本類別 docstring 的既有限制）：
          1. TWSE T86 這支 API 只涵蓋「上市」股票，上櫃（TPEx）用的是不同的
             資料源，暫不支援，所以這份排行榜不包含上櫃股票。
          2. 只在交易日有資料；若查詢日為假日或盤後資料尚未公布，會自動
             往前找最近的交易日（跟其他 ChipEngine 方法一致）。
          3. 若 TWSE 官方介面改版或服務異常，會回傳 status='unavailable'，
             不影響呼叫端（app.py）繼續運作。

        ⚠️ 新增（快取）：這份「全市場當日排行」資料只跟日期有關，跟呼叫者
        是誰、要看第幾名都無關，很適合共用快取——原本每次點擊「抓取最新
        排行榜」都會重新對 TWSE 發一次請求，即使同一個交易日內反覆點擊也是
        如此。現在把「原始全市場資料 + 資料日期」快取進 DatabaseEngine 的
        kv_cache（預設新鮮期限 4 小時，盤中/盤後資料本來就不會頻繁更新，
        不需要比這更短的快取窗口），Top-N 排序與切片則每次都重新計算
        （便宜、且允許同一份快取資料配合不同 top_n 使用，不需要因為 top_n
        不同就重抓一次）。use_cache=False 可強制略過快取重新抓取。

        ⚠️ 新增（標的標籤）：加上「標的」欄位（格式 "[代碼] 名稱"），跟
        ScannerEngine/NameEngine 其他頁面的顯示格式一致。這裡直接用 TWSE
        當天回傳的官方名稱組字串，而不是查 NameEngine.NAME_MAP（那份只
        涵蓋本專案內建觀察名單的幾十檔，這份排行榜涵蓋全部上市股票，用
        NAME_MAP 會讓大部分股票顯示成「未知名稱」，反而更不準確）。

        回傳格式：
            {
                "status": "ok",
                "date": "2026-07-08",
                "total_stocks": 950,
                "rankings": {
                    "外資": {"買超前N名": df, "賣超前N名": df},
                    "投信": {"買超前N名": df, "賣超前N名": df},
                    "自營商": {"買超前N名": df, "賣超前N名": df},
                    "三大法人合計": {"買超前N名": df, "賣超前N名": df},
                }
            }
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "chip_market_wide_institutional_raw"
        work = None
        date_str = None

        if use_cache:
            try:
                cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours, db_path=db_path)
                if cached:
                    payload = cached["payload"]
                    work = pd.DataFrame(payload["records"])
                    date_str = payload["date"]
            except Exception:
                work = None

        if work is None:
            for delta in range(lookback_days):
                d = datetime.now() - timedelta(days=delta)
                df = ChipEngine._fetch_institutional_single_day(d.strftime("%Y%m%d"))
                if df.empty:
                    continue

                code_col = ChipEngine._find_col(df.columns, "證券代號")
                name_col = ChipEngine._find_col(df.columns, "證券名稱")
                if code_col is None:
                    ChipEngine._record_fetch_error(
                        "ranking",
                        f"{d.strftime('%Y-%m-%d')}: 找不到「證券代號」欄位，實際欄位：{list(df.columns)}"
                    )
                    continue

                # ⚠️ 同樣的欄位改名問題（見 get_institutional_snapshot 的說明）：
                # TWSE 已把「外資及陸資買賣超股數」改名為「外陸資買賣超股數
                # (不含外資自營商)」，依序嘗試新舊兩種命名。
                foreign_col = ChipEngine._find_col(df.columns, "外陸資買賣超股數") \
                    or ChipEngine._find_col(df.columns, "外資及陸資買賣超股數") \
                    or ChipEngine._find_col(df.columns, "外資買賣超股數")
                trust_col = ChipEngine._find_col(df.columns, "投信買賣超股數")
                dealer_col = ChipEngine._find_col(df.columns, "自營商買賣超股數(自行買賣)") \
                    or ChipEngine._find_col(df.columns, "自營商買賣超股數")
                total_col = ChipEngine._find_col(df.columns, "三大法人買賣超股數")

                if not all([foreign_col, trust_col, dealer_col]):
                    # 當天頁面欄位對不上（例如TWSE偶爾微調格式），換下一個交易日再試，
                    # 不要用錯誤對應的欄位硬算，避免排行榜資料失真。記下實際欄位名稱，
                    # 方便比對是哪個欄位對不上（不是籠統的「格式異動」）。
                    ChipEngine._record_fetch_error(
                        "ranking",
                        f"{d.strftime('%Y-%m-%d')}: 找不到法人買賣超欄位 "
                        f"(外資={foreign_col}, 投信={trust_col}, 自營商={dealer_col})，"
                        f"實際欄位：{list(df.columns)}"
                    )
                    continue

                candidate = pd.DataFrame({
                    "代碼": df[code_col].astype(str).str.strip(),
                    "名稱": df[name_col].astype(str).str.strip() if name_col else "",
                    "外資買賣超": df[foreign_col].apply(ChipEngine._to_int),
                    "投信買賣超": df[trust_col].apply(ChipEngine._to_int),
                    "自營商買賣超": df[dealer_col].apply(ChipEngine._to_int),
                })
                candidate["三大法人合計買賣超"] = (
                    df[total_col].apply(ChipEngine._to_int) if total_col
                    else candidate["外資買賣超"] + candidate["投信買賣超"] + candidate["自營商買賣超"]
                )

                # 過濾掉代碼格式異常的雜項列（權證、受益證券等非普通股/ETF代碼）
                candidate = candidate[candidate["代碼"].str.match(r"^[0-9]{4,6}$")].reset_index(drop=True)
                if candidate.empty:
                    ChipEngine._record_fetch_error("ranking", f"{d.strftime('%Y-%m-%d')}: 過濾代碼格式後沒有剩下任何資料列")
                    continue

                ChipEngine._last_fetch_error.pop("ranking", None)
                work = candidate
                date_str = d.strftime("%Y-%m-%d")
                break

            if work is None:
                # 優先顯示這個方法自己記錄的「欄位比對」診斷（比較精確指出卡在
                # 哪個交易日、哪個欄位對不上），沒有的話才退回顯示
                # _fetch_institutional_single_day 自己的連線層級診斷。
                detail = ChipEngine._last_fetch_error.get("ranking") \
                    or ChipEngine._last_fetch_error.get("institutional", "未知原因")
                return {
                    "status": "unavailable",
                    "message": (
                        "⚠️ 近期無法取得全市場三大法人買賣超排行榜（可能是連續假日、TWSE服務異常，或近期非交易日）。"
                        f"技術細節（最近一次嘗試）：{detail}"
                    ),
                }

            if use_cache:
                try:
                    DatabaseEngine.set_cache(
                        cache_key,
                        {"date": date_str, "records": work.to_dict(orient="records")},
                        db_path=db_path,
                    )
                except Exception:
                    pass

        work = work.copy()
        work["標的"] = "[" + work["代碼"].astype(str) + "] " + work["名稱"].astype(str)

        rankings = {}
        for label, col in [
            ("外資", "外資買賣超"), ("投信", "投信買賣超"),
            ("自營商", "自營商買賣超"), ("三大法人合計", "三大法人合計買賣超"),
        ]:
            buy_top = work.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
            sell_top = work.sort_values(col, ascending=True).head(top_n).reset_index(drop=True)
            rankings[label] = {"買超前N名": buy_top, "賣超前N名": sell_top}

        return {
            "status": "ok",
            "date": date_str,
            "total_stocks": len(work),
            "rankings": rankings,
        }

    # ==========================================
    # 大戶持股／千張大戶分析（v2.9 新增，資料源：TDCC 集保戶股權分散表）
    # ==========================================
    # 記錄最近一次 TDCC/TWSE 開放資料抓取失敗的技術細節（HTTP狀態碼／例外
    # 訊息／欄位不符等），讓 Dashboard 可以把「為什麼失敗」講清楚，而不是
    # 只顯示一句通用的「暫時無法取得資料」。單一進程內共用，Streamlit
    # 屬單使用者互動模式，這裡的簡化夠用，不需要額外的執行緒隔離機制。
    _last_fetch_error = {}

    @staticmethod
    def _fetch_shareholding_distribution_full(use_cache: bool = True, max_age_hours: float = 20) -> pd.DataFrame:
        """
        抓取 TDCC 集保戶股權分散表「全市場單一CSV」快照（每週更新，涵蓋
        上市/上櫃/興櫃全部證券）。這份CSV涵蓋全市場所有證券、每檔證券
        17個級距列，逐股票查詢的話沒辦法只抓單一股票，每次都要下載全部——
        因此這裡快取「整份原始資料」而不是「單一股票的結果」，同一週期
        (max_age_hours預設20小時，略短於一天，避免跨日快取到舊的一週資料)
        內重複查詢不同股票都只需要下載一次。
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "tdcc_shareholding_distribution_raw"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    return pd.DataFrame(cached["payload"]["records"])
                except Exception:
                    pass

        try:
            resp = requests.get(ChipEngine.TDCC_SHAREHOLDING_URL, headers=ChipEngine.HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8-sig"
            df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        except Exception as e:
            ChipEngine._record_fetch_error("shareholding", f"{type(e).__name__}: {e}")
            return pd.DataFrame()

        if df.empty:
            ChipEngine._record_fetch_error("shareholding", "TDCC回應內容為空（連線成功但沒有資料，可能是TDCC服務端暫時異常）")
            return df

        df.columns = [str(c).strip() for c in df.columns]
        expected_cols = {"資料日期", "證券代號", "持股分級", "人數", "股數", "占集保庫存數比例%"}
        if not expected_cols.issubset(set(df.columns)):
            # TDCC 改版欄位名稱或格式跟預期不符，如實回傳空表，不要用錯位的
            # 欄位硬算，避免產生看似正常、實則張冠李戴的數字。
            missing = expected_cols - set(df.columns)
            ChipEngine._record_fetch_error(
                "shareholding",
                f"欄位格式與預期不符，缺少欄位：{missing}，實際欄位：{list(df.columns)[:10]}"
                f"（TDCC可能已改版頁面格式，需要更新解析邏輯）"
            )
            return pd.DataFrame()

        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        for col in ["持股分級", "人數", "股數"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["占集保庫存數比例%"] = pd.to_numeric(df["占集保庫存數比例%"], errors="coerce")
        df = df.dropna(subset=["持股分級"])
        df["持股分級"] = df["持股分級"].astype(int)

        ChipEngine._last_fetch_error.pop("shareholding", None)

        if use_cache:
            try:
                DatabaseEngine.set_cache(
                    cache_key, {"records": df.to_dict(orient="records")}, db_path=None
                )
            except Exception:
                pass

        return df

    @staticmethod
    def get_shareholding_distribution(stock_code: str, use_cache: bool = True) -> dict:
        """
        🏦 大戶持股／千張大戶分析 (Shareholding Concentration Analysis)

        資料源：TDCC（臺灣集中保管結算所）集保戶股權分散表開放資料
        （https://opendata.tdcc.com.tw/getOD.ashx?id=1-5），涵蓋上市/上櫃/
        興櫃全市場證券，每週更新。

        ⚠️ 誠實揭露：
          1. 「大戶」定義採業界慣例：持股 1,000,001股以上（= 1,000張以上，
             對應第15級），這是市場慣用定義，不是嚴謹的法定分類。
          2. 這裡只反映「集保帳戶」的持股分布，不是「實質受益人」——同一
             個人可能透過信託、代操等方式分散在不同集保帳戶，實際集中度
             可能比表面數字更高或更低，此為 TDCC 資料本身的固有限制，
             不是本引擎的實作問題。
          3. 每週更新一次，不是即時資料；當週資料通常會落後1~2個交易日
             （TDCC官方說明的結算時間差），不適合用來判斷「今天」的籌碼
             變化，是中長期籌碼集中度的參考指標。
          4. 歷史趨勢（見 get_shareholding_trend()）是本系統自己累積的
             快照記錄，不是 TDCC 官方提供的歷史資料——TDCC 這份開放資料
             CSV 本身只給「最新一週」，沒有歷史區間查詢，所以只能從
             「系統開始使用之後」才逐週累積出趨勢，剛啟用時只會有一筆。
        """
        stock_code_clean = str(stock_code).split(".")[0].strip()
        full_df = ChipEngine._fetch_shareholding_distribution_full(use_cache=use_cache)
        if full_df.empty:
            detail = ChipEngine._last_fetch_error.get("shareholding", "未知原因")
            return {
                "status": "unavailable",
                "message": f"⚠️ 暫時無法取得集保戶股權分散表資料。技術細節：{detail}",
            }

        match = full_df[full_df["證券代號"] == stock_code_clean].copy()
        if match.empty:
            return {"status": "unavailable", "message": f"⚠️ 查無代碼 {stock_code_clean} 的股權分散表資料，請確認代碼是否正確。"}

        match = match.sort_values("持股分級")
        total_row = match[match["持股分級"] == 17]
        detail_tiers = match[match["持股分級"] <= 15].copy()
        detail_tiers["級距"] = detail_tiers["持股分級"].map(ChipEngine.SHAREHOLDING_TIER_LABELS)

        large_holder_row = match[match["持股分級"] == ChipEngine.LARGE_HOLDER_TIER]
        large_holder_pct = float(large_holder_row["占集保庫存數比例%"].iloc[0]) if not large_holder_row.empty else 0.0
        large_holder_count = ChipEngine._safe_int(large_holder_row["人數"].iloc[0]) if not large_holder_row.empty else 0

        total_holders = ChipEngine._safe_int(total_row["人數"].iloc[0]) if not total_row.empty else ChipEngine._safe_int(detail_tiers["人數"].sum())
        total_shares = ChipEngine._safe_int(total_row["股數"].iloc[0]) if not total_row.empty else ChipEngine._safe_int(detail_tiers["股數"].sum())

        raw_date = str(match["資料日期"].iloc[0])
        date_fmt = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else raw_date

        flags = []
        if large_holder_pct >= 60:
            flags.append(f"🔴 千張大戶(≥1,000張)持股佔比高達 {large_holder_pct:.1f}%，籌碼高度集中在大戶手上")
        elif large_holder_pct >= 40:
            flags.append(f"🟡 千張大戶持股佔比 {large_holder_pct:.1f}%，籌碼集中度中等偏高")
        else:
            flags.append(f"ℹ️ 千張大戶持股佔比 {large_holder_pct:.1f}%，籌碼相對分散")
        flags.append("⚠️ 每週更新一次、反映集保帳戶而非實質受益人，僅供中長期籌碼集中度參考，不適合判斷單日籌碼變化。")

        # 累積這一週的摘要進本地歷史記錄（見上方 class docstring 的趨勢限制說明）
        try:
            from engines.db_engine import DatabaseEngine
            DatabaseEngine.save_shareholding_snapshot(
                date_fmt, stock_code_clean, large_holder_pct, total_holders, total_shares
            )
        except Exception:
            pass

        return {
            "status": "ok",
            "date": date_fmt,
            "total_holders": total_holders,
            "total_shares": total_shares,
            "tiers": detail_tiers[["級距", "人數", "股數", "占集保庫存數比例%"]].reset_index(drop=True),
            "large_holder_pct": round(large_holder_pct, 2),
            "large_holder_count": large_holder_count,
            "flags": flags,
        }

    @staticmethod
    def get_shareholding_trend(stock_code: str, weeks: int = 12) -> pd.DataFrame:
        """
        回傳本系統自己累積的「大戶持股佔比」歷史趨勢（見
        get_shareholding_distribution() docstring 對趨勢資料來源限制的說明）。
        剛開始使用本系統時歷史筆數會很少，屬正常現象，會隨著每週重複
        查詢逐漸累積。
        """
        from engines.db_engine import DatabaseEngine
        stock_code_clean = str(stock_code).split(".")[0].strip()
        try:
            return DatabaseEngine.load_shareholding_history(stock_code_clean, weeks=weeks)
        except Exception:
            return pd.DataFrame()

    # ==========================================
    # 董監事／大股東持股與設質分析（v2.9 新增，資料源：證交所董監持股開放資料）
    # ==========================================
    @staticmethod
    def _fetch_insider_holdings_full(use_cache: bool = True, max_age_hours: float = 20) -> pd.DataFrame:
        """
        抓取證交所「上市公司董監事持股餘額明細資料」全市場單一CSV快照
        （每月更新）。同一份資料涵蓋全部上市公司的每一位董監事/經理人/
        大股東，逐股票查詢沒辦法只抓單一公司，因此整份快取，同一週期內
        重複查詢不同股票只需下載一次（比照 get_shareholding_distribution
        的快取設計）。
        """
        from engines.db_engine import DatabaseEngine

        cache_key = "twse_insider_holdings_raw"
        if use_cache:
            cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
            if cached is not None:
                try:
                    return pd.DataFrame(cached["payload"]["records"])
                except Exception:
                    pass

        try:
            resp = requests.get(ChipEngine.TWSE_INSIDER_HOLDINGS_URL, headers=ChipEngine.HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8-sig"
            df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        except Exception as e:
            ChipEngine._record_fetch_error("insider", f"{type(e).__name__}: {e}")
            return pd.DataFrame()

        if df.empty:
            ChipEngine._record_fetch_error("insider", "TWSE回應內容為空（連線成功但沒有資料，可能是TWSE服務端暫時異常）")
            return df

        df.columns = [str(c).strip() for c in df.columns]
        expected_cols = {
            "出表日期", "資料年月", "公司代號", "公司名稱", "職稱", "姓名",
            "選任時持股", "目前持股", "設質股數", "設質股數佔持股比例",
            "內部人關係人目前持股合計", "內部人關係人設質股數", "內部人關係人設質比例",
        }
        if not expected_cols.issubset(set(df.columns)):
            # TWSE改版欄位跟預期不符，如實回傳空表，不用錯位欄位硬算。
            missing = expected_cols - set(df.columns)
            ChipEngine._record_fetch_error(
                "insider",
                f"欄位格式與預期不符，缺少欄位：{missing}，實際欄位：{list(df.columns)[:10]}"
            )
            return pd.DataFrame()

        df["公司代號"] = df["公司代號"].astype(str).str.strip()
        for col in ["選任時持股", "目前持股", "設質股數",
                    "內部人關係人目前持股合計", "內部人關係人設質股數"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for pct_col in ["設質股數佔持股比例", "內部人關係人設質比例"]:
            df[pct_col] = pd.to_numeric(
                df[pct_col].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )

        ChipEngine._last_fetch_error.pop("insider", None)

        if use_cache:
            try:
                DatabaseEngine.set_cache(cache_key, {"records": df.to_dict(orient="records")}, db_path=None)
            except Exception:
                pass

        return df

    @staticmethod
    def get_insider_holdings(stock_code: str, use_cache: bool = True) -> dict:
        """
        🏛️ 董監事／大股東持股與設質分析 (Insider Holdings & Pledge Analysis)

        資料源：證交所「上市公司董監事持股餘額明細資料」開放資料
        (https://mopsfin.twse.com.tw/opendata/t187ap11_L.csv)，已實際抓取
        驗證格式屬實，每月更新，涵蓋全部上市公司的董事/監察人/經理人/大股東。

        ⚠️ 誠實揭露：
          1. 只支援上市股票，上櫃另有不同的資料檔案，尚未驗證格式，暫不
             支援（沿用本類別既有的上市/上櫃資料源不對稱限制）。
          2. 每月更新一次，反映「申報當下」的持股，不是即時資料。
          3. 法人董事在原始資料裡會同時列出「法人本身」與「其法人代表人」
             兩筆，個別加總持股數容易重複計算——這裡改用官方已經算好的
             「內部人關係人目前持股合計」欄位取最大值代表整體內部人持股
             規模，避免自行加總時的重複計算問題。
          4. 「設質股數佔持股比例」是股票質押借款的比例，過高代表資金壓力
             較大，是常見的公司治理風險指標之一，但不代表一定有問題，
             需要搭配財報、產業狀況等其他資訊綜合判斷，不構成投資建議。
        """
        stock_code_clean = str(stock_code).split(".")[0].strip()
        full_df = ChipEngine._fetch_insider_holdings_full(use_cache=use_cache)
        if full_df.empty:
            detail = ChipEngine._last_fetch_error.get("insider", "未知原因")
            return {
                "status": "unavailable",
                "message": f"⚠️ 暫時無法取得董監事持股資料（僅支援上市股票）。技術細節：{detail}",
            }

        match = full_df[full_df["公司代號"] == stock_code_clean].copy()
        if match.empty:
            return {"status": "unavailable", "message": f"⚠️ 查無代碼 {stock_code_clean} 的董監事持股資料（可能是上櫃股票，目前僅支援上市）。"}

        data_month = str(match["資料年月"].iloc[0])
        company_name = str(match["公司名稱"].iloc[0])

        # ⚠️ 修正：原本 int(match[...].max()) 如果該公司這個欄位全部是缺值，
        # .max() 會回傳 NaN，int(NaN) 會直接拋出 ValueError 讓整個查詢崩潰
        # （不像下面的設質比例只是誤導顯示，這裡是真的會讓函式壞掉）。
        # 統一改用 _safe_int()，缺值時明確給 0，不是假裝有資料。
        total_insider_holding = ChipEngine._safe_int(match["內部人關係人目前持股合計"].max())
        individual_max_pledge_pct = float(match["設質股數佔持股比例"].max())
        relationship_max_pledge_pct = float(match["內部人關係人設質比例"].max())
        # ⚠️ 修正：headline 的「設質風險」判斷取「個人」與「關係人彙總」兩個
        # 欄位的較高者，避免出現「headline 顯示健康，但明細表同時列出某位
        # 個人質押比例逼近100%」這種互相矛盾的畫面——這兩個欄位本來就是
        # 衡量不同範圍（個人 vs 含配偶/法人代表關係人的彙總），只看其中一個
        # 容易漏掉風險。
        candidates = [v for v in (individual_max_pledge_pct, relationship_max_pledge_pct) if pd.notna(v)]
        max_pledge_pct = max(candidates) if candidates else None

        high_pledge = match[match["設質股數佔持股比例"] >= 50][
            ["職稱", "姓名", "目前持股", "設質股數", "設質股數佔持股比例"]
        ].sort_values("設質股數佔持股比例", ascending=False).reset_index(drop=True)

        flags = []
        if max_pledge_pct is None:
            flags.append("ℹ️ 這批資料的設質比例欄位皆為缺值，無法判斷設質風險（不代表沒有風險，只是資料缺失）。")
        elif max_pledge_pct >= 50:
            flags.append(f"🔴 內部人關係人設質比例高達 {max_pledge_pct:.1f}%，資金壓力風險需留意")
        elif max_pledge_pct >= 20:
            flags.append(f"🟡 內部人關係人設質比例 {max_pledge_pct:.1f}%，中等偏高")
        else:
            flags.append(f"✅ 內部人關係人設質比例 {max_pledge_pct:.1f}%，尚屬健康水位")

        if not high_pledge.empty:
            flags.append(f"⚠️ 有 {len(high_pledge)} 位個別內部人設質比例超過50%，詳見明細表")

        flags.append("⚠️ 每月更新，反映申報當下持股，不是即時資料；設質比例高不代表一定有問題，需搭配其他資訊綜合判斷，不構成投資建議。")

        return {
            "status": "ok",
            "data_month": data_month,
            "company_name": company_name,
            "detail": match[[
                "職稱", "姓名", "選任時持股", "目前持股", "設質股數",
                "設質股數佔持股比例", "內部人關係人目前持股合計", "內部人關係人設質比例",
            ]].reset_index(drop=True),
            "total_insider_holding": total_insider_holding,
            "max_pledge_pct": None if max_pledge_pct is None else round(max_pledge_pct, 2),
            "high_pledge_table": high_pledge,
            "flags": flags,
        }

    # ==========================================
    # 整合報告（供 Dashboard 顯示）
    # ==========================================
    @staticmethod
    def build_chip_report(stock_code: str):
        stock_code_clean = str(stock_code).split(".")[0].strip()

        try:
            institutional = ChipEngine.get_institutional_snapshot(stock_code_clean)
        except Exception:
            institutional = None

        try:
            margin = ChipEngine.get_margin_snapshot(stock_code_clean)
        except Exception:
            margin = None

        if institutional is None and margin is None:
            return {
                "status": "unavailable",
                "message": "⚠️ 暫時無法取得籌碼資料（可能為上櫃股票、TWSE 服務異常，或近期非交易日）。",
            }

        flags = []
        if institutional:
            f_net, t_net = institutional["foreign_net"], institutional["trust_net"]
            if f_net > 0 and t_net > 0:
                flags.append("🟢 外資與投信同步買超，籌碼面偏多")
            elif f_net < 0 and t_net < 0:
                flags.append("🔴 外資與投信同步賣超，籌碼面偏空")
            elif f_net * t_net < 0:
                flags.append("🟡 外資與投信買賣方向分歧，籌碼面不明確")

        if margin:
            if margin["margin_change"] < 0 and margin["short_change"] > 0:
                flags.append("⚠️ 融資減少、融券增加，散戶轉趨保守／看空")
            elif margin["margin_change"] > 0 and margin["short_change"] < 0:
                flags.append("ℹ️ 融資增加、融券減少，散戶轉趨樂觀")

        if not flags:
            flags.append("ℹ️ 籌碼動向中性，無明顯一致訊號")

        return {
            "status": "ok",
            "institutional": institutional,
            "margin": margin,
            "flags": flags,
        }