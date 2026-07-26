import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

from engines.logging_config import get_logger
from engines.db_engine import DatabaseEngine

logger = get_logger(__name__)


class NewsEngine:
    """
    📰 新聞情緒中心 (News & Sentiment Center) — Phase 1（含專業投資觀點強化）

    ⚠️ 範圍與誠實揭露（請先讀這段再串接到 app.py）：

    1. 新聞來源：改用 Google 新聞 RSS（關鍵字搜尋），不是原提案講的
       「Yahoo股市個股新聞RSS」。Yahoo奇摩股市目前沒有穩定、可用代碼
       查詢的公開RSS端點（舊端點多半已失效／改版），Google新聞RSS
       用關鍵字搜尋（股票代碼 + 公司名稱）覆蓋率與穩定性都更好，完全
       免費、免金鑰，且用 stdlib xml.etree 解析，不需要額外安裝
       feedparser。缺點：搜尋式來源，非「純個股官方新聞源」，偶爾會
       混入不相關結果（例如代碼剛好是其他語境的數字），這裡沒有做
       額外的相關性過濾，使用時請自行留意。

    2. 情緒分析：規則式關鍵字比對是「一定會執行」的 baseline，免成本、
       零延遲、可稽核（回傳命中了哪些關鍵字），但本質上不是語意理解，
       容易誤判反諷、條件句、「調升目標價但已反映」這類情境，不建議
       單獨當作交易依據。LLM 分類是「選用加強」：只有呼叫端明確傳入
       llm_classify_fn 時才會啟用，失敗或沒傳入就自動 fallback 回規則式，
       且結果一律誠實標示 method 是 'rule_based' 還是 'llm'，不假裝
       兩者準確度相同。這支 engine 刻意不直接寫死呼叫任何LLM
       API——金鑰管理與呼叫額度控制屬於 app 層責任，engine 保持純邏輯、
       方便測試與替換。

    3. 不做「技術面+籌碼面+基本面+新聞面」四合一總分（例如「93分」）。
       不同性質的分數用固定權重相加會製造虛假的精確感，這裡新聞情緒
       只回傳獨立的 sentiment_label／confidence，呼叫端(app.py)應該
       並列呈現、不要自動加總成單一分數。

    4. 不做即時推播通知。Streamlit 是請求-回應式架構，沒有常駐
       process，做不到真正的push；如果之後真的需要，屬於 Phase 3，
       要另外寫獨立排程腳本 + Telegram/LINE Bot，不在這支engine範圍內。

    ── 專業投資觀點強化（這次新增，以「達人視角」補的判斷邏輯）──

    5. 關鍵字分兩層權重，不是齊頭式平等：
       - Tier1（硬資訊，權重2）：法人買賣超金額、目標價調整、營收/獲利
         數字創高創低，這些是「有具體數字或機構動作背書」的新聞。
       - Tier2（軟敘述，權重1）：看好/看壞/強勢/受惠這類形容詞式報導，
         記者主觀敘述成分高，訊號強度較弱。
       這樣「單純一篇形容詞式報導」不會跟「外資實際買超金額」的訊號
       強度被當成一樣重，比較貼近專業判讀新聞時的直覺。

    6. 加入時間衰減（recency decay）：越新的新聞對 overall_bias 的
       影響力越大，24小時前的舊新聞権重會明顯降低——新聞是會過期的
       資訊，專業判讀不會把三天前的利多和剛剛發布的利空同等看待。
       如果 RSS 沒給可解析的發布時間，該篇新聞退回權重1.0（不放大也
       不縮小），不會讓解析失敗直接讓那篇新聞消失。

    7. 加入新聞量異動偵測（news volume change）：跟上一次抓到的新聞
       篇數比較（不論上次快取是否已過期），篇數明顯增加可能代表市場
       正在消化新事件、值得多留意，但這只是「量」的訊號，不代表方向，
       不能直接當成利多或利空。

    8. get_investor_notes()：固定提供一組「達人視角」的使用提醒，
       app.py 應該把這些提醒顯示在新聞區塊旁邊，而不是只丟數字給
       使用者。核心立場：新聞情緒是落後或同步指標，很多時候消息出來
       時股價已經反映一部分，新聞面只適合當作「確認」或「留意」的
       輔助工具，不是獨立的進出場訊號，更不能取代技術面/籌碼面/
       基本面的交叉驗證。

    ── Phase 2 新增（這次追加，方法說明見各自 docstring）──

    9. get_related_concept_stocks()：從新聞文字比對「概念股對照表」
       （CONCEPT_STOCK_MAP），列出可能受影響的個股與受惠程度星等。
       ⚠️ 這是手動整理的小型對照表，範例性質、非全市場覆蓋，需要
       定期維護更新，不是自動化的產業關聯分析，僅供輔助參考。

    10. AI 摘要（每檔個股）：get_news_with_sentiment() 現在會多回傳
        'ai_summary'（重點條列 + 星等）。預設用規則式（列出命中最多次
        的事件標籤/關鍵字），可選傳入 llm_summarize_fn 換成真正的AI
        摘要，簽章見 get_news_with_sentiment() docstring。

    11. compute_resonance()：新聞面 × 技術面（可選加籌碼/成交量）的
        共振檢查。只是「訊號一致性」的簡單規則判斷，不是量化模型，
        更不是進出場訊號本身。

    12. build_daily_market_summary()：彙總「自選股清單」的新聞，做成
        每日市場摘要。⚠️ 範圍限定在你自選股清單裡的個股新聞，不是
        大盤指數新聞或美股/Fed這類總經新聞——後者需要另外接國際財經
        新聞來源，屬於下面第13點講的 Phase 3 範圍，這裡沒有做。

    13. get_watchlist_news_overview()：直接讀你資料庫裡的
        watchlist_status 表（既有的自選股/持股狀態表，沒有另外開新表），
        逐檔查詢新聞篇數與情緒，做成「自選股新聞中心」總覽。

    ⚠️ 目前新聞來源仍只有 Google新聞RSS 這一種，只有標題與摘要，涵蓋
    不到公開資訊觀測站重大訊息、法說會官方公告、營收公告，也沒有接
    美股/Fed等國際財經新聞源。這些屬於「多來源整合」，資料格式、更新
    頻率、可靠性都跟RSS完全不同，需要各自獨立開發與測試，這次沒有做，
    是刻意的範圍界定，不是遺漏——避免為了衝功能數量而交出品質沒把關
    過的整合。

    快取：沿用 DatabaseEngine.set_cache/get_cache（kv_cache表），
    key 格式為 "news:{code}"。新聞時效性高，預設 max_age_hours=1，
    比股價資料常用的6小時短很多。
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }

    GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

    DEFAULT_MAX_AGE_HOURS = 1
    DEFAULT_MAX_ITEMS = 15

    # -------------------------------------------------
    # 新聞來源可信度對照表（Phase 2c 新增）
    # ⚠️ 誠實揭露：這是主觀評估的可信度權重，不是根據任何正式的媒體
    # 評鑑機構打分，用「比對來源名稱裡是否包含關鍵字」的方式匹配，
    # 找不到比對的來源一律給預設權重（既不特別加分也不特別扣分）。
    # 用途：讓 weighted_bias_score 的計算不會把小型部落格/論壇轉載跟
    # 通訊社/主要財經媒體的原始報導視為同等可信，但這只是「權重」，
    # 不是「只信任特定媒體」——低權重來源的新聞依然會顯示、依然算分，
    # 只是影響力打折。
    # -------------------------------------------------
    SOURCE_WEIGHT = {
        "路透": 1.00, "reuters": 1.00, "彭博": 0.98, "bloomberg": 0.98,
        "中央社": 0.95, "中央通訊社": 0.95,
        "工商時報": 0.90, "經濟日報": 0.90, "moneydj": 0.88,
        "自由財經": 0.80, "自由時報": 0.75, "中時": 0.75, "聯合新聞網": 0.80,
        "鉅亨網": 0.80, "yahoo": 0.75, "鏡週刊": 0.70, "三立": 0.65, "民視": 0.65,
        "商業周刊": 0.85, "天下雜誌": 0.85, "今周刊": 0.82, "財訊": 0.82,
    }
    DEFAULT_SOURCE_WEIGHT = 0.6  # 找不到比對來源時的中性預設值

    # 事件去重（Cluster）判定用的標題相似度門檻（Phase 2c 新增）
    # ⚠️ 用 difflib 的字串相似度比對標題，不是語意理解，同一事件但標題
    # 寫法差異很大的新聞可能無法被歸為同一群，這是簡化heuristic的已知
    # 限制，不是嚴謹的NLP去重。
    DEDUP_TITLE_SIMILARITY_THRESHOLD = 0.55

    # -------------------------------------------------
    # 規則式情緒關鍵字：分兩層權重（Tier1硬資訊=2分／Tier2軟敘述=1分）
    # 也是 LLM 失敗時的 fallback baseline。
    # -------------------------------------------------
    BULLISH_TIER1 = [
        "外資買超", "投信買超", "法人買超", "調高目標價", "上修財測",
        "營收創新高", "獲利優於預期", "營收年增", "調升評等",
    ]
    BULLISH_TIER2 = [
        "買超", "調升", "上修", "創新高", "追加訂單", "大單", "利多",
        "獲利成長", "營收創", "轉單", "擴產", "供不應求", "訂單能見度",
        "強勢", "看好", "受惠",
    ]
    BEARISH_TIER1 = [
        "外資賣超", "投信賣超", "法人賣超", "調降目標價", "下修財測",
        "營收年減", "財報不如預期", "調降評等",
    ]
    BEARISH_TIER2 = [
        "賣超", "調降", "下修", "利空", "衰退", "裁員", "罰款", "訴訟",
        "違約", "跌停", "停產", "砍單", "去化", "庫存過高", "看壞",
        "警訊", "違規",
    ]
    # 為了不重複判斷同一個詞（例如「外資買超」同時是Tier1完整詞、又包含
    # 在Tier2的「買超」裡），比對時Tier1優先命中後，同一個詞不會再重複
    # 計進Tier2，避免雙重計分膨脹權重。
    BULLISH_KEYWORDS = BULLISH_TIER1 + BULLISH_TIER2  # 供舊版呼叫端相容用
    BEARISH_KEYWORDS = BEARISH_TIER1 + BEARISH_TIER2  # 供舊版呼叫端相容用

    # 事件分類關鍵字（比照使用者提案的分類清單，一則新聞可能命中多個標籤）
    EVENT_TAGS = {
        "法人買超": ["外資買超", "投信買超", "法人買超"],
        "法人賣超": ["外資賣超", "投信賣超", "法人賣超"],
        "法說會": ["法說會", "法人說明會"],
        "營收公布": ["營收", "月營收"],
        "除權息": ["除權", "除息", "配息", "配股"],
        "庫藏股": ["庫藏股"],
        "新產品": ["新產品", "新品發表", "量產"],
        "AI": ["人工智慧", "AI伺服器", "AI晶片", "生成式AI"],
        "CPO": ["CPO", "共同封裝光學"],
        "矽光子": ["矽光子"],
        "軍工": ["軍工", "國防"],
        "PCB": ["PCB", "印刷電路板"],
        "CoWoS": ["CoWoS"],
        "關稅": ["關稅", "貿易戰"],
        "車用": ["車用", "電動車", "自動駕駛"],
        "機器人": ["機器人", "人形機器人"],
        "聯準會": ["聯準會", "Fed", "FOMC", "升息", "降息"],
    }

    # -------------------------------------------------
    # 概念股對照表（Phase 2 新增）— 手動整理範例，非全市場自動化關聯
    # ⚠️ 誠實揭露：這是示範性質的小型對照表，涵蓋常見熱門主題，不是
    # 完整產業關聯分析，個股清單需要你自行核實與定期維護，不構成
    # 投資建議。stars(1-5) 是主觀評估的「相關程度／受惠程度」，不是
    # 量化計算結果。
    # -------------------------------------------------
    CONCEPT_STOCK_MAP = {
        "NVIDIA": [("2330", "台積電", 5), ("3711", "日月光投控", 4),
                   ("6669", "緯穎", 4), ("3017", "奇鋐", 3), ("2454", "聯發科", 3)],
        "輝達": [("2330", "台積電", 5), ("3711", "日月光投控", 4),
                 ("6669", "緯穎", 4), ("3017", "奇鋐", 3)],
        "CoWoS": [("2330", "台積電", 5), ("3711", "日月光投控", 4), ("6239", "力成", 3)],
        "矽光子": [("3363", "上詮", 4), ("6237", "驊訊", 3)],
        "CPO": [("3363", "上詮", 4), ("2345", "智邦", 4), ("3596", "智易", 3)],
        "AI伺服器": [("2317", "鴻海", 4), ("6669", "緯穎", 5), ("2382", "廣達", 5)],
        "電動車": [("2308", "台達電", 4), ("1590", "亞德客-KY", 3)],
        "機器人": [("2308", "台達電", 3), ("1590", "亞德客-KY", 4)],
        "軍工": [("2634", "漢翔", 4), ("3540", "曜越", 2)],
    }

    @staticmethod
    def _build_query(code: str, name: str = None) -> str:
        code = str(code).strip()
        parts = [code]
        if name:
            name = str(name).strip()
            if name:
                parts.append(name)
        return " OR ".join(parts) if len(parts) > 1 else parts[0]

    @staticmethod
    def fetch_news_for_stock(code: str, name: str = None, max_items: int = None,
                              use_cache: bool = True, max_age_hours: float = None) -> dict:
        """
        抓取指定股票代碼(+可選公司名稱)的相關新聞列表。

        回傳:
          {
            'code': str, 'query': str, 'source': 'google_news_rss',
            'fetched_at': ISO時間字串, 'from_cache': bool,
            'items': [{'title','link','published','source','summary'}, ...],
            'status': 'ok' | 'empty' | 'error', 'error': str or None,
          }
        status='error' 時 items 會是空list，呼叫端應該顯示錯誤訊息而不是
        誤判成「今天真的沒新聞」。
        """
        max_items = max_items or NewsEngine.DEFAULT_MAX_ITEMS
        max_age_hours = max_age_hours if max_age_hours is not None else NewsEngine.DEFAULT_MAX_AGE_HOURS
        code = str(code).strip()
        cache_key = f"news:{code}"

        if use_cache:
            try:
                cached = DatabaseEngine.get_cache(cache_key, max_age_hours=max_age_hours)
                if cached is not None:
                    payload = cached["payload"]
                    payload["from_cache"] = True
                    return payload
            except Exception:
                pass  # 快取查詢失敗不影響主流程，繼續往下重新抓取

        query = NewsEngine._build_query(code, name)
        params = {"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}

        result = {
            "code": code, "query": query, "source": "google_news_rss",
            "fetched_at": datetime.now().isoformat(), "from_cache": False,
            "items": [], "status": "ok", "error": None,
        }

        try:
            resp = requests.get(NewsEngine.GOOGLE_NEWS_RSS_URL, params=params,
                                 headers=NewsEngine.HEADERS, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            raw_items = root.findall(".//item")[:max_items]
            for item in raw_items:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                source_el = item.find("source")
                source_name = source_el.text.strip() if source_el is not None and source_el.text else ""
                description = (item.findtext("description") or "").strip()
                result["items"].append({
                    "title": title, "link": link, "published": pub_date,
                    "source": source_name, "summary": description,
                })
            if not result["items"]:
                result["status"] = "empty"
        except Exception as e:
            logger.warning(f"NewsEngine.fetch_news_for_stock 抓取失敗 ({code}): {e}")
            result["status"] = "error"
            result["error"] = str(e)

        if use_cache and result["status"] in ("ok", "empty"):
            try:
                DatabaseEngine.set_cache(cache_key, result)
            except Exception:
                pass  # 寫快取失敗不影響本次回傳結果

        return result

    @staticmethod
    def classify_sentiment_rule_based(title: str, summary: str = "") -> dict:
        """
        規則式關鍵字情緒分類（分層加權版），永遠可用（不需要網路/API金鑰），
        也是 classify_sentiment() 在 LLM 不可用時的 fallback。

        Tier1（硬資訊：法人買賣超金額、目標價/財測調整、營收獲利數字）
        權重2分，Tier2（看好/看壞這類形容詞式軟敘述）權重1分，加總後
        算出 weighted_score，這是專業判讀新聞時常見的直覺：「外資買超」
        比「市場看好」的訊號強度更高，不該被當成同一件事。

        回傳的 confidence 只是「加權分數的粗略正規化」，不是機率，
        不要拿來跟 LLM 回傳的 confidence 直接比大小。
        """
        text = f"{title} {summary}"

        matched_bull_t1 = [kw for kw in NewsEngine.BULLISH_TIER1 if kw in text]
        # Tier2 只比對「還沒被Tier1命中覆蓋」的部分，避免「外資買超」裡的
        # 「買超」在Tier2重複計分，人為膨脹權重。
        remaining_text = text
        for kw in matched_bull_t1:
            remaining_text = remaining_text.replace(kw, "")
        matched_bull_t2 = [kw for kw in NewsEngine.BULLISH_TIER2 if kw in remaining_text]

        matched_bear_t1 = [kw for kw in NewsEngine.BEARISH_TIER1 if kw in text]
        remaining_text_bear = text
        for kw in matched_bear_t1:
            remaining_text_bear = remaining_text_bear.replace(kw, "")
        matched_bear_t2 = [kw for kw in NewsEngine.BEARISH_TIER2 if kw in remaining_text_bear]

        bull_score = len(matched_bull_t1) * 2 + len(matched_bull_t2) * 1
        bear_score = len(matched_bear_t1) * 2 + len(matched_bear_t2) * 1
        net_score = bull_score - bear_score

        if bull_score == 0 and bear_score == 0:
            label, confidence = "neutral", 0.0
        elif net_score > 0:
            label, confidence = "bullish", min(1.0, net_score / 4)
        elif net_score < 0:
            label, confidence = "bearish", min(1.0, abs(net_score) / 4)
        else:
            label, confidence = "neutral", 0.3

        return {
            "label": label, "method": "rule_based", "confidence": round(confidence, 2),
            "weighted_score": net_score,
            "matched_bullish": matched_bull_t1 + matched_bull_t2,
            "matched_bearish": matched_bear_t1 + matched_bear_t2,
            "matched_bullish_tier1": matched_bull_t1, "matched_bullish_tier2": matched_bull_t2,
            "matched_bearish_tier1": matched_bear_t1, "matched_bearish_tier2": matched_bear_t2,
        }

    @staticmethod
    def classify_sentiment(title: str, summary: str = "", llm_classify_fn=None) -> dict:
        """
        情緒分類主入口。優先嘗試呼叫端注入的 llm_classify_fn
        （簽章需為 fn(title, summary) -> {'label': 'bullish'/'bearish'/'neutral',
        'confidence': float} 或 None/丟例外），失敗或未提供就 fallback 回規則式。

        Phase 2 要接LLM時，在 app.py 寫一個包好 Anthropic API 呼叫的函式
        傳進來即可，不需要改這支 engine（金鑰管理、額度控制留在app層）。
        """
        if llm_classify_fn is not None:
            try:
                llm_result = llm_classify_fn(title, summary)
                if llm_result and llm_result.get("label") in ("bullish", "bearish", "neutral"):
                    llm_result.setdefault("method", "llm")
                    return llm_result
            except Exception as e:
                logger.warning(f"NewsEngine.classify_sentiment LLM分類失敗，fallback回規則式: {e}")

        return NewsEngine.classify_sentiment_rule_based(title, summary)

    @staticmethod
    def classify_event_tags(title: str, summary: str = "") -> list:
        """回傳命中的事件分類標籤（一則新聞可能同時命中多個標籤）。"""
        text = f"{title} {summary}"
        return [tag for tag, keywords in NewsEngine.EVENT_TAGS.items()
                if any(kw in text for kw in keywords)]

    @staticmethod
    def _recency_weight(pub_date_str: str) -> float:
        """
        時間衰減權重：剛發布的新聞權重接近1.0，每過24小時權重打七折，
        最低不會低於0.15（避免完全歸零導致舊新聞被當作「沒發生過」）。
        解析失敗（RSS沒給標準RFC822時間格式）時回傳中性權重1.0，不放大
        也不縮小，避免因為解析失敗而讓那篇新聞的意見消失。
        """
        if not pub_date_str:
            return 1.0
        try:
            pub_dt = parsedate_to_datetime(pub_date_str)
            if pub_dt.tzinfo is not None:
                pub_dt = pub_dt.replace(tzinfo=None)
            hours_old = max(0.0, (datetime.utcnow() - pub_dt).total_seconds() / 3600)
        except Exception:
            return 1.0
        days_old = hours_old / 24
        weight = 0.7 ** days_old
        return max(0.15, round(weight, 3))

    @staticmethod
    def get_investor_notes() -> list:
        """
        固定的「達人視角」使用提醒，app.py 應該把這些提醒顯示在新聞區塊
        旁邊，不要只丟數字/標籤給使用者看。
        """
        return [
            "新聞情緒是落後或同步指標：報導出來時，股價往往已經反映了一部分，"
            "看到「利多」才追進場，經常是幫別人出貨。",
            "規則式關鍵字判讀只看字面，判斷不了反諷、條件句、「已利多出盡」"
            "這類語境，遇到重大新聞務必自己點進原文確認。",
            "新聞面建議當作「確認」或「留意」工具，跟技術面轉強、籌碼面法人"
            "同步買超一起出現時，訊號才比較可信；單一新聞面利多不足以構成"
            "進場理由。",
            "新聞量異常增加只代表「市場正在討論這檔股票」，不代表方向，"
            "增加的新聞裡有可能利多利空各半，仍需要看內容判斷。",
        ]

    @staticmethod
    def get_related_concept_stocks(text: str) -> list:
        """
        從新聞文字比對 CONCEPT_STOCK_MAP，列出可能受影響的個股與受惠程度
        星等（1-5，主觀評估，非量化計算）。同一檔股票被多個關鍵字命中時，
        取最高的 stars。回傳依 stars 由高到低排序。

        ⚠️ 範例性質對照表，不是全市場自動化的產業關聯分析，見class
        docstring第9點的完整揭露。
        """
        best = {}
        for keyword, stocks in NewsEngine.CONCEPT_STOCK_MAP.items():
            if keyword in text:
                for code, name, stars in stocks:
                    if code not in best or stars > best[code]["stars"]:
                        best[code] = {"code": code, "name": name, "stars": stars,
                                      "matched_keyword": keyword}
        return sorted(best.values(), key=lambda x: x["stars"], reverse=True)

    @staticmethod
    def _score_to_stars(weighted_bias_score: float) -> int:
        """把加權分數轉成 1~5 星（絕對值越大星數越多），只是方便UI呈現，
        不是嚴謹的量化分級，門檻是主觀設定的。"""
        abs_score = abs(weighted_bias_score)
        if abs_score >= 8:
            return 5
        elif abs_score >= 5:
            return 4
        elif abs_score >= 2:
            return 3
        elif abs_score > 0:
            return 2
        return 1

    @staticmethod
    def _rule_based_stock_summary(items: list) -> list:
        """
        規則式重點條列（Phase 2 baseline，也是 llm_summarize_fn 不可用
        時的 fallback）：列出命中次數最多的事件標籤，附一則代表性新聞
        標題當例子。不是語意摘要，只是「出現頻率最高的主題」統計。
        """
        from collections import Counter
        tag_counter = Counter()
        tag_example = {}
        for item in items:
            for tag in item.get("event_tags", []):
                tag_counter[tag] += 1
                tag_example.setdefault(tag, item["title"])

        bullets = []
        for tag, count in tag_counter.most_common(4):
            bullets.append(f"「{tag}」相關新聞 {count} 篇（例如：{tag_example[tag]}）")
        if not bullets:
            bullets.append("近期新聞未命中已知事件分類，建議直接看下方原始新聞列表。")
        return bullets

    @staticmethod
    def compute_resonance(news_stars: int, tech_stars: int,
                           volume_stars: int = None, chip_stars: int = None) -> dict:
        """
        新聞面 × 技術面（可選加成交量／籌碼面）共振檢查。

        輸入的 *_stars 都是 1~5 星，由呼叫端(app.py)自行從各自的既有
        分析結果換算過來（例如技術面可以用你的 DecisionEngine/
        StrategyEngine 現有的評分換算成星等）。這裡只做「訊號一致性」
        的簡單規則判斷，不是量化模型，也不是進出場訊號本身——只是提醒
        使用者「這些面向講的是同一個故事，還是互相矛盾」。

        回傳 {'avg_stars': float, 'consistent': bool, 'message': str}
        """
        provided = [s for s in [news_stars, tech_stars, volume_stars, chip_stars] if s is not None]
        avg_stars = round(sum(provided) / len(provided), 2) if provided else 0
        spread = (max(provided) - min(provided)) if provided else 0
        consistent = spread <= 1

        if news_stars >= 4 and tech_stars >= 4 and consistent:
            message = "📈 高度共振，值得關注（但仍請自行確認新聞內容與技術面細節，非自動化進場訊號）。"
        elif news_stars >= 4 and tech_stars <= 2:
            message = "📰 消息面偏多，但技術面尚未確認，建議先觀察，不建議單獨依消息面追價。"
        elif news_stars <= 2 and tech_stars >= 4:
            message = "📊 技術面轉強，但新聞面尚未出現對應消息，留意是否為提前反應或籌碼面獨立行情。"
        elif news_stars <= 2 and tech_stars <= 2:
            message = "😐 新聞面與技術面皆偏弱，暫無積極訊號。"
        else:
            message = "🔍 訊號不一致或強度中等，建議持續觀察，不建議單獨依此做進出場決策。"

        return {"avg_stars": avg_stars, "consistent": consistent, "message": message}

    @staticmethod
    def _get_source_weight(source: str) -> float:
        """依 SOURCE_WEIGHT 對照表換算來源可信度權重，找不到比對時回傳
        DEFAULT_SOURCE_WEIGHT（中性值，不是懲罰值）。"""
        if not source:
            return NewsEngine.DEFAULT_SOURCE_WEIGHT
        source_lower = str(source).lower()
        for key, weight in NewsEngine.SOURCE_WEIGHT.items():
            if key in source_lower:
                return weight
        return NewsEngine.DEFAULT_SOURCE_WEIGHT

    @staticmethod
    def _title_similarity(title_a: str, title_b: str) -> float:
        """用 difflib 計算兩個標題的相似度（0~1），供事件去重使用。
        stdlib 做法，不需要額外套件，但不是語意理解，只是字面相似度。"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, title_a or "", title_b or "").ratio()

    @staticmethod
    def _event_confidence(cluster: list) -> dict:
        """
        事件可信度（Phase 2d 新增）：把「有幾家獨立媒體報導」跟「報導
        來源的可信度」合成一個 0~1 的信心分數，供 UI／Decision Engine
        判斷「這則新聞值得信任的程度」，不只是「方向是多是空」。

        算法（刻意簡單、可解釋，不是統計模型）：
          confidence = 0.6 × 最高來源可信度 + 0.4 × min(獨立來源數/4, 1)
        獨立來源數用「群組內不重複的 source 名稱數」而非篇數——同一家
        媒體同一事件轉發兩次不該被當成兩個獨立來源在加強可信度。

        回傳 {'score': float(0~1), 'level': '高'/'中'/'低',
              'source_count': int(獨立來源數)}
        """
        top_weight = max(NewsEngine._get_source_weight(it.get("source", "")) for it in cluster)
        distinct_sources = len({str(it.get("source", "")).strip() for it in cluster if it.get("source")})
        distinct_sources = max(distinct_sources, 1)
        score = round(0.6 * top_weight + 0.4 * min(distinct_sources / 4, 1.0), 2)
        if score >= 0.75:
            level = "高"
        elif score >= 0.5:
            level = "中"
        else:
            level = "低"
        return {"score": score, "level": level, "source_count": distinct_sources}

    @staticmethod
    def _cluster_news(items: list) -> None:
        """
        新聞事件去重（Phase 2c 新增）：同一事件被多家媒體轉載報導時，
        只挑一篇「代表新聞」（可信度最高的來源）計入情緒分數，避免
        「12篇都在報同一場法說會」被當成12個獨立利多訊號重複加分。

        直接在傳入的 items（list of dict，就地修改）上加幾個欄位：
          - 'is_representative': bool，該篇是否為所屬事件群組的代表新聞
          - 'duplicate_count': 該事件群組除了代表新聞外還有幾篇其他報導
            （只有代表新聞這個欄位有意義，非代表新聞固定是0）
          - 'event_confidence'（Phase 2d 新增，只有代表新聞有意義）：
            見 _event_confidence()，反映「這則新聞有幾家獨立可信來源
            佐證」，不是情緒方向的信心，是「這件事有沒有發生、報導
            廣泛程度」的信心。

        分群方式：貪婪法（greedy）——依序比對每篇新聞的標題跟目前已存在
        的各群組代表標題的相似度，超過 DEDUP_TITLE_SIMILARITY_THRESHOLD
        就歸進同一群，否則自成一群。群組代表新聞 = 群內來源可信度
        (source_weight) 最高者，同分時取較早出現(RSS排序較前)的那篇。

        ⚠️ 這是簡化heuristic，不是嚴謹NLP聚類，同一事件但標題寫法差異
        很大的新聞可能沒被歸在一起，這種情況下等於沒去重，不會出錯，
        只是效果打折。
        """
        clusters = []  # list of list[item]
        for item in items:
            item["_source_weight_tmp"] = NewsEngine._get_source_weight(item.get("source", ""))
            placed = False
            for cluster in clusters:
                if NewsEngine._title_similarity(item["title"], cluster[0]["title"]) >= NewsEngine.DEDUP_TITLE_SIMILARITY_THRESHOLD:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        for cluster in clusters:
            representative = max(cluster, key=lambda it: it["_source_weight_tmp"])
            event_confidence = NewsEngine._event_confidence(cluster)
            for it in cluster:
                it["is_representative"] = (it is representative)
                it["duplicate_count"] = (len(cluster) - 1) if it is representative else 0
                it["source_weight"] = it.pop("_source_weight_tmp")
                it["event_confidence"] = event_confidence if it is representative else None

    @staticmethod
    def get_news_with_sentiment(code: str, name: str = None, max_items: int = None,
                                 use_cache: bool = True, llm_classify_fn=None,
                                 llm_summarize_fn=None) -> dict:
        """
        整合入口，app.py 應該主要呼叫這個方法（其他方法留給單元測試/
        單獨呼叫用）。抓新聞 + 逐篇加情緒標籤(含時間衰減權重) + 事件分類
        + 概念股關聯 + 事件去重(Dedup) + 來源可信度加權 + 新聞量異動偵測
        + 當日統計摘要 + AI摘要 + 達人視角使用提醒。

        Phase 2c 新增（事件去重／來源可信度）：weighted_bias_score 的計算
        現在只採計每個事件群組裡的「代表新聞」（可信度最高的來源）×
        來源可信度權重 × 時間衰減權重，同一事件被10幾家媒體轉載不會被
        當成10幾個獨立利多/利空訊號重複加分。summary_stats 裡的
        bullish/bearish/neutral 篇數統計同理，只算代表新聞，'total' 則
        維持「實際抓到幾篇原始報導」的真實篇數，兩者刻意分開呈現，UI
        請標示清楚差異，不要混用。

        llm_summarize_fn（選用）：簽章為 fn(items, code, name) ->
        {'bullets': [str,...], 'label': 'bullish'/'bearish'/'neutral'}
        或 None/丟例外。提供時優先使用，失敗或未提供則 fallback 回
        _rule_based_stock_summary()（列出命中最多次的事件標籤），
        ai_summary 裡的 'method' 欄位會誠實標示是 'llm' 還是
        'rule_based'，不假裝兩者品質相同。

        回傳結構同 fetch_news_for_stock()，但每個 item 多了
        'sentiment'（含weighted_score）、'event_tags'、'recency_weight'、
        'related_concepts'、'source_weight'、'is_representative'、
        'duplicate_count'，並多以下欄位：
          - 'summary_stats': {'total'(原始篇數),'unique_events'(去重後事件數),
             'bullish','bearish','neutral'(以上三者為去重後代表新聞的統計),
             'overall_bias','weighted_bias_score'}
          - 'volume_change': {'previous_total', 'current_total', 'trend'}
             trend 為 'increase'/'decrease'/'flat'/'unknown'（第一次查詢
             沒有歷史可比較時是 unknown）
          - 'ai_summary': {'bullets': [...], 'stars': int, 'label': str,
             'method': 'rule_based'/'llm'}
          - 'investor_notes': 達人視角提醒清單

        overall_bias／weighted_bias_score 是規則式加權多數決的粗略偏向，
        不是精確分數，UI上請標示清楚這只是「今天抓到的新聞裡，加權後
        利多/利空哪邊比較強」，不是嚴謹的量化訊號，更不能單獨依此下單。
        """
        # 在覆蓋快取前，先讀一次「不管新鮮度」的舊資料，只為了比較新聞量，
        # 不影響本次回傳的實際新聞內容。
        previous_total = None
        if use_cache:
            try:
                stale_cached = DatabaseEngine.get_cache(f"news:{str(code).strip()}", max_age_hours=24 * 365)
                if stale_cached is not None:
                    previous_total = len(stale_cached["payload"].get("items", []))
            except Exception:
                previous_total = None

        news = NewsEngine.fetch_news_for_stock(code, name=name, max_items=max_items, use_cache=use_cache)

        for item in news["items"]:
            sentiment = NewsEngine.classify_sentiment(
                item["title"], item.get("summary", ""), llm_classify_fn=llm_classify_fn
            )
            recency_weight = NewsEngine._recency_weight(item.get("published", ""))
            item["sentiment"] = sentiment
            item["event_tags"] = NewsEngine.classify_event_tags(item["title"], item.get("summary", ""))
            item["recency_weight"] = recency_weight
            item["related_concepts"] = NewsEngine.get_related_concept_stocks(
                f"{item['title']} {item.get('summary', '')}"
            )

        # 事件去重（Phase 2c）：就地幫每個item加上 is_representative /
        # duplicate_count / source_weight，只有代表新聞會被計入下面的
        # 加權分數與偏多/偏空篇數統計。
        NewsEngine._cluster_news(news["items"])

        weighted_bull, weighted_bear = 0.0, 0.0
        representative_items = [it for it in news["items"] if it.get("is_representative")]
        for item in representative_items:
            raw_score = item["sentiment"].get("weighted_score", 0)
            combined_weight = item["recency_weight"] * item["source_weight"]
            if raw_score > 0:
                weighted_bull += raw_score * combined_weight
            elif raw_score < 0:
                weighted_bear += abs(raw_score) * combined_weight

        bull_count = sum(1 for it in representative_items if it["sentiment"]["label"] == "bullish")
        bear_count = sum(1 for it in representative_items if it["sentiment"]["label"] == "bearish")
        unique_events = len(representative_items)
        total = len(news["items"])
        neutral_count = unique_events - bull_count - bear_count

        weighted_bias_score = round(weighted_bull - weighted_bear, 2)
        if unique_events == 0:
            overall_bias = "無新聞"
        elif weighted_bias_score > 1:
            overall_bias = "偏多"
        elif weighted_bias_score < -1:
            overall_bias = "偏空"
        else:
            overall_bias = "中性"

        news["summary_stats"] = {
            "total": total, "unique_events": unique_events,
            "bullish": bull_count, "bearish": bear_count,
            "neutral": neutral_count, "overall_bias": overall_bias,
            "weighted_bias_score": weighted_bias_score,
            "avg_event_confidence": (
                round(sum(it["event_confidence"]["score"] for it in representative_items) / unique_events, 2)
                if unique_events else 0
            ),
        }

        if not news["from_cache"] and previous_total is not None:
            if total > previous_total:
                trend = "increase"
            elif total < previous_total:
                trend = "decrease"
            else:
                trend = "flat"
        else:
            trend = "unknown"
        news["volume_change"] = {
            "previous_total": previous_total, "current_total": total, "trend": trend,
        }

        news["investor_notes"] = NewsEngine.get_investor_notes()

        # ── AI摘要（Phase 2）：優先LLM，失敗/未提供則fallback規則式 ──
        ai_summary = None
        if llm_summarize_fn is not None and representative_items:
            try:
                llm_result = llm_summarize_fn(representative_items, code, name)
                if llm_result and llm_result.get("bullets"):
                    ai_summary = {
                        "bullets": llm_result["bullets"],
                        "label": llm_result.get("label", overall_bias),
                        "stars": NewsEngine._score_to_stars(weighted_bias_score),
                        "method": "llm",
                    }
            except Exception as e:
                logger.warning(f"NewsEngine.get_news_with_sentiment LLM摘要失敗，fallback回規則式: {e}")

        if ai_summary is None:
            ai_summary = {
                "bullets": NewsEngine._rule_based_stock_summary(representative_items) if representative_items else
                           ["今日無相關新聞可摘要。"],
                "label": overall_bias,
                "stars": NewsEngine._score_to_stars(weighted_bias_score),
                "method": "rule_based",
            }
        news["ai_summary"] = ai_summary

        return news

    @staticmethod
    def build_daily_market_summary(codes: list, use_cache: bool = True,
                                    max_items_per_stock: int = 5) -> dict:
        """
        彙總「自選股清單」的新聞，做成每日市場摘要。

        ⚠️ 範圍限定在傳入的 codes 清單（例如你的自選股），不是大盤指數
        新聞或美股/Fed這類國際總經新聞——後者需要另外接國際財經新聞
        來源，這裡沒有做（見 class docstring 第13點）。

        回傳 {'per_stock': [...], 'top_event_tags': [(tag,count),...],
              'market_weighted_bias_score': float, 'market_stars': int,
              'bullets': [...], 'generated_at': ISO時間字串}
        """
        from collections import Counter

        per_stock = []
        tag_counter = Counter()
        total_weighted = 0.0

        for code in codes:
            try:
                report = NewsEngine.get_news_with_sentiment(
                    code, max_items=max_items_per_stock, use_cache=use_cache
                )
            except Exception as e:
                logger.warning(f"NewsEngine.build_daily_market_summary 查詢 {code} 失敗: {e}")
                continue

            stats = report.get("summary_stats", {})
            per_stock.append({
                "code": code, "total": stats.get("total", 0),
                "unique_events": stats.get("unique_events", 0),
                "overall_bias": stats.get("overall_bias", "無新聞"),
                "weighted_bias_score": stats.get("weighted_bias_score", 0),
            })
            total_weighted += stats.get("weighted_bias_score", 0)
            for item in report.get("items", []):
                if not item.get("is_representative", True):
                    continue  # 事件去重：非代表新聞不重複計入市場層級的主題統計
                for tag in item.get("event_tags", []):
                    tag_counter[tag] += 1

        top_tags = tag_counter.most_common(6)
        bullets = [f"「{tag}」相關新聞共 {count} 篇" for tag, count in top_tags]
        if not bullets:
            bullets = ["自選股清單目前沒有可彙總的新聞，或清單為空。"]

        market_stars = NewsEngine._score_to_stars(total_weighted)
        if total_weighted > 1:
            market_bias = "偏多"
        elif total_weighted < -1:
            market_bias = "偏空"
        else:
            market_bias = "中性"

        return {
            "per_stock": sorted(per_stock, key=lambda x: x["total"], reverse=True),
            "top_event_tags": top_tags,
            "market_weighted_bias_score": round(total_weighted, 2),
            "market_bias": market_bias,
            "market_stars": market_stars,
            "bullets": bullets,
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def get_watchlist_news_overview(status_filter=None, use_cache: bool = True,
                                     max_items_per_stock: int = 5) -> list:
        """
        自選股新聞中心：直接讀既有的 watchlist_status 表（沒有另外開新
        表），逐檔查詢新聞篇數與情緒，依新聞篇數由多到少排序回傳。

        回傳 [{'code','name','total','overall_bias','stars'}, ...]
        股票名稱查詢失敗時 name 會是空字串，不影響其餘欄位。
        """
        from engines.name_engine import NameEngine

        try:
            watchlist_df = DatabaseEngine.list_watchlist(status_filter=status_filter)
        except Exception as e:
            logger.warning(f"NewsEngine.get_watchlist_news_overview 讀取自選股清單失敗: {e}")
            return []

        overview = []
        for ticker in watchlist_df.get("ticker", []):
            try:
                report = NewsEngine.get_news_with_sentiment(
                    ticker, max_items=max_items_per_stock, use_cache=use_cache
                )
                stats = report.get("summary_stats", {})
                try:
                    stock_name = NameEngine.get_name(ticker) or ""
                except Exception:
                    stock_name = ""
                overview.append({
                    "code": ticker, "name": stock_name,
                    "total": stats.get("total", 0),
                    "overall_bias": stats.get("overall_bias", "無新聞"),
                    "stars": NewsEngine._score_to_stars(stats.get("weighted_bias_score", 0)),
                })
            except Exception as e:
                logger.warning(f"NewsEngine.get_watchlist_news_overview 查詢 {ticker} 失敗: {e}")
                continue

        return sorted(overview, key=lambda x: x["total"], reverse=True)
