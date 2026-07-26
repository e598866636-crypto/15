import pandas as pd

class IndustryEngine:
    """
    🏭 產業中心 (Industry Center) - TQAI Pro 跨界生醫與前沿科技擴充版

    將 ScannerEngine 的全台股掃描結果，依產業分類聚合成「產業排名」。

    ⚠️ 修正說明（v2.6）：
    原版把台積電(2330)、廣達(2382)等主業明顯是半導體代工／電子代工的公司，
    直接歸類為「生物晶片與先進封測」「AI醫療與智慧生醫」——這是把公司的
    次要業務線或想像空間當成主分類，容易讓使用者誤以為某類股有生醫題材
    支撐、扭曲產業排名的參考價值。

    現在拆成兩張表：
      - INDUSTRY_MAP：公司的「主要營收/實際核心業務」分類，用於預設的
        產業排名（rank_industries 預設行為），避免誤導。
      - THEME_MAP：原本的「跨界生醫／前沿科技」敘事標籤，保留給想用這個
        主題鏡頭觀察的人，但預設不用於排名，且清楚標示為「題材觀察」
        而非嚴謹產業分類。
    """

    # === 主要產業分類（依實際核心業務，用於預設排名） ===
    INDUSTRY_MAP = {
        "2330": "晶圓代工",
        "3374": "半導體封裝與影像感測",
        "6223": "半導體測試設備(探針卡)",
        "3711": "半導體封測",
        "6841": "醫療AI軟體",
        "2382": "電子代工(伺服器/NB)",
        "3231": "電子代工",
        "2356": "電子代工",
        "6472": "製藥CDMO",
        "6901": "創投/投資控股",
        "4743": "新藥研發",
        "6712": "再生醫療/細胞治療",

        "2317": "電子代工", "2454": "半導體(IC設計)", "2308": "AI Server／散熱",
        "2379": "半導體", "3034": "半導體", "2412": "電信", "2881": "金融",
        "2603": "航運", "1515": "重電與綠能", "1101": "傳產水泥",

        # === ETF（v2.7 興櫃/ETF資料擴充新增）===
        # ETF 本質上是一籃子股票的組合，不屬於任何單一產業，因此獨立歸為
        # 「ETF」類別，而不是硬塞進某個看似相關的產業（避免污染產業平均分數）。
        "0050": "ETF", "0056": "ETF", "00878": "ETF",
        "006208": "ETF", "00631L": "ETF", "00713": "ETF",
    }

    # === 題材標籤（原始的跨界生醫敘事分類，僅供主題觀察，非嚴謹產業分類） ===
    THEME_MAP = {
        "2330": "生物晶片與先進封測(題材)", "3374": "生物晶片與先進封測(題材)",
        "6223": "生物晶片與先進封測(題材)", "3711": "生物晶片與先進封測(題材)",
        "6841": "AI醫療與智慧生醫(題材)", "2382": "AI醫療與智慧生醫(題材)",
        "3231": "AI醫療與智慧生醫(題材)", "2356": "AI醫療與智慧生醫(題材)",
        "6472": "合成生物與前沿創投(題材)", "6901": "合成生物與前沿創投(題材)",
        "4743": "合成生物與前沿創投(題材)", "6712": "合成生物與前沿創投(題材)",
    }

    @staticmethod
    def get_industry(stock_code: str, use_theme: bool = False) -> str:
        code = str(stock_code).split(".")[0].strip()
        if use_theme:
            return IndustryEngine.THEME_MAP.get(code, IndustryEngine.INDUSTRY_MAP.get(code, "其他/未分類"))
        return IndustryEngine.INDUSTRY_MAP.get(code, "其他/未分類")

    @staticmethod
    def _to_stars(score: float) -> str:
        if score >= 75: return "★★★★★"
        elif score >= 65: return "★★★★☆"
        elif score >= 55: return "★★★☆☆"
        elif score >= 45: return "★★☆☆☆"
        else: return "★☆☆☆☆"

    @staticmethod
    def rank_industries(scan_result_df: pd.DataFrame, use_theme: bool = False) -> pd.DataFrame:
        """
        依產業分類聚合排名。

        use_theme=False（預設）：用實際核心業務分類，適合判斷「哪個真實產業
        資金動能較強」。
        use_theme=True：用原本的跨界生醫題材標籤，適合觀察「特定敘事主題」，
        但請注意這不是嚴謹的產業分類，公司主業可能與標籤完全無關。
        """
        if scan_result_df is None or scan_result_df.empty:
            return pd.DataFrame()

        df = scan_result_df.copy()
        df["產業"] = df["代碼"].apply(lambda code: IndustryEngine.get_industry(code, use_theme=use_theme))

        grouped = df.groupby("產業").agg(
            平均AIScore=("AI Score", "mean"),
            成分股數=("代碼", "count"),
            強勢股數=("AI Score", lambda s: int((s >= 70).sum())),
            弱勢股數=("AI Score", lambda s: int((s <= 45).sum())),
        ).reset_index()

        grouped["強勢比例(%)"] = (grouped["強勢股數"] / grouped["成分股數"] * 100).round(1)
        grouped["平均AIScore"] = grouped["平均AIScore"].round(1)
        grouped["產業評級"] = grouped["平均AIScore"].apply(IndustryEngine._to_stars)

        grouped = grouped.sort_values("平均AIScore", ascending=False).reset_index(drop=True)
        grouped.insert(0, "排名", range(1, len(grouped) + 1))

        return grouped[["排名", "產業", "產業評級", "平均AIScore", "強勢比例(%)", "成分股數", "強勢股數", "弱勢股數"]]

    @staticmethod
    def get_industry_constituents(scan_result_df: pd.DataFrame, industry_name: str, use_theme: bool = False) -> pd.DataFrame:
        if scan_result_df is None or scan_result_df.empty:
            return pd.DataFrame()

        df = scan_result_df.copy()
        df["產業"] = df["代碼"].apply(lambda code: IndustryEngine.get_industry(code, use_theme=use_theme))
        result = df[df["產業"] == industry_name].sort_values("AI Score", ascending=False).reset_index(drop=True)
        return result.drop(columns=["產業"])