import numpy as np
import pandas as pd


class CanslimEngine:
    """
    📋 CAN SLIM 量化評分引擎 (CANSLIM Engine)

    William O'Neil 的 CAN SLIM 七項條件量化評分，組裝自本專案既有的
    FundamentalEngine / ChipEngine / StrategyEngine(market_regime) /
    RSRatingEngine，本身不重新呼叫任何外部 API，純粹是「既有資料的
    重新組合與評分」，設計上跟 StockAcademyEngine（五維度評分）同一種
    「組合既有報告」的模式。

    ⚠️ 誠實揭露（務必先讀，這比評分數字本身更重要）：
    原版 CAN SLIM 的 C（當季 EPS 年增率）與 A（近3年 EPS 年複合成長率）
    需要「季度財報 EPS」這種細顆粒度資料。本專案的資料來源是 yfinance
    的 `.info`（見 fundamental_engine.py），對台股/興櫃股票**沒有穩定的
    季度 EPS 成長率欄位可用**，只有 TTM（近四季合計）等級的數字。
    因此本引擎：
      - C 項：用「營收年增率」(revenue_growth_yoy) 代理「當季獲利年增率」，
        兩者不是同一件事（營收成長不等於獲利成長，尤其毛利率會變動的
        公司），純粹是資料可得性下的最佳近似值。
      - A 項：用 ROE 水準代理「長期獲利成長趨勢」，同樣不是嚴謹的
        3年EPS複合成長率。
      - 兩項評分在 detail 裡都會明確標注「代理指標，非原版定義」，
        UI 端請完整顯示這段警語，不要只顯示分數本身。
      - N（新產品/新高）、S（供需/量能）、L（領導股/RS Rating）、
        I（法人認養）、M（大盤方向）這五項是技術面/籌碼面資料，
        本專案既有資料可以支撐，評分較貼近原版精神。
    """

    LETTER_MAX = {'C': 15, 'A': 15, 'N': 15, 'S': 15, 'L': 20, 'I': 15, 'M': 5}
    # L (Leader/RS Rating) 權重刻意設得比其他項目高一些，因為 Minervini/O'Neil
    # 都反覆強調「RS Rating 是最重要的單一篩選條件」，這裡用權重反映這個優先順序。

    @staticmethod
    def _score_c(snapshot: dict) -> dict:
        rev_growth = snapshot.get('revenue_growth_yoy') if snapshot else None
        if snapshot is None or snapshot.get('status') not in (None, 'ok') or pd.isna(rev_growth):
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足（無季度EPS成長率資料，且營收年增率代理值也缺失）'}
        pct = rev_growth * 100
        if pct >= 25:
            score = 15
        elif pct >= 15:
            score = 11
        elif pct >= 5:
            score = 6
        elif pct >= 0:
            score = 3
        else:
            score = 0
        return {'score': score, 'max': 15,
                'note': f"⚠️ 代理指標（非原版季度EPS年增率）：營收年增率 {pct:.1f}%"}

    @staticmethod
    def _score_a(snapshot: dict) -> dict:
        roe = snapshot.get('roe') if snapshot else None
        if snapshot is None or pd.isna(roe):
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足（無3年EPS複合成長率資料，ROE代理值也缺失）'}
        pct = roe * 100
        if pct >= 20:
            score = 15
        elif pct >= 15:
            score = 11
        elif pct >= 10:
            score = 6
        elif pct >= 0:
            score = 3
        else:
            score = 0
        return {'score': score, 'max': 15,
                'note': f"⚠️ 代理指標（非原版3年EPS複合成長率）：ROE {pct:.1f}%"}

    @staticmethod
    def _score_n(df: pd.DataFrame) -> dict:
        """新高（技術面代理「新產品/新管理階層/新高價」的 N）。"""
        if df is None or 'close' not in df.columns or len(df) < 20:
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足'}
        window = min(252, len(df))
        recent_high = df['close'].tail(window).max()
        current = float(df['close'].iloc[-1])
        pct_from_high = (current - recent_high) / recent_high * 100 if recent_high > 0 else np.nan
        if pd.isna(pct_from_high):
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足'}
        if pct_from_high >= -2:
            score, note = 15, f"股價距離{window}日高點僅 {pct_from_high:.1f}%，符合『創新高附近』條件"
        elif pct_from_high >= -10:
            score, note = 8, f"股價距離{window}日高點 {pct_from_high:.1f}%，尚未站上新高"
        else:
            score, note = 0, f"股價距離{window}日高點 {pct_from_high:.1f}%，離新高有相當距離"
        return {'score': score, 'max': 15, 'note': note}

    @staticmethod
    def _score_s(df: pd.DataFrame) -> dict:
        """供需：用近期量能相對均量（RVOL）代理『籌碼供需吃緊』。"""
        if df is None or 'rvol' not in df.columns or df.empty:
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足（無 rvol 欄位，請確認已跑過 IndicatorEngine）'}
        rvol = df['rvol'].iloc[-1]
        if pd.isna(rvol):
            return {'score': 0, 'max': 15, 'note': '⚠️ 資料不足'}
        rvol = float(rvol)
        if rvol >= 1.5:
            score, note = 15, f"近期量能為均量 {rvol:.2f} 倍，買盤積極"
        elif rvol >= 1.1:
            score, note = 8, f"近期量能為均量 {rvol:.2f} 倍，溫和放大"
        else:
            score, note = 0, f"近期量能為均量 {rvol:.2f} 倍，量能未見擴張"
        return {'score': score, 'max': 15, 'note': note}

    @staticmethod
    def _score_l(rs_rating) -> dict:
        """領導股：直接採用 RSRatingEngine 算出的 RS Rating。"""
        if rs_rating is None or pd.isna(rs_rating):
            return {'score': 0, 'max': 20, 'note': '⚠️ RS Rating 尚未計算（需搭配全台股掃描才有排名母體）'}
        rs_rating = float(rs_rating)
        if rs_rating >= 90:
            score = 20
        elif rs_rating >= 80:
            score = 16
        elif rs_rating >= 70:
            score = 10
        elif rs_rating >= 50:
            score = 4
        else:
            score = 0
        return {'score': score, 'max': 20, 'note': f"RS Rating = {rs_rating:.0f}"}

    @staticmethod
    def _score_i(chip_report: dict) -> dict:
        """法人認養：外資/投信同步買超給分。"""
        if not chip_report or chip_report.get('status') != 'ok':
            return {'score': 0, 'max': 15, 'note': '⚠️ 籌碼資料不可用，以0分計（非真的認養度差）'}
        inst = chip_report.get('institutional')
        if not inst:
            return {'score': 0, 'max': 15, 'note': '⚠️ 無三大法人資料'}
        f_net = inst.get('foreign_net', 0)
        t_net = inst.get('trust_net', 0)
        if f_net > 0 and t_net > 0:
            score, note = 15, "外資與投信同步買超，法人認養度高"
        elif f_net > 0 or t_net > 0:
            score, note = 8, "外資或投信其中一方買超"
        else:
            score, note = 0, "外資與投信皆未買超"
        return {'score': score, 'max': 15, 'note': note}

    @staticmethod
    def _score_m(market_regime: str) -> dict:
        """大盤方向：沿用 strategy_engine.py 既有的 market_regime 分類。"""
        if not market_regime:
            return {'score': 0, 'max': 5, 'note': '⚠️ 無市場狀態資料'}
        if '多頭' in market_regime:
            score, note = 5, f"大盤狀態：{market_regime}，順勢操作條件成立"
        elif '空頭' in market_regime:
            score, note = 0, f"大盤狀態：{market_regime}，逆勢操作風險較高"
        else:
            score, note = 2, f"大盤狀態：{market_regime}，方向不明確"
        return {'score': score, 'max': 5, 'note': note}

    @staticmethod
    def analyze(df: pd.DataFrame, fundamental_report: dict = None, chip_report: dict = None,
                market_regime: str = None, rs_rating=None) -> dict:
        """
        主要進入點：組合出七項 CAN SLIM 分數與總評。

        參數皆為既有引擎的輸出結果，呼叫端通常已經算好，直接傳進來即可：
            fundamental_report : FundamentalEngine.build_fundamental_report() 的回傳值
            chip_report        : ChipEngine.build_chip_report() 的回傳值
            market_regime      : df['market_regime'].iloc[-1]（StrategyEngine 算好的欄位）
            rs_rating          : RSRatingEngine 排名結果裡該股票的 rs_rating（1~99 或 None）
        """
        snapshot = None
        etf_mode = bool(fundamental_report and fundamental_report.get('status') == 'not_applicable')
        if fundamental_report and fundamental_report.get('status') == 'ok':
            snapshot = fundamental_report.get('snapshot')

        letters = {
            'C': CanslimEngine._score_c(snapshot),
            'A': CanslimEngine._score_a(snapshot),
            'N': CanslimEngine._score_n(df),
            'S': CanslimEngine._score_s(df),
            'L': CanslimEngine._score_l(rs_rating),
            'I': CanslimEngine._score_i(chip_report),
            'M': CanslimEngine._score_m(market_regime),
        }

        # v2.9.11 修正：C（獲利年增率）與 A（EPS複合成長率）的代理指標都是
        # 「公司財報」概念，ETF 本質上沒有這兩項（不是「資料剛好缺失」，
        # 是「這個概念對 ETF 結構性不適用」）。修正前，ETF 一律被扣掉
        # 30/100 的滿分（C、A 各 0/15），卻只在逐項備註裡用一句話帶過，
        # 使用者看到的總分/百分比/等第完全沒有反映「這 30 分本來就不該
        # 算進去」，容易誤讀成「這檔ETF成長性很差」。這裡改成：偵測到
        # ETF（fundamental_report.status == 'not_applicable'）時，直接把
        # C、A 排除在總分與滿分之外，只用 N/S/L/I/M 五項（滿分 70）計算
        # 百分比，並在 disclosure 中明確說明，而不是讓兩項注定拿不到的
        # 分數持續稀釋總評。一般股票只是「暫時查不到財報」（snapshot 為
        # None 但 status 不是 'not_applicable'）則維持原本 0 分的誠實
        # 懲罰——那種情況資料缺失本身就是一個值得注意的訊號，不該比照
        # ETF 排除。
        if etf_mode:
            scored_letters = {k: v for k, v in letters.items() if k not in ('C', 'A')}
        else:
            scored_letters = letters

        total = sum(v['score'] for v in scored_letters.values())
        max_total = sum(v['max'] for v in scored_letters.values())
        pct = (total / max_total * 100) if max_total else 0.0

        if pct >= 80:
            grade = "🟢 A（高度符合 CAN SLIM 條件）"
        elif pct >= 65:
            grade = "🟢 B（大致符合）"
        elif pct >= 45:
            grade = "🟡 C（部分符合，需個別檢視弱項）"
        else:
            grade = "🔴 D（多數條件不符合）"

        disclosure = (
            "⚠️ C（當季獲利年增率）與 A（3年EPS複合成長率）兩項使用「營收年增率」"
            "與「ROE」作為資料可得性下的代理指標，不是原版 CAN SLIM 的季度EPS定義，"
            "解讀時請把這兩項的權重打折扣，並優先參考 N/S/L/I/M 五項技術面與籌碼面評分。"
        )
        if etf_mode:
            disclosure = (
                "⚠️ 偵測到此標的為 ETF：C（獲利年增率）與 A（EPS複合成長率）兩項"
                "對 ETF 結構性不適用（ETF 本身沒有「公司獲利」），已從總分與滿分中"
                "排除，本評分僅反映 N（新高）/S（籌碼供需）/L（相對強度）/I（法人動向）/"
                "M（大盤狀態）五項技術面與籌碼面條件，滿分70分換算百分比，"
                "不是完整7項CAN SLIM評分，僅供技術面/籌碼面參考。"
            )

        return {
            'total_score': round(total, 1),
            'max_score': max_total,
            'pct': round(pct, 1),
            'grade': grade,
            'letters': letters,
            'etf_mode': etf_mode,
            'disclosure': disclosure,
        }
