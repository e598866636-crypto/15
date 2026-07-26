import numpy as np
import pandas as pd


class RSRatingEngine:
    """
    🏆 相對強度評等引擎 (RS Rating Engine)

    ⚠️ 這是 v2.9.5 新增模組，補上原本 MomentumEngine 缺少的一塊：
    MomentumEngine 裡的「相對強度」用的是 RSI —— 那是股票「跟自己過去比」
    的強弱指標，不是 Minervini / IBD 系統裡真正的 RS Rating（一檔股票的
    價格表現，跟『同一批股票池裡的其他股票』比較後的百分位排名）。這兩者
    容易被誤認為同一件事，這裡刻意用不同名稱（RS Rating vs RSI）並列，
    避免混淆。

    ⚠️ 誠實揭露（重要，請務必閱讀）：
    IBD 官方的 RS Rating 是拿一檔股票的表現去跟「全美/全市場所有掛牌股票」
    比較，母體是幾千檔股票。這個引擎受限於資料來源（僅能對 scanner_engine
    當次掃描的股票池計算），排名母體是「這次掃描名單」，不是全市場。
    股票池越小，排名的統計意義越弱（例如只掃 20 檔，排名前 5 名不代表在
    全市場也是前段班）。回傳結果一律標注 `universe_size`，UI 端應該把這個
    數字一起顯示出來，不要讓使用者誤以為這是官方全市場 RS Rating。

    方法論（簡化版 IBD 加權動能）：
        raw_score = 40% × 63日報酬 + 20% × 126日報酬 + 20% × 189日報酬 + 20% × 252日報酬
    越近期的窗口權重越高（跟 IBD 官方公式的精神一致：近期表現更重要），
    資料不足 252 日的股票，會用可取得的最長窗口等比例重新分配權重
    （而不是直接排除，興櫃/新股常常不到一年資料）。
    """

    _WINDOWS_WEIGHTS = [(63, 0.40), (126, 0.20), (189, 0.20), (252, 0.20)]

    @staticmethod
    def compute_raw_score(df: pd.DataFrame) -> dict:
        """
        對單一股票計算「原始加權動能分數」（尚未跟其他股票比較排名）。

        回傳 None 或 status != 'ok' 代表資料不足，呼叫端應該把該股票排除在
        排名母體之外，而不是硬塞一個 0 分（0 分在百分位排名裡會被誤讀成
        「表現最差」，但實際上只是「資料不足」）。
        """
        if df is None or 'close' not in df.columns or len(df) < 20:
            return {'status': 'insufficient_data', 'raw_score': np.nan}

        close = df['close']
        current = float(close.iloc[-1])

        weighted_sum = 0.0
        weight_used = 0.0
        detail = {}

        for window, weight in RSRatingEngine._WINDOWS_WEIGHTS:
            if len(close) > window:
                past = float(close.iloc[-1 - window])
                if past > 0:
                    ret_pct = (current - past) / past * 100
                    detail[f'return_{window}d_pct'] = round(ret_pct, 2)
                    weighted_sum += ret_pct * weight
                    weight_used += weight

        if weight_used == 0:
            return {'status': 'insufficient_data', 'raw_score': np.nan}

        # 資料不足一年時，把用得到的窗口權重等比例放大回 100%，
        # 而不是讓分數被「資料不足」系統性拉低。
        raw_score = weighted_sum / weight_used

        return {'status': 'ok', 'raw_score': raw_score, 'detail': detail,
                'windows_used': int(weight_used * 100)}

    @staticmethod
    def rank_universe(raw_scores: dict) -> dict:
        """
        輸入 {ticker: raw_score}（只放 status=='ok' 的股票），
        回傳 {ticker: {'rs_rating': 1~99, 'percentile': 0~100, 'universe_size': n}}。

        RS Rating 採用 IBD 慣例的 1~99 分制（而非 0~100），百分位排名後
        線性映射到 1~99。母體只有 1 檔股票時無法排名，回傳 rs_rating=None。
        """
        valid = {k: v for k, v in raw_scores.items() if pd.notna(v)}
        n = len(valid)
        if n < 2:
            return {k: {'rs_rating': None, 'percentile': None, 'universe_size': n} for k in raw_scores}

        tickers = list(valid.keys())
        scores = np.array([valid[t] for t in tickers])
        # 百分位排名：分數越高，百分位越高（越接近 99）
        ranks = pd.Series(scores).rank(pct=True).values  # 0~1

        result = {}
        for t, pct in zip(tickers, ranks):
            rs_rating = int(np.clip(round(pct * 98) + 1, 1, 99))
            result[t] = {'rs_rating': rs_rating, 'percentile': round(float(pct) * 100, 1),
                         'universe_size': n}

        # 資料不足以計算 raw_score 的股票，明確標示，不要跟著給假排名
        for k in raw_scores:
            if k not in result:
                result[k] = {'rs_rating': None, 'percentile': None, 'universe_size': n}

        return result

    @staticmethod
    def grade_from_rating(rs_rating) -> str:
        """把 1~99 的 RS Rating 轉成 IBD 慣用的文字分級，方便 UI 顯示。"""
        if rs_rating is None or pd.isna(rs_rating):
            return "N/A（母體不足或資料不足）"
        if rs_rating >= 90:
            return "🟢 極強（前10%，Minervini/IBD 選股門檻通常要求 ≥80~90）"
        if rs_rating >= 80:
            return "🟢 強勢（前20%）"
        if rs_rating >= 70:
            return "🟡 中上"
        if rs_rating >= 50:
            return "🟡 中等"
        return "🔴 弱勢（後段班，落後掃描池內大多數股票）"
