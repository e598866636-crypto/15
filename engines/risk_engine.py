import numpy as np
import pandas as pd


class RiskEngine:
    """
    🛡️ Risk Center 風險中心

    補齊規劃文件中的風險量化模組：
      - Volatility      年化波動率
      - Maximum Drawdown 滾動最大回撤
      - Beta             相對大盤（加權指數）的系統性風險
      - VaR              歷史法風險值 (95% / 99%)
      - Reward/Risk Ratio 報酬風險比（依 ATR 停損停利推算）
      - Liquidity        流動性風險（成交值代理，v2.9.7 新增，見下方說明）

    ⚠️ v2.9.7 新增（以專業投資角度覆核既有引擎後，補上的一塊實務缺口）：
    原本這裡涵蓋的都是「價格波動」風險（波動率、回撤、Beta、VaR），完全
    沒有「流動性」風險——這在台股中小型股/興櫃股是實務上很關鍵的一塊：
    一檔股票就算技術面/基本面訊號再好，成交值太低時，「進得去、出不來」
    本身就是風險（想停利/停損時，市價單可能大幅打到不利價位，甚至根本
    沒有對手盤）。這裡用「近N日平均成交值」（收盤價×成交量，新台幣）
    當作流動性的代理指標——不是完整的市場微結構分析（買賣價差、委託簿
    深度都拿不到，yfinance 沒有這些資料），純粹是「量能規模」層次的粗略
    防線，但已經足以濾掉最危險的極端案例（例如日均成交值不到幾百萬元的
    冷門股），且已知這個代理指標的意義：它衡量的是「歷史上平均有多少人
    在交易」，不是「你想賣的當下實際能用多少價格成交」，兩者不完全等同。
    """

    # ==========================================
    # 1. 向量化逐日風險指標（接在 IndicatorEngine 之後皆可呼叫）
    # ==========================================
    @staticmethod
    def add_risk_metrics(df: pd.DataFrame, vol_window: int = 20, mdd_window: int = 60, var_window: int = 252):
        df = df.copy()
        returns = df['close'].pct_change()

        # 年化波動率：近 vol_window 日報酬標準差 × √252
        df['volatility_annualized'] = returns.rolling(vol_window, min_periods=5).std() * np.sqrt(252) * 100

        # 滾動最大回撤：近 mdd_window 日，相對期間內高點的最大跌幅
        rolling_max = df['close'].rolling(mdd_window, min_periods=1).max()
        drawdown_pct = (df['close'] - rolling_max) / rolling_max * 100
        df['drawdown_pct'] = drawdown_pct
        df['rolling_mdd_60d'] = drawdown_pct.rolling(mdd_window, min_periods=1).min()

        # 歷史法 VaR：近 var_window 日報酬分布的 5% / 1% 分位數
        min_periods = min(60, var_window)
        df['var_95_pct'] = returns.rolling(var_window, min_periods=min_periods).quantile(0.05) * 100
        df['var_99_pct'] = returns.rolling(var_window, min_periods=min_periods).quantile(0.01) * 100

        return df

    # ==========================================
    # 1b. 流動性風險 (Liquidity Risk) — v2.9.7 新增
    # ==========================================
    # 台股實務門檻參考（非官方硬性標準，屬業界常用經驗法則，已標明來源
    # 為經驗法則而非監管定義）：
    #   日均成交值 < 5,000萬元：流動性偏低，大額買賣容易顯著影響價格
    #   日均成交值 < 1,000萬元：流動性極低，一般建議避開或僅用極小部位試單
    _LIQUIDITY_LOW_NTD = 50_000_000
    _LIQUIDITY_VERY_LOW_NTD = 10_000_000

    @staticmethod
    def add_liquidity_metrics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        加入近 window 日平均成交值（新台幣）與流動性等級欄位。
        需求欄位：close, volume（DataEngine 已提供）。volume 單位若為「股數」
        則成交值單位為新台幣元；若上游資料源改用「張數」，這裡的絕對數字
        門檻需要對應調整——目前假設沿用 yfinance 回傳的股數單位（與本專案
        其他地方一致），不在此重新驗證上游單位假設。
        """
        df = df.copy()
        if 'close' not in df.columns or 'volume' not in df.columns:
            df['avg_trading_value_20d'] = np.nan
            df['liquidity_level'] = "資料不足"
            return df

        trading_value = df['close'] * df['volume']
        df['avg_trading_value_20d'] = trading_value.rolling(window, min_periods=5).mean()

        df['liquidity_level'] = np.select(
            [df['avg_trading_value_20d'] < RiskEngine._LIQUIDITY_VERY_LOW_NTD,
             df['avg_trading_value_20d'] < RiskEngine._LIQUIDITY_LOW_NTD],
            ["🔴 極低流動性", "🟡 流動性偏低"],
            default="🟢 流動性正常"
        )
        return df

    # ==========================================
    # 2. Beta：個股相對大盤（加權指數）的系統性風險係數
    # ==========================================
    @staticmethod
    def compute_beta(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, window: int = 120):
        """
        ⚠️ 修正說明：個股資料若最後一筆是「盤中即時估計值」
        （DataEngine 用即時報價覆蓋收盤價），但大盤基準資料
        （get_benchmark_data）沒有做同樣的即時覆蓋，兩邊最後一天
        的「新鮮程度」會不對稱（個股是盤中價，大盤是前一天收盤價），
        這樣算出來的相關係數會失真。這裡改成：若個股資料有
        is_intraday_estimate 標記，計算 Beta 時排除掉那一筆尚未
        收盤定案的資料，確保兩邊都是正式收盤價再做比較。
        """
        if stock_df is None or benchmark_df is None or stock_df.empty or benchmark_df.empty:
            return np.nan
        if 'date' not in stock_df.columns or 'date' not in benchmark_df.columns:
            return np.nan

        stock_df = stock_df.copy()
        if 'is_intraday_estimate' in stock_df.columns:
            stock_df = stock_df[~stock_df['is_intraday_estimate'].fillna(False).astype(bool)]

        s = stock_df[['date', 'close']].rename(columns={'close': 'stock_close'})
        b = benchmark_df[['date', 'close']].rename(columns={'close': 'bench_close'})
        merged = pd.merge(s, b, on='date', how='inner').sort_values('date')
        merged['stock_ret'] = merged['stock_close'].pct_change()
        merged['bench_ret'] = merged['bench_close'].pct_change()
        merged = merged.dropna()

        if len(merged) < 20:
            return np.nan

        recent = merged.tail(window)
        cov_matrix = np.cov(recent['stock_ret'], recent['bench_ret'])
        bench_var = np.var(recent['bench_ret'])

        if bench_var == 0 or np.isnan(bench_var):
            return np.nan
        return float(cov_matrix[0][1] / bench_var)

    # ==========================================
    # 3. 整合風險報告（給最新一筆使用，供 Dashboard 顯示）
    # ==========================================
    @staticmethod
    def build_risk_report(df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        latest = df.iloc[-1]
        close = float(latest['close'])
        atr = latest.get('atr_14', np.nan)

        stop_loss = latest.get('stop_loss', np.nan)
        target_1 = latest.get('target_1', np.nan)
        if pd.isna(stop_loss) and pd.notna(atr):
            stop_loss = close - 1.5 * atr
        if pd.isna(target_1) and pd.notna(atr):
            target_1 = close + 2.0 * atr

        risk_amt = close - stop_loss if pd.notna(stop_loss) else np.nan
        reward_amt = target_1 - close if pd.notna(target_1) else np.nan
        rr_ratio = (reward_amt / risk_amt) if (pd.notna(reward_amt) and pd.notna(risk_amt) and risk_amt > 0) else np.nan

        beta = RiskEngine.compute_beta(df, benchmark_df) if benchmark_df is not None else np.nan

        avg_trading_value = latest.get('avg_trading_value_20d', np.nan)
        liquidity_level = latest.get('liquidity_level', "資料不足")

        report = {
            'volatility_annualized': latest.get('volatility_annualized', np.nan),
            'max_drawdown_60d': latest.get('rolling_mdd_60d', np.nan),
            'var_95_pct': latest.get('var_95_pct', np.nan),
            'var_99_pct': latest.get('var_99_pct', np.nan),
            'beta': beta,
            'reward_risk_ratio': rr_ratio,
            'avg_trading_value_20d': avg_trading_value,
            'liquidity_level': liquidity_level,
        }

        # 風險等級提示旗標
        flags = []
        vol = report['volatility_annualized']
        mdd = report['max_drawdown_60d']

        if pd.notna(mdd) and mdd < -20:
            flags.append("⚠️ 近 60 日最大回撤超過 20%，下檔風險偏高")
        if pd.notna(beta) and beta > 1.5:
            flags.append(f"⚠️ Beta = {beta:.2f}，相對大盤波動劇烈（系統性風險高）")
        if pd.notna(beta) and beta < 0.5 and beta > -10:
            flags.append(f"ℹ️ Beta = {beta:.2f}，相對大盤連動性低（防禦型）")
        if pd.notna(rr_ratio) and rr_ratio < 1.0:
            flags.append(f"⚠️ 報酬風險比 {rr_ratio:.2f} < 1，潛在虧損大於潛在獲利")
        if pd.notna(vol) and vol > 60:
            flags.append(f"⚠️ 年化波動率 {vol:.1f}%，波動劇烈，建議降低部位")
        # v2.9.7 新增：流動性風險提示（見 add_liquidity_metrics 說明）
        if pd.notna(avg_trading_value) and avg_trading_value < RiskEngine._LIQUIDITY_VERY_LOW_NTD:
            flags.append(f"🔴 近20日日均成交值僅約 {avg_trading_value/1e6:.0f} 百萬元，流動性極低，"
                         f"買賣容易顯著影響價格，建議避開或僅用極小部位試單")
        elif pd.notna(avg_trading_value) and avg_trading_value < RiskEngine._LIQUIDITY_LOW_NTD:
            flags.append(f"🟡 近20日日均成交值約 {avg_trading_value/1e6:.0f} 百萬元，流動性偏低，"
                         f"大額買賣建議分批，並預留較大的滑價緩衝")

        if not flags:
            flags.append("✅ 各項風險指標均處於合理範圍")

        report['risk_flags'] = flags

        # 綜合風險等級
        risk_points = 0
        if pd.notna(mdd) and mdd < -20:
            risk_points += 1
        if pd.notna(beta) and beta > 1.5:
            risk_points += 1
        if pd.notna(rr_ratio) and rr_ratio < 1.0:
            risk_points += 1
        if pd.notna(vol) and vol > 60:
            risk_points += 1
        # v2.9.7：流動性極低單獨計 1 點；「偏低」不計點（避免過度懲罰中型股，
        # 僅在文字說明提醒），刻意跟其他四項用同樣的「一項=1點」邏輯疊加，
        # 不做額外加權，維持既有風險等級公式的簡單、可解釋性。
        if pd.notna(avg_trading_value) and avg_trading_value < RiskEngine._LIQUIDITY_VERY_LOW_NTD:
            risk_points += 1

        if risk_points >= 3:
            report['risk_level'] = "🔴 高風險"
        elif risk_points >= 1:
            report['risk_level'] = "🟡 中度風險"
        else:
            report['risk_level'] = "🟢 低風險"

        return report

    # ==========================================
    # 4. ATR 部位配置 (ATR-based Position Sizing)
    # ==========================================
    # ⚠️ v2.9.5 新增：原本 RiskEngine 只有用 ATR 算「停損/停利價位」，卻沒有
    # 用 ATR 反推「這筆交易應該買多少股／多少部位」——停損價位算得再精準，
    # 沒有搭配部位大小，總風險暴露還是不可控（停損抓對了，但買太多股一樣
    # 會爆賠）。這裡補上「先定義每筆交易願意虧多少錢，再回推部位大小」的
    # 標準做法。
    @staticmethod
    def compute_atr_position_size(account_equity: float, close: float, atr: float,
                                   risk_pct_per_trade: float = 1.0, atr_multiplier: float = 2.0,
                                   max_position_pct: float = 100.0) -> dict:
        """
        依 ATR 停損距離反推部位大小，讓每筆交易的「最大可能虧損」固定為
        帳戶淨值的一個百分比，而不是每筆都用同樣股數/同樣資金比例進出
        （後者會讓波動大的股票承擔不成比例的風險）。
        """
        if pd.isna(atr) or atr <= 0 or pd.isna(close) or close <= 0 or account_equity <= 0:
            return {'status': 'insufficient_data', 'shares': 0,
                    'note': '⚠️ 資料不足（缺 ATR/現價/帳戶淨值），無法計算部位大小'}

        stop_distance = atr * atr_multiplier
        risk_amount = account_equity * (risk_pct_per_trade / 100.0)
        raw_shares = risk_amount / stop_distance

        position_value = raw_shares * close
        max_position_value = account_equity * (max_position_pct / 100.0)
        capped = position_value > max_position_value
        if capped:
            position_value = max_position_value
            raw_shares = position_value / close

        return {
            'status': 'ok',
            'shares': int(raw_shares),
            'position_value': round(position_value, 0),
            'position_pct_of_equity': round(position_value / account_equity * 100, 1),
            'stop_distance': round(stop_distance, 2),
            'risk_amount': round(risk_amount, 0),
            'capped_by_max_position_pct': capped,
            'note': (f"以單筆最大虧損 {risk_pct_per_trade}% 帳戶淨值、{atr_multiplier}x ATR 停損距離反推，"
                     f"若停損確實在 {atr_multiplier}x ATR 觸發，實際虧損約為帳戶淨值的 {risk_pct_per_trade}%"
                     + ("（已受 max_position_pct 上限封頂，實際風險暴露會低於設定的risk_pct_per_trade）" if capped else "")),
        }

    # ==========================================
    # 5. Kelly 準則部位配置 (Kelly Criterion Position Sizing)
    # ==========================================
    # ⚠️ v2.9.5 新增：Kelly 準則需要「勝率」與「平均賺賠比」這兩個歷史統計量，
    # 本引擎刻意不自己重新統計交易紀錄，而是直接吃 BacktestEngine.run_backtest()
    # 算出來的 summary，避免同一組統計邏輯在兩個引擎裡各自實作、可能兜不攏。
    @staticmethod
    def compute_kelly_fraction(win_rate_pct: float, avg_win_pct: float, avg_loss_pct: float,
                                kelly_fraction_cap: float = 0.5) -> dict:
        """
        標準 Kelly 公式：f* = W - (1-W)/R
            W = 勝率（0~1）；R = 平均獲利% / 平均虧損%（賺賠比，恆為正數）

        ⚠️ 重要警告（務必在 UI 顯示）：
        1. 完整 Kelly（f*）在真實市場常常過度激進——它假設歷史勝率/賺賠比
           會準確重演，樣本數不足或未來與過去統計特性不同時，容易造成
           過度下注。業界慣例是用「半凱利」甚至更保守，本方法預設用
           kelly_fraction_cap=0.5 封頂，f* 為負值時一律不建議進場。
        2. 這是「建議動用的資金比例上限」，不是「一定要動用這麼多」——
           實務上通常還會再疊加 ATR 部位配置的風險上限，兩者取更保守的一個。
        """
        if pd.isna(win_rate_pct) or pd.isna(avg_win_pct) or pd.isna(avg_loss_pct) or avg_loss_pct <= 0:
            return {'status': 'insufficient_data', 'kelly_pct': 0.0,
                    'note': '⚠️ 資料不足（需要歷史勝率與平均賺賠比，建議先跑 BacktestEngine）'}

        w = win_rate_pct / 100.0
        r = avg_win_pct / avg_loss_pct if avg_loss_pct != 0 else np.nan
        if pd.isna(r) or r <= 0:
            return {'status': 'insufficient_data', 'kelly_pct': 0.0, 'note': '⚠️ 賺賠比資料無效'}

        f_full = w - (1 - w) / r
        f_capped = max(0.0, min(f_full, 1.0)) * kelly_fraction_cap

        if f_full <= 0:
            note = "🔴 Kelly 公式算出負值，代表依歷史勝率/賺賠比，這個策略的『數學期望值』不足以支撐下注，建議不進場或重新檢視策略。"
        else:
            note = (f"歷史統計：勝率 {win_rate_pct:.1f}%、賺賠比 {r:.2f}。"
                    f"全凱利建議動用 {f_full*100:.1f}% 資金，此處採用 {kelly_fraction_cap:.0%} 的保守係數"
                    f"（半凱利或更保守），實際建議部位為 {f_capped*100:.1f}% 資金。")

        return {
            'status': 'ok',
            'full_kelly_pct': round(f_full * 100, 1),
            'kelly_pct': round(f_capped * 100, 1),
            'win_rate_pct': win_rate_pct,
            'win_loss_ratio': round(r, 2),
            'kelly_fraction_cap': kelly_fraction_cap,
            'note': note,
        }

    # ==========================================
    # 6. 吊燈出場 (Chandelier Exit)
    # ==========================================
    # ⚠️ v2.9.5 新增：跟 StrategyEngine 現有的固定倍數 ATR 停損（進場時就
    # 訂死 stop_loss，之後不再往上調整）不同，Chandelier Exit 是「移動式」
    # 停損——停損價會隨著持倉期間的新高持續上移，讓獲利部位可以放大盈利，
    # 同時把已經到手的獲利鎖住一部分，可與現有的固定 ATR 停損並列比較使用，
    # 而非取代。
    @staticmethod
    def add_chandelier_exit(df: pd.DataFrame, window: int = 22, atr_multiplier: float = 3.0) -> pd.DataFrame:
        """
        多頭吊燈出場 = 近 window 日最高價 - atr_multiplier × ATR(14)。
        需求欄位：high, atr_14（需已跑過 IndicatorEngine）。資料不足時該欄位
        為 NaN，不會拋出例外（沿用本專案一貫的防禦設計）。
        """
        df = df.copy()
        if 'high' not in df.columns or 'atr_14' not in df.columns:
            df['chandelier_exit_long'] = np.nan
            return df

        highest_high = df['high'].rolling(window, min_periods=1).max()
        df['chandelier_exit_long'] = highest_high - atr_multiplier * df['atr_14']
        return df