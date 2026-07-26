"""
Golden Dataset — 核心 Pipeline（單一事實來源）

⚠️ 重要：這個檔案的 run_core_pipeline() 必須手動跟 app.py 第 1013~1024 行
（df = DataEngine.get_stock_data(...) 之後那一段 try 區塊）保持一致。
這裡刻意只複製「純 df-in-df-out、不需要網路/外部報告」的那 9 個 Engine：

    IndicatorEngine.add_indicators
    StructureEngine.add_swing_points
    PatternEngine.add_patterns
    StageEngine.add_stage_analysis
    RiskEngine.add_risk_metrics
    RiskEngine.add_liquidity_metrics
    DivergenceEngine.add_defense_signals
    StrategyEngine.generate_signals   ← ai_score / market_regime 在這裡產生
    MomentumEngine.add_momentum_score
    EvidenceEngine.add_evidence

刻意不包含的層（原因：需要即時網路資料，凍結它們會讓回歸測試失去意義，
只會凍結「當天抓到的籌碼/基本面資料」，不是「Engine 邏輯」本身）：
    ChipEngine / FundamentalEngine（即時籌碼、財報）
    BreakoutEngine.analyze / CanslimEngine.analyze（需要 chip_report/theme_score）
    RSRatingEngine（需要跨股票 watchlist 排名，且 MomentumEngine 會用到）
    DecisionEngine.build_consensus（需要 canslim_report + breakout_report）

如果之後 app.py 這段流水線的順序或參數改變，這裡也要跟著更新，
否則 Golden Dataset 驗證的就不是「現在的 app.py 實際在跑的邏輯」。
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.indicator_engine import IndicatorEngine
from engines.structure_engine import StructureEngine
from engines.pattern_engine import PatternEngine
from engines.stage_engine import StageEngine
from engines.risk_engine import RiskEngine
from engines.divergence_engine import DivergenceEngine
from engines.strategy_engine import StrategyEngine
from engines.momentum_engine import MomentumEngine
from engines.evidence_engine import EvidenceEngine


def run_core_pipeline(df):
    """輸入原始 OHLCV df，依 app.py 相同順序跑過九個核心 Engine，回傳完整 df。"""
    df = df.copy()
    df = IndicatorEngine.add_indicators(df)
    df = StructureEngine.add_swing_points(df)
    df = PatternEngine.add_patterns(df)
    df = StageEngine.add_stage_analysis(df)
    df = RiskEngine.add_risk_metrics(df)
    df = RiskEngine.add_liquidity_metrics(df)
    df = DivergenceEngine.add_defense_signals(df)
    df = StrategyEngine.generate_signals(df)
    df = MomentumEngine.add_momentum_score(df)
    df = EvidenceEngine.add_evidence(df)
    return df


# ==========================================
# Snapshot 欄位分層（對應 doc 建議的五層報告；Decision/Risk 層
# 目前只能取「核心 pipeline 內」算得出來的部分，不含需要外部報告的
# consensus / breakout / canslim）
# ==========================================
SNAPSHOT_FIELDS = {
    "feature": [
        "ema_8", "ema_21", "sma_20", "sma_60", "sma_120", "sma_200",
        "rsi_14", "macd_dif", "macd_dea", "macd_hist", "atr_14",
        "obv", "obv_sma", "k_9", "d_9", "mtm", "pagoda_trend",
        "bb_upper", "bb_lower",
    ],
    "structure_pattern_stage": [
        "stage", "stage_label", "stage_note",
    ],
    "evidence": [
        "data_quality_pct", "confidence_pct", "confidence_label",
    ],
    "decision_partial": [
        "market_regime", "ai_score", "entry_signal", "exit_signal",
        "stop_loss", "target_1", "target_2",
    ],
    "risk": [
        "volatility_annualized", "drawdown_pct", "rolling_mdd_60d",
        "var_95_pct", "var_99_pct",
    ],
    "momentum": [
        "momentum_score", "momentum_grade", "momentum_score_complete",
    ],
}

# 每個欄位的比對容差，從 tolerance.yaml 讀取（Tolerance Registry，獨立設定檔，
# 不需要改程式碼就能調整容差）。float 用數值容差，null／None 要求完全相等。
def _load_tolerance_registry():
    import yaml
    tolerance_path = os.path.join(os.path.dirname(__file__), "tolerance.yaml")
    with open(tolerance_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("fields", {}), config.get("default_float_tolerance", 1e-9)


FIELD_TOLERANCE, DEFAULT_FLOAT_TOLERANCE = _load_tolerance_registry()
