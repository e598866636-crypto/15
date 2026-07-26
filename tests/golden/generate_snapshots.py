"""
Golden Dataset — Step 2：凍結目前 Engine 的輸出（Snapshot）

⚠️ 只有在你確認「目前這個版本的 Engine 輸出是你信任、可以當作基準的結果」
時才執行這支腳本 —— 它會覆蓋 tests/golden/snapshots/ 底下的既有快照。
每次要凍結新的基準版本前，建議先跑 run_golden.py 確認目前狀態，或者
直接用 git 保留舊快照的版本歷史，才不會不小心蓋掉上一個可信的基準。

用法（在 generate_golden.py 之後執行）：
    python tests/golden/generate_snapshots.py
"""
import datetime
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline import run_core_pipeline, SNAPSHOT_FIELDS  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# 影響數值計算結果的套件（rolling/ewm/quantile 等實作可能隨版本改變浮點結果），
# 只記錄真正會影響 pipeline 計算的，不編造專案沒用到的套件（例如 ta-lib）。
ENVIRONMENT_PACKAGES = ["pandas", "numpy", "scipy"]


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_environment_fingerprint():
    return {
        "python": platform.python_version(),
        **{pkg: _package_version(pkg) for pkg in ENVIRONMENT_PACKAGES},
    }

# 參與 run_core_pipeline() 的所有 Engine 檔案 —— 用 SHA256 而不是人工版本號，
# 因為專案目前沒有機制保證有人改了程式碼會記得手動升版；hash 一定誠實反映
# 「這份檔案的內容是否跟凍結 Golden 時完全一樣」。
PIPELINE_ENGINE_FILES = [
    "engines/indicator_engine.py",
    "engines/structure_engine.py",
    "engines/pattern_engine.py",
    "engines/stage_engine.py",
    "engines/risk_engine.py",
    "engines/divergence_engine.py",
    "engines/strategy_engine.py",
    "engines/momentum_engine.py",
    "engines/evidence_engine.py",
    "engines/feature_provider.py",  # 尚未接上 pipeline，但一起追蹤方便 Phase 2B 之後比對
    "tests/golden/pipeline.py",
]


def _sha256_of(relative_path):
    full_path = os.path.join(REPO_ROOT, relative_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_manifest(symbols):
    return {
        "golden_created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "symbols": symbols,
        "environment": build_environment_fingerprint(),
        "engine_file_hashes": {
            path: _sha256_of(path) for path in PIPELINE_ENGINE_FILES
        },
        "note": (
            "engine_file_hashes 是凍結當下每個檔案的 SHA256。"
            "之後跑 run_golden.py 前，可以重新算一次 hash 比對，"
            "任何差異都代表這份 Golden 基準已經不是對應目前程式碼版本，"
            "PASS 的意義要打折扣（應該重新凍結而不是照樣信任舊基準）。"
            "environment 記錄了凍結當下的 Python/pandas/numpy/scipy 版本——"
            "如果之後這些套件升版，rolling()/ewm()/quantile() 等實作可能有"
            "浮點層級的差異，run_golden.py 若 FAIL 但 engine_file_hashes 完全"
            "沒變，先檢查是不是這裡的版本不一致，而不是急著懷疑 Engine 邏輯。"
        ),
    }


def _json_safe(value):
    """把 pandas/numpy 純量轉成 json 可序列化型別，NaN 統一轉成 None。"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def build_snapshot(df):
    latest = df.iloc[-1]
    snapshot = {"_row_count": len(df), "_last_date": _json_safe(latest.get("date"))}
    for layer, cols in SNAPSHOT_FIELDS.items():
        snapshot[layer] = {}
        for col in cols:
            if col in df.columns:
                snapshot[layer][col] = _json_safe(latest[col])
            else:
                snapshot[layer][col] = "__MISSING_COLUMN__"
    return snapshot


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not raw_files:
        print(f"⚠️ {RAW_DIR} 底下沒有 .parquet 檔，請先執行 generate_golden.py")
        return

    symbols = []
    for raw_path in raw_files:
        ticker = os.path.splitext(os.path.basename(raw_path))[0]
        df = pd.read_parquet(raw_path)
        try:
            result_df = run_core_pipeline(df)
            snapshot = build_snapshot(result_df)
            out_path = os.path.join(SNAPSHOT_DIR, f"{ticker}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            print(f"{ticker}: snapshot 已寫入 {out_path}")
            symbols.append(ticker)
        except Exception as e:
            print(f"{ticker}: FAIL，pipeline 執行失敗 — {e}")

    manifest = build_manifest(symbols)
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest.json 已寫入 {manifest_path}（記錄了每個 Engine 檔案的 SHA256）")

    print("完成。建議把 tests/golden/snapshots/ + manifest.json 一起 commit 進版控，")
    print("作為未來所有重構的回歸基準。")


if __name__ == "__main__":
    main()
