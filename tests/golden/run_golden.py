"""
Golden Dataset — Step 3：回歸測試（每次改動 Engine 後執行這支）

用法：
    python tests/golden/run_golden.py

流程：讀取 tests/golden/raw/{ticker}.parquet（凍結的原始資料，不重新連網），
重新跑一次 run_core_pipeline()，跟 tests/golden/snapshots/{ticker}.json
（凍結的基準輸出）逐欄位比對。

Exit code：全部 PASS 回傳 0，任何一項 FAIL 回傳 1（方便串進 CI 或
pre-commit hook）。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline import (  # noqa: E402
    run_core_pipeline,
    SNAPSHOT_FIELDS,
    FIELD_TOLERANCE,
    DEFAULT_FLOAT_TOLERANCE,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def _json_safe(value):
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


def compare_field(col, expected, actual):
    """回傳 (is_pass: bool, detail: str)"""
    if expected is None and actual is None:
        return True, "None == None"
    if expected == "__MISSING_COLUMN__" or actual == "__MISSING_COLUMN__":
        if expected == actual:
            return True, "皆缺欄位（一致）"
        return False, f"欄位缺失狀態改變: golden={expected!r} now={actual!r}"

    tol = FIELD_TOLERANCE.get(col, DEFAULT_FLOAT_TOLERANCE)
    if tol is None:
        ok = expected == actual
        return ok, ("完全一致" if ok else f"golden={expected!r} now={actual!r}")

    # 數值容差比對
    try:
        if expected is None or actual is None:
            return expected == actual, f"golden={expected!r} now={actual!r}"
        diff = abs(float(expected) - float(actual))
        ok = diff <= tol
        return ok, f"diff={diff:.10g} (tol={tol}) golden={expected} now={actual}"
    except (TypeError, ValueError):
        ok = expected == actual
        return ok, f"golden={expected!r} now={actual!r}"


def run_for_ticker(ticker, raw_path, snapshot_path):
    df = pd.read_parquet(raw_path)
    with open(snapshot_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    result_df = run_core_pipeline(df)
    latest = result_df.iloc[-1]

    layer_results = {}
    overall_pass = True
    for layer, cols in SNAPSHOT_FIELDS.items():
        layer_pass = True
        field_details = []
        for col in cols:
            expected = golden.get(layer, {}).get(col, "__MISSING_COLUMN__")
            actual = _json_safe(latest[col]) if col in result_df.columns else "__MISSING_COLUMN__"
            ok, detail = compare_field(col, expected, actual)
            field_details.append((col, ok, detail))
            layer_pass = layer_pass and ok
        layer_results[layer] = (layer_pass, field_details)
        overall_pass = overall_pass and layer_pass

    return overall_pass, layer_results


def print_manifest_diff():
    """資訊性質，不影響 PASS/FAIL：列出哪些 Engine 檔案的 SHA256 跟凍結
    Golden 當下不同了。這是預期會發生的事（Phase 2B 本來就是要改 Engine），
    只是讓你清楚知道「這次 PASS/FAIL 是在哪些檔案已經變動的前提下跑出來的」。"""
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    from generate_snapshots import _sha256_of  # noqa: E402 (延用同一份 hash 邏輯)

    changed = []
    for rel_path, golden_hash in manifest.get("engine_file_hashes", {}).items():
        current_hash = _sha256_of(rel_path)
        if current_hash != golden_hash:
            changed.append(rel_path)

    print(f"Golden 凍結時間: {manifest.get('golden_created_utc', '未知')}")

    golden_env = manifest.get("environment", {})
    if golden_env:
        from generate_snapshots import build_environment_fingerprint  # noqa: E402
        current_env = build_environment_fingerprint()
        env_diffs = {
            k: (golden_env.get(k), current_env.get(k))
            for k in golden_env
            if golden_env.get(k) != current_env.get(k)
        }
        if env_diffs:
            print("⚠️ 執行環境跟凍結 Golden 當下不同（可能造成浮點層級的計算差異，")
            print("   跟 Engine 邏輯本身無關）：")
            for pkg, (golden_v, current_v) in env_diffs.items():
                print(f"  ~ {pkg}: golden={golden_v} current={current_v}")
        else:
            print("執行環境（Python/pandas/numpy/scipy）跟凍結 Golden 當下一致。")

    if changed:
        print("自 Golden 凍結後，以下 Engine 檔案已變動（預期中的重構才會有變動）：")
        for path in changed:
            print(f"  ~ {path}")
    else:
        print("自 Golden 凍結後，所有追蹤的 Engine 檔案都沒有變動。")
    print()


def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not raw_files:
        print(f"⚠️ {RAW_DIR} 底下沒有 .parquet 檔，請先執行 generate_golden.py + generate_snapshots.py")
        sys.exit(1)

    print("=" * 60)
    print("Golden Regression")
    print("=" * 60)
    print_manifest_diff()

    all_pass = True
    for raw_path in raw_files:
        ticker = os.path.splitext(os.path.basename(raw_path))[0]
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"{ticker}.json")
        if not os.path.exists(snapshot_path):
            print(f"{ticker}: ⚠️ 找不到 snapshot，請先執行 generate_snapshots.py")
            all_pass = False
            continue

        ok, layer_results = run_for_ticker(ticker, raw_path, snapshot_path)
        all_pass = all_pass and ok
        print(f"\n--- {ticker} ---")
        for layer, (layer_pass, field_details) in layer_results.items():
            status = "PASS" if layer_pass else "FAIL"
            print(f"{layer:24s} {status}")
            if not layer_pass:
                for col, field_ok, detail in field_details:
                    if not field_ok:
                        print(f"    ✗ {col}: {detail}")

    print("\n" + "=" * 60)
    print(f"Total: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
