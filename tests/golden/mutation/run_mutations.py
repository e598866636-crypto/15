"""
Golden Dataset — Mutation Test Suite

目的：光是「Golden Regression 現在 PASS」不能證明測試本身有效——也可能是
容差設太寬、比對的欄位不對，或根本沒跑到修改的程式碼。Mutation Test
反過來驗證：如果真的改壞一個公式，Golden Regression 保證會 FAIL。

做法：對每一種「已知的參數變更」，暫時把 Engine 原始碼改壞、跑一次
run_golden.py（用全新的 subprocess，避免 Python import cache 造成假象），
預期結果必須是 FAIL（exit code 1）。跑完不論成功或例外都會還原原始檔案。

用法（需要先有 tests/golden/raw/ + tests/golden/snapshots/，
也就是先跑過 generate_golden.py + generate_snapshots.py）：
    python tests/golden/mutation/run_mutations.py

⚠️ 這支腳本會暫時覆寫 engines/ 底下的檔案內容（跑完會自動還原）。
建議只在乾淨的 git working tree 上執行，跑之前先 git status 確認沒有
未 commit 的變更，萬一腳本中途被強制中斷，可以用 git checkout 救回來。
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 每一項：(相對路徑, 原始字串, 改壞後字串, 說明)
# 原始字串必須是該檔案裡「唯一」出現一次的內容，否則會拒絡執行（避免
# 意外改到別的地方）。
MUTATIONS = [
    (
        "engines/feature_provider.py",
        "df[col] = df[source_col].ewm(span=span, adjust=False).mean()",
        "df[col] = df[source_col].ewm(span=span + 1, adjust=False).mean()",
        "EMA 週期公式改成 span+1（EMA 邏輯已於 Phase 2B 遷移至 FeatureProvider，改測這裡）",
    ),
    (
        "engines/feature_provider.py",
        "df[col] = df[source_col].rolling(window, min_periods=min_periods).mean()",
        "df[col] = df[source_col].rolling(window + 1, min_periods=min_periods).mean()",
        "SMA 視窗公式改成 window+1（SMA 邏輯已於 Phase 2C 遷移至 FeatureProvider，改測這裡）",
    ),
    (
        "engines/feature_provider.py",
        "dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()",
        "dif = c.ewm(span=fast - 2, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()",
        "MACD 快線週期(fast)改成 fast-2（MACD 邏輯已於 Phase 2D 遷移至 FeatureProvider，改測這裡；"
        "原本這條測試指向 indicator_engine.py 的內聯公式，遷移後那段程式碼已不存在，"
        "若不更新會變成 occurrences=0 被跳過，等於這個 Mutation 沒有真的在測任何東西）",
    ),
    (
        "engines/indicator_engine.py",
        'df["atr_14"] = tr.rolling(14).mean()',
        'df["atr_14"] = tr.rolling(10).mean()',
        "ATR14 視窗改成 10",
    ),
    (
        "engines/indicator_engine.py",
        "gain = delta.clip(lower=0).rolling(14).mean()",
        "gain = delta.clip(lower=0).rolling(10).mean()",
        "RSI14 視窗改成 10（只改 gain，製造 gain/loss 視窗不一致的 bug）",
    ),
]


def run_golden_subprocess():
    """用全新的 python process 跑 run_golden.py，避免 import cache 讓修改後
    的程式碼沒有真的被重新載入、造成偽陽性的 PASS。"""
    result = subprocess.run(
        [sys.executable, os.path.join(GOLDEN_DIR, "run_golden.py")],
        cwd=GOLDEN_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def main():
    if not os.path.isdir(os.path.join(GOLDEN_DIR, "raw")) or not os.listdir(os.path.join(GOLDEN_DIR, "raw")):
        print("⚠️ tests/golden/raw/ 是空的，請先跑 generate_golden.py + generate_snapshots.py")
        sys.exit(1)

    print("=" * 60)
    print("Mutation Test Suite")
    print("=" * 60)

    results = []
    for rel_path, old_str, new_str, description in MUTATIONS:
        full_path = os.path.join(REPO_ROOT, rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        occurrences = original_content.count(old_str)
        if occurrences != 1:
            print(f"⚠️ 跳過「{description}」：old_str 在 {rel_path} 出現 {occurrences} 次（預期剛好 1 次），拒絕執行避免改錯地方。")
            results.append((description, "SKIPPED", None))
            continue

        mutated_content = original_content.replace(old_str, new_str, 1)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(mutated_content)
            returncode, stdout, stderr = run_golden_subprocess()
            detected = returncode != 0
            results.append((description, "DETECTED" if detected else "MISSED", returncode))
        finally:
            # 無論成功或例外，一律還原原始檔案內容
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(original_content)

    print()
    all_detected = True
    for description, status, returncode in results:
        marker = "✅" if status == "DETECTED" else ("⚠️" if status == "SKIPPED" else "❌")
        print(f"{marker} {description:45s} {status}")
        if status == "MISSED":
            all_detected = False

    print()
    print("=" * 60)
    if all_detected:
        print("Total: 所有 Mutation 都被 Golden Regression 抓到 PASS")
    else:
        print("Total: 有 Mutation 沒被抓到 —— Golden Regression 的覆蓋範圍或容差需要檢討 FAIL")
    print("=" * 60)
    sys.exit(0 if all_detected else 1)


if __name__ == "__main__":
    main()
