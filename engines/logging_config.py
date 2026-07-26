"""
engines/logging_config.py

📝 集中式 Logging 設定

⚠️ 為什麼需要這個：
這次開發過程中，除錯的模式一直是「使用者點擊按鈕 → 截圖 Streamlit 顯示的錯誤
訊息 → 貼給開發者」，這個循環雖然最終都能定位問題，但每次都要重現操作、
反覆截圖，效率不高。如果失敗當下就寫進本地 log 檔案，之後排查問題可以直接
翻 log，不需要使用者每次都重新操作一次來重現錯誤。

使用方式：
    from engines.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("...")
    logger.warning("...")
    logger.exception("...")   # 在 except 區塊內用，會自動附上完整 traceback

⚠️ 設計取捨：
  1. 這裡用 Python 內建的 logging + RotatingFileHandler（單檔最大5MB，
     保留3份備份），不需要額外安裝套件，也不需要外部日誌服務。
  2. Streamlit 每次互動都會重新執行整支 app.py，若不做防護，同一個
     logger 每次重跑都會被重複加上 handler，導致同一則訊息被寫入log
     檔案好幾次。這裡用 `if logger.handlers: return logger` 防止重複
     加入 handler。
  3. Log 檔案位置：專案根目錄下的 `logs/tqai.log`（跟 `data/tqai.db`
     同一層概念，都是「執行期產生的資料」，不是原始碼，不需要納入版本
     控制；若使用 git，建議把 `logs/` 加進 `.gitignore`）。
  4. 這裡刻意不記錄任何個人身分資訊或敏感資料——只記錄技術層面的錯誤
     細節（例外類型、股票代碼、呼叫的功能名稱），股票代碼本身不是
     敏感資訊。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "tqai.log")


def get_logger(name: str) -> logging.Logger:
    """
    取得一個設定好的 logger。同一個 name 重複呼叫只會設定一次 handler
    （見上方 class docstring 的 Streamlit rerun 防護說明）。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception:
        # ⚠️ 連 log 都寫不進去（例如檔案系統唯讀、權限問題），不應該讓這個
        # 問題反過來讓主程式崩潰——logging 本身失敗時安靜放棄，退回
        # Python 預設的 logger 行為（不寫檔案，但呼叫 logger.xxx() 也不會
        # 拋例外中斷呼叫端）。
        pass

    return logger
