import numpy as np
import pandas as pd
import yfinance as yf


class ETFEngine:
    """
    📊 ETF 專屬分析引擎 (ETF-Specific Analysis Engine) - TQAI Pro v2.9.4

    背景與動機：
    FundamentalEngine 的 EPS/本益比/ROE 等公司基本面指標對 ETF（一籃子
    股票的組合）沒有意義，偵測到 ETF 後會回傳 status='not_applicable'——
    但這樣使用者查詢 ETF 時，個股分析頁面的「基本面」區塊會直接消失，
    沒有提供任何「ETF 該看什麼」的替代資訊，等於功能上的空洞。這個引擎
    補上這一塊。

    ⚠️ 專業風控角度的核心提醒（槓桿/反向ETF的「波動耗損」，這是散戶最
    容易誤解、卻很少有工具主動提醒的地方）：
    槓桿型ETF（例如代碼結尾L、名稱含「正2」）與反向型ETF（代碼結尾R、
    名稱含「反1」）是「每日」重新平衡到目標槓桿倍數，不是「長期」維持
    固定倍數。這代表在震盪盤（標的指數上下來回，但一段時間後淨變化不大）
    中，槓桿/反向ETF長期持有的實際報酬，通常會低於「大盤報酬 × 槓桿倍數」
    這種直覺推算的期望值，這個現象稱為「波動耗損 (volatility decay /
    beta slippage)」，是每日重新平衡機制下的數學複利效果，不是ETF公司的
    錯、也不是詐騙。但很多散戶並不清楚這件事，容易把槓桿ETF當成「長期
    加倍賺大盤」的存股工具持有，這正是專業風控角度認為最需要主動示警的
    地方——本引擎偵測到槓桿/反向ETF時一定會顯示這個提醒。

    ⚠️ 誠實揭露（資料來源限制，務必先讀完再使用）：
      1. ETF 的類別/規模/配息率/費用率等中繼資料完全依賴 yfinance 的
         `Ticker.info`，Yahoo Finance 對台股 ETF 這類欄位的覆蓋率不一定
         完整，缺值一律顯示「資料不足」，不假造、不用其他ETF的數字頂替。
      2. 沒有淨值(NAV)即時資料，因此**無法計算折溢價 (premium/discount
         to NAV)**——這是機構評估ETF很重要的一項指標（尤其是流動性較差
         或連結海外標的的ETF，折溢價可能明顯偏離），但需要即時NAV資料源，
         yfinance 對台股ETF不提供這個欄位，本引擎老實承認做不到，不是
         忽略這個維度的重要性。
      3. 沒有追蹤誤差 (tracking error) 資料——需要ETF淨值歷史與標的指數
         歷史對照計算，同樣受限於免費資料源，暫不支援。
      4. 技術面（IndicatorEngine/StrategyEngine）、籌碼面（三大法人買賣超,
         ChipEngine）分析，ETF 本身跟一般股票走同一套既有引擎——ETF 在
         集中市場的交易方式跟股票相同，這些既有分析本來就適用，不需要
         另外重做，本引擎只補「基本面」被判定不適用之後空出來的那塊。
      5. 槓桿/反向的判斷是用代碼字母尾碼與中文名稱關鍵字比對，不是讀取
         官方的槓桿倍數分類欄位——多數台股槓桿/反向ETF確實遵循「代碼結尾
         L=槓桿、R=反向」與「名稱含正2/反1」的市場慣例，但不保證涵蓋所有
         邊緣案例，如有疑問請以公開說明書為準。
    """

    _LEVERAGE_KEYWORDS = ["正2", "正3", "槓桿"]
    _INVERSE_KEYWORDS = ["反1", "反2", "反向"]

    @staticmethod
    def classify_etf_type(code: str, name: str = "") -> dict:
        """依代碼字母尾碼與中文名稱關鍵字，判斷是否為槓桿/反向ETF（見
        class docstring 限制第5點）。"""
        code = str(code).strip().upper()
        name = name or ""

        is_leveraged = code.endswith("L") or any(kw in name for kw in ETFEngine._LEVERAGE_KEYWORDS)
        is_inverse = code.endswith("R") or any(kw in name for kw in ETFEngine._INVERSE_KEYWORDS)

        if is_leveraged:
            return {"type": "leveraged", "label": "槓桿型 ETF"}
        elif is_inverse:
            return {"type": "inverse", "label": "反向型 ETF"}
        return {"type": "plain", "label": "一般型 ETF"}

    @staticmethod
    def get_etf_info(stock_code: str) -> dict:
        """
        用 yfinance 的 Ticker.info 取得 ETF 中繼資料（類別/規模/配息率/
        費用率）。任何欄位缺值都顯示 None，呼叫端不應假設一定有值——
        Yahoo Finance 對台股ETF的欄位覆蓋率明顯不如美股ETF完整。
        """
        code = str(stock_code).split(".")[0].strip()

        info = {}
        resolved = None
        try:
            for suffix in [".TW", ".TWO"]:
                candidate = code + suffix
                tkr = yf.Ticker(candidate)
                try:
                    candidate_info = tkr.info if hasattr(tkr, "info") else {}
                except Exception:
                    candidate_info = {}
                if candidate_info and (
                    candidate_info.get("totalAssets") is not None
                    or candidate_info.get("category") is not None
                    or candidate_info.get("longName") is not None
                ):
                    info = candidate_info
                    resolved = candidate
                    break
        except Exception:
            info = {}

        if not info:
            return {
                "status": "unavailable",
                "message": "⚠️ 暫時無法取得ETF中繼資料（yfinance對台股ETF的欄位覆蓋率有限，或近期服務異常）。",
            }

        def g(key):
            return info.get(key)

        yield_raw = g("yield")
        ytd_raw = g("ytdReturn")
        expense_raw = g("annualReportExpenseRatio")

        return {
            "status": "ok",
            "resolved_symbol": resolved,
            "long_name": g("longName"),
            "category": g("category"),
            "fund_family": g("fundFamily"),
            "total_assets": g("totalAssets"),
            "yield_pct": (yield_raw * 100) if yield_raw is not None else None,
            "ytd_return_pct": (ytd_raw * 100) if ytd_raw is not None else None,
            "expense_ratio_pct": (expense_raw * 100) if expense_raw is not None else None,
        }

    @staticmethod
    def build_etf_report(stock_code: str, etf_name: str = "") -> dict:
        """
        整合方法：判斷槓桿/反向類型 + 抓中繼資料 + 組成附帶專業風控提醒
        的報告，供 app.py 顯示。任何內部失敗都優雅降級，不拋例外中斷
        呼叫端。
        """
        code = str(stock_code).split(".")[0].strip()

        try:
            etf_type = ETFEngine.classify_etf_type(code, etf_name)
        except Exception:
            etf_type = {"type": "plain", "label": "一般型 ETF"}

        try:
            info = ETFEngine.get_etf_info(code)
        except Exception as e:
            info = {"status": "unavailable", "message": f"⚠️ ETF中繼資料查詢時發生錯誤：{e}"}

        flags = []
        if etf_type["type"] in ("leveraged", "inverse"):
            direction = "正向兩倍" if etf_type["type"] == "leveraged" else "反向"
            flags.append(
                f"🔴 這是{etf_type['label']}（每日重新平衡至{direction}曝險）："
                f"槓桿/反向倍數是「每日」重設，不是「長期」固定——在震盪盤（標的指數上下"
                f"來回、一段時間後淨變化不大）中，長期持有的實際報酬通常會低於"
                f"「大盤報酬×倍數」的直覺推算，這個現象稱為「波動耗損」，是每日重新平衡"
                f"機制下的數學複利效果，不是ETF公司的問題。這類ETF設計上比較適合"
                f"「短期／戰術性」操作，不是傳統「長期持有分散風險」的存股工具，長期持有"
                f"前務必理解這個機制，不要只看槓桿倍數就假設長期報酬會等比例放大。"
            )
        else:
            flags.append("ℹ️ 一般型ETF，沒有每日重新平衡的槓桿/反向機制，長期持有的複利耗損疑慮較低。")

        if info.get("status") == "ok":
            if info.get("expense_ratio_pct") is not None:
                flags.append(
                    f"總管理費用率約 {info['expense_ratio_pct']:.2f}%/年，長期持有會直接侵蝕報酬，"
                    f"建議跟同類型ETF比較費用率高低。"
                )
            if info.get("category"):
                flags.append(f"yfinance分類：{info['category']}" + (f"（{info['fund_family']}）" if info.get("fund_family") else ""))
        else:
            flags.append(info.get("message", "⚠️ ETF中繼資料暫時無法取得。"))

        flags.append(
            "⚠️ 本分析沒有淨值(NAV)資料，無法計算折溢價；也沒有追蹤誤差資料——這兩項是"
            "機構評估ETF的重要指標，但受限於免費資料源，本引擎老實承認做不到，不是忽略"
            "其重要性。技術面／籌碼面（三大法人買賣超）分析請參考本頁面其他區塊——ETF本身"
            "跟一般股票用同一套市場交易，既有引擎同樣適用，不需要另外重做。"
        )

        return {
            "status": "ok",
            "etf_type": etf_type,
            "info": info,
            "flags": flags,
        }
