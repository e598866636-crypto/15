import pandas as pd


class DecisionEngine:
    """
    🧭 決策共識層 (Decision Consensus Engine)

    ⚠️ v2.9.6 新增，範圍界定（請務必先讀）：
    這是對「AI Score 應該是彙總結果，而不是黑箱單一數字」這個訴求的
    **誠實、可落地版本**，不是完整的「多代理 AI 決策系統」。它做的事情
    很單純：把本專案既有引擎（已經算好的市場狀態、CAN SLIM、RS Rating、
    Stage、Breakout 評分、Risk 等級）讀出來，用**寫死的規則**（不是機器
    學習、不是LLM推理、不會自我學習調整權重）組成一個「共識儀表板」：
    幾個維度是偏多、幾個偏空、有沒有互相矛盾。

    這裡刻意不做、也不建議在這個專案基礎上做的事情（附上原因，而不是
    只說「以後再做」）：
      - 「AI 自動學習優化權重」的迴圈：需要大量已標記結果的歷史交易
        紀錄與嚴謹的樣本外驗證框架，沒有這些基礎前先做「自動調整」，
        很容易變成對雜訊過擬合，反而比固定規則更不可靠。
      - 自動加碼/減碼/下單執行（Trade Manager 的執行層）：這已經涉及
        真實資金的自動化操作，需要券商API整合、額外的權限與錯誤處理
        機制，风险與既有「產生建議、人工決定」的設計完全不同量級，
        不應該在沒有額外討論配套風控的情況下直接加進去。
      - 多套 LLM Agent 互相辯論：本專案 strategy_engine.py 的 Bull/Bear/
        Risk「Agent」是規則式評分函式，不是真的獨立 LLM 推理，這裡沿用
        同樣的誠實原則，不假裝這是真正的多代理 AI 系統。
      - 把整個共識儀表板改成單一「0-100分固定權重加總」（例如技術面
        30%+籌碼25%+基本面20%+新聞15%+市場環境10%）：這種寫法會製造
        虛假的精確感——這些權重通常是主觀拍板，沒有經過樣本外驗證，
        「93分」聽起來很精確，但背後只是幾個數字相加，使用者容易誤以為
        這是嚴謹的量化結果。維持「幾個維度同意/不同意」的共識票數，
        比硬湊一個分數更誠實，也更容易看出「哪個維度在唱反調」。

    ⚠️ v2.9.13 更新：新聞面（NewsEngine）已經接入共識儀表板，見下方
    build_consensus() 的 news_report 參數與第11點投票邏輯。舊版
    docstring 曾寫「本專案沒有新聞資料源，硬做這塊需要額外串接」，這句
    話已經過時（NewsEngine 用 Google新聞RSS 免費/免金鑰做到了），這裡
    一併修正，避免文件跟程式碼對不上。

    這個引擎能誠實提供的價值：把分散在好幾個頁面的訊號，整理成一個
    「這幾個維度同意/不同意」的總表，方便快速判斷要不要再深入看，
    不是自動化交易決策。

    ⚠️ v2.9.7 新增（以專業投資角度覆核既有引擎後，補上「已經算好但沒接進
    共識儀表板」的缺口，不是新的資料源）：
      - 籌碼面（三大法人合計買賣超方向）：ChipEngine 早就算好，但共識層
        原本完全沒讀取，導致「籌碼」這個台股實務上很重要的維度被排除在
        共識判斷之外。
      - 產業輪動：SectorRotationEngine 同樣早就存在，但只在「產業輪動」
        頁面獨立顯示，沒有接回個股的共識判斷。
      - 流動性：新增為第二個「否決」維度（跟風險等級同一邏輯）——極低
        流動性的股票就算其他訊號再一致偏多，也可能因為進出場價格滑價過大
        而不適合交易，這是可交易性的硬限制，不是可以被其他訊號抵銷的
        方向性判斷。
      - 總經背景：只做「附帶顯示」，刻意不計入偏多/偏空票數（理由見
        build_consensus 的 macro_flags 參數說明），避免全市場共同的環境
        變數被誤讀為「這檔個股的總經面訊號」。

    ⚠️ v2.9.9 新增：
      - 公司治理（董監事設質比例）：ChipEngine.get_insider_holdings() 早就
        算好設質比例與紅旗判斷，但同樣沒接回共識儀表板。這裡加入「公司
        治理」維度，設質比例過高時計為偏空——這不是股價方向的技術訊號，
        是「這家公司未來會不會因為內部人資金壓力而發生非經營面的籌碼/
        股價衝擊」的治理風險，跟 CAN SLIM/RS Rating 這種價格/財報訊號
        性質不同，但同樣值得放進共識判斷；資料每月才更新一次，不是即時，
        且是相關性指標而非因果判斷，這點會在票數旁的文字說明中一併揭露。
    """

    @staticmethod
    def build_consensus(latest: pd.Series, canslim_report: dict = None, breakout_report: dict = None,
                         rs_rating=None, risk_level: str = None, chip_report: dict = None,
                         sector_signal: dict = None, macro_flags: list = None,
                         liquidity_level: str = None, insider_report: dict = None,
                         news_report: dict = None) -> dict:
        """
        參數皆為既有引擎已經算好的輸出，這個方法不重新計算任何指標，
        純粹讀取與彙整：
            latest           單一股票最新一列（含 market_regime, stage_label,
                             ai_score, momentum_grade 等既有欄位）
            canslim_report   CanslimEngine.analyze() 的回傳值
            breakout_report  BreakoutEngine.analyze() 的回傳值
            rs_rating        RSRatingEngine 排名結果的 rs_rating（1~99）
            risk_level       latest 對應的 risk_report['risk_level']（字串）
            chip_report      ChipEngine.build_chip_report() 的回傳值（v2.9.7 新增）
            sector_signal    該股所屬產業在 SectorRotationEngine 輪動表中的
                             一列（dict，需含 'signal' 欄位），沒有值代表
                             使用者尚未在「產業輪動」頁面執行過計算，或該股
                             不在本次輪動計算的名單/產業分類中（v2.9.7 新增）
            macro_flags      MacroEngine.build_macro_flags() 的回傳值（list of
                             str）。⚠️ 刻意不當作獨立投票維度：總經背景是
                             全市場共同的環境變數，不是這檔個股獨有的訊號，
                             跟其他維度用同一套加減票邏輯混在一起會誤導成
                             「這檔股票的總經面很強/很弱」，這裡只原樣附帶
                             在輸出裡供顯示，不計入 bullish/bearish 計數
                             （v2.9.7 新增）
            liquidity_level  latest 對應的 risk_report['liquidity_level']
                             （字串，v2.9.7 新增，跟風險等級一樣走「否決」
                             邏輯，見下方說明）
            insider_report   ChipEngine.get_insider_holdings() 的回傳值
                             （v2.9.9 新增）。需要使用者已在個股頁面按過
                             「查詢董監事持股與設質狀況」按鈕才會有值，
                             沒有值時如實顯示「資料不足」，不會假裝查過。
            news_report      NewsEngine.get_news_with_sentiment() 的回傳值
                             （v2.9.13 新增）。需要使用者已在「📰 新聞情緒
                             中心」展開區按過「抓取最新相關新聞並分析情緒」
                             才會有值，沒有值時如實顯示「資料不足」。
                             ⚠️ 只用規則式加權分數(weighted_bias_score)的
                             正負號判斷方向，不管LLM摘要的星等——LLM摘要
                             是選用功能，投票邏輯必須在沒接LLM時也能穩定
                             運作。
        """
        votes = {}

        # 1. 市場狀態
        market_regime = latest.get('market_regime', '') if latest is not None else ''
        if '多頭' in str(market_regime):
            votes['市場狀態'] = ('偏多', 1)
        elif '空頭' in str(market_regime):
            votes['市場狀態'] = ('偏空', -1)
        else:
            votes['市場狀態'] = ('中性', 0)

        # 2. Stage Analysis
        stage_label = str(latest.get('stage_label', '')) if latest is not None else ''
        if 'Stage 2' in stage_label:
            votes['階段分析'] = ('偏多', 1)
        elif 'Stage 4' in stage_label:
            votes['階段分析'] = ('偏空', -1)
        else:
            votes['階段分析'] = ('中性', 0)

        # 3. RS Rating
        if rs_rating is not None and pd.notna(rs_rating):
            if rs_rating >= 80:
                votes['相對強度'] = ('偏多', 1)
            elif rs_rating < 50:
                votes['相對強度'] = ('偏空', -1)
            else:
                votes['相對強度'] = ('中性', 0)
        else:
            votes['相對強度'] = ('資料不足', 0)

        # 4. CAN SLIM
        if canslim_report and canslim_report.get('pct') is not None:
            pct = canslim_report['pct']
            if pct >= 65:
                votes['CAN SLIM'] = ('偏多', 1)
            elif pct < 45:
                votes['CAN SLIM'] = ('偏空', -1)
            else:
                votes['CAN SLIM'] = ('中性', 0)
        else:
            votes['CAN SLIM'] = ('資料不足', 0)

        # 5. Breakout（VCP/突破結構）
        if breakout_report and 'error' not in breakout_report:
            grade = str(breakout_report.get('grade', ''))
            if any(g in grade for g in ['A', 'S']):
                votes['突破結構'] = ('偏多', 1)
            elif 'D' in grade or 'F' in grade:
                votes['突破結構'] = ('偏空', -1)
            else:
                votes['突破結構'] = ('中性', 0)
        else:
            votes['突破結構'] = ('資料不足', 0)

        # 6. 籌碼面（三大法人買賣超方向，v2.9.7 新增）
        # ⚠️ 只用「三大法人合計淨買賣」判斷方向，不單獨拆外資/投信/自營商，
        # 避免跟 CAN SLIM 的 I（法人認養）項目在細節上重複判斷、卻可能給出
        # 不一致結論；融資融券（散戶槓桿）刻意不放進這個方向性投票——那是
        # 「風險/情緒」屬性的資料，不是清楚的多空方向，見 chip_engine.py
        # get_margin_trend() 的雙向解讀說明，只在 UI 端另外呈現，不在這裡
        # 算成一票。
        if chip_report and chip_report.get('status') == 'ok' and chip_report.get('institutional'):
            total_net = chip_report['institutional'].get('total_net')
            if total_net is not None and total_net > 0:
                votes['籌碼面'] = ('偏多', 1)
            elif total_net is not None and total_net < 0:
                votes['籌碼面'] = ('偏空', -1)
            else:
                votes['籌碼面'] = ('中性', 0)
        else:
            votes['籌碼面'] = ('資料不足', 0)

        # 7. 產業輪動（該股所屬產業目前資金動能方向，v2.9.7 新增）
        # ⚠️ 這是「產業」層級的訊號，不是個股專屬——同產業的股票會拿到
        # 相同的輪動投票結果，這是刻意的（輪動本來就是產業層級的概念），
        # 但也代表這一票的「獨立資訊量」比其他個股專屬維度低，解讀共識
        # 程度時可將這點納入考量。sector_signal 需要使用者先在「產業輪動」
        # 頁面按下計算按鈕才會有值，見 sector_rotation_engine.py。
        if sector_signal and sector_signal.get('status') == 'ok':
            # 對應 SectorRotationEngine.rank_rotation() 的「輪動訊號」欄位文字
            # （🔄 疑似資金輪入 / 📤 疑似資金輪出 / ➖ 排名相對穩定 / ℹ️ 資料不足），
            # 這裡直接比對關鍵字，兩邊文字修改時需同步維護。
            signal_text = str(sector_signal.get('signal', ''))
            if '輪入' in signal_text:
                votes['產業輪動'] = ('偏多', 1)
            elif '輪出' in signal_text:
                votes['產業輪動'] = ('偏空', -1)
            elif '資料不足' in signal_text:
                votes['產業輪動'] = ('資料不足', 0)
            else:
                votes['產業輪動'] = ('中性', 0)
        else:
            votes['產業輪動'] = ('資料不足', 0)

        # 10. 公司治理 — 董監事/內部人設質比例（v2.9.9 新增）
        # ⚠️ 這不是價格/財報訊號，是「內部人資金壓力」的治理風險：設質比例
        # 過高時，一旦股價大跌，內部人可能被要求補提擔保品或被斷頭賣股，
        # 形成與公司經營面無關的籌碼/股價衝擊。這裡計為偏空票，但權重
        # 邏輯上跟「風險等級/流動性」的否決邏輯不同——不做成否決，因為
        # 這是相關性指標、每月才更新一次、且「設質高不代表一定有問題」
        # （ChipEngine 自己的揭露原則），沒有強到可以否決其他所有訊號。
        if insider_report and insider_report.get('status') == 'ok' and insider_report.get('max_pledge_pct') is not None:
            pledge_pct = insider_report['max_pledge_pct']
            if pledge_pct >= 50:
                votes['公司治理'] = (f'偏空（設質{pledge_pct:.0f}%）', -1)
            elif pledge_pct >= 20:
                votes['公司治理'] = (f'中性偏謹慎（設質{pledge_pct:.0f}%）', 0)
            else:
                votes['公司治理'] = ('健康', 0)
        else:
            votes['公司治理'] = ('資料不足', 0)

        # 11. 新聞面（v2.9.13 新增）
        # ⚠️ 只用規則式加權分數的正負號判斷方向，票數只有1（跟其他維度
        # 同等權重），不因為「新聞很多篇」就加重這一票——見class docstring
        # 對「固定權重加總」的反對理由，這裡刻意維持「一個維度一票」的
        # 設計，不做成新聞面單獨佔15%這種寫法。
        if news_report and news_report.get('status') in ('ok', 'empty') and news_report.get('summary_stats'):
            _wscore = news_report['summary_stats'].get('weighted_bias_score', 0)
            _conf = news_report['summary_stats'].get('avg_event_confidence')
            _conf_str = f"，事件可信度{_conf:.0%}" if _conf is not None else ""
            if _wscore > 1:
                votes['新聞面'] = (f'偏多（加權分數{_wscore}{_conf_str}）', 1)
            elif _wscore < -1:
                votes['新聞面'] = (f'偏空（加權分數{_wscore}{_conf_str}）', -1)
            else:
                votes['新聞面'] = ('中性', 0)
        else:
            votes['新聞面'] = ('資料不足', 0)

        # 8. 風險等級（風險是「否決」維度，不是加分維度——高風險不會因為
        #    其他維度偏多就被抵銷，這是刻意設計，不是統計上的加權平均）
        risk_veto = False
        if risk_level and '🔴' in str(risk_level):
            votes['風險等級'] = ('高風險（否決）', -1)
            risk_veto = True
        elif risk_level and '🟡' in str(risk_level):
            votes['風險等級'] = ('中度風險', 0)
        else:
            votes['風險等級'] = ('可接受', 0)

        # 9. 流動性（v2.9.7 新增，跟風險等級一樣是「否決」維度——一檔股票
        #    就算其他所有維度都偏多，極低流動性代表實際上「進得去、出不來」，
        #    這是可交易性的硬限制，不是可以被其他偏多訊號抵銷的方向性判斷）
        liquidity_veto = False
        if liquidity_level and '🔴' in str(liquidity_level):
            votes['流動性'] = ('極低流動性（否決）', -1)
            liquidity_veto = True
        elif liquidity_level and '🟡' in str(liquidity_level):
            votes['流動性'] = ('流動性偏低', 0)
        else:
            votes['流動性'] = ('正常', 0)

        _veto_dims = ('風險等級', '流動性')
        score_values = [v[1] for k, v in votes.items() if k not in _veto_dims]
        bullish_count = sum(1 for v in score_values if v == 1)
        bearish_count = sum(1 for v in score_values if v == -1)
        neutral_count = sum(1 for v in score_values if v == 0)
        total_dims = len(score_values)

        net_score = sum(score_values)
        agreement_pct = round(max(bullish_count, bearish_count) / total_dims * 100, 0) if total_dims else 0

        any_veto = risk_veto or liquidity_veto
        if risk_veto and liquidity_veto:
            decision = "🔴 NO TRADE（風險等級與流動性同時否決，不論其他維度結果）"
        elif risk_veto:
            decision = "🔴 NO TRADE（風險等級否決，不論其他維度結果）"
        elif liquidity_veto:
            decision = "🔴 NO TRADE（流動性極低否決——訊號再好也可能進得去出不來）"
        elif net_score >= 3 and bullish_count >= 4:
            decision = "🟢 BUY READY（多維度一致偏多，可進一步檢視進場時機）"
        elif net_score <= -3:
            decision = "🔴 AVOID（多維度一致偏空）"
        elif bullish_count > 0 and bearish_count > 0 and abs(bullish_count - bearish_count) <= 1:
            decision = "🟡 WATCH（訊號分歧，各維度意見不一致，不建議貿然進場）"
        else:
            decision = "🟡 WATCH（訊號中性或證據不足）"

        return {
            'decision': decision,
            'votes': votes,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_dims': total_dims,
            'agreement_pct': agreement_pct,
            'risk_veto': risk_veto,
            'liquidity_veto': liquidity_veto,
            'any_veto': any_veto,
            'macro_context': macro_flags or [],
            'disclosure': (
                "⚠️ 這是「規則式共識儀表板」，不是機器學習或LLM推理的決策系統，"
                "每個維度的判斷邏輯都是上方程式碼裡寫死的固定規則，不會自我學習調整；"
                "任何一項『資料不足』代表對應引擎沒有算出結果，不是真的中性訊號，"
                "解讀共識程度時請把資料不足的維度排除在外看待；「產業輪動」是產業層級"
                "而非個股專屬訊號，「總經背景」僅供參考顯示、不計入偏多/偏空計數，"
                "整體共識不構成投資建議。"
            ),
        }
