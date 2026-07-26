import pandas as pd
import numpy as np

class StructureEngine:
    """
    📐 市場結構引擎 - TQAI Pro 高階版
    修正原版 ZigZag 演算法在單邊趨勢中會產生的「連續標記漂移」問題。
    採用標準的 Peak/Trough 探索機制：在同方向動能中動態更新極值，在反向偏離時確立轉折。

    ⚠️ 重要限制（Repainting Warning）：
    ZigZag 的本質是「事後才確認」的轉折指標 —— 某根 K 棒是否為高/低點，
    要等到價格反轉超過 deviation 之後才會回頭確立。這代表：
      1. 圖表上看到的最後一段 zigzag 線，隨時可能因為新資料進來而被「重畫」
         (repaint)，不是最終定案的結構。
      2. 【絕對不可】把 zigzag / zigzag_ffill 當作特徵餵給 StrategyEngine 或
         BacktestEngine 用來產生交易訊號 —— 這樣做等於讓策略在訓練/回測時
         看到了「未來才會確認」的轉折點，構成嚴重的前視偏誤 (look-ahead bias)，
         回測績效會不合理地被高估。
      3. 此引擎僅供「視覺化圖表輔助判讀」使用，本專案目前的 pipeline
         （見 ScannerEngine._run_single_pipeline）沒有把 zigzag 接進
         StrategyEngine，維持這個邊界是刻意設計，未來擴充時請勿破壞它。
    """
    @staticmethod
    def add_swing_points(df: pd.DataFrame, deviation: float = 0.04):
        df = df.copy()
        if len(df) < 3:
            df['zigzag'] = np.nan
            df['zigzag_ffill'] = df['close'].ffill()
            return df
            
        c = df['close']
        pivots_idx = [0]
        pivots_val = [c.iloc[0]]
        
        # 追蹤當前尋找狀態： 0 = 初始判定, 1 = 尋找波段最高點, -1 = 尋找波段最低點
        state = 0 
        
        for i in range(1, len(df)):
            current_price = c.iloc[i]
            last_pivot_val = pivots_val[-1]
            dev = (current_price - last_pivot_val) / (last_pivot_val + 1e-9)
            
            if state == 0:
                # 初始階段：看哪邊先突破偏離度
                if dev > deviation:
                    state = 1  # 確立向上，開始尋找最高點
                    pivots_idx.append(i)
                    pivots_val.append(current_price)
                elif dev < -deviation:
                    state = -1 # 確立向下，開始尋找最低點
                    pivots_idx.append(i)
                    pivots_val.append(current_price)
                    
            elif state == 1:
                # 尋找波段高點中：
                if current_price > last_pivot_val:
                    # 價格更高，動態「更新」當前高點的位置與數值
                    pivots_idx[-1] = i
                    pivots_val[-1] = current_price
                elif dev < -deviation:
                    # 從最高點回檔超過設定值，轉折確立！確認高點，並轉向尋找低點
                    state = -1
                    pivots_idx.append(i)
                    pivots_val.append(current_price)
                    
            elif state == -1:
                # 尋找波段低點中：
                if current_price < last_pivot_val:
                    # 價格更低，動態「更新」當前低點的位置與數值
                    pivots_idx[-1] = i
                    pivots_val[-1] = current_price
                elif dev > deviation:
                    # 從最低點反彈超過設定值，轉折確立！確認低點，並轉向尋找高點
                    state = 1
                    pivots_idx.append(i)
                    pivots_val.append(current_price)
                    
        # 記錄哪些 pivot 是真正被「反轉確立」的轉折（confirmed），
        # 用來跟下面強制補上的最後一筆（僅供 ffill 有基底、不代表轉折已確立）區分開。
        confirmed_flags = [True] * len(pivots_idx)

        # 強制將最後一個交易日納入，確保即時資料有基底做 ffill 比較
        # 注意：這一筆不是「轉折已確立」，只是讓 zigzag_ffill 在最新資料上有值可畫，
        # 隨時可能因為明天的新資料而被覆蓋/重畫（repaint）。
        if pivots_idx[-1] != len(df) - 1:
            pivots_idx.append(len(df) - 1)
            pivots_val.append(c.iloc[-1])
            confirmed_flags.append(False)
        else:
            # 若最後一筆本身就是動態更新中的極值（state 1 或 -1 尚未反轉），
            # 同樣視為未確認
            confirmed_flags[-1] = False

        df['zigzag'] = np.nan
        df['zigzag_confirmed'] = pd.array([pd.NA] * len(df), dtype="boolean")
        # 使用 iloc 绝对位置賦值，徹底根除 Index 錯位或 SettingWithCopy 警告
        df.iloc[pivots_idx, df.columns.get_loc('zigzag')] = pivots_val
        df.iloc[pivots_idx, df.columns.get_loc('zigzag_confirmed')] = confirmed_flags
        df['zigzag_ffill'] = df['zigzag'].ffill()

        return df