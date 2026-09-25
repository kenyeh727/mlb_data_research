# MLB Whiff Prediction — 揮空率預測

用 MLB Statcast 逐球追蹤數據，預測「打者揮棒會不會揮空」。End-to-end 數據科學專案：數據抓取 → 特徵工程 → 建模 → 視覺化。

## 問題定義

給定一顆打者**出棒**的球，預測它會不會揮空（swinging strike / foul tip = 1）。

## 數據

- 來源：MLB Statcast（經由 `pybaseball`，官方免費數據）
- 範圍：2026-09-01 至 2026-09-21，共 80,851 球，其中 38,429 次揮棒
- 基準揮空率：25.1%

## 方法

特徵：球速、轉速、位移（pfx_x/pfx_z）、進壘點（plate_x/plate_z）、距好球帶中心距離、球數、球種、打者慣用手等。

| 模型 | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression | 0.721 | 0.534 |
| Gradient Boosting | **0.747** | **0.555** |

最重要的特徵：距好球帶中心距離、進壘高度、垂直位移、球數、球速。

## 主要發現

- Splitter（35.6%）和 Slider（32.8%）最容易讓打者揮空；Sinker（13.4%）最低（以製造滾地球為主）
- 好球帶邊緣與好球帶外的球揮空率最高——「追打壞球」是揮空的主因
- 兩好球後揮空率明顯上升

## 執行方式

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py
```

## 檔案結構

```
src/
  data.py    # Statcast 抓取 + 快取 + whiff 資料集建構
  model.py   # 特徵管線 + 模型訓練 + 評估
  plots.py   # 視覺化
run_experiment.py  # 一鍵執行全流程
figures/     # 輸出圖表
data/        # 快取的 parquet 數據
```

## 下一步可延伸

- 加入投手/打者層級隨機效應（mixed-effects model）
- 用 GAM 畫球種-位置的非線性揮空曲面
- 預測擊球初速/預期打擊率（xwOBA 方向）
- 投手 clustering：找出球路型態相似的投手
