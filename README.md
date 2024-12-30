# predict_power
CYUT Machine Learning Class Final Report for AI CUP 2024 Fall: 根據區域微氣候資料預測發電量競賽
## 簡介
使用 Linear Regression 、LightGBM及Multi-layer Perceptron Regressor來預測太陽能發電量，由於AICUP官方還沒有給最後解答，所以本篇以資料集內每個月的三十日的早上九點到下午三點來當作預測的目標並比較這三種模型所預測出的結果跟實際結果。
## Runtime Environment
- Python Version: 3.11.0
- NumPy: 1.26.4
- Pandas: 2.2.3
- Scikit-learn: 1.5.2
- Joblib: 1.4.2
- Matplotlib: 3.9.2
- LightGBM: 4.5.0
## Dataset
- 來源
  - AICUP官網 L1~L17_Train.csv 以及 L2,4,7,8,9,10,12_Train_2.csv
- 資料集
  - ML_finalReport_train.csv : 訓練的資料
  - ML_finalReport_test.csv : 預測的資料
- 特徵(處理前)
  - `LocationCode`: 位置
  - `DateTime`: 日期時間
  - `WindSpeed(m/s)`: 風速
  - `Pressure(hpa)`: 氣壓
  - `Temperature(°C)`: 氣溫
  - `Humidity(%)`: 濕度
  - `Sunlight(Lux)`: 光照強度
  - `Power(mW)`: 發電量
- 特徵(處理後)
  - LocationCode
  - DateTime
  - WindSpeed(m/s)
  - Pressure(hpa)
  - Temperature(°C),
  - Humidity(%)
  - Sunlight(Lux)
  - Power(mW)
  - year
  - month
  - day
  - hour
  - minute
  - weekday
  - hour_sin
  - hour_cos
  - minute_sin
  - minute_cos
  - month_sin
  - month_cos
  - quarter
  - day_of_year
  - week_of_year
  - hour_squared
  - time_of_day
- 資料筆數及維度
  - 處理前 : (1375028, 8)
  - 處理後
    - 訓練集 : (1335199, 25)
    - 測試集 : (20393, 25)
- 腳本 : `ML_finalReport_traindata.py`
- 步驟
  1. 剔除不合理的數據（如缺失值、異常值等）。
  2. 拆分 `DateTime`將其轉換為 `year`, `month`, `day`, `hour`, `minute`, `weekday`。
  3. 利用拆分的時間特徵，計算時間的週期性特徵：`hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`, `month_sin`, `month_cos`。
  4. 增加其他時間周期性特徵：`quarter`, `day_of_year`, `week_of_year`, `hour_squared`, `time_of_day`。
  5. 將每月30日的9點至15點的數據存入 `test.csv`，其他數據存入 `train.csv` 進行訓練。
## Train
  - **Linear Regression**
    - 腳本 : `ML_finalReport_linear.py`
    - 結果 :
      
  ![linear](images/linear.png)
  - **LightGBM**
    - 腳本 : `ML_finalReport_lightgbm.py`
    - 結果 :
      
  ![lightgbm](images/LightGBM.png)
  - **Multi-layer Perceptron Regressor**
    - 腳本 : `ML_finalReport_mlp.py`
    - 結果 :

  ![lightgbm](images/MLP.png)
  - **LightGBM stacking**
    - 腳本 : `ML_finalReport_lightgbm_stack.py`
    - 結果 :
      
  ![lightgbm](images/LightGBM_stacking.png)
## 結論
  - 四個模型比較

      | 指標 | linear  | LightGBM  | MLP  | stacking  |
      |:--------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
      | 均方誤差 (MSE) | 151088.971 | 22723.869  | 52031.444 | 17736.343 |
      | 平均絕對值誤差 (MAE) | 256.771 | 67.911 | 131.756 | 51.626 |
      | R² 分數 | 0.244  | 0.886 | 0.739 | 0.911 |
      | 均方根誤差 (RMSE) | 388.701 | 150.744 | 228.104 | 133.177 |
      | 總發電量(9842244mW) | 總誤差(9280692mW) | 總誤差(5967945mW) | 總誤差(8951724mW) | 總誤差(5913228mW) |
  
  - LightGBM_stacking的結果最好，以此方法實際去比賽AICUP秋季賽根據區域微氣候資料預測發電量競賽。
    - Private Leaderboard 22 名。
    - Private Leaderboard 成績(總誤差)619536.9。
