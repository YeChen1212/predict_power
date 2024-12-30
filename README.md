# predict_power
CYUT Machine Learning Class Final Report for AI CUP 2024 Fall: Predicting Solar Energy Power
## 簡介
使用 Linear Regression 、LightGBM及Multi-layer Perceptron Regressor來預測太陽能發電量，由於AICUP官方還沒有給最後解答，所以本篇以資料集內每個月的三十日的早上九點到下午三點來當作預測的目標並比較這三種模型所預測出的結果跟實際結果。
## Runtime Environment
python3.11.0, numpy1.26.4, pandas2.2.3, scikit-learn1.5.2, joblib1.4.2, matplotlib3.9.2, lightgbm4.5.0
## Dataset
- 來源
  - AICUP官網 L1~L17_Train.csv 以及 L2,4,7,8,9,10,12_Train_2.csv
- 資料集
  - ML_finalReport_train.csv : 訓練的資料
  - ML_finalReport_test.csv : 預測的資料
- 腳本
  - ML_finalReport_traindata.py
