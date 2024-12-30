import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import lightgbm as lgb
import joblib
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\program\python\ML\ML_finalReport_train.csv')
test_data = pd.read_csv(r'D:\program\python\ML\ML_finalReport_test.csv')

X = train_data[['year', 'month', 'day', 'hour', 'minute', 'weekday', 'LocationCode',
                'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'month_sin', 'month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]
y = train_data['Power(mW)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features = ['LocationCode','quarter','time_of_day']

params = {
    'objective': 'regression',   # 回歸
    'metric': 'rmse',            # 評估指標
    'learning_rate': 0.01,       # 學習率
    'max_depth': 25,             # 最大樹深度
    'num_leaves': 355,           # 樹葉節點數
    'feature_fraction': 0.8,     # 每次使用部分特徵
    'bagging_fraction': 0.8,     # 每次使用部分數據
    'bagging_freq': 5,           # 每5次疊代重新抽樣數據
    'verbose': -1                # 禁止冗長輸出
}

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=lgb_train)

lgb_reg = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=[lgb_test],
    num_boost_round=2000,
    callbacks=[lgb.log_evaluation(period=50)]
)

y_pred = lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mse)
print(f'均方誤差 (MSE): {mse}')
print(f'平均絕對值誤差 (MAE): {mae}')
print(f'R² 分數: {r2}')
print(f'均方根誤差 (RMSE): {rmse}')

joblib.dump(lgb_reg, r'D:\program\python\ML\ML_finalReport_lightGBM.pkl')

Xpred = test_data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode',
             'hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]

predPower = lgb_reg.predict(Xpred)
predPower = np.clip(predPower, 0, None)

plt.figure(figsize=(20, 5))
originalPower = test_data[['Power(mW)']].values
plt.plot(originalPower, label='original', color='blue', linewidth=2)
plt.plot(predPower, label='predict', color='orange', linestyle='--', linewidth=2)
plt.title('original vs predict', fontsize=16)
plt.xlabel('index', fontsize=14)
plt.ylabel('Power(mW)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

error = abs(originalPower.flatten() - predPower)
total_error = np.sum(error)
print(f'總誤差: {total_error}')
