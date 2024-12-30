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

lgb_reg1 = lgb.train(
    params=params,
    train_set=lgb_train,             
    valid_sets=[lgb_test],           
    num_boost_round=1500,             
    callbacks=[lgb.log_evaluation(period=50)]
)

y_pred1 = lgb_reg1.predict(X_test, num_iteration=lgb_reg1.best_iteration)

X_train_meta = X_train.copy()
X_test_meta = X_test.copy()
X_train_meta['pred1'] = lgb_reg1.predict(X_train)
X_test_meta['pred1'] = y_pred1

lgb_train_meta = lgb.Dataset(X_train_meta, label=y_train, categorical_feature=cat_features)
lgb_test_meta = lgb.Dataset(X_test_meta, label=y_test, categorical_feature=cat_features, reference=lgb_train_meta)

lgb_reg2 = lgb.train(
    params=params,
    train_set=lgb_train_meta,             
    valid_sets=[lgb_test_meta],           
    num_boost_round=700,             
    callbacks=[lgb.log_evaluation(period=50)]
)

y_pred2 = lgb_reg2.predict(X_test_meta, num_iteration=lgb_reg2.best_iteration)

mse = mean_squared_error(y_test, y_pred2)
mae = mean_absolute_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)
rmse = sqrt(mse)
print(f'均方誤差 (MSE): {mse}')
print(f'平均絕對值誤差 (MAE): {mae}')
print(f'R² 分數: {r2}')
print(f'均方根誤差 (RMSE): {rmse}')

joblib.dump(lgb_reg1, r'D:\program\python\ML\ML_finalReport_lightGBM_stack1.pkl')
joblib.dump(lgb_reg2, r'D:\program\python\ML\ML_finalReport_lightGBM_stack2.pkl')

Xpred = test_data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode',
             'hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]

predPower1 = lgb_reg1.predict(Xpred)
predPower1 = np.clip(predPower1, 0, None)

X_meta = Xpred.copy()
X_meta['pred1'] = predPower1

predPower2 = lgb_reg2.predict(X_meta)
predPower2 = np.clip(predPower2, 0, None)

plt.figure(figsize=(20, 5))
originalPower = test_data[['Power(mW)']].values
plt.plot(originalPower, label='original', color='blue', linewidth=2)
plt.plot(predPower2, label='predict', color='orange', linestyle='--', linewidth=2)
plt.title('original vs predict', fontsize=16)
plt.xlabel('index', fontsize=14)
plt.ylabel('Power(mW)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

error = abs(originalPower.flatten() - predPower2)
total_error = np.sum(error)
print(f'總誤差: {total_error}')