import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'D:\program\python\ML\ML_finalReport_train.csv')
pred_data = pd.read_csv(r'D:\program\python\ML\ML_finalReport_test.csv')

X = train_data[['year', 'month', 'day', 'hour', 'minute', 'weekday', 'LocationCode',
                'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'month_sin', 'month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]
y = train_data['Power(mW)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mse)
print(f'均方誤差 (MSE): {mse}')
print(f'平均絕對值誤差 (MAE): {mae}')
print(f'R² 分數: {r2}')
print(f'均方根誤差 (RMSE): {rmse}')

joblib.dump(lin_reg, r'D:\program\python\ML\ML_finalReport_linear_regression.pkl')

Xpred = pred_data[['year', 'month', 'day', 'hour', 'minute','weekday', 'LocationCode',
                    'hour_sin','hour_cos','minute_sin','minute_cos','month_sin','month_cos','hour_squared','quarter','time_of_day','day_of_year','week_of_year']]

predPower = lin_reg.predict(Xpred)
predPower = np.clip(predPower, 0, None)

plt.figure(figsize=(20, 5))
originalPower = pred_data[['Power(mW)']].values
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

print(np.sum(originalPower))