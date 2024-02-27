import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler

"""
1 Лінійна регресія з 1 змінною:
Провести базовий аналіз даних використовуючи pandas. Визначити набір колонок, які можуть бути використані для multivariate linear regression.
Обрати будь яку колонку з датасету (sqft_living для наведеного датасету) для моделі лінійної регресії з 1 змінною. За допомогою pandas або matplotlib візуалізувати зв'зок між обраної змінною і набором міток (sqft_living & price). Зберегти результат у файл
Використовуючи sklearn побудувати і натренувати модель Linear Regression для обраної змінної. Порахувати точність. як метрику використати середнє квадратичне відхилення. За допомогою sklearn зберегти модель.
"""
data = pd.read_csv('kc_house_data.csv')
# first 5 row
print(data.head())
# info
print(data.info())
# describe
print(data.describe())
# max in column
print(data.max())
# min in column
print(data.min())
# count on unique in column
print(data.nunique())
# count space in column
print(data.isnull().sum)

num_features = data.select_dtypes(include=['float64', 'int64']).columns
print(num_features)

# choose column sqft_living
X = data[['sqft_living']].values
# choose column price
y = data['price'].values

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating linear regression and training
model = LinearRegression()
model.fit(X_train, y_train)

# predicting the test set result
predictions = model.predict(X_test)

# calculating cost function
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error: {mse}")

# creating plot
plt.scatter(X_test, y_test, color='red', label='Real Value')
plt.plot(X_test, predictions, color='blue', linewidth=2, label='Predicted Value')
plt.title('Linear regression: sqft_living vs price')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend()
plt.savefig('regression.png')
plt.show()

# saving model
joblib.dump(model, 'univariate_linear_regression.joblib')

print('\n--------------------------------------------------------------------------------------------------------')
"""
2 Лінійна регресія з багатьма змінними:
Обрати максимальну кількість колонок, які можуть бути використані для регресії.
Привести дані до схожого скейлу (нормалізація чи стандартизація) за допомогою методів sklearn
Побудувати модель лінійної регресії з багатьма змінними. Порахувати точність. як метрику використати середнє квадратичне відхилення. За допомогою sklearn зберегти модель.
Викорисати normal equation для обчислення аналітичного рішення. Порівняти результат із ітеративною моделлю (із поперднього пункту). Відобразити точність обох моделей.
"""
X_m = data.iloc[:, 3:15].values
y_m = data['price'].values

# splitting dataset into training set and test set
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

# feature normalization
scaler = StandardScaler()
X_m_train_scaled = scaler.fit_transform(X_m_train)
X_m_test_scaled = scaler.transform(X_m_test)

# print(X_m_standard)

# creating linear regression and training
model_mult = LinearRegression()
model_mult.fit(X_m_train_scaled, y_m_train)

# predicting set test result
mult_predictions = model_mult.predict(X_m_test_scaled)

# cost function
mult_mse = mean_squared_error(y_m_test, mult_predictions)
print(f'Cost function for gradient descent: {mult_mse}')

# saving model
joblib.dump(model_mult, 'multivariate_linear_regression.joblib')

# adding column with ones and merge for train
X_m_train_ones = np.c_[np.ones(X_m_train_scaled.shape[0]), X_m_train_scaled]

# normal equation train
theta = np.linalg.inv(X_m_train_ones.T @ X_m_train_ones) @ X_m_train_ones.T @ y_m_train

# adding column with ones and merge for test
X_m_test_ones = np.c_[np.ones(X_m_test_scaled.shape[0]), X_m_test_scaled]

# predict
normal_mult_predictions = X_m_test_ones @ theta

# cost function for normal equation
normal_mult_mse = mean_squared_error(y_m_test, normal_mult_predictions)
print(f'Cost function for normal equation: {normal_mult_mse}')

cost_ratio = mult_mse / normal_mult_mse
print(f"Cost function ratio (gradient descent / normal equation): {cost_ratio}")

# plot for model with gradient descent
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_m_test)), y_m_test, color='blue', label='test value')
plt.scatter(range(len(mult_predictions)), mult_predictions, color='red', label='predict value')
plt.title('model with gradient descent')
plt.xlabel('index')
plt.ylabel('price')
plt.legend()
plt.show()

# plot for model with normal equation
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_m_test)), y_m_test, color='blue', label='real value')
plt.scatter(range(len(normal_mult_predictions)), normal_mult_predictions, color='green', label='predict value')
plt.title('model with normal equation')
plt.xlabel('index')
plt.ylabel('price')
plt.legend()
plt.show()

print('\n--------------------------------------------------------------------------------------------------------')
"""
3 Написати функцію яка приймає на вхід 2 вектори - набір міток 'y' (рандомний вектор )і набір предіктнутих значень `h_x` (рандомний вектор ). 
Ця функція має обчислювати loss J для даних. Використати можливості numpy. Цикли в обчисленнях заборонені.
"""


def cost_function(y, h_x):
    # quantity
    m = len(y)

    # func
    J = np.sum((y - h_x) ** 2) / m
    return J


y = np.random.rand(10)
h_x = np.random.rand(10)

loss = cost_function(y, h_x)

print("Mean Squared Error:", loss)
