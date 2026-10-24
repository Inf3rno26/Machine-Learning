import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 0,
    test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train) , color = 'blue')
plt.title("Salary Vs Experience (training) ")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary Vs Experience (test) ")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()