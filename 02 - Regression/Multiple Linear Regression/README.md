---
tags:
  - ML
  - Regression
  - MLR
---
---

# Multiple Linear Regression (MLR)

## 1. What is Multiple Linear Regression?

Multiple Linear Regression is a **supervised machine learning algorithm** used to predict a continuous numerical value using two or more independent variables (features).

For example, in the `50_Startups.csv` dataset, we predict a startup's **Profit** using:

- R&D Spend
    
- Administration
    
- Marketing Spend
    
- State


Unlike Simple Linear Regression, which uses one independent variable, Multiple Linear Regression uses multiple independent variables.

## 2. Mathematical Equation

The general equation of Multiple Linear Regression is:

$$  
\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_nx_n  
$$

Where:

|Symbol|Meaning|
|---|---|
|$\hat{y}$|Predicted output|
|$b_0$|Intercept|
|$b_1, b_2, \ldots, b_n$|Learned coefficients|
|$x_1, x_2, \ldots, x_n$|Input feature values|
|$n$|Number of input features|

### Understanding the terms

- **Intercept ($b_0$):** The model's predicted output when all input features are zero.
    
- **Coefficient ($b_i$):** Represents how the prediction changes with a one-unit increase in the corresponding feature, while the other features remain constant.
    
- **Feature ($x_i$):** An input variable used to make the prediction.
    
- **Prediction ($\hat{y}$):** The output calculated by substituting the input values into the equation.

The model learns the coefficients and intercept from the training data by minimizing the sum of squared residuals.

$$  
SSE = \sum_{i=1}^{m}(y_i-\hat{y}_i)^2  
$$

Here, $y_i$ is the actual output and $\hat{y}_i$ is the predicted output.

## 3. Dataset Example: 50 Startups

The dataset contains the following features and target:

|Feature|Description|
|---|---|
|R&D Spend|Money spent on research and development|
|Administration|Administrative expenditure|
|Marketing Spend|Marketing expenditure|
|State|Startup's state|
|Profit|Target variable to predict|

Here, $X$ contains the input features, while $y$ contains the actual Profit values.

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

The State column is categorical, so it is encoded before training. Refer to the separate preprocessing note ([[Encoding Categorical Data]]) for details.

## 4. Training the Model

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
```

### What happens here?

1. `LinearRegression()` creates the regression model.
    
2. `fit(X_train, y_train)` learns the coefficients and intercept from the training data.
    
3. `predict(X_test)` uses the learned equation to predict outputs for unseen test features.
 

Scikit-Learn automatically calculates the coefficients and intercept. We do not need to derive them manually for normal model training.

## 5. Understanding the Learned Coefficients

After training, we can inspect the model's parameters:

```python
print("Intercept:", regressor.intercept_)
print("Coefficients:", regressor.coef_)
print("Features:", ct.get_feature_names_out())
```

- `regressor.intercept_` returns $b_0$.
    
- `regressor.coef_` returns the learned coefficients.
    
- `ct.get_feature_names_out()` identifies the corresponding transformed features.


**Important:** Always match coefficients to the transformed feature names in their exact order.

In the example, the one-hot encoded State columns appear before the numerical spending columns. Therefore, the first coefficient corresponds to the California indicator, not R&D Spend.

## 6. Mathematical Calculation Using Actual Coefficients

Suppose the model returns the following approximate values:

$$  
b_0 = 42467.5292  
$$

The coefficients shown by the model were approximately:

|Transformed feature|Approximate coefficient|
|---|---|
|California indicator|86.6|
|Florida indicator|-873|
|New York indicator|786|
|R&D Spend|0.773|
|Administration|0.0329|
|Marketing Spend|0.0366|

These values are rounded for illustration. Use full-precision values for accurate calculations.

For a startup in New York, the encoded State values are:

- California = 0
    
- Florida = 0
    
- New York = 1
    

For the first dataset row:

- R&D Spend = 165349.2
    
- Administration = 136897.8
    
- Marketing Spend = 471784.1
    

Substituting into the equation:

$$  
\begin{aligned}  
\hat{y}={}&42467.5292+86.6(0)\  
&-873(0)+786(1)\  
&+0.773(165349.2)\  
&+0.0329(136897.8)\  
&+0.0366(471784.1)  
\end{aligned}  
$$

Using these rounded coefficients gives an approximate prediction. The exact prediction must be calculated using the model's full-precision coefficients.

### Key takeaway

Each coefficient is multiplied by its corresponding feature value. All these terms are added to the intercept to calculate the predicted output.

## 7. Comparing Predictions with Actual Values

```python
import numpy as np

np.set_printoptions(precision=2)

comparison = np.concatenate((
    y_pred.reshape(len(y_pred), 1),
    y_test.reshape(len(y_test), 1)
), axis=1)

print(comparison)
```

The output contains two columns:

|Column|Meaning|
|---|---|
|First|Predicted Profit ($\hat{y}$)|
|Second|Actual Profit ($y$)|

The prediction error, also called the residual, is:

$$  
e = y-\hat{y}  
$$

- A residual close to zero indicates a prediction close to the actual value.
    
- A large absolute residual indicates a larger prediction error.


The test dataset helps evaluate how well the model generalizes to unseen data.

## 8. Important Points to Remember

- MLR predicts a continuous numerical target using multiple features.
    
- $b_0$ is the intercept.
    
- Each $b_i$ is the coefficient associated with feature $x_i$.
    
- The equation calculates predictions by multiplying each feature by its coefficient and adding the terms together.
    
- Coefficients must be matched to the correct feature names and order.
    
- Categorical features can contribute through one-hot encoded indicator variables.
    
- Coefficients describe relationships in the fitted model; they do not automatically establish causation.
    
- The actual target and predicted target are different: $y$ is actual, while $\hat{y}$ is predicted.
    
- Preprocessing must be handled appropriately to avoid data leakage.

# Related Links:

[[Introduction - Regression]]
[[Importing Dataset & Selecting Data with `iloc`|Importing Dataset]]
[[Train-Test Split & Feature Scaling]]
[[Encoding Categorical Data]]
