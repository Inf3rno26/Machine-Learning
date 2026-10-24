---
tags:
  - ML
  - Regression
  - Polynomial
aliases:
  - Polynomial Linear Regression
---
---
# Polynomial Regression

## 1. What is Polynomial Regression?

Polynomial Regression is a supervised machine learning technique used to predict a continuous numerical output when the relationship between the input and output is nonlinear.

Unlike Simple Linear Regression, which fits a straight line, Polynomial Regression can fit a curved relationship by adding powers of the input feature.

### Simple Linear Regression

$$  
\hat{y} = b_0 + b_1x  
$$

### Polynomial Regression

For degree 2:

$$  
\hat{y} = b_0 + b_1x + b_2x^2  
$$

For degree 4:

$$  
\hat{y} = b_0 + b_1x + b_2x^2 + b_3x^3 + b_4x^4  
$$

Where:

- $\hat{y}$ = predicted output
    
- $b_0$ = intercept
    
- $b_1,b_2,\ldots$ = learned coefficients
    
- $x$ = input feature
    
- Degree = highest power of the input feature

**Important:** Polynomial Regression is nonlinear in the input feature $x$, but linear in its coefficients. Therefore, Scikit-Learn's `LinearRegression` can still be used to train it.

## 2. Dataset: Position Salaries

We uses `Position_Salaries.csv`.

|Position|Level|Salary|
|---|---|---|
|Business Analyst|1|45,000|
|Junior Consultant|2|50,000|
|Senior Consultant|3|60,000|
|Manager|4|80,000|
|Country Manager|5|110,000|
|Region Manager|6|150,000|
|Partner|7|200,000|
|Senior Partner|8|300,000|
|C-level|9|500,000|
|CEO|10|1,000,000|

The salary increases slowly at first and then grows rapidly. A straight line may not capture this pattern well, so Polynomial Regression is used to model the curvature.

## 3. Importing the Dataset

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

- `X` contains the Position Level.
    
- `y` contains the Salary.
    
- The first column, `Position`, contains text labels and is not used as the numerical input feature.

## 4. Training Simple Linear Regression

```
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
```

The model learns a straight line:

$$  
\hat{y} = b_0 + b_1x  
$$

This model acts as a baseline for comparison. It attempts to represent all salary observations using one straight line.

## 5. Creating Polynomial Features

```
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)
```

`PolynomialFeatures` creates additional features from the original input.

With `degree=4` and the default `include_bias=True`, an input value $x$ is transformed into:

$$  
[1,\ x,\ x^2,\ x^3,\ x^4]  
$$

For example, if $x=2$:

$$  
[1,\ 2,\ 4,\ 8,\ 16]  
$$

The additional powers allow the model to learn a curved relationship.

## 6. Training Polynomial Regression

```
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```

The model learns the equation:

$$  
\hat{y} = b_0 + b_1x + b_2x^2 + b_3x^3 + b_4x^4  
$$

### Why are there two Linear Regression models?

|Model|Input|Result|
|---|---|---|
|`lin_reg`|Original feature $x$|Straight line|
|`lin_reg_2`|Expanded features $1,x,x^2,x^3,x^4$|Potentially curved line|

Both models use the same actual salary values, but they receive different input features.

The two models are trained separately so their fits can be compared. The first equation is not added to the second equation.

**Key concept:** Polynomial Regression uses the Linear Regression estimator after transforming the input features.

## 7. Visualizing the Results

### Simple Linear Regression

```
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

- Red dots represent actual salary observations.
    
- The blue line represents predictions from the straight-line model.

### Polynomial Regression

```
plt.scatter(X, y, color='red')
plt.plot(
    X,
    lin_reg_2.predict(poly_reg.transform(X)),
    color='blue'
)

plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

The polynomial model can follow the curved pattern more closely than a straight line.

### Why use `X_grid`?

`X_grid` generates many intermediate input values, allowing the plotted polynomial curve to look smoother.

It is optional and is only a plotting convenience. It is not required to train the model or make predictions.

For a single-feature dataset, the grid can be created with:

```
X_grid = np.arange(
    X.min(),
    X.max(),
    0.1
).reshape(-1, 1)

plt.scatter(X, y, color='red')
plt.plot(
    X_grid,
    lin_reg_2.predict(poly_reg.transform(X_grid)),
    color='blue'
)

plt.show()
```

**Why did** `**min(X)**` **cause a** `**TypeError**`**?**

`X` is a two-dimensional array with shape `(10, 1)`. Python's built-in `min(X)` returns a row array, not a scalar. `np.arange()` expects scalar start and stop values.

Using `X.min()` and `X.max()` returns scalar values for this single-feature dataset.

For datasets with multiple features, a one-dimensional grid cannot represent all possible combinations of inputs directly. Other features must be fixed at chosen values or a suitable multidimensional visualization must be used.

## 8. Predicting a New Salary

Suppose we want to predict the salary for Position Level 6.5.

### Simple Linear Regression

```
lin_reg.predict([[6.5]])
```

This predicts using the straight-line equation.

### Polynomial Regression

```
lin_reg_2.predict(poly_reg.transform([[6.5]]))
```

The input is transformed into:

$$  
[1,\ 6.5,\ 6.5^2,\ 6.5^3,\ 6.5^4]  
$$

The trained polynomial model uses these features to calculate the predicted salary.

**Important:** Use `transform()` for new inputs because the polynomial transformer has already been fitted. There is no need to call `fit_transform()` again for prediction.

## 9. Important Points to Remember

- Polynomial Regression can model nonlinear relationships between input and output.
    
- The degree determines the highest power of the input feature.
    
- `PolynomialFeatures` creates powers such as $x^2$, $x^3$, and $x^4$.
    
- `LinearRegression` learns the coefficients using the expanded feature matrix.
    
- Two models are used in the course example to compare a straight-line fit against a polynomial fit.
    
- A degree-2 polynomial is quadratic; a degree-4 polynomial can form more complex curves.
    
- `X_grid` is optional and only improves the smoothness of the plotted curve.
    
- Higher polynomial degrees can overfit, especially with small datasets.
    
- The best degree should be selected based on model evaluation, not simply because a curve follows the training observations closely.

## 10. Quick Revision

**Q1. What is the main purpose of Polynomial Regression?**

To model nonlinear relationships using powers of input features.

**Q2. What does** `**PolynomialFeatures(degree=4)**` **create for a single input feature?**

With the default bias term: $[1,x,x^2,x^3,x^4]$.

**Q3. Why do we create two regression models in the course example?**

To compare a straight-line model with a polynomial model.

**Q4. Why can we still use** `**LinearRegression**` **for Polynomial Regression?**

Because the model is linear in its learned coefficients, even though it uses powers of the input feature.

**Q5. Is** `**X_grid**` **required for prediction?**

No. It is optional and is mainly used to draw a smoother-looking curve.

**Q6. What can happen if the polynomial degree is too high?**

The model may overfit the training data and generalize poorly to new observations.

---
# Related Links:

[[Introduction - Regression]]
[[SLR|Simple Linear Regression]]
[[Importing Dataset & Selecting Data with `iloc`|Importing Dataset]]