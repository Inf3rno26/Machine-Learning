---
tags:
  - ML
  - Regression
  - Supervised-Learning
  - SLR
aliases:
  - Simple Linear Regression
  - Linear Regressio
---

---
# Simple Linear Regression

## 1. What is Simple Linear Regression?

**Simple Linear Regression (SLR)** is a supervised learning algorithm used to predict a **continuous numerical value** using **one independent variable**.

Example:

```text
Years of Experience (X)
          ↓
   Linear Regression
          ↓
     Salary (y)
````

In the course's example:

```text
Fertilizer (X) → Potato Yield (y)
```

In our Python example:

```text
Years of Experience (X) → Salary (y)
```

> **Simple = one independent variable (X).**

---

## 2. Goal of SLR

The goal is to find the **best-fitting straight line** through the data points.

```text
Salary
  │
  │              ●
  │           ●
  │        ●
  │     ●
  │  ●
  └────────────────────
       Experience
```

The line represents the relationship between `X` and `y`.

---

# 3. Equation of the Regression Line

The basic equation is:

y^=b0+b1x\hat{y} = b_0 + b_1x

Where:

|Symbol|Meaning|
|---|---|
|`x`|Independent variable / input|
|`y`|Actual target value|
|`ŷ`|Predicted target value|
|`b₀`|Intercept|
|`b₁`|Slope / coefficient|

### Example

If:

y^=30000+9000x\hat{y} = 30000 + 9000x

and experience is:

x=5x = 5

then:

y^=30000+(9000)(5)\hat{y}=30000+(9000)(5) y^=75000\hat{y}=75000

So the predicted salary is **₹75,000**.

---

# 4. Slope (`b₁`)

The **slope** tells us how much the predicted `y` changes when `x` increases by 1 unit.

Example:

b1=9000b_1 = 9000

means:

> For every 1-year increase in experience, predicted salary increases by ₹9,000.

```text
       y
       │
       │        /
       │      /
       │    /  ← Slope
       │  /
       │/
       └──────────── x
```

---

# 5. Intercept (`b₀`)

The **intercept** is the predicted value of `y` when:

x=0x=0

For:

y^=30000+9000x\hat{y}=30000+9000x

when:

x=0x=0

then:

y^=30000\hat{y}=30000

So the line crosses the `y`-axis at **30,000**.

---

# 6. Actual vs Predicted Values

The dataset contains the **actual values**:

```text
y
↓
Actual Salary
```

The model produces:

```text
ŷ
↓
Predicted Salary
```

The difference between them is called the **residual/error**.

e=y−y^e = y-\hat{y}

```text
Actual value
     ●
     │
     │ ← Residual / Error
     │
     ●
Predicted value
```

---

# 7. Least Squares

There can be many possible straight lines through the data.

SLR tries to find the line with the **minimum total squared error**.

SSE=∑(y−y^)2SSE = \sum (y-\hat{y})^2

Where:

- `y` → actual value
    
- `ŷ` → predicted value
    
- `y - ŷ` → error/residual
    
- Squaring prevents positive and negative errors from cancelling out.
    

> **The best-fit line is the line that minimizes the Sum of Squared Errors (SSE).**

The mathematical calculations for finding `b₀` and `b₁` are handled by Scikit-Learn when we use `fit()`.

---

# 8. Implementing SLR in Python

## Import and Load Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
```

`pandas` → handles the dataset.

`matplotlib` → visualizes the data.

---

## Separate X and y

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

```text
X → Years of Experience
y → Salary
```

### `X`

```python
dataset.iloc[:, :-1]
```

```text
:    → all rows
:-1  → all columns except last
```

### `y`

```python
dataset.iloc[:, -1]
```

```text
:   → all rows
-1  → last column
```

---

# 9. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0,
    test_size=0.30
)
```

The dataset is divided into:

```text
Dataset
   │
   ├── 70% → Training
   │
   └── 30% → Testing
```

### `test_size=0.30`

30% of the data is used for testing.

### `random_state=0`

Makes the random split reproducible.

---

# 10. Create the Model

```python
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
```

This creates a Linear Regression model.

At this point, the model has **not learned the relationship yet**.

```text
LinearRegression()
      ↓
Empty/untrained model
```

---

# 11. Train the Model

```python
regression.fit(X_train, y_train)
```

This is where the model learns from the training data.

Conceptually:

```text
X_train + y_train
       ↓
      fit()
       ↓
Least Squares
       ↓
Find b₀ and b₁
       ↓
Best-fit regression line
```

The model learns:

```python
regression.coef_
```

→ slope (`b₁`)

and:

```python
regression.intercept_
```

→ intercept (`b₀`)

---

# 12. Make Predictions

```python
y_pred = regression.predict(X_test)
```

The model uses the learned regression equation to predict salaries for the **unseen `X_test` values**.

```text
X_test
   ↓
Regression Model
   ↓
y_pred
```

We can compare:

```text
y_test → Actual salary
y_pred → Predicted salary
```

---

# 13. Visualizing the Training Set

```python
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### `scatter()`

Shows the actual training data points.

```python
plt.scatter(X_train, y_train)
```

### `plot()`

Draws the regression line learned by the model.

```python
plt.plot(
    X_train,
    regression.predict(X_train)
)
```

---

# 14. Visualizing the Test Set

```python
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

Here:

```text
Red points  → Actual test data
Blue line   → Regression line learned from training data
```

The important point is that **we don't create a new regression line using the test data**.

The same line learned from the training data is used to see how well it fits the unseen test points.

---

# 15. Complete Code

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")

# Separate X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0,
    test_size=0.30
)

# Train Simple Linear Regression Model
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

# Prediction
y_pred = regression.predict(X_test)

# Training Set Visualization
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Test Set Visualization
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

---

# 🧠 SLR Mental Model

```text
Dataset
   ↓
Separate X and y
   ↓
Train/Test Split
   ↓
X_train + y_train
   ↓
LinearRegression()
   ↓
fit()
   ↓
Find best b₀ and b₁
   ↓
Best-Fit Line
   ↓
predict(X_test)
   ↓
y_pred
   ↓
Compare with y_test
```

---

# 🔑 Key Points

### For Training Graph

Pehle `X_train` aur `y_train` ke actual data ko red dots se plot karte hain. Phir same `X_train` ko trained regression model mein daal ke predicted Y values nikalte hain. `X_train` ko X-axis aur predicted Y ko Y-axis pe plot karke blue regression line banate hain. Isse hum dekh sakte hain ki learned line actual training data ko kitna closely fit kar rahi hai.

### Testing graph

> **Testing graph mein hum `X_test` aur `y_test` ke actual data ko red dots se plot karte hain. Ye woh data hai jo model ne training ke time nahi dekha tha.**
> 
> **Phir hum `regressor.predict(X_train)` se wahi regression line plot karte hain jo training data se learn hui thi. `X_train` X-axis pe hota hai aur uske predicted Y values Y-axis pe, isliye blue line banti hai.**
> 
> **Ab hum dekhte hain ki unseen test data ke red dots blue regression line ke kitne close hain. Red dot aur blue line ke beech ka vertical gap prediction error ko represent karta hai.**



- **SLR predicts a continuous numerical value.**
    
- **Simple = one independent variable.**
    
- The regression equation is:
    

y^=b0+b1x\boxed{\hat{y}=b_0+b_1x}

- `b₀` → intercept
    
- `b₁` → slope
    
- `fit()` → learns the best `b₀` and `b₁`
    
- `predict()` → uses the learned line to make predictions
    
- **Least Squares** finds the line that minimizes:
    

SSE=∑(y−y^)2\boxed{SSE=\sum(y-\hat{y})^2}

- `y_test` → actual values
    
- `y_pred` → predicted values
    
- The regression line is learned **only from training data**.
    

### One-line Mental Model

> **Simple Linear Regression finds the best straight line between one input `X` and a continuous target `y`, then uses that line to predict new values.**



# Related Links:

[[Introduction - Regression]]
[[Importing Dataset & Selecting Data with `iloc`|Importing Dataset]]
[[Train-Test Split & Feature Scaling]]