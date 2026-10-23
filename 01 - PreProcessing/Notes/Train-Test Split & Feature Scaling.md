---
aliases:
  - Train Test Split
  - Feature Scaling
  - StandardScaler
tags:
  - ML
  - Data-Preprocessing
  - Train-Test-Split
  - Feature-Scaling
  - Python
---
---
# Train-Test Split & Feature Scaling

## 1. Splitting the Dataset

We split the dataset into **training data** and **testing data** before training the ML model.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)
````

### What happens?

```text
                    X, y
                     │
                     ▼
              train_test_split
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Training Data          Testing Data
      X_train, y_train       X_test, y_test
          80%                    20%
```

### `test_size=0.2`

20% of the data is used for testing and the remaining 80% for training.

### `random_state=1`

Controls the random split.

Using the same value gives the **same split every time**, making the result reproducible.

---

## 2. Why Split the Dataset?

The model needs data it has **never seen during training** to evaluate how well it performs.

```text
Training data
     ↓
Model learns

Testing data
     ↓
Model is evaluated
```

> **Training set → Learn**
> 
> **Testing set → Evaluate**

---

# 3. Feature Scaling

After splitting, we scale the numerical features.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
```

`StandardScaler` transforms features so they have approximately:

```text
Mean = 0
Standard Deviation = 1
```

### Why Feature Scaling?

Different features can have very different ranges.

```text
Age       → 18 – 60
Salary    → 20,000 – 100,000
```

Salary has much larger numerical values than Age.

Some ML algorithms are affected by these differences in scale.

Scaling puts numerical features on a **similar scale**.

```text
Before:

Age       → 18 – 60
Salary    → 20,000 – 100,000

          ↓ Scaling

Age       → around -2 to +2
Salary    → around -2 to +2
```

> **Feature scaling prevents differences in numerical ranges from unfairly influencing the model.**

---

# 4. Scaling the Training Data

```python
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
```

### Why `[:, 3:]`?

After One-Hot Encoding, `X` looks approximately like:

```text
Index
  0       1        2       3      4
  ↓       ↓        ↓       ↓      ↓
France  Germany  Spain    Age   Salary
```

Therefore:

```python
X_train[:, 3:]
```

means:

```text
:   → all rows
3:  → column 3 onwards
```

So we select:

```text
Age
Salary
```

and **do not scale the one-hot encoded Country columns**.

---

## `fit_transform()` on Training Data

```python
sc.fit_transform(X_train[:, 3:])
```

This does two things:

```text
fit
 ↓
Learn mean and standard deviation from X_train

transform
 ↓
Scale X_train using those values
```

> **`fit()` = Learn**  
> **`transform()` = Apply**

---

# 5. Scaling the Test Data

```python
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

Notice that we use:

```python
transform()
```

not:

```python
fit_transform()
```

The scaler has already learned the mean and standard deviation from the **training data**.

We use those same parameters (from X_test with fit_transform()) to scale the test data.

### Correct Procedure

```text
TRAINING DATA → fit_transform()
TEST DATA     → transform()
```

```text
X_train
   ↓
fit_transform()
   ↓
Learn mean + SD
   │
   │ same parameters
   ▼
X_test
   ↓
transform()
   ↓
Scale using training parameters
```

---

## What If We Use `fit_transform()` on `X_test`?

❌ Incorrect:

```python
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
```

The second `fit_transform()` calculates **new mean and standard deviation from the test data**.

For example:

```text
Training:
Mean = 50
Std  = 10

Test:
Mean = 60
Std  = 15
```

A value of `60` would become:

```text
Using training parameters:
(60 - 50) / 10 = 1

Using test parameters:
(60 - 60) / 15 = 0
```

Now training and testing data are scaled using **different parameters**.

More importantly, preprocessing has used information from the test set, which is a form of **data leakage**.

> **Never `fit()` your scaler on the test data.**

---

# 6. Why Scale After Splitting?

This is very important.

### ❌ Incorrect

```text
Entire Dataset
      ↓
Calculate mean/std
      ↓
Split into Train/Test
```

The mean and standard deviation would include information from the test set.

This causes **data leakage**.

### ✅ Correct

```text
             Dataset
                ↓
          Train/Test Split
           ↙           ↘
      X_train          X_test
         ↓                ↓
   fit_transform()    transform()
         │                │
         └──────┬─────────┘
                ↓
          Ready for ML
```

> **Always learn preprocessing parameters from the training set only.**

---

# 7. Complete Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
```

---

# 🧠 Key Points

```text
train_test_split()
        ↓
80% Training + 20% Testing
```

```text
StandardScaler()
        ↓
Mean ≈ 0
Standard Deviation ≈ 1
```

```text
X_train → fit_transform()
X_test  → transform()
```

### Most Important Rule

> **Split first → fit scaler on training data → transform both training and test data using the training scaler.**

### Mental Model

```text
Split
  ↓
Train ──→ fit_transform()
  │
  │ learn scaling parameters
  ↓
Test ───→ transform()
```


### Related Links

[[Importing Dataset & Selecting Data with `iloc`|Importing Dataset]]
[[Fixing Missing Values - SimpleImputer]]
[[Encoding Categorical Data]]
[[Train-Test Split & Feature Scaling|Feature Scaling]]
