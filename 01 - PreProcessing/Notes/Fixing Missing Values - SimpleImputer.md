
---
## tags: [ML, Python, Scikit-Learn, Data-Preprocessing]  
aliases: [SimpleImputer, Fixing Missing Values]


---

# Fixing Missing Values

Real-world datasets can contain missing values (`NaN`). Most ML algorithms cannot directly work with them, so we need to replace them with suitable values.

Our example dataset has missing values in **Age** and **Salary**.

---

## SimpleImputer

`SimpleImputer` from Scikit-Learn is used to replace missing values.

```python
from sklearn.impute import SimpleImputer
````

Create an imputer:

```python
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)
```

- `missing_values=np.nan` → identifies missing values. Its a default parameter, can be not written.
    
- `strategy='mean'` → replaces each missing value with the **==mean== of its column**
    

Other strategies:

```text
mean           → column mean
median         → column median
most_frequent  → most frequent value
constant       → a specified value
```

---

## Selecting the Columns

```python
X[:, 1:3]
```

Remember:

```text
X[row, column]
```

Therefore:

```text
:     → all rows
1:3   → columns 1 and 2
```

Our `X`:

```text
Column 0 → Country
Column 1 → Age
Column 2 → Salary
```

So:

```python
X[:, 1:3]
```

selects **Age and Salary**, where the ==missing numerical values== are present, and ==NO Strings== should be present.

---

## `fit()`

```python
imputer.fit(X[:, 1:3])
```

`fit()` **learns the required values from the data**.

With `strategy='mean'`, it calculates the mean of each column:

```text
Age     → mean Age
Salary  → mean Salary
```

It does **not modify `X`**.

> `fit()` = **Learn**

---

## `transform()`

```python
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

`transform()` uses the values learned by `fit()` and replaces the missing values.

```text
NaN
 ↓
Column mean
 ↓
Replaced value
```

> `transform()` = **Apply**

---

## `fit_transform()`

Since we usually want to learn and apply the transformation together:

```python
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
```

This is equivalent to:

```python
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

```text
fit()            → Learn
transform()      → Apply
fit_transform()  → Learn + Apply
```

---

## Complete Code

```python
import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)

X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
```

### Processing Flow

```text
X
│
├── Country
├── Age ──────┐
└── Salary ───┤
              ↓
        Select X[:, 1:3]
              ↓
        SimpleImputer
              ↓
       Calculate column mean
              ↓
        Replace NaN values
              ↓
          Clean X
```

### 🧠 Key Points

- **`SimpleImputer`** → handles missing values
    
- **`np.nan`** → represents missing numerical data
    
- **`X[:, 1:3]`** → selects Age and Salary
    
- **`strategy='mean'`** → replaces missing values with column mean
    
- **`fit()`** → learns the replacement values
    
- **`transform()`** → applies the learned values
    
- **`fit_transform()`** → does both
    
- The original non-missing values are **not changed**
    
- After imputation, `X` contains **no missing values** in the selected columns

## One line mental model 
```
NaN → SimpleImputer → calculate mean → replace NaN → clean numerical data
```


## Related links:

[[Importing Dataset & Selecting Data with `iloc`]]
[[The Machine learning Process]]