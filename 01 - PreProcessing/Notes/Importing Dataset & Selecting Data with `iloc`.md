---
tags:
  - ML
  - Python
  - Pandas
  - Data-Preprocessing
aliases:
  - Pandas iloc
  - Importing Dataset
---

--- 
## 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
````

- `numpy` → Numerical operations and arrays
    
- `pandas` → Working with datasets and CSV files
    
- `matplotlib.pyplot` → Data visualization

---

## 2. Importing the Dataset

```python
dataset = pd.read_csv("Data.csv")
```

`pd.read_csv()` reads a CSV file and stores it as a Pandas **DataFrame**.

```text
Data.csv
   ↓
pd.read_csv()
   ↓
dataset (DataFrame)
```

Useful commands:

```python
dataset.head()       # First 5 rows
dataset.tail()       # Last 5 rows
dataset.shape        # Number of rows and columns
dataset.columns      # Column names
dataset.info()       # Dataset information
```

---

## 3. `iloc`

`iloc` is used to select data using **integer/index positions** from the csv.

### Syntax

```python
dataset.iloc[rows, columns]
```

> **Before the comma → Rows**  
> **After the comma → Columns**

### `:` means "all"

```python
dataset.iloc[:, :]
```

```text
             rows       columns
               ↓          ↓
          all rows     all columns
```

---

## 4. Selecting `X` — Features

```python
X = dataset.iloc[:, :-1].values
```

Breakdown:

```text
dataset.iloc[:, :-1]
             │   │
             │   └── All columns except the last
             │
             └────── All rows
```

`X` contains the **input features** used by the ML model.

For example:

```text
X
┌─────────┬─────┬────────┐
│ Country │ Age │ Salary │
├─────────┼─────┼────────┤
│ France  │ 44  │ 72000  │
│ Spain   │ 27  │ 48000  │
│ Germany │ 30  │ 54000  │
│ ...     │ ... │ ...    │
└─────────┴─────┴────────┘
```

### Why `:-1`?

Python uses negative indexing:

```text
0        1       2          3
│        │       │          │
Country  Age    Salary   Purchased
                           ↑
                          -1
```

`:-1` means:

> Start from the beginning and stop before the last column.

Therefore, `Purchased` is excluded from `X`.

---

## 5. Selecting `y` — Target

```python
y = dataset.iloc[:, -1].values
```

Breakdown:

```text
dataset.iloc[:, -1]
             │   │
             │   └── Last column
             │
             └────── All rows
```

`y` contains the **target/output** that the model needs to predict.

```text
y
│
├── No
├── Yes
├── No
├── No
├── Yes
└── ...
```

---

## 6. X and y in Machine Learning

```text
                 DATASET
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
          X                   y
      FEATURES              TARGET
          │                   │
   Country, Age, Salary    Purchased
          │                   │
          └─────────┬─────────┘
                    ↓
                ML MODEL
```

### `X`

The **inputs/features** given to the model.

```python
X = dataset.iloc[:, :-1].values
```

### `y`

The **target/output** that the model tries to predict.

```python
y = dataset.iloc[:, -1].values
```

---

## 7. `.values`

```python
X = dataset.iloc[:, :-1].values
```

`.values` converts the Pandas DataFrame/Series into a **NumPy array**.

```text
Pandas DataFrame
       ↓
   .values
       ↓
NumPy Array
```

This is useful because many ML libraries work with NumPy arrays.

---

## 8. Printing X and y

```python
print(X)
print(y)
```

This shows what was extracted from the dataset.

```text
X → Features / Inputs
y → Target / Output
```

---

# `iloc` Cheat Sheet

```python
.iloc[rows, columns]
```

|Code|Meaning|
|---|---|
|`iloc[:, :]`|All rows, all columns|
|`iloc[:, :-1]`|All rows, all columns except last|
|`iloc[:, -1]`|All rows, last column|
|`iloc[0, :]`|First row, all columns|
|`iloc[:, 0]`|All rows, first column|
|`iloc[0:5, :]`|First 5 rows, all columns|
|`iloc[:, 1:3]`|All rows, columns 1 and 2|

---

## 🧠 Key Rule

```text
.iloc[ ROWS , COLUMNS ]
          ↑
      first part

Before comma → Rows
After comma  → Columns
```

### Most important ML pattern

```python
X = dataset.iloc[:, :-1].values    # Features
y = dataset.iloc[:, -1].values     # Target
```

> **X = Everything except the target**  
> **y = The target we want to predict**

## Related Links

[[The Machine learning Process]]