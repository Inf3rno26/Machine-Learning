---
tags:
  - Data-Preprocessing
  - Encoding
aliases:
  - OneHotEncoder
  - Categorical Data
  - LabelEncoder
---
---
# Encoding Categorical Data

ML algorithms generally work with **numerical data**, so categorical/text values need to be converted into numbers.

Our dataset contains:

```
Country     → Categorical feature
Age         → Numerical feature
Salary      → Numerical feature
Purchased   → Categorical target
```

We use different encoders for `X` and `y`.

## 1. Why Encoding?

Our `X` initially looks like:

```text
Country    Age    Salary
France     44     72000
Spain      27     48000
Germany    30     54000
```

ML models cannot directly understand:

```text
France
Spain
Germany
```

as meaningful numerical values.

So we convert categorical data into numerical data.

```text
Categorical data
       ↓
    Encoding
       ↓
Numerical data
       ↓
    ML Model
```

---

# 2. `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```

`ColumnTransformer` allows us to apply **different preprocessing operations to different columns**.

For our dataset:

```text
Country ──→ OneHotEncoder
Age     ──→ Keep unchanged
Salary  ──→ Keep unchanged
```

---

## Understanding `transformers`

```python
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough'
)
```

The general structure is:

```python
transformers=[
    ('name', transformer, columns)
]
```

There are 3 parts:

```text
('encoder', OneHotEncoder(), [0])
      │            │           │
      │            │           └── Which columns?
      │            │
      │            └────────────── What transformation?
      │
      └─────────────────────────── Name
```

### `'encoder'`

A name/identifier for this transformation.

It could also be:

```python
'country_encoder'
```

The name does not perform the transformation.

### `OneHotEncoder()`

The actual transformation that converts categories into binary columns.

### `[0]`

Specifies which column to transform.

Our columns are:

```text
Index     0          1       2
          ↓          ↓       ↓
       Country      Age    Salary
```

Therefore:

```python
[0]
```

means:

> Apply `OneHotEncoder` to column 0 → `Country`.

---

# 3. `remainder='passthrough'`

```python
remainder='passthrough'
```

Means:

> Keep all columns that were not selected for transformation.

So:

```text
Country ──→ OneHotEncoder ──→ Encoded Country
Age     ────────────────────→ Unchanged
Salary  ────────────────────→ Unchanged
```

Without `passthrough`, the columns that aren't specified in `transformers` would not be included in the output.

---

# 4. Applying `OneHotEncoder`

```python
X = np.array(ct.fit_transform(X))
```

`fit_transform()` does two things:

```text
fit       → Learn the categories
transform → Convert the categories
```

For example:

```text
Country

France
Spain
Germany

        ↓ OneHotEncoder

France    Germany    Spain
   1         0          0
   0         0          1
   0         1          0
```

The numerical columns are passed through unchanged.

So `X` changes from:

```text
Country    Age    Salary
```

to something like:

```text
France    Germany    Spain    Age    Salary
   1         0          0      44     72000
   0         0          1      27     48000
   0         1          0      30     54000
```

`np.array()` converts the transformed result into a NumPy array.

---

# 5. Why One-Hot Encoding?

We should **not** simply assign:

```text
France  → 0
Spain   → 1
Germany → 2
```

because this can make the model think:

```text
Germany > Spain > France
```

or that the numerical difference has meaning.

But countries have **no natural numerical order**.

One-hot encoding avoids this by creating separate binary columns:

```text
France   → [1, 0, 0]
Spain    → [0, 0, 1]
Germany  → [0, 1, 0]
```

---

# 6. Encoding `y` with `LabelEncoder`

`Purchased` is our target:

```text
No
Yes
No
Yes
```

We can convert it into:

```text
No  → 0
Yes → 1
```

Import:

```python
from sklearn.preprocessing import LabelEncoder
```

Create encoder:

```python
le = LabelEncoder()
```

Apply it:

```python
y = le.fit_transform(y)
```

`fit_transform()`:

```text
fit       → Learn the unique labels
transform → Convert them into numbers
```

Result:

```text
Before       After

No      →      0
Yes     →      1
No      →      0
Yes     →      1
```

---

# 7. Why `LabelEncoder` for `y`?

`y` contains the **target/output**, not a feature.

For our binary target:

```text
Purchased

No
Yes
```

representing the classes as:

```text
No  → 0
Yes → 1
```

is appropriate.

We use:

```text
X → OneHotEncoder
y → LabelEncoder
```

---

# 8. OneHotEncoder vs LabelEncoder

|Encoder|Used for|Example|
|---|---|---|
|`OneHotEncoder`|Categorical features (`X`)|Country → `[1,0,0]`|
|`LabelEncoder`|Target (`y`)|Yes → `1`, No → `0`|

### Important

`LabelEncoder` is generally used for the **target**, not for ordinary categorical input features, because assigning numbers to input categories can introduce an artificial order.

---

# 9. Complete Code

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
```

---

# Processing Flow

```text
                    DATASET
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
             X                   y
      Country, Age, Salary    Purchased
             │                   │
             ↓                   ↓
    ColumnTransformer       LabelEncoder
             │                   │
             ↓                   ↓
       Country → OneHot       Yes → 1
             │                No  → 0
             ↓                   │ 
       Age & Salary              │  
       passthrough               │ 
             │                   │ 
             └─────────┬─────────┘
                       ↓
                Numerical X & y
                       ↓
                   ML Model
```

---

# 🧠 Key Points

- **Encoding** → Converts categorical data into numerical data.
    
- **`ColumnTransformer`** → Applies transformations to specific columns.
    
- **`transformers`** → Contains the preprocessing instructions.
    
- `('encoder', OneHotEncoder(), [0])` means:
    
    - `'encoder'` → name
        
    - `OneHotEncoder()` → transformation
        
    - `[0]` → column to transform
        
- **`remainder='passthrough'`** → keeps unselected columns unchanged.
    
- **`OneHotEncoder`** → used for categorical features in `X`.
    
- **`LabelEncoder`** → converts target classes in `y` into numerical labels.
    
- **`fit_transform()`** → learns the required information and applies the transformation.
    

### Mental Model

```text
ColumnTransformer
       ↓
WHAT transformation?
       ↓
WHERE to apply it?
       ↓
WHAT happens to the rest?
       ↓
Processed X
```


## Related Links

[[Importing Dataset & Selecting Data with `iloc`|Importing Dataset]]
[[Fixing Missing Values - SimpleImputer]]