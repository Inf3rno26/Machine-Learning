---
tags:
  - ML
  - Regression
  - Supervised-Learning
---

---
# Regression

## What is Regression?

**Regression** is a supervised machine learning technique used to predict a **real/continuous numerical value**.

Examples:

- Salary → ₹75,000
- House Price → ₹80,00,000
- Temperature → 32.5°C
- Potato Yield → 25 kg

```text
Input (X)
   ↓
Regression Model
   ↓
Predicted numerical value (ŷ)
````

---

## Prediction vs Forecasting

The difference depends on the independent variable.

### Prediction

If `X` is **not time**, the model predicts a present but unknown value.

Example:

```text
Fertilizer → Potato Yield
```

We use the amount of fertilizer to predict the potato yield.

### Forecasting

If the independent variable is **time**, the model is used to predict future values.

Example:

```text
Past Sales Data
      ↓
Regression Model
      ↓
Future Sales
```

> **Time as X → Forecasting future values**
> 
> **Non-time X → Predicting an unknown value**

---

# Types of Regression

Regression models can be broadly:

### Linear Regression

The relationship is represented using a **straight line**.

```text
y
│
│       •
│     •
│   •
│ •
└──────────── x
```

### Non-Linear Regression

The relationship is represented using a **curve or more complex function**.

```text
y
│
│    •
│  •
│ •
│   •
│      •
└──────────── x
```

---

# Regression Models

The main regression algorithms covered are:

1. **Simple Linear Regression**
    
2. **Multiple Linear Regression**
    
3. **Polynomial Regression**
    
4. **Support Vector Regression (SVR)**
    
5. **Decision Tree Regression**
    
6. **Random Forest Regression**


---

## 1. Simple Linear Regression

Uses **one independent variable** to predict the target.

Example:

```text
Fertilizer (X)
      ↓
Simple Linear Regression
      ↓
Potato Yield (y)
```

Basic equation:



---

## 2. Multiple Linear Regression

Uses **multiple independent variables** to predict one target.

Example:

```text
Fertilizer ──┐
Rainfall ────┤
Temperature ─┤──→ Regression Model ──→ Potato Yield
Soil Quality ┘
```

---

## 3. Polynomial Regression

Used when the relationship between the variables is **curved rather than a straight line**.

```text
x → Polynomial features → Regression Model → y
```

---

## 4. Support Vector Regression (SVR)

SVR uses the principles of **Support Vector Machines** to perform regression.

Instead of finding a classification boundary, it tries to find a suitable function that predicts continuous values while controlling prediction errors.

---

## 5. Decision Tree Regression

Uses a **decision tree** to divide the data into different regions and make numerical predictions.

```text
             Fertilizer > 50?
                /       \
              Yes        No
              /           \
         Higher yield   Lower yield
```

---

## 6. Random Forest Regression

Combines multiple **Decision Trees** and uses their predictions together to produce the final prediction.

```text
             Dataset
                ↓
      ┌────┬────┬────┬────┐
      ↓    ↓    ↓    ↓    ↓
    Tree Tree Tree Tree Tree
      ↓    ↓    ↓    ↓    ↓
      └────┴────┬───┴─────┘
                ↓
         Combined Prediction
```

---

# 🧠 Key Points

- **Regression → predicts continuous numerical values.**
    
- **Linear regression → straight-line relationship.**
    
- **Non-linear regression → curved/complex relationship.**
    
- **Prediction → unknown present value.**
    
- **Forecasting → future value when time is the independent variable.**
    
- Different regression algorithms are suitable for different relationships and datasets.
    

### Regression Roadmap

```text
Regression
    │
    ├── Simple Linear Regression
    │
    ├── Multiple Linear Regression
    │
    ├── Polynomial Regression
    │
    ├── Support Vector Regression
    │
    ├── Decision Tree Regression
    │
    └── Random Forest Regression
```

