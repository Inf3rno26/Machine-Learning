---
aliases:
  - Feature
  - Scaling
tags:
  - Data-Preprocessing
  - ML
  - Python
---

---

## It is applied to make the data more comparable with the same units 
or 
Feature scaling is ==a data preprocessing step used in machine learning to bring all numerical features onto a consistent value range or distribution==. 
# Always applied to the Separate Columns and not the rows.

# Two Main types of Feature Scaling:

## 1. Normalization - 

Normalization is the process of taking the minimum inside a column, subtracting that minimum from every single value inside that column, and then dividing by the difference between the maximum and the minimum.

![[Pasted image 20260901211515.png]]

- Every single value in a column is adjusted this way and you will end up with a new column or an adjusted column with values which are all between 0 and 1.


# 2. Standardization - 
Standardization, on the other hand, the process is similar, but instead of subtracting the minimum, we subtract the average, and we divide by the standard deviation. 
- As a result, all of the values, or almost all of the values inside the column will be between -3 and 3.
- If you have some extreme values or some outliers,  then they can end up outside of these -3 and 3 boundaries.
- ==Effective== for data approximately normally distributed
![[Pasted image 20260901211745.png]]





[[The Machine learning Process]]