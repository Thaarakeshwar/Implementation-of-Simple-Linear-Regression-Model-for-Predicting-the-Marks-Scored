<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ba0c2486-90e6-4c16-b36e-8a8d9d29d41f" /># Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.
   

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Thaarakeshwar
RegisterNumber:25014935(212225040466)
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset (Hours studied vs Marks scored)
data = {
    "Hours_Studied": [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7],
    "Marks_Scored":  [21,47,27,75,30,20,88,60,81,25]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
# Step 3: Split into Features and Target
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Visualization
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
```
## Output:

<img width="495" height="169" alt="Screenshot (85)" src="https://github.com/user-attachments/assets/03fb9911-b9c5-4a1d-a770-b0fe29d8701b" />
<img width="373" height="88" alt="Screenshot (86)" src="https://github.com/user-attachments/assets/3d20763d-3c24-41fa-97fa-7cc41ea18ae9" />
<img width="421" height="87" alt="Screenshot (87)" src="https://github.com/user-attachments/assets/4a081be2-ce0b-499e-923f-bc460a373572" />
![Uploading Screenshot (88).png…]()



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
