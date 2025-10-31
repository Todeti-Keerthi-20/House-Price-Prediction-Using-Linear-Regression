import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Step 1: Create the dataset
data = {
    'SquareFeet': [1200, 1500, 1800, 2000, 2200, 2500, 2800],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 4],
    'Bathrooms': [2, 3, 2, 4, 3, 4, 3],
    'Price': [200000, 250000, 300000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Step 3: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Step 7: Print model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 8: Plot actual Vs predicted
# 2D plot: Actual vs Predicted Prices
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()