# Simple Linear Regression

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('../data/raw/salary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.8, \
                                                    random_state=0)

# Fitting Simple Linear Regression to the Training set
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model_lr.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, model_lr.predict(X_train), color='blue')

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, model_lr.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
