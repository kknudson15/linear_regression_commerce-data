'''
Linear Regression Project
Ecommerce company based in New York 
Determine whether to focus on mobile or website platforms
'''
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('Ecommerce Customers')
print(df.head())

'''
Exploratory Data Analysis
Use Seaborn to create a jointplot to compare the Time on website and yearly amount spent columns
Does the correlation make sense?

This correlation makes some sense because the longer time spent on the website should correspond to 
more money being spent however there is a limit to this trend.  The data surronds 37 and this seems to be the peak
for this relationship. 
'''
sns.jointplot('Time on Website', 'Yearly Amount Spent', data = df)
#plt.show()

'''
Do the same but with Time on App column instead.
Less time is needed on the App to produce about the same amount of money spent for the year. ~12 compared 
to 37.
'''
sns.jointplot('Time on App', 'Yearly Amount Spent', data = df)
#plt.show()

'''
Use joinplot to create a 2D heax bin plot comparing time on app and length of membership
'''
sns.jointplot('Time on App', 'Length of Membership', data = df, kind = "hex bin")
#plt.show()

'''
use pairplot to recreate plot in notebook
Based off this plot what looks to be the most correlated feature with yearly amount spent?
Length of membership appears to have the most correlated feature in this data set.

Create a linear model plot (using seaborn's implot) of yearly amount spent vs length of membership
'''

sns.pairplot(data = df)
#plt.show()

sns.lmplot('Yearly Amount Spent', 'Length of Membership', df)
#plt.show()


'''
Training and Testing the Data
Set a Variable X equal to the numerical features of the customers and a 
varibale y equal to the "Yearly Amount Spent" column

split the data into training and testing sets.  Set test_size = 0.3 and random_state = 101
'''
#print(df.columns)

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

'''
Training the model 
Import LinearRegression from sklearn.linear_model

Create an instance of a Linear Regression() model named lm

train/fit lm on the training data 

print the coefficients of the model
'''

from sklearn.linear_model import LinearRegression 

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.coef_)


'''
Predicting Test Data 
Use lm.predict() to predict off the X_test set of the data 

Create a scatterplot of the real test values versus the predicted values
'''
predictions = lm.predict(X_test)
sns.jointplot(predictions,y_test, kind= "scatter")
#plt.show()



'''
Evaluating the Model 

Calculate the Mean Absolute Error, Mean Squared Error and the Root Mean Squared Error.

'''
from sklearn import metrics

print(f"Mean absolute Error: {metrics.mean_absolute_error(y_test, predictions)}")
print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, predictions)}")

sq = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print(f"Root Mean Squared Error: {sq}")


'''
Residuals
Plot a histogram of the residuals and make sure it looks normally distributed. use 
either seabornd distplot or just plt.hist()
'''
sns.distplot(y_test-predictions)
#plt.show()


coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coef'])
print(coef)
'''
Conclusion 
How can you interpret these coefficients?
With an increase in unit by one the amount of money spent wil increase by this much. 

Do You think the company should focus more on their mobile app or on their website?

THe company should focus on their mobile app because increasing by one unit will result in 387.590159 dollar increase in 
revenue where as an increase in one unit for the time on website will result very little increase, 0.190405 
'''

