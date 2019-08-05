'''
Linear Regression Project
Ecommerce company based in New York 
Determine whether to focus on mobile or website platforms
'''
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def gather_data(file):
    df = pd.read_csv(file)
    print(df.head())
    return df

def explore_Data(data_frame):
    sns.jointplot('Time on Website', 'Yearly Amount Spent', data = data_frame)
    plt.show()
    sns.jointplot('Time on App', 'Yearly Amount Spent', data = data_frame)
    plt.show()
    sns.jointplot('Time on App', 'Length of Membership', data = data_frame, kind = "hex bin")
    plt.show()
    sns.pairplot(data = data_frame)
    plt.show()
    sns.lmplot('Yearly Amount Spent', 'Length of Membership', data_frame)
    plt.show()


def split_data(data_frame):
    X = data_frame[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = data_frame['Yearly Amount Spent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test, X


def train_model(X_train, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    return lm

def evaluate_model(model, X_test, y_test):
    print('Coefficient:', model.coef_)
    predictions = model.predict(X_test)
    sq = np.sqrt(metrics.mean_squared_error(y_test,predictions))
    print(f"Mean absolute Error: {metrics.mean_absolute_error(y_test, predictions)}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, predictions)}")
    print(f"Root Mean Squared Error: {sq}")
    data_explore = input("Do you want to see a graph of the data?")
    if data_explore == 'yes':
        sns.jointplot(predictions,y_test, kind= "scatter")
        plt.show()
    return predictions


def examine_residuals(model,X):
    coef = pd.DataFrame(model.coef_, X.columns, columns = ['Coef'])
    print(coef)

if __name__ == '__main__':
    filename = 'Ecommerce Customers'
    data_frame = gather_data(filename)
    explore = input('Would you like to explore the data further?')
    if explore == 'yes':
        explore_Data(data_frame)
    data = split_data(data_frame)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    X = data[4]
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    examine_residuals(model, X)

