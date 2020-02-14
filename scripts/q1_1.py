#!/bin/python3
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def preprocess(path_to_pickle):
    dataset = pickle.load(open(path_to_pickle, 'rb'))
    X = dataset[:,0]
    y = dataset[:,1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    try:
        X_trains = np.split(X_train, 10)
        y_trains = np.split(y_train, 10)
    except ValueError:
        print('train data split resulted in unequal division')
        return 
    return X_trains, X_test, y_trains, y_test

def prepare_reg(X_trains, y_trains, X_test, y_test, degree_of_polynomial):
    y_predict = np.zeros(y_test.shape)
    for X_train, y_train in zip(X_trains, y_trains):
        poly = PolynomialFeatures(degree=degree_of_polynomial)
        X_train_, X_test_ = X_train.reshape(-1, 1), X_test.reshape(-1,1)
        X_train_, X_test_ = poly.fit_transform(X_train_), poly.fit_transform(X_test_)
        reg = LinearRegression(n_jobs=-1).fit(X_train_, y_train)
        y_predict += reg.predict(X_test_)
    y_predict=y_predict/len(X_trains[0])
    reg_bias_square = np.mean((np.mean(y_predict)-y_test)**2)
    reg_var = np.var(y_predict)
    return reg_bias_square, reg_var

def plot_bias_variance(X_trains, y_trains, X_test, y_test, degrees):
    bias_square, var = [],[]
    for deg in tqdm(degrees):
        reg_deg_bias_square, reg_deg_var = prepare_reg(X_trains, y_trains, X_test, y_test, deg)
        bias_square.append(reg_deg_bias_square)
        var.append(reg_deg_var)
    print(pd.DataFrame(data=list(zip(bias_square, var)), index=degrees, columns=['bias square', 'variance'])) 
    plt.plot(degrees, bias_square, 'b')
    plt.plot(degrees, var, 'r')
    plt.show()

if __name__ == "__main__":
    X_trains, X_test, y_trains, y_test = preprocess('../resources/Q1_data/data.pkl')
    plot_bias_variance(X_trains, y_trains, X_test, y_test, list([i for i in range(1, 10)])) 
