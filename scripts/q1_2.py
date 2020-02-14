#!/bin/python3
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from q1_1 import preprocess

def prepare_reg(X_train, y_train, X_test, y_test, degree_of_polynomial):
    poly = PolynomialFeatures(degree=degree_of_polynomial)
    X_train_, X_test_ = X_train.reshape(-1, 1), X_test.reshape(-1,1)
    X_train_, X_test_ = poly.fit_transform(X_train_), poly.fit_transform(X_test_)
    reg = LinearRegression(n_jobs=-1).fit(X_train_, y_train)
    y_predict = reg.predict(X_test_)
    reg_bias_square = np.mean((np.mean(y_predict)-y_test)**2)
    reg_var = np.var(y_predict)
    return reg_bias_square, reg_var

def plot_bias_variance(X_trains, y_trains, X_test, y_test, degrees):
    bias_square, var = [],[]
    for deg in tqdm(degrees):
        deg_bias_square, deg_var = [], []
        for X_train, y_train in zip(X_trains, y_trains):
            tmp_deg_bias_square, tmp_deg_var = prepare_reg(X_train, y_train, X_test, y_test, deg)
            deg_bias_square.append(tmp_deg_bias_square)
            deg_var.append(tmp_deg_var)
        bias_square.append(np.mean(deg_bias_square))
        var.append(np.mean(deg_var))
    print(pd.DataFrame(data=list(zip(bias_square, var)), index=degrees, columns=['bias square', 'variance'])) 
    plt.plot(degrees, bias_square, 'b')
    plt.plot(degrees, var, 'r')
    plt.show()

if __name__ == "__main__":
    X_trains, X_test, y_trains, y_test = preprocess('../resources/Q1_data/data.pkl')
    plot_bias_variance(X_trains, y_trains, X_test, y_test, list([i for i in range(1, 10)])) 
