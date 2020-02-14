#!/bin/python3
import pickle
from q1_2 import prepare_reg, plot_bias_variance
from q2_1 import preprocess

if __name__ == "__main__":
    X_trains, X_test, y_trains, y_test = preprocess('../resources/Q2_data/X_train.pkl', '../resources/Q2_data/Y_train.pkl', '../resources/Q2_data/X_test.pkl', '../resources/Q2_data/Fx_test.pkl')
    plot_bias_variance(X_trains, y_trains, X_test, y_test, list([i for i in range(1, 10)]))
