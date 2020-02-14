#!/bin/python3
import pickle
from q1_1 import prepare_reg, plot_bias_variance

def preprocess(path_to_X_trains, path_to_y_trains, path_to_X_test, path_to_y_test):
    X_trains = pickle.load(open(path_to_X_trains, 'rb'))
    X_test = pickle.load(open(path_to_X_test, 'rb'))
    y_trains = pickle.load(open(path_to_y_trains, 'rb'))
    y_test= pickle.load(open(path_to_y_test, 'rb'))

    return X_trains, X_test, y_trains, y_test

if __name__ == "__main__":
    X_trains, X_test, y_trains, y_test = preprocess('../resources/Q2_data/X_train.pkl', '../resources/Q2_data/Y_train.pkl', '../resources/Q2_data/X_test.pkl', '../resources/Q2_data/Fx_test.pkl')
    plot_bias_variance(X_trains, y_trains, X_test, y_test, list([i for i in range(1, 10)]))
