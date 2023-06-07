
import utilities
import regression_utilities
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle as pk

def baseline_random_forest_regression():
    clf = RandomForestRegressor()
    regression_utilities.regress(clf, "rf_baseline", "x_train", "x_test", True)

def baseline_pca_random_forest_regression():
    clf = RandomForestRegressor()
    regression_utilities.regress(clf, "rf_baseline_pca", "x_pca_train", "x_pca_test", False)

def hyper_parameter_tuning_random_forest_regression():

    n_estimators = [int(x) for x in np.linspace(start=80, stop=200, num=5)]
    max_depth = [int(x) for x in np.linspace(5, 15, num=3)]
    # min_samples_split = [1, 5, 10]
    min_samples_leaf = [1, 5, 10]
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth' : max_depth,
        'min_samples_leaf': min_samples_leaf
    }
    #  in this function I set the output variable(ov) manually and run this function to get the result
    ov = 'OP1'
    technique='rf_tuned'
    X_train = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":")
    y_train = pd.read_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":")
    X_test = pd.read_csv(f"../data/test_training_data/{ov}/x_test.csv", sep=":")

    model = RandomForestRegressor()
    rf_Grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, cv=3, n_jobs=1)
    rf_Grid.fit(X_train, y_train)
    rf_regressor = rf_Grid.best_estimator_
    y_predict = rf_regressor.predict(X_test)

    with open(f'../models/{technique}/{ov}.pkl', 'wb') as model_file:
        pk.dump(rf_regressor, model_file)
    with open(f'../results/{technique}/{ov}.pkl', 'wb') as prediction_file:
        pk.dump(y_predict, prediction_file)

    regression_utilities.print_save_metrics(y_predict, technique, ov)
    regression_utilities.feature_importance(rf_regressor.feature_importances_, technique, ov)
    regression_utilities.print_save_regressor_params(rf_regressor, technique, ov)
