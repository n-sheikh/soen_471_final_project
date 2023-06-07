import pandas as pd
import numpy as np
import utilities
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import csv
import pickle as pk


"""
Based on Shahreh's function random_forest()
Refactored by Nadia to ensure usability for both Random Forests and Decision Trees
"""


def regress_ov(regressor, technique, ov, x_train, x_test):
    x_train = pd.read_csv(f"../data/test_training_data/{ov}/{x_train}.csv", sep=":")
    print(len(x_train))
    y_train = pd.read_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":")
    x_test = pd.read_csv(f"../data/test_training_data/{ov}/{x_test}.csv", sep=":")
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    with open(f'../models/{technique}/{ov}.pkl', 'wb') as model_file:
        pk.dump(regressor, model_file)
    with open(f'../results/{technique}/{ov}.pkl', 'wb') as prediction_file:
        pk.dump(y_pred, prediction_file)
    return regressor, y_pred


"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code 
Modified by Nadia 
"""


def print_save_metrics(y_pred, technique, ov):
    y_test = pd.read_csv(f"../data/test_training_data/{ov}/y_test.csv", sep=":")
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    with open(f"../results/{technique}/metrics_{ov}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["r2_score", "Mean Absolute Error (MAE)", 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'])
        writer.writerow([r2_score, mae, mse, rmse])
    print("r2_score:" + ov, r2_score)
    print('Mean Absolute Error (MAE):' + ov, mae)
    print('Mean Squared Error (MSE):' + ov, mse)
    print('Root Mean Squared Error (RMSE):' + ov, rmse)


"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code 
Modified to save feature importance
"""


def feature_importance(regressor_feature_importance, technique, ov):
    x_test_columns = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", ":").columns
    column_map = utilities.init_column_maps()
    important_features = sorted(list(zip(regressor_feature_importance, x_test_columns)), key=lambda x: x[0],
                                reverse=True)[:10]
    readable_important_features = []
    for feature in important_features:
        feature = (feature[0], feature[1].replace('_imputed', ''))
        column_map_tuple = list(filter(lambda column_name: feature[1] == column_name[0], column_map))[0]
        readable_important_features.append((feature[0], column_map_tuple[2]))
    file = open(f'../results/{technique}/feature_importance{ov}.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(readable_important_features)
    print(readable_important_features)
    return None


def print_save_regressor_params(regressor, technique, ov):
    with open(f"../results/{technique}/regressor_params_{ov}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value"])
        for param, value in regressor.get_params(deep=True).items():
            writer.writerow([param, value])


def regress(regressor, technique, x_train, x_test, f_imp):
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for ov in output_variables:
            ov = ov.replace("\n", "")
            regressor, y_pred = regress_ov(regressor, technique, ov, x_train, x_test)
            print_save_metrics(y_pred, technique, ov)
            if f_imp:
                feature_importance(regressor.feature_importances_, technique, ov)
            print_save_regressor_params(regressor, technique, ov)


def hyper_parameter_tuning_ov(p_grid, regressor, ov, technique):
    x_train = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":")
    y_train = pd.read_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":")
    x_test = pd.read_csv(f"../data/test_training_data/{ov}/x_test.csv", sep=":")
    grid = GridSearchCV(estimator=regressor,
                        param_grid=p_grid,
                        cv=10,
                        n_jobs=6)
    grid.fit(x_train, y_train)
    best_regressor = grid.best_estimator_
    y_pred = best_regressor.predict(x_test)
    with open(f'../models/{technique}/{ov}.pkl', 'wb') as model_file:
        pk.dump(best_regressor, model_file)
    with open(f'../results/{technique}/{ov}.pkl', 'wb') as prediction_file:
        pk.dump(y_pred, prediction_file)
    print_save_metrics(y_pred, technique, ov)
    feature_importance(best_regressor.feature_importances_, technique, ov)
    print_save_regressor_params(best_regressor, technique, ov)
