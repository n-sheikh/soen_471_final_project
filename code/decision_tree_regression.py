import regression_utilities
import utilities
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pickle as pk
import pandas as pd


def baseline_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    regression_utilities.regress(clf, "dt_baseline", "x_train", "x_test", True)


def baseline_pca_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    regression_utilities.regress(clf, "dt_baseline_pca", "x_pca_train", "x_pca_test", False)


def print_and_save_decision_trees(technique):
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for ov in output_variables:
            ov = ov.replace("\n", "")
            df = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":")
            df_cols = df.columns
            print(len(df_cols))
            column_map = utilities.init_column_maps()
            readable_feature_names = []
            for col in df_cols:
                col = col.replace('_imputed', '')
                column_map_list = list(filter(lambda column_tup: col == column_tup[0], column_map))
                if len(column_map_list) > 0:
                    column_map_tuple = column_map_list[0]
                    col = column_map_tuple[2]
                readable_feature_names.append(col)
            with open(f'../models/{technique}/{ov}.pkl', "rb") as f:
                dt = pk.load(f)
                dt_len = dt.tree_.max_depth
                dt_text_rep = tree.export_text(dt, feature_names=readable_feature_names, show_weights=True, max_depth=dt_len)
                with open(f"../results/{technique}/{ov}_dt_structure.txt", "a+") as dtf:
                    dtf.write(f"DT LEN: {dt_len}\n")
                    dtf.write(dt_text_rep)


def hyper_parameter_tuning_decision_tree_regression():
    regressor = DecisionTreeRegressor(random_state=0)

