import utilities
import preprocess_raw_data
import exploratory_analysis
import generate_test_training_sets
import encode_and_impute_data
import regression_utilities
import decision_tree_regression
import numpy as np
import pandas as pd
import pca
import random_forest_regression


def main():
    #preprocess_raw_data.preprocess_all_data()
    #exploratory_analysis.generate_histograms_for_output_variable()
    #exploratory_analysis.generate_histograms_for_output_variable_avg_by_town()
    #generate_test_training_sets.combine_census_features()
    #generate_test_training_sets.combine_all_features()
    #generate_test_training_sets.generate_test_training_sets()
    #encode_and_impute_data.encode_and_impute_data()
    #generate_test_training_sets.generate_x_y_splits()
    #decision_tree_regression.baseline_decision_tree_regression()
    #decision_tree_regression.print_and_save_decision_trees("dt_baseline")
    #decision_tree_regression.hyper_parameter_tuning_decision_tree_regression()
    #decision_tree_regression.print_and_save_decision_trees("dt_tuned")
    #decision_tree_regression.baseline_pca_decision_tree_regression()
    # random_forest_regression.baseline_pca_random_forest_regression()
    # pca.perform_pca()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
