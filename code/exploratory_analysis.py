import matplotlib.pyplot as plt
import math
from tabulate import tabulate
import utilities
import seaborn as sns
import pandas as pd


def generate_histograms_for_output_variable():
    spark = utilities.init_spark()
    output_df = spark.read.parquet(utilities.preprocessed_output_variables_file_path_pq)
    column_maps = utilities.init_column_maps()
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for output_variable in output_variables:
            output_variable = output_variable.replace("\n", "")
            readable_output_variable = list(filter(lambda col_map: col_map[0] == output_variable, column_maps))[0][2]
            title = readable_output_variable + " vs Number of Schools"
            output_np = output_df.select(output_variable).dropna().toPandas()
            n = output_np.count()
            nos_of_intervals = math.ceil(math.sqrt(n))
            plt.hist(output_np, bins=nos_of_intervals)
            plt.xlabel(readable_output_variable)
            plt.ylabel("Number of Schools")
            plt.title(title)
            plt.savefig(f"../figures/exploratory_analysis/output_variables/{output_variable}_vs_Number_of_Schools.png")
            plt.clf()
    return None


def generate_histograms_for_output_variable_avg_by_town():
    spark = utilities.init_spark()
    output_df = spark.read.parquet(utilities.preprocessed_output_variables_file_path_pq)
    column_maps = utilities.init_column_maps()
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for output_variable in output_variables:
            output_variable = output_variable.replace("\n", "")
            output_variable_df = output_df.select("School_Code", "Town", output_variable).dropna()
            output_variable_pd = output_variable_df.groupBy("Town").agg({output_variable: "mean"}).drop("Town")\
                .toPandas()
            n = output_variable_pd.count()
            nos_of_intervals = math.ceil(math.sqrt(n))
            plt.hist(output_variable_pd, bins=nos_of_intervals)
            readable_output_variable = list(filter(lambda col_map: col_map[0] == output_variable, column_maps))[0][2]
            title = "Average " + readable_output_variable + " vs Number of Towns"
            plt.xlabel(readable_output_variable)
            plt.ylabel("Number of Towns")
            plt.title(title)
            plt.savefig(f"../figures/exploratory_analysis/output_variables/{output_variable}_vs_Number_of_Towns.png")
            plt.clf()


def generate_box_plots_for_census_input_parameters(parameter_df):
    parameters = parameter_df.schema.names
    parameters.remove("PLACE")
    for param in parameters:
        parameter_np = parameter_df.select(param).dropna().toPandas()
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(parameter_np)
        title = param
        plt.title(title)
        plt.savefig(f"../figures/{param}.png")
        plt.clf()


def generate_box_plots_for_school_input_parameters(parameter_df):
    parameters = parameter_df.schema.names
    with open("../data/school_categorical_characteristics.txt") as f:
        categorical_parameters = f.readlines()
        for cat_param in categorical_parameters:
            cat_param = cat_param.replace("\n", "")
            print(cat_param)
            parameters.remove(cat_param)
    for param in parameters:
        parameter_np = parameter_df.select(param).dropna().toPandas()
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(parameter_np)
        title = param
        plt.title(title)
        plt.savefig(f"../figures/{param}.png")
        plt.clf()


def generate_schools_per_town():
    spark = utilities.init_spark()
    school_df = spark.read.parquet("../data/school_characteristics.parquet")
    with open("../figures/Count_Of_School_Town.txt", "x") as f:
        f.write(tabulate(school_df.groupBy("Town").count().orderBy("count").toPandas(), headers=["Town", "Count of Schools"]))


def generate_heat_map(ov,x_train):

    x_num_train = pd.read_csv(f"../data/test_training_data/{ov}/{x_train}.csv", sep=':')
    corr_pearson = x_num_train.corr(method='pearson')
    # print(x_num_train)
    heat=sns.heatmap(corr_pearson)
    plt.savefig(f'../figures/PCA_OP/heatmap_{x_train}' + ov, dpi=400)
    plt.close()
