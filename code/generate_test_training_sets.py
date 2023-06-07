from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, _lit_doc, regexp_replace, col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import utilities


def combine_census_features():
    spark = utilities.init_spark()
    social_characteristics_df = spark.read.parquet(utilities.preprocessed_social_characteristics_file_path_pq)
    economic_characteristics_df = spark.read.parquet(utilities.preprocessed_economic_characteristics_file_path_pq)
    housing_characteristics_df = spark.read.parquet(utilities.preprocessed_housing_characteristics_file_path_pq)
    demographic_characteristics_df = spark.read.parquet(utilities.preprocessed_demographic_characteristics_file_path_pq)
    social_characteristics_df = social_characteristics_df.withColumnRenamed("PLACE", "SC_PLACE")
    economic_characteristics_df = economic_characteristics_df.withColumnRenamed("PLACE", "EC_PLACE")
    housing_characteristics_df = housing_characteristics_df.withColumnRenamed("PLACE", "HC_PLACE")
    demographic_characteristics_df = demographic_characteristics_df.withColumnRenamed("PLACE", "DC_PLACE")
    df = social_characteristics_df.join(economic_characteristics_df,
                                        social_characteristics_df.SC_PLACE == economic_characteristics_df.EC_PLACE)
    df = df.join(housing_characteristics_df, df.SC_PLACE == housing_characteristics_df.HC_PLACE)
    df = df.join(demographic_characteristics_df, df.SC_PLACE == demographic_characteristics_df.DC_PLACE)
    df = df.withColumnRenamed("SC_PLACE", "PLACE").drop("EC_PLACE").drop("HC_PLACE").drop("DC_PLACE")
    df.write.mode("overwrite").parquet(utilities.preprocessed_combined_census_characteristics_file_path_pq, "overwrite")
    df.toPandas().to_csv(utilities.preprocessed_combined_census_characteristics_file_path_csv
                         , sep=":", index=False)


def combine_all_features():
    spark = utilities.init_spark()
    census_df = spark.read.parquet(utilities.preprocessed_combined_census_characteristics_file_path_pq)
    school_df = spark.read.parquet(utilities.preprocessed_school_characteristics_file_path_pq)
    features_df = school_df.join(census_df, census_df.PLACE == school_df.Town).drop("PLACE")
    features_df.write.mode("overwrite").parquet(utilities.preprocessed_combined_characteristics_file_path_pq)
    features_df.toPandas().to_csv(utilities.preprocessed_combined_characteristics_file_path_csv, sep=":", index=False)


def generate_test_train_split(output_variable, stratification_bin):
    column_maps = utilities.init_column_maps()
    readable_output_variable = list(filter(lambda col_map: col_map[0] == output_variable, column_maps))
    spark = utilities.init_spark()
    output_df = spark.read.parquet(utilities.preprocessed_output_variables_file_path_pq)
    var_df = output_df.select("Town", "School_Code", output_variable).dropna()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    ov_df = output_df.select("Town", output_variable).dropna()
    ov_df = ov_df.groupby("Town").agg({output_variable: 'mean'})
    output_variable = ov_df.drop("Town").schema.names[0]
    ov_pd = ov_df.filter(ov_df[output_variable] > 0).toPandas()
    ov_cat = output_variable + "_cat"
    ov_pd[ov_cat] = pd.cut(ov_pd[output_variable], bins=stratification_bin, labels=range(1, len(stratification_bin)))
    ax = ov_pd[ov_cat].hist()
    ax.set_title(f"Stratified {readable_output_variable}")
    ax.set_xlabel("Strata")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    fig.savefig(f"../figures/exploratory_analysis/output_variables/{output_variable}_stratification.png")
    fig.clf()
    for train_index, test_index in split.split(ov_pd, ov_pd[ov_cat]):
        train = ov_pd.loc[train_index]
        test = ov_pd.loc[test_index]
    train_df = spark.createDataFrame(train).drop(ov_cat).drop(ov_cat.replace("_cat", ""))
    train_df = train_df.join(var_df, on="Town", how="inner").drop("Town").dropna()
    test_df = spark.createDataFrame(test).drop(ov_cat).drop(ov_cat.replace("_cat", ""))
    test_df = test_df.join(var_df, on="Town", how="inner").drop("Town")
    return [test_df, train_df]


def generate_preprocessed_training_test_data(features_df, ov_test_split, ov_train_split, folder_name):
    test_df = ov_test_split.join(features_df, on="School_Code", how="inner")
    test_df.write.mode("overwrite").parquet(f"../data/test_training_data/{folder_name}/preprocessed_test_data.parquet",
                                            "overwrite")
    test_df.toPandas().to_csv(f"../data/test_training_data/{folder_name}/preprocessed_test_data.csv", sep=":",
                              index=False)
    train_df = ov_train_split.join(features_df, on="School_Code", how="inner")
    train_df.write.mode("overwrite").parquet(f"../data/test_training_data/{folder_name}/preprocessed_training_data"
                                             f".parquet", "overwrite")
    train_df.toPandas().to_csv(f"../data/test_training_data/{folder_name}/preprocessed_training_data.csv", sep=":",
                               index=False)


def generate_stratification_bins():
    with open(utilities.stratification_bins_file_path) as f:
        strata_specifications = f.readlines()
        strata = dict()
        for strata_bin in strata_specifications:
            strata_bin = strata_bin.replace("\n", "")
            strata_bin = strata_bin.split(":")
            strata_key = strata_bin[0]
            strata_bin.remove(strata_key)
            strata_bin = [int(x) for x in strata_bin]
            strata_bin.append(np.inf)
            strata[strata_key] = strata_bin
    return strata


def generate_test_training_sets():
    spark = utilities.init_spark()
    features_df = spark.read.parquet(utilities.preprocessed_combined_characteristics_file_path_pq)
    strata = generate_stratification_bins()
    for ov in strata.keys():
        test_df, train_df = generate_test_train_split(ov, strata[ov])
        generate_preprocessed_training_test_data(features_df, test_df, train_df, ov)


"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code
Modified by Nadia  to save generated train_test_split
"""


def x_y_split(ov, source):
    training = pd.read_csv(f"../data/test_training_data/{ov}/final_training_data.csv", sep=':')
    test = pd.read_csv(f"../data/test_training_data/{ov}/final_test_data.csv", sep=':')
    y_train = training.filter(regex=ov)
    x_train = training.drop(y_train, axis=1).drop(["School_Code"], axis=1)
    y_test = test.filter(regex=ov)
    x_test = test.drop(y_test, axis=1).drop(["School_Code"], axis=1)
    x_train.to_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":", index=False)
    y_train.to_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":", index=False)
    x_test.to_csv(f"../data/test_training_data/{ov}/x_test.csv", sep=":", index=False)
    y_test.to_csv(f"../data/test_training_data/{ov}/y_test.csv", sep=":", index=False)


def generate_x_y_splits():
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for ov in output_variables:
            ov = ov.replace("\n", "")
            x_y_split(ov)


