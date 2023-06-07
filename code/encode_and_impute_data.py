from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, _lit_doc, regexp_replace, col, udf
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import utilities
import preprocess_raw_data


def one_hot_encoding(input_df, cat_var, uniq_id):
    vals = input_df.select(cat_var).drop_duplicates().dropna().rdd.map(lambda x: x[0]).collect()
    enc_df = input_df.groupBy(uniq_id) \
        .pivot(cat_var, values=vals) \
        .agg(f.lit(1))
    for val in vals:
        new_column_name = cat_var + "_" + val
        column_map = utilities.init_column_maps()
        map_tuple = list(filter(lambda map: map[0] == cat_var, column_map))[0]
        readable_name = map_tuple[2] + f": {val}"
        with open(utilities.categorical_column_map_file_path, "a") as fi:
            fi.write(f"{new_column_name}:{map_tuple[1]}:{readable_name}\n")
        enc_df = enc_df.withColumnRenamed(val, new_column_name)
    enc_df = enc_df.toDF(*(c.replace(" ", "_") for c in enc_df.columns))
    enc_df = enc_df.toDF(*(c.replace(",", "") for c in enc_df.columns))
    enc_df = enc_df.drop(vals[0]).na.fill(0)
    enc_df.printSchema()
    input_df = input_df.drop(cat_var).join(enc_df, on=uniq_id, how="inner")
    return input_df

# def clean_training_and_test_data_and_save(folder_name):
#     spark = init_spark()
#     test_df = spark.read.parquet(f"../{folder_name}/test_data.parquet")
#     imp_test_df = impute_df(test_df)
#     imp_test_df.toPandas().to_csv(f"../{folder_name}/imputed_test_data.csv")
#     imp_test_df.write.parquet(f"../{folder_name}/imputed_test_data.parquet")
#     train_df = spark.read.parquet(f"../{folder_name}/training_data.parquet")
#     imp_train_df = impute_df(train_df)
#     imp_train_df.toPandas().to_csv(f"../{folder_name}/imputed_training_data.csv")
#     imp_train_df.write.parquet(f"../{folder_name}/imputed_training_data.parquet")


def impute_df(df):
    df_columns = df.schema.names
    df_columns.remove("School_Code")
    imp_columns = [f"{x}_imputed" for x in df_columns]
    impute_cols = Imputer(inputCols=df_columns, outputCols=imp_columns)
    imp_df = impute_cols.fit(df).transform(df)
    for col_name in df_columns:
        imp_df = imp_df.drop(col_name)
    return imp_df


def split_categorical_and_numerical_data(ov, split):
    spark = utilities.init_spark()
    ov_df = spark.read.parquet(f"../data/test_training_data/{ov}/preprocessed_{split}_data.parquet")
    ov_num_df = ov_df
    with open(utilities.categorical_school_characteristics_file_path) as f:
        categorical_features = f.readlines()
        for feature in categorical_features:
            print(feature)
            feature = feature.replace("\n", "")
            ov_num_df = ov_num_df.drop(feature)
        select_param = preprocess_raw_data.generate_select_parameter(utilities.categorical_school_characteristics_file_path)
        select_param.append("School_Code")
        ov_cat_df = ov_df.select(select_param)
    return ov_num_df, ov_cat_df


def encode_and_impute_data():
    spark = utilities.init_spark()
    with open(utilities.output_variables_file_path) as fi:
        output_variables = fi.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for ov in output_variables:
            ov = ov.replace("\n", "")
            train_num_df, train_cat_df = split_categorical_and_numerical_data(ov, "training")
            test_num_df, test_cat_df = split_categorical_and_numerical_data(ov, "test")
            cat_df = train_cat_df.union(test_cat_df)
            categorical_columns = cat_df.schema.names
            categorical_columns.remove("School_Code")
        for category in categorical_columns:
            cat_df = one_hot_encoding(cat_df, category, "School_Code")
        test_num_df = impute_df(test_num_df)
        train_num_df = impute_df(train_num_df)
        test_num_df.write.mode("overwrite").parquet(f"../data/test_training_data/{ov}/final_num_test_data.parquet", "overwrite")
        test_num_df.toPandas().to_csv(f"../data/test_training_data/{ov}/final_num_test_data.csv", sep=":", index=False)
        train_num_df.write.mode("overwrite").parquet(f"../data/test_training_data/{ov}/final_num_training_data.parquet",
                                                 "overwrite")
        train_num_df.toPandas().to_csv(f"../data/test_training_data/{ov}/final_num_training_data.csv", sep=":", index=False)
        test_df = test_num_df.join(cat_df, how="inner", on="School_Code")
        train_df = train_num_df.join(cat_df, how="inner", on="School_Code")
        test_df.write.mode("overwrite").parquet(f"../data/test_training_data/{ov}/final_test_data.parquet", "overwrite")
        test_df.toPandas().to_csv(f"../data/test_training_data/{ov}/final_test_data.csv", sep=":", index=False)
        train_df.write.mode("overwrite").parquet(f"../data/test_training_data/{ov}/final_training_data.parquet", "overwrite")
        train_df.toPandas().to_csv(f"../data/test_training_data/{ov}/final_training_data.csv", sep=":", index=False)




