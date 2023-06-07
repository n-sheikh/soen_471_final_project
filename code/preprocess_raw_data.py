# spark imports
from pyspark.sql.functions import row_number, _lit_doc, regexp_replace, col, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType, IntegerType
import utilities


def rename_columns(df):
    column_maps = utilities.init_column_maps()
    original_names = df.schema.names
    for original_name in original_names:
        column_map = list(filter(lambda map: map[1] == original_name, column_maps))
        if len(column_map) != 0:
            column_map = column_map[0]
            df = df.withColumnRenamed(original_name, column_map[0])
    return df


def remove_percentage(col_val):
    if '%' in col_val:
        col_val = col_val.replace('%', "")
        return float(col_val)
    else:
        return col_val


def generate_select_parameter(parameter_file):
    with open(parameter_file, encoding="utf-8") as parameters:
        parameter_list = parameters.readlines()
        parameter_list = list(map(lambda x: x.replace("\n", ""), parameter_list))
    return parameter_list


def shift_town_labels(input_df, value_label):
    df_values = input_df.filter(input_df["Label"] == value_label)
    df_towns = input_df.filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Estimate") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Margin of Error") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Percent") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Percent Margin of Error").select("Label")
    w = Window().orderBy("Label")
    df_values = df_values.withColumn("row_num", row_number().over(w))
    df_towns = df_towns.withColumn("row_num", row_number().over(w))
    df_towns = df_towns.withColumnRenamed("Label", "PLACE")
    df = df_towns.join(df_values, df_values.row_num == df_towns.row_num).drop("row_num").drop(
        "Label")
    df = df.filter(df["PLACE"] != 'undefined')
    return df


def preprocess_school_characteristics(raw_data_file_path, feature_file_path, integer_feature_file_path,
                                      double_feature_file_path):
    spark = utilities.init_spark()
    df = spark.read.csv(raw_data_file_path, header=True, encoding="utf-8")
    df = rename_columns(df)
    parameter_list = generate_select_parameter(feature_file_path)
    df = df.select(parameter_list)
    with open(double_feature_file_path) as parameters:
        parameter_list = parameters.readlines()
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(DoubleType()))
    with open(integer_feature_file_path) as parameters:
        parameter_list = parameters.readlines()
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(IntegerType()))
    df.write.mode("overwrite").parquet(utilities.preprocessed_school_characteristics_file_path_pq, "overwrite")
    df.toPandas().to_csv(utilities.preprocessed_school_characteristics_file_path_csv, sep=":",
                         index=False)


def preprocess_output_variables(raw_data_file_path, output_variable_file_path, integer_variable_file_path,
                                double_variable_file_path):
    spark = utilities.init_spark()
    df = spark.read.csv(raw_data_file_path, header=True, encoding="utf-8")
    df = rename_columns(df)
    parameter_list = generate_select_parameter(output_variable_file_path)
    df = df.select(parameter_list)
    with open(double_variable_file_path) as parameters:
        parameter_list = parameters.readlines()
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(DoubleType()))
    with open(integer_variable_file_path) as parameters:
        parameter_list = parameters.readlines()
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(IntegerType()))
    df.write.mode("overwrite").parquet(utilities.preprocessed_output_variables_file_path_pq, "overwrite")
    df.toPandas().to_csv(utilities.preprocessed_output_variables_file_path_csv, sep=":",
                         index=False)


def preprocess_census_data(raw_data_file_path, aspect, percentage_parameter_file=None, estimate_parameter_file=None):
    spark = utilities.init_spark()
    df = spark.read.csv(raw_data_file_path, header=True, encoding="utf-8")
    df = rename_columns(df)
    if percentage_parameter_file is not None:
        parameter_list = generate_select_parameter(percentage_parameter_file)
        df_percentage = df.select(parameter_list)
        df_percentage = shift_town_labels(df_percentage, "\xa0\xa0\xa0\xa0Percent")
        cast_to_float = udf(lambda x: remove_percentage(x), StringType())
        with open(percentage_parameter_file) as parameters:
            parameter_list = parameters.readlines()
            parameter_list.remove("Label\n")
            for parameter in parameter_list:
                parameter = parameter.replace("\n", "")
                df_percentage = df_percentage.withColumn(parameter, cast_to_float(col(parameter)).cast(DoubleType()))
    if estimate_parameter_file is not None:
        parameter_list = generate_select_parameter(estimate_parameter_file)
        df_estimate = df.select(parameter_list)
        df_estimate = shift_town_labels(df_estimate, "\xa0\xa0\xa0\xa0Estimate")
        remove_comma = udf(lambda x: x.replace(",", ""), StringType())
        with open(estimate_parameter_file) as parameters:
            parameter_list = parameters.readlines()
            parameter_list.remove("Label\n")
            for parameter in parameter_list:
                parameter = parameter.replace("\n", "")
                df_estimate = df_estimate.withColumn(parameter, remove_comma(col(parameter)).cast(IntegerType()))
    if percentage_parameter_file is None:
        df = df_estimate
    elif estimate_parameter_file is None:
        df = df_percentage
    else:
        df_estimate = df_estimate.withColumnRenamed("PLACE", "ESTIMATE_PLACE")
        df = df_percentage.join(df_estimate, df_estimate.ESTIMATE_PLACE == df_percentage.PLACE)
        df = df.drop("ESTIMATE_PLACE")
    df = df.withColumn("PLACE", regexp_replace("PLACE", " CDP, Massachusetts", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " city, Massachusetts", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Center", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Corner", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Town", ""))
    df.write.mode("overwrite").parquet(
        f"../data/preprocessed_data/{aspect}_characteristics/{aspect}_characteristics.parquet", "overwrite")
    df.toPandas().to_csv(f"../data/preprocessed_data/{aspect}_characteristics/{aspect}_characteristics.csv",
                         sep=":", index=False)


def preprocess_all_data():
    preprocess_school_characteristics(utilities.raw_school_data_file_path,
                                      utilities.school_characteristics_file_path,
                                      utilities.integer_school_characteristics_file_path,
                                      utilities.double_school_characteristics_file_path)
    preprocess_census_data(utilities.raw_social_characteristics_file_path, aspect="social",
                           percentage_parameter_file=utilities.percentage_social_characteristics_file_path)
    preprocess_census_data(utilities.raw_demographic_characteristics_file_path, aspect="demographic",
                           percentage_parameter_file=utilities.percentage_demographic_characteristics_file_path,
                           estimate_parameter_file=utilities.estimate_demographic_characteristics_file_path)
    preprocess_census_data(utilities.raw_housing_characteristics_file_path, aspect="housing",
                           percentage_parameter_file=utilities.percentage_housing_characteristics_file_path)
    preprocess_census_data(utilities.raw_economic_characteristics_file_path, aspect="economic",
                           percentage_parameter_file=utilities.percentage_economic_characteristics_file_path,
                           estimate_parameter_file=utilities.estimate_economic_characteristics_file_path)
    preprocess_output_variables(utilities.raw_school_data_file_path,
                                utilities.output_variables_file_path,
                                utilities.integer_output_variables_file_path,
                                utilities.double_output_variables_file_path)
