from pyspark.sql import SparkSession


raw_social_characteristics_file_path = "../data/raw_data/ACSDP5Y2017.DP02-2021-03-23T230834.csv"
percentage_social_characteristics_file_path = "../parameters/social_characteristics/percentage_social_characteristics" \
                                              ".txt"
preprocessed_social_characteristics_file_path_pq = "../data/preprocessed_data/social_characteristics" \
                                                   "/social_characteristics.parquet"
preprocessed_social_characteristics_file_path_csv = "../data/preprocessed_data/social_characteristics/" \
                                                    "social_characteristics.csv"

raw_demographic_characteristics_file_path = "../data/raw_data/ACSDP5Y2017.DP05-2021-03-24T134209.csv"
estimate_demographic_characteristics_file_path = "../parameters/demographic_characteristics" \
                                                 "/estimate_demographic_characteristics.txt"
percentage_demographic_characteristics_file_path = "../parameters/demographic_characteristics" \
                                                   "/percentage_demographic_characteristics.txt"
preprocessed_demographic_characteristics_file_path_pq = "../data/preprocessed_data/demographic_characteristics" \
                                                     "/demographic_characteristics.parquet"
preprocessed_demographic_characteristics_file_path_csv = "../data/preprocessed_data/demographic_characteristics" \
                                                     "/demographic_characteristics.csv"

raw_housing_characteristics_file_path = "../data/raw_data/ACSDP5Y2017.DP04-2021-03-24T123953.csv"
percentage_housing_characteristics_file_path = "../parameters/housing_characteristics" \
                                               "/percentage_housing_characteristics.txt"
preprocessed_housing_characteristics_file_path_pq = "../data/preprocessed_data/housing_characteristics" \
                                                 "/housing_characteristics.parquet"
preprocessed_housing_characteristics_file_path_csv = "../data/preprocessed_data/housing_characteristics" \
                                                 "/housing_characteristics.csv"

raw_economic_characteristics_file_path = "../data/raw_data/ACSDP5Y2017.DP03-2021-03-24T112004.csv"
estimate_economic_characteristics_file_path = "../parameters/economic_characteristics" \
                                              "/estimate_economic_characteristics.txt"
percentage_economic_characteristics_file_path = "../parameters/economic_characteristics" \
                                                "/percentage_economic_characteristics.txt"
preprocessed_economic_characteristics_file_path_pq = "../data/preprocessed_data/economic_characteristics" \
                                                  "/economic_characteristics.parquet"
preprocessed_economic_characteristics_file_path_csv = "../data/preprocessed_data/economic_characteristics" \
                                                  "/economic_characteristics.csv"


preprocessed_combined_census_characteristics_file_path_pq = "../data/preprocessed_data/combined_census_characteristics" \
                                                         "/combined_census_characteristics.parquet"
preprocessed_combined_census_characteristics_file_path_csv = "../data/preprocessed_data/combined_census_characteristics" \
                                                         "/combined_census_characteristics.csv"

raw_school_data_file_path = "../data/raw_data/MA_Public_Schools_2017.csv"
school_characteristics_file_path = "../parameters/school_characteristics/school_characteristics.txt"
double_school_characteristics_file_path = "../parameters/school_characteristics/school_double_characteristics.txt"
integer_school_characteristics_file_path = "../parameters/school_characteristics/school_integer_characteristics.txt"
categorical_school_characteristics_file_path = "../parameters/school_characteristics/school_categorical_characteristics.txt"
preprocessed_school_characteristics_file_path_pq = "../data/preprocessed_data/school_characteristics" \
                                                "/school_characteristics.parquet"
preprocessed_school_characteristics_file_path_csv = "../data/preprocessed_data/school_characteristics" \
                                                "/school_characteristics.csv"


preprocessed_combined_characteristics_file_path_pq = "../data/preprocessed_data/combined_characteristics" \
                                                         "/combined_characteristics.parquet"
preprocessed_combined_characteristics_file_path_csv = "../data/preprocessed_data/combined_characteristics" \
                                                         "/combined_characteristics.csv"

output_variables_file_path = "../parameters/output_characteristics/output_variables.txt"
double_output_variables_file_path = "../parameters/output_characteristics/double_output_characteristics.txt"
integer_output_variables_file_path = "../parameters/output_characteristics/integer_output_characteristics.txt"
preprocessed_output_variables_file_path_pq = "../data/preprocessed_data/output_variables/output_variables.parquet"
preprocessed_output_variables_file_path_csv = "../data/preprocessed_data/output_variables/output_variables.csv"

stratification_bins_file_path = "../parameters/stratification_bins.txt"
column_map_file_path = "../parameters/column_map.txt"
categorical_column_map_file_path = "../parameters/categorical_column_map.txt"


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.submit.pyFiles", "/home/nadia/Documents/WINTER2021/S471/FinalProject/code/") \
        .config("spark.driver.memory", "2g")\
        .getOrCreate()
    return spark


def init_column_maps():
    column_maps = []
    with open(column_map_file_path) as f:
        parameters = f.readlines()
        for p in parameters:
            p = p.replace("\n", "")
            p = tuple(p.split(":"))
            column_maps.append(p)
    with open(categorical_column_map_file_path) as cf:
        parameters = cf.readlines()
        for p in parameters:
            p = p.replace("\n", "")
            p = p.split(":")
            p[0] = p[0].replace(" ", "_")
            p = tuple(p)
            column_maps.append(p)
    return column_maps



