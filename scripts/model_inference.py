import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print('Done importing libraries')

def setup_config():
    snapshot_date_str = "2024-01-01"
    model_name = "credit_model_2024_09_01.pkl"

    config = {}
    config["snapshot_date_str"] = snapshot_date_str
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")

    config["model_name"] = model_name
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

    print("Config setup complete")
    pprint.pprint(config)

    return config


def read_gold_table(table, gold_db, spark):
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df



def main():
    print('\n\n--- starting job ~ model_training ---\n\n')
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # == Setup configuration ===
    config = setup_config()


    # == Load Model Artefact ===
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
        
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # ==== Load Feature Data ====
    X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
    X_spark = X_spark.filter(col('snapshot_date') == config["snapshot_date_str"])
    X_df = X_spark.toPandas().sort_values(by='customer_id')
    print('X_df shape:', X_df.shape)


    # ==== Preprocess Data ====
    print('starting model training...')
    x_inference = X_df.drop(columns=['customer_id', 'snapshot_date'])
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    x_inference = transformer_stdscaler.transform(x_inference)

    print('x_inference shape:', x_inference.shape)


    # ==== Make Predictions ====
    model = model_artefact["model"]

    y_inference = model.predict_proba(x_inference)[:, 1]

    y_inference_pdf = X_df[['customer_id', 'snapshot_date']].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["prediction"] = y_inference

    print(y_inference_pdf.head(10))


    # ==== Save Predictions ====
    gold_directory = f'datamart/gold/predictions/{config["model_name"]}/'

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    partition_name = config["model_name"][:-4]+"_predictions_"+"2024_01_01"+'.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)

    print(f"Predictions saved to {filepath}")


main()