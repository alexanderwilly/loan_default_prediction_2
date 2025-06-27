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
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
print('done importing')


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
    spark = SparkSession \
    .builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    
    # == Setup configuration ===
    config = setup_config()
    
    # ==== Load Label Store ====
    y_spark = read_gold_table('label_store', 'datamart/gold', spark)
    y_df = y_spark.toPandas().sort_values(by='snapshot_date')
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])
    
    # ==== Load Inference ========
    inference_path = f"datamart/gold/predictions/{config['model_name']}/{config['model_name'][:-4]}_predictions_{config['snapshot_date_str'].replace('-', '_')}.parquet"
    y_inference = spark.read.option("header", "true").parquet(inference_path)
    y_inference_pdf = y_inference.toPandas().sort_values(by='snapshot_date')
    y_inference_pdf['snapshot_date'] = pd.to_datetime(y_inference_pdf['snapshot_date'])
    
    
    # ==== Monitor with performance metrics ====
    df_monitor = pd.merge(
        y_inference_pdf,
        y_df[['customer_id', 'snapshot_date', 'label']],
        on=['customer_id'],
        how='inner'
    )
    
    df_monitor = df_monitor[['customer_id', 'model_name', 'prediction', 'label', 'snapshot_date_x']]
    df_monitor['snapshot_date'] = df_monitor['snapshot_date_x']
    df_monitor = df_monitor.drop('snapshot_date_x', axis = 1)
    
    
    threshold = 0.35 # set threshold to 0.35
    y_pred_train = (df_monitor['prediction'] >= threshold).astype(int)
    df_monitor['pred_label'] = y_pred_train
    
    monitor_accuracy = accuracy_score(df_monitor['label'], df_monitor['pred_label'])
    monitor_precision = precision_score(df_monitor['label'], df_monitor['pred_label'])
    monitor_recall = recall_score(df_monitor['label'], df_monitor['pred_label'])
    monitor_f1 = f1_score(df_monitor['label'], df_monitor['pred_label'])


    print('accuracy:', monitor_accuracy)
    print('precision:', monitor_precision)
    print('recall:', monitor_recall)
    print('f1 score:', monitor_f1)
    
    
    
main()