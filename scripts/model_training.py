import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
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
from pyspark.sql.functions import col, to_date, count, min, max, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, sum as spark_sum, when
from pyspark.sql import SparkSession

print('Done importing libraries')

def setup_config():
    model_train_date_str = "2024-09-01"
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 

    print('Config:')
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

    # setup config
    config = setup_config()


    # ========== read gold tables ==========
    print('reading gold tables...')
    X_spark = read_gold_table('feature_store', 'datamart/gold', spark)
    y_spark = read_gold_table('label_store', 'datamart/gold', spark)

    X_df = X_spark.toPandas().sort_values(by='customer_id')
    y_df = y_spark.toPandas().sort_values(by='customer_id')
    print('X_df shape:', X_df.shape)
    print('y_df shape:', y_df.shape)


    # ========== modelling ==========
    print('starting model training...')
    # Consider data from model training date
    y_model_df = y_df[(y_df['snapshot_date'] >= config['train_test_start_date']) & (y_df['snapshot_date'] <= config['model_train_date'])]
    X_model_df = X_df[np.isin(X_df['customer_id'], y_model_df['customer_id'].unique())]
    
    # split data into train - test - oot
    y_oot = y_model_df[(y_model_df['snapshot_date'] >= config['oot_start_date']) & (y_model_df['snapshot_date'] <= config['oot_end_date'])]
    X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]
    
    
    y_traintest = y_model_df[y_model_df['snapshot_date'] <= config['train_test_end_date']]
    X_traintest = X_model_df[np.isin(X_model_df['customer_id'], y_traintest['customer_id'].unique())]

    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, 
                                                    test_size=1-config['train_test_ratio'], 
                                                    random_state=1, 
                                                    shuffle=True, 
                                                    stratify=y_traintest['label']
                                                )
    
    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0])
    print('y_test', y_test.shape[0])
    print('y_oot', y_oot.shape[0])

    # ========== Preprocess data ==========
    X_train_arr = X_train.drop(columns=['customer_id', 'snapshot_date']).values
    X_test_arr = X_test.drop(columns=['customer_id', 'snapshot_date']).values
    X_oot_arr = X_oot.drop(columns=['customer_id', 'snapshot_date']).values

    y_train_arr = y_train['label'].values
    y_test_arr = y_test['label'].values
    y_oot_arr = y_oot['label'].values


    scaler = StandardScaler()

    transformer_stdscaler = scaler.fit(X_train_arr)

    X_train_arr = transformer_stdscaler.transform(X_train_arr)
    X_test_arr = transformer_stdscaler.transform(X_test_arr)
    X_oot_arr = transformer_stdscaler.transform(X_oot_arr)



    # ========== Model training ==========
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=1)    
    
    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=3,  
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=1,
        n_jobs=-1   # Use all available cores
    )

    # Perform the random search
    random_search.fit(X_train_arr, y_train_arr)
    best_model = random_search.best_estimator_

    # Predict and evaluate
    # AUC
    y_pred_proba_train = best_model.predict_proba(X_train_arr)[:, 1]
    train_auc = roc_auc_score(y_train_arr, y_pred_proba_train)

    y_pred_proba_test = best_model.predict_proba(X_test_arr)[:, 1]
    test_auc = roc_auc_score(y_test_arr, y_pred_proba_test)

    y_pred_proba_oot = best_model.predict_proba(X_oot_arr)[:, 1]
    oot_auc = roc_auc_score(y_oot_arr, y_pred_proba_oot)

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"OOT AUC: {oot_auc:.4f}")
    
    
    
    y_pred_train = best_model.predict(X_train_arr)
    y_pred_test = best_model.predict(X_test_arr)
    y_pred_oot = best_model.predict(X_oot_arr)
    
    # Accuracy
    train_accuracy = accuracy_score(y_train_arr, y_pred_train)
    test_accuracy = accuracy_score(y_test_arr, y_pred_test)
    oot_accuracy = accuracy_score(y_oot_arr, y_pred_oot)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"OOT Accuracy: {oot_accuracy:.4f}")

    # Precision
    train_precision = precision_score(y_train_arr, y_pred_train)
    test_precision = precision_score(y_test_arr, y_pred_test)
    oot_precision = precision_score(y_oot_arr, y_pred_oot)

    print(f"Train Precision: {train_precision:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"OOT Precision: {oot_precision:.4f}")

    # Recall
    train_recall = recall_score(y_train_arr, y_pred_train)
    test_recall = recall_score(y_test_arr, y_pred_test)
    oot_recall = recall_score(y_oot_arr, y_pred_oot)
    print(f"Train Recall: {train_recall:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"OOT Recall: {oot_recall:.4f}")
    
    # F1 Score
    train_f1 = f1_score(y_train_arr, y_pred_train)
    test_f1 = f1_score(y_test_arr, y_pred_test)
    oot_f1 = f1_score(y_oot_arr, y_pred_oot)

    print(f"Train F1 Score: {train_f1:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"OOT F1 Score: {oot_f1:.4f}")


    # ========== Save model artefacts ==========
    model_artefact = {}

    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')

    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)


    model_artefact['results'] = {}
    model_artefact['results']['train_auc'] = train_auc
    model_artefact['results']['test_auc'] = test_auc
    model_artefact['results']['oot_auc'] = oot_auc
    model_artefact['results']['train_f1'] = train_f1
    model_artefact['results']['test_f1'] = test_f1
    model_artefact['results']['oot_f1'] = oot_f1
    model_artefact['results']['train_accuracy'] = train_accuracy
    model_artefact['results']['test_accuracy'] = test_accuracy
    model_artefact['results']['oot_accuracy'] = oot_accuracy
    model_artefact['results']['train_precision'] = train_precision
    model_artefact['results']['test_precision'] = test_precision
    model_artefact['results']['oot_precision'] = oot_precision
    model_artefact['results']['train_recall'] = train_recall
    model_artefact['results']['test_recall'] = test_recall
    model_artefact['results']['oot_recall'] = oot_recall

    model_artefact['hp_params'] = random_search.best_params_

    pprint.pprint(model_artefact)



    # ======== Save model ==========
    model_bank_directory = "model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")
    









main()