import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from pyspark.sql import SparkSession
from tqdm import tqdm
import time  # to simulate loading for tqdm
import sys
import os
import glob
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, to_date, count, min, max, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, sum as spark_sum, when



def main():
    print("\n\n--- starting job ~ dep_check_source_feature_data---\n\n")

    # Create a Spark session
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    file_path = 'data/lms_loan_daily.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Exiting.")
        sys.exit(1)

    # Read the CSV file into a Spark DataFrame
    print(f"Reading data from {file_path}...")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print("Data read successfully.")
    print(f"Number of rows in the DataFrame: {df.count()}")

    # end spark session
    spark.stop()

    print('\n\n---Done Dependency Check---\n\n')



main()