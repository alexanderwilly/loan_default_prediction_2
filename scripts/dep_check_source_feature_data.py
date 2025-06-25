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


def dependency_check(file_path, spark):
    # load data - IRL ingest from back end source system
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    row_count = df.count()
    
    print(f'{file_path} row count:', row_count)



def main():
    print("\n\n--- starting job ~ dep_check_source_feature_data---\n\n")

    # Create a Spark session
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")


    file_names = ['feature_clickstream.csv', 'features_attributes.csv', 'features_financials.csv']
    
    for file_name in file_names:
        file_path = f'data/{file_name}'
        dependency_check(file_path, spark)




    # end spark session
    spark.stop()

    print('\n\n---Done Dependency Check---\n\n')



main()