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

import utils.data_processing_bronze_table
print('Done Importing!')


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates




def main():
    print('\n\n--- starting job ~ bronze_attributes ---\n\n')

    # ~~~~ Setup Spark Session ~~~~
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")


    # ~~~~ Generate Partitions ~~~~
    # Setup Config
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"

    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print("Done generate list of dates!")
    print(dates_str_lst)

    # ~~~~ Create Bronze Datalake ~~~~
    bronze_directory = "datamart/bronze"

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table('attributes', 'data/features_attributes.csv', bronze_directory, date_str, spark)

   

    # end spark session
    spark.stop()
    print('\n\n--- finished job ~ bronze_attributes ---\n\n')
    





main()