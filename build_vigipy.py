# Databricks notebook source
# # Uninstall vigipy if it has been installed
# %pip uninstall -y vigipy

# # Install wheel package
# %pip install wheel 

# # Build the package
# !python setup.py bdist_wheel

# # Install the built package
# %pip install ./dist/vigipy-1.4.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd 
from vigipy import convert, convert_multi_item_pipeline, gps
import time as time

# COMMAND ----------

# read in dataset
df=pd.read_csv("/dbfs/mnt/data/caers/openebgm_dataset.csv", header=0)
df.rename(columns={'var1': 'name'}, inplace=True)
df.rename(columns={'var2': 'AE'}, inplace=True)
df['count'] = 1

# clean out duplicate cases, which would have been handled fairly natively by OpenEBGM
df = df.drop_duplicates(subset=['id', 'name', 'AE'], keep='first')

# COMMAND ----------

time1 = time.time()
vigipy_data = convert(df, count_unique_ids=False, ae_label='AE')
time2 = time.time()
print("TIME FOR PROCESSING = ", time2-time1)

# COMMAND ----------

# carry out vigipy data optimisation
# results = gps(vigipy_data, 
#               prior_init={'alpha1': 2, 'beta1': 1, 'alpha2': 2, 'beta2': 1, 'w': 0.05},
#               minimization_method='Nelder-Mead',
#               message=True, 
#               opt_likelihood=True,
#               truncate=True,
#               truncate_thres=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Trying to convert processing to pyspark for speed

# COMMAND ----------

import pyspark.pandas as ps 
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from vigipy.utils import Container

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

def convert_ps(
    data_frame,  # Now this is a PySpark DataFrame
    margin_threshold=1,
    product_label="name",
    count_label="count",
    ae_label="AE",
    count_unique_ids=False,
    id_label="id",
    opt=True
):
    col_list = data_frame.columns

    if ((id_label not in col_list) and (count_unique_ids == True)):
        raise ValueError(f"Column '{id_label}' does not exist in the DataFrame")
    if product_label not in col_list:
        raise ValueError(f"Column '{product_label}' does not exist in the DataFrame")
    if ae_label not in col_list:
        raise ValueError(f"Column '{ae_label}' does not exist in the DataFrame")
    if count_label not in col_list:
        raise ValueError(f"Column '{count_label}' does not exist in the DataFrame")

    # Call the compute_contingency function for the contingency table
    data_cont = compute_contingency_ps(
        data_frame, product_label, count_label, ae_label, margin_threshold
    )

    if not count_unique_ids:
        # Compute column and row sums using PySpark
        col_sums = data_cont.groupBy().sum().collect()[0]
        row_sums = data_cont.agg(F.sum(data_cont.columns)).collect()[0]

        if opt:
            data_df = count_optimized_ps(data_cont, row_sums, col_sums)  # Need to convert this function for PySpark
        else:
            data_df = count_ps(data_cont, row_sums, col_sums)  # Need to convert this function for PySpark
        total_report_number = data_df.agg(F.sum('events')).collect()[0][0]

    else:
        # Compute the flattened table directly from the data with unique ID counts
        actual = data_frame.groupBy([product_label, ae_label]).agg(F.countDistinct(id_label).alias('events'))
        product_marg = data_frame.groupBy(product_label).agg(F.countDistinct(id_label).alias('product_aes'))
        ae_marg = data_frame.groupBy(ae_label).agg(F.countDistinct(id_label).alias('count_across_brands'))

        # Merging in PySpark
        data_df = actual.join(product_marg, on=product_label).join(ae_marg, on=ae_label)
        data_df = data_df.select('events', 'product_aes', 'count_across_brands', ae_label, product_label)
        total_report_number = data_frame.select(F.countDistinct(id_label)).collect()[0][0]

    # Initialize the container object and assign the data
    DC = Container()
    DC.contingency = data_cont
    DC.data = data_df
    DC.N = total_report_number
    return DC


# COMMAND ----------

def compute_contingency_ps(data_frame, product_label, count_label, ae_label, margin_threshold):
    # Create a contingency table using PySpark's pivot operation
    data_cont = (
        data_frame.groupBy(product_label)
        .pivot(ae_label)
        .agg(F.sum(count_label))  # Sum the counts
        .fillna(0)  # Fill missing values with 0
    )

    # # Calculate row sums and filter rows based on margin_threshold
    # # Compute row sums
    # row_sum_exprs = [F.sum(F.col(c)).alias(c) for c in data_cont.columns if c != product_label]
    # row_sums = data_cont.withColumn("row_sum", sum(F.col(c) for c in data_cont.columns if c != product_label))

    # # Filter rows with row sums below margin_threshold
    # rows_to_drop = row_sums.filter(F.col("row_sum") < margin_threshold).select(product_label)

    # # Drop rows with low counts
    # data_cont = data_cont.join(rows_to_drop, on=product_label, how="left_anti")

    # # Now check columns based on margin threshold
    # col_sum_exprs = [F.sum(F.col(c)).alias(c) for c in data_cont.columns if c != product_label]
    # col_sums_df = data_cont.agg(*col_sum_exprs)
    # cols_to_drop = [k for k, v in col_sums_df.first().asDict().items() if v < margin_threshold]

    # # Drop columns that have sums below the threshold
    # data_cont = data_cont.drop(*cols_to_drop)

    return data_cont


# COMMAND ----------

def count_optimized_ps(data, row_sums, col_sums):
    """
    Convert the input contingency table to a flattened table using PySpark

    Arguments:
        data (PySpark DataFrame): A contingency table of brands and events
        row_sums (PySpark DataFrame): A DataFrame showing how often each product appears
        col_sums (PySpark DataFrame): A DataFrame showing how often each event appears

    Returns:
        df: A PySpark DataFrame with the count information
    """

    # Convert the contingency table from wide to long format (equivalent to Pandas stack())
    # Here we "melt" the table by pivoting each event (column) into a row under the 'ae_name' column
    df_long = data.select(
        F.expr(f"stack({len(data.columns) - 1}, " + 
               ", ".join([f"'{col}', {col}" for col in data.columns[1:]]) + ")")
    ).withColumnRenamed('col0', 'ae_name').withColumnRenamed('col1', 'events')

    # Add the product_name from the original data
    df_long = df_long.withColumn('product_name', F.col(data.columns[0]))

    # Filter out rows where events are zero
    df_long = df_long.filter(F.col('events') > 0)

    # Convert row_sums and col_sums to Spark DataFrames (if they're not already)
    # Assuming row_sums and col_sums are passed in as PySpark DataFrames

    # Join to map rows (product_name) to product_aes (from row_sums)
    df_long = df_long.join(row_sums.withColumnRenamed("sum", "product_aes"), "product_name", "left")

    # Join to map columns (ae_name) to count_across_brands (from col_sums)
    df_long = df_long.join(col_sums.withColumnRenamed("sum", "count_across_brands"), "ae_name", "left")

    # Return the final DataFrame
    return df_long


# COMMAND ----------

df_spark = spark.createDataFrame(df)

time1 = time.time()
vigipy_data2 = convert_ps(df_spark, count_unique_ids=False, ae_label='AE')
time2 = time.time()
print("TIME FOR PROCESSING PYSPARK = ", time2-time1)

# COMMAND ----------

dataset1 = vigipy_data.data 
dataset2 = vigipy_data2.data.toPandas()

# COMMAND ----------

dataset1.sort_values(by=['AE', 'name'])

# COMMAND ----------

dataset2.sort_values(by=['AE', 'name'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Working/not working so far
# MAGIC - My optimised version of the `convert` function that is like openEBGM does work and seems to produce consistent answers to other stuff 
# MAGIC - Count and compute_contingency functions need a look at, because they currently fail
# MAGIC - Multi-item conversion also needs to be done
