# Databricks notebook source
# MAGIC %md
# MAGIC # Linear Regression Consulting Project

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! You've been contracted by Hyundai Heavy Industries to help them build a predictive model for some ships. [Hyundai Heavy Industries](http://www.hyundai.eu/en) is one of the world's largest ship manufacturing companies and builds cruise liners.
# MAGIC 
# MAGIC You've been flown to their headquarters in Ulsan, South Korea to help them give accurate estimates of how many crew members a ship will require.
# MAGIC 
# MAGIC They are currently building new ships for some customers and want you to create a model and use it to predict how many crew members the ships will need.
# MAGIC 
# MAGIC Here is what the data looks like so far:
# MAGIC 
# MAGIC     Description: Measurements of ship size, capacity, crew, and age for 158 cruise
# MAGIC     ships.
# MAGIC 
# MAGIC 
# MAGIC     Variables/Columns
# MAGIC     Ship Name     1-20
# MAGIC     Cruise Line   21-40
# MAGIC     Age (as of 2013)   46-48
# MAGIC     Tonnage (1000s of tons)   50-56
# MAGIC     passengers (100s)   58-64
# MAGIC     Length (100s of feet)  66-72
# MAGIC     Cabins  (100s)   74-80
# MAGIC     Passenger Density   82-88
# MAGIC     Crew  (100s)   90-96
# MAGIC     
# MAGIC It is saved in a csv file for you called "cruise_ship_info.csv". Your job is to create a regression model that will help predict how many crew members will be needed for future ships. The client also mentioned that they have found that particular cruise lines will differ in acceptable crew counts, so it is most likely an important feature to include in your analysis! 
# MAGIC 
# MAGIC Once you've created the model and tested it for a quick check on how well you can expect it to perform, make sure you take a look at why it performs so well!

# COMMAND ----------

dataset = spark.read.format("csv").load("/FileStore/tables/mllib_sample_data/cruise_ship_info.csv", header = True, inferSchema = True)

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dealing with Cruise Line column: turn string to index and apply onehot encoding

# COMMAND ----------

# StringIndex + Onehot for the Cruiseline Column:
indexer = StringIndexer(inputCol = 'Cruise_line',
                       outputCol = 'Cruiseline_Index')
indexed = indexer.fit(dataset).transform(dataset)

encoder = OneHotEncoder(inputCols = ['Cruiseline_Index'], outputCols = ['Cruiseline_Onehot'])
encoded = encoder.fit(indexed).transform(indexed)

encoded.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Construct the Feature column for the model by using VectorAssembler

# COMMAND ----------

assembler = VectorAssembler(
inputCols = ['Age', 'Tonnage','passengers','length','cabins','passenger_density', 'Cruiseline_Onehot'],
outputCol = 'features')
output = assembler.transform(encoded)

# COMMAND ----------

final_data = output.select('features', 'crew')

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7, 0.3])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

lr = LinearRegression(labelCol = 'crew')
lrModel = lr.fit(train_data)

# COMMAND ----------

test_results = lrModel.evaluate(test_data)

# COMMAND ----------

print('RMSE:{}'.format(test_results.rootMeanSquaredError))
print('R2:{}'.format(test_results.r2adj))

# COMMAND ----------

