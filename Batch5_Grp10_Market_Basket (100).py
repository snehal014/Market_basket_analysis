# Databricks notebook source
# MAGIC %md
# MAGIC # ## Market Basket Analysis Notebook
# MAGIC
# MAGIC This notebook analyzes transaction data to uncover associations between products using the Market Basket Analysis technique. We will perform data preprocessing, apply the Apriori algorithm, generate association rules, and visualize key findings.

# COMMAND ----------

# MAGIC %md
# MAGIC Market Basket Analysis Notebook
# MAGIC This notebook analyzes transaction data to uncover associations between products using the Market Basket Analysis technique. We will perform data preprocessing, apply the Apriori algorithm, generate association rules, and visualize key findings.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # ## Steps Covered in this Notebook:
# MAGIC
# MAGIC
# MAGIC 1. **Load and Preprocess Data:** Import dataset, clean data, and transform into a suitable format.
# MAGIC 2. **Apply Apriori Algorithm:** Identify frequent itemsets using association rule mining.
# MAGIC 3. **Generate Association Rules:** Extract meaningful relationships between products.
# MAGIC 4. **Visualize Findings:** Use data visualization techniques to interpret results.

# COMMAND ----------

# MAGIC %md
# MAGIC # ## Market Basket Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### ## # Import necessary libraries

# COMMAND ----------

from pyspark.sql.functions import col
from functools import reduce
from pyspark.sql.functions import col, when


# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating DataFrame

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/tables/market_basket")

# COMMAND ----------

dbutils.fs.ls("/FileStore/tables/market_basket/")

# COMMAND ----------

data=spark.read.format("csv")\
     .option("header",True)\
     .option("inferschema",True)\
     .load("/FileStore/tables/Groceries_data.csv")
     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Print the Schema 

# COMMAND ----------

# schema of dataframe
data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Records of Data

# COMMAND ----------

# DBTITLE 1,print schema
# 5 recrods from data
data.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Null Values of Data

# COMMAND ----------

from functools import reduce


# COMMAND ----------


# Filter rows where any column has a null value
df_with_nulls = data.filter(reduce(lambda a, b: a | b, (col(c).isNull() for c in data.columns)))
df_with_nulls.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Category Maping

# COMMAND ----------

from pyspark.sql.functions import when


# COMMAND ----------

result = data.withColumn(
    "category", 
    when(col("itemDescription").isin('tropical fruit', 'pip fruit', 'citrus fruit', 'berries', 'grapes', 'fruit/vegetable juice'), 'Fruits')
    .when(col("itemDescription").isin('other vegetables', 'root vegetables', 'packaged fruit/vegetables', 'frozen vegetables', 'potato products', 'pickled vegetables', 'specialty vegetables'), 'Vegetables')
    .when(col("itemDescription").isin('whole milk', 'butter', 'yogurt', 'curd cheese', 'cream cheese', 'hard cheese', 'processed cheese', 'sliced cheese', 'spread cheese', 'butter milk', 'UHT-milk', 'specialty cheese', 'soft cheese'), 'Dairy')
    .when(col("itemDescription").isin('sparkling wine', 'misc. beverages', 'bottled water', 'coffee', 'bottled beer', 'beverages', 'white wine', 'red/blush wine', 'soda', 'liquor', 'liquor (appetizer)', 'wine', 'rum', 'prosecco', 'tea', 'cocoa drinks', 'instant coffee'), 'Beverages')
    .when(col("itemDescription").isin('beef', 'chicken', 'pork', 'hamburger meat', 'frankfurter', 'sausage', 'fish', 'turkey', 'bacon', 'organic sausage', 'meat', 'liver loaf'), 'Meat')
    .when(col("itemDescription").isin('chips', 'popcorn', 'chocolate', 'specialty bar', 'chocolate marshmallow', 'cookies', 'sweet spreads', 'salty snack', 'candy', 'snack products', 'tidbits'), 'Snacks')
    .when(col("itemDescription").isin('bread', 'rolls/buns', 'brown bread', 'pastry', 'semi-finished bread', 'long life bakery product', 'waffles', 'cakes'), 'Bakery')
    .when(col("itemDescription").isin('detergent', 'cleaner', 'dish cleaner', 'abrasive cleaner', 'bathroom cleaner', 'house keeping products', 'shopping bags', 'toilet cleaner', 'kitchen towels'), 'Household')
    .when(col("itemDescription").isin('dental care', 'skin care', 'male cosmetics', 'female sanitary products', 'baby cosmetics', 'cosmetics'),'Personal Care')
    .when(col("itemDescription").isin('canned beer', 'canned fish', 'canned vegetables', 'canned fruit', 'canned food'),'Canned')
    .when(col("itemDescription").isin('frozen potato products', 'frozen meals', 'frozen fish', 'frozen fruits', 'frozen chicken', 'frozen vegetables', 'frozen dessert'),'Frozen')
    .otherwise('Other')
)

result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Association Rule Mining Algorithms

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, explode
from pyspark.ml.fpm import FPGrowth

# Initialize Spark Session
spark = SparkSession.builder.appName("MarketBasketAnalysis").getOrCreate()

# Load Data (Ensure it has 'Member_number', 'Date', 'itemDescription')
# Example: data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Group transactions by Member_number and Date to collect items as sets
tx_data = data.groupBy("Member_number", "Date").agg(collect_set("itemDescription").alias("items"))

# Check item frequency in transactions
print("Top 20 Most Frequent Items:")
tx_data.selectExpr("explode(items) as item").groupBy("item").count().orderBy(col("count").desc()).show(20, truncate=False)

# Apply FP-Growth with optimized parameters
fp_growth = FPGrowth(itemsCol="items", minSupport=0.003, minConfidence=0.05)
model = fp_growth.fit(tx_data)

# Display Association Rules
rules_count = model.associationRules.count()
print("Total Association Rules:", rules_count)

if rules_count > 0:
    print("Association Rules:")
    model.associationRules.show(20, truncate=False)
else:
    print("No association rules found. Try adjusting minSupport and minConfidence.")


# COMMAND ----------

# List available tables in Databricks
spark.sql("SHOW TABLES").show()

# If stored in a CSV file inside Databricks
tx_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/Groceries_data.csv")

# Check the first few rows
tx_data.show(5)

# Display schema to understand data types
tx_data.printSchema()


# COMMAND ----------

tx_data.show(5)
tx_data.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC Top 10 purchase

# COMMAND ----------

from pyspark.sql.functions import col

# Group by itemDescription and count the number of purchases
top_items = data.groupBy("itemDescription") \
                   .count() \
                   .orderBy(col("count").desc()) \
                   .limit(10)

# Display the result in Databricks
display(top_items)


# COMMAND ----------

# MAGIC %md
# MAGIC Monthly Sales Trend (Line Chart)

# COMMAND ----------

display(data.orderBy("year", "month"))

# COMMAND ----------

# MAGIC %md
# MAGIC Most Popular Product Categories

# COMMAND ----------

display(result.groupBy("category").count().orderBy("count", ascending=False))



# COMMAND ----------

# MAGIC %md
# MAGIC Seasonal buying pattern

# COMMAND ----------

from pyspark.sql.functions import to_date, month, count

# Convert 'Date' column to proper date format
data = data.withColumn("Date", to_date(data["Date"], "yyyy-MM-dd"))

# Extract Month
data = data.withColumn("Month", month(data["Date"]))

# Compute seasonal sales count
seasonal_sales = data.groupBy("Month").agg(count("itemDescription").alias("Total_Transactions")).orderBy("Month")

# Display the visualization in Databricks
display(seasonal_sales)


# COMMAND ----------

# MAGIC %md
# MAGIC Most Frequent Item Pairs (Bar Chart)

# COMMAND ----------

from pyspark.sql.functions import collect_list, explode, col, count, udf
from pyspark.sql.types import ArrayType, StringType
from itertools import combinations

# Step 1: Group transactions by Member_number
transactions_df = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("items"))

# Step 2: Define UDF to generate item pairs
def generate_pairs(items):
    return [",".join(pair) for pair in combinations(sorted(items), 2)] if len(items) > 1 else []

pair_udf = udf(generate_pairs, ArrayType(StringType()))

# Step 3: Apply UDF to generate item pairs
pairs_df = transactions_df.withColumn("item_pairs", pair_udf(col("items")))

# Step 4: Explode the pairs into separate rows
pairs_df = pairs_df.select(explode(col("item_pairs")).alias("Item_Pair"))

# Step 5: Count item pair frequency
pair_counts = pairs_df.groupBy("Item_Pair").agg(count("*").alias("Count"))

# Step 6: Get the top 10 most frequent item pairs
top_pairs = pair_counts.orderBy(col("Count").desc()).limit(10)

# Step 7: Display as a bar chart in Databricks
display(top_pairs)


# COMMAND ----------

# MAGIC %md
# MAGIC Product Popularity Over Time (Heatmap)

# COMMAND ----------

from pyspark.sql.functions import concat_ws, year, month, count, col

# Step 1: Extract YearMonth from Date
data = data.withColumn("YearMonth", concat_ws("-", year(data["Date"]), month(data["Date"])))

# Step 2: Count purchases per YearMonth and itemDescription
heatmap_data = data.groupBy("YearMonth", "itemDescription").agg(count("*").alias("Total_Sales"))

# Step 3: Find top 10 most sold items
top_items = data.groupBy("itemDescription").agg(count("*").alias("Total_Sales"))
top_items = top_items.orderBy(col("Total_Sales").desc()).limit(10)

# Step 4: Filter only top 10 items in heatmap data
heatmap_filtered = heatmap_data.join(top_items.select("itemDescription"), on="itemDescription", how="inner")

# Step 5: Display as heatmap in Databricks
display(heatmap_filtered)


# COMMAND ----------

from pyspark.sql.functions import month, count, udf
from pyspark.sql.types import StringType

# Step 1: Extract Month from Date
data = data.withColumn("month", month("Date"))

# Step 2: Define UDF to categorize seasons
def categorize_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

season_udf = udf(categorize_season, StringType())

# Step 3: Apply UDF to create season column
data = data.withColumn("season", season_udf(data["month"]))

# Step 4: Count purchases per season and itemDescription
seasonal_sales = data.groupBy("season", "itemDescription").agg(count("*").alias("count"))

# Step 5: Pivot the data to analyze trends
seasonal_trends = seasonal_sales.groupBy("season").pivot("itemDescription").sum("count").fillna(0)

# Step 6: Display in Databricks
display(seasonal_trends)


# COMMAND ----------


from pyspark.sql.functions import count
import matplotlib.pyplot as plt

# Step 1: Count Purchases Per Season
season_counts = data.groupBy("season").agg(count("*").alias("count"))

# Step 2: Convert to Pandas for Visualization
season_counts_pd = season_counts.toPandas()

# Step 3: Create Pie Chart
plt.figure(figsize=(8, 8))
colors = ["lightblue", "lightgreen", "orange", "pink"]
plt.pie(season_counts_pd["count"], labels=season_counts_pd["season"], autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Seasonal Buying Distribution")
plt.show()


# COMMAND ----------

from pyspark.sql.functions import month, when

# Add a 'season' column based on the 'month' column
data = result.withColumn(
    "season",
    when((month(result.Date) == 12) | (month(result.Date) == 1) | (month(result.Date) == 2), "Winter")
    .when((month(result.Date) >= 3) & (month(result.Date) <= 5), "Spring")
    .when((month(result.Date) >= 6) & (month(result.Date) <= 8), "Summer")
    .otherwise("Autumn")  # Months 9, 10, 11
)

# Display the updated DataFrame
data.select("Date", "month", "season").show(10)


# COMMAND ----------

from pyspark.sql.functions import count

# Count purchases per season and category
seasonal_category_sales = data.groupBy("season", "category").agg(count("*").alias("count"))

# Display in Databricks as a Bar Chart
display(seasonal_category_sales)


# COMMAND ----------

# MAGIC %md
# MAGIC 3.Sales Trend Over Time

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Most Frequent item bought

# COMMAND ----------

display(data.groupBy("itemDescription").count().orderBy("count", ascending=False))


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Product Recommendation Engines

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, udf
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.types import ArrayType, DoubleType
 
# Initialize Spark Session
spark = SparkSession.builder.appName("Content-Based Recommendation").getOrCreate()
 
# Load dataset
data = spark.read.csv("/FileStore/tables/Groceries_data.csv", header=True, inferSchema=True)
 
# Step 1: Tokenize item descriptions
tokenizer = Tokenizer(inputCol="itemDescription", outputCol="words")
tokenized_data = tokenizer.transform(data)
 
# Step 2: Compute Term Frequency (TF)
hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)
tf_data = hashing_tf.transform(tokenized_data)
 
# Step 3: Compute Inverse Document Frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(tf_data)
tfidf_data = idf_model.transform(tf_data)
 
# Convert SparseVector to DenseVector
def sparse_to_dense(vector):
    if isinstance(vector, SparseVector):
        return vector.toArray().tolist()
    elif isinstance(vector, DenseVector):
        return vector.tolist()
    return []
 
sparse_to_dense_udf = udf(sparse_to_dense, ArrayType(DoubleType()))
tfidf_data = tfidf_data.withColumn("denseFeatures", sparse_to_dense_udf(col("features")))
 
# Self-join to compute similarity using dot product
similarity_df = (
    tfidf_data.alias("a")
    .join(tfidf_data.alias("b"), col("a.itemDescription") != col("b.itemDescription"))
    .select(
        col("a.itemDescription").alias("Item_A"),
        col("b.itemDescription").alias("Item_B"),
        (col("a.denseFeatures")[0] * col("b.denseFeatures")[0]).alias("Similarity")  # Simple dot product
    )
    .orderBy(col("Similarity").desc())
)
 
# Recommendation function
def recommend_items(purchased_items, top_n=2):
    recommendations = (
        similarity_df.filter(col("Item_A").isin(purchased_items))
        .select("Item_B", "Similarity")
        .orderBy(col("Similarity").desc())
        .limit(top_n)
        .rdd.map(lambda row: row.Item_B)
        .collect()
    )
    return recommendations
 
# 🚀 Test with a sample purchase
sample_purchase = ["pip fruit", "berries"]
recommended = recommend_items(sample_purchase)
 
print(f"If a customer buys {sample_purchase}, recommend: {recommended}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Store Layout Optimization

# COMMAND ----------

# MAGIC %md
# MAGIC Priority-Based Store Layout using FP-Growth in PySpark

# COMMAND ----------

from pyspark.sql.functions import collect_set

# Example usage: Group by 'Member_number' and collect unique 'itemDescription'
df_grouped = data.groupBy("Member_number").agg(collect_set("itemDescription").alias("unique_items"))

# Show results
df_grouped.show(10, False)


# COMMAND ----------

from pyspark.ml.fpm import FPGrowth


# COMMAND ----------



# Initialize Spark session
spark = SparkSession.builder.appName("Priority-Based Store Layout").getOrCreate()

# Group transactions by Member_number
transactions_df = data.groupBy("Member_number").agg(collect_set("itemDescription").alias("items"))

# Apply FP-Growth for frequent item mining
fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.3)
model = fp_growth.fit(transactions_df)

# Extract frequent itemsets
frequent_items = model.freqItemsets.toPandas()

# Define priority categories
high_priority = frequent_items.sort_values(by="freq", ascending=False).head(10)["items"].tolist()
low_priority = frequent_items.sort_values(by="freq", ascending=True).head(10)["items"].tolist()

# Generate priority-based placement
priority_placement = []
for i in range(min(len(high_priority), len(low_priority))):
    priority_placement.append((str(high_priority[i]), str(low_priority[i])))

# Convert to DataFrame for visualization
priority_df = spark.createDataFrame(priority_placement, ["High Priority Item", "Low Priority Item"])

# Display the suggested layout based on priority
priority_df.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Optimized Store Layout Suggestions Using Frequent Itemsets

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType




# COMMAND ----------

from pyspark.sql.functions import explode


# COMMAND ----------

import numpy as np

# Ensure high_priority and low_priority contain only plain Python strings
high_priority = [item.tolist() if isinstance(item, np.ndarray) else item for item in high_priority]
low_priority = [item.tolist() if isinstance(item, np.ndarray) else item for item in low_priority]


# Suggest item placements based on priority
layout_suggestions = [
    ("Central Placement", [str(item) for item in high_priority[:5]]),
    ("Near Checkouts", [str(item) for item in high_priority[5:]]),
    ("Near Each Other", [str((high_priority[i], high_priority[i+1])) for i in range(len(high_priority)-1)]),
    ("Low Priority Corners", [str(item) for item in low_priority])
]

# Define schema
schema = StructType([
    StructField("Section", StringType(), True),
    StructField("Items", ArrayType(StringType()), True)
])

# Convert layout suggestions to DataFrame
layout_df = spark.createDataFrame(layout_suggestions, schema=schema)

# Display the optimized store layout
layout_df.select("Section", explode("Items").alias("Item")).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Store Layout Optimization and Purchase Recommendations Using FP-Growth

# COMMAND ----------

# Extract frequent itemsets and association rules
frequent_items = model.freqItemsets.select("items", "freq").collect()
association_rules = model.associationRules.select("antecedent", "consequent").collect()

# Convert to Python lists
frequent_items = [(list(row["items"]), row["freq"]) for row in frequent_items]
association_rules = [(list(row["antecedent"]), list(row["consequent"])) for row in association_rules]

# Define store sections
sections = [
    ("Central Aisle", ["bread", "milk", "eggs", "butter"]),
    ("Near Checkout", ["chips", "chocolate", "soda"]),
    ("Essentials Section", ["bread", "butter", "milk", "jam", "cheese"]),
    ("Beverages Section", ["coffee", "tea", "bottled water"]),
    ("Frozen Section", ["frozen meals", "ice cream", "frozen vegetables"]),
    ("Snacks Section", ["cookies", "chips", "chocolate"]),
    ("Low Priority Corner", ["toilet cleaner", "detergent"])
]

# Define schema
schema = StructType([
    StructField("Section", StringType(), True),
    StructField("Items", ArrayType(StringType()), True)
])

# Convert sections to a Spark DataFrame
layout_df = spark.createDataFrame(sections, schema=schema)

# **Fix the explode issue**
layout_df = layout_df.withColumn("Item", explode(col("Items")))

# Display the optimized store layout
layout_df.select("Section", "Item").show(truncate=False)

# Convert association rules to DataFrame
recommendation_df = spark.createDataFrame(association_rules, ["Item Bought", "Recommended Item"])

# Display purchase recommendations
recommendation_df.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC real time promotional avtivities

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/tables/stream_output") 
dbutils.fs.mkdirs("/FileStore/tables/stream_read")   #making directory for streaming output 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, year, month, dayofweek, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, TimestampType

# ✅ Initialize Spark Session
spark = SparkSession.builder.appName("RealTimePromotion").getOrCreate()

# ✅ Define Schema (Including Timestamp for Time-Based Offers)
schema = StructType([
    StructField("Member_number", IntegerType(), True),
    StructField("Date", DateType(), True),  # Purchase date
    StructField("ItemDescription", StringType(), True),
    StructField("year", IntegerType(), True),  # Extracted year from Date
    StructField("month", IntegerType(), True),  # Extracted month from Date
    StructField("day", IntegerType(), True),  # Extracted day from Date
    StructField("day_of_week", IntegerType(), True)  # Extracted day of the week (1=Sunday, 7=Saturday)
])


# ✅ Read Streaming Data
naya_df = spark.readStream.format("csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("/FileStore/tables/stream_read/")

#/FileStore/tables/stream_read

# COMMAND ----------

# Extract Date & Time Features
naya_df = naya_df.withColumn("year", year(col("Date"))) \
                 .withColumn("month", month(col("Date"))) \
                 .withColumn("day", dayofmonth(col("Date"))) \
                 .withColumn("day_of_week", dayofweek(col("Date"))) \
                  # Extract hour from timestamp


# COMMAND ----------

# ✅ Apply Promotions Based on Day, Month, Time & Weekday
promotions_df = naya_df.withColumn("promotion",
    when(col("day") == 1, "1st Day Special: 5% Off")  # Special discount on 1st of the month
    .when(col("month") == 12, "15% Christmas Discount")  # Christmas Offer
    .when(col("month") == 11, "Black Friday Deal: 30% Off")  # Black Friday in November
    .when(col("day_of_week").isin([6, 7]), "Weekend Discount: 20%")  # Saturday & Sunday
    .when(col("day_of_week") == 2, "Tuesday Special: 5% Off")  # Tuesday Offer
    .otherwise("No Special Discount")  # Default case
)



# COMMAND ----------

(promotions_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("mergeSchema", "true")
    .option("checkpointLocation", "/FileStore/tables/stram_checkpoint")
    .start("/FileStore/tables/stream_output"))
#query.awaitTermination()# Keep the stream running


# COMMAND ----------

display(promotions_df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, month, year, dayofweek, lit

# ✅ Initialize Spark Session
spark = SparkSession.builder.appName("MarketBasketPromotions").getOrCreate()

# ✅ Load Data (Replace with Your Data Location)
df = spark.read.format("csv").option("header", "true").load("/FileStore/tables/Groceries_data.csv")

# ✅ Convert Date Column to Proper Format
df = df.withColumn("year", year(col("Date"))) \
       .withColumn("month", month(col("Date"))) \
       .withColumn("day_of_week", dayofweek(col("Date")))

# ✅ Identify Bulk Purchases (More than 5 items per transaction)
basket_size_df = df.groupBy("Member_number", "Date").count()

# ✅ Add Bulk Purchase Promotion
basket_size_df = basket_size_df.withColumn("promotion",
    when(col("count") > 5, "Bulk Purchase Discount: 15% Off").otherwise("No Discount")
)

# ✅ Combo Deal: Identify Frequently Bought Together Items
from pyspark.sql.functions import collect_set
basket_items = df.groupBy("Member_number", "Date").agg(collect_set("itemDescription").alias("items"))

# Example: If "Milk" & "Bread" appear in the same basket, offer a Combo Deal
basket_items = basket_items.withColumn("promotion",
    when(col("items").contains("Milk") & col("items").contains("Bread"), "Combo Offer: Buy 1 Get 1 Free")
)

# ✅ Loyalty Bonus: Identify Frequent Shoppers (10+ purchases in 3 months)
loyal_customers = df.groupBy("Member_number").count().filter(col("count") >= 10)
df = df.join(loyal_customers, "Member_number", "left_outer") \
       .withColumn("promotion", when(col("count") >= 10, "Loyalty Bonus: Extra 10% Off"))

# ✅ Display Promotions
display(df)


# COMMAND ----------

from pyspark.sql.functions import col, array_contains, when, collect_set

# Group transactions by customer and date, collecting items into an array
basket_items = df.groupBy("Member_number", "Date").agg(collect_set("itemDescription").alias("items"))

# Apply promotion when a customer buys both "Milk" and "Bread"
basket_items = basket_items.withColumn("promotion",
    when(array_contains(col("items"), "Milk") & array_contains(col("items"), "Bread"), 
         "Combo Offer: Buy 1 Get 1 Free")
    .otherwise("No Discount")
)

# ✅ Display the results
display(basket_items)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, when, array_contains, year, month, dayofweek, dayofmonth
from pyspark.ml.fpm import FPGrowth

# ✅ Initialize Spark Session
spark = SparkSession.builder.appName("RealTime_Promotion_FPGrowth").getOrCreate()

# ✅ Define Streaming Data Schema
schema = StructType([
    StructField("Member_number", IntegerType(), True),
    StructField("Date", DateType(), True),
    StructField("itemDescription", StringType(), True)
])

# ✅ Read Real-Time Streaming Data (Simulating Transactional Data)
naya_df = (spark.readStream
           .format("csv")
           .option("header", "true")
           .schema(schema)
           .load("/FileStore/tables/stream_read/"))

# ✅ Extract Time Features
naya_df = (naya_df.withColumn("year", year(col("Date")))
                  .withColumn("month", month(col("Date")))
                  .withColumn("day", dayofmonth(col("Date")))
                  .withColumn("day_of_week", dayofweek(col("Date"))))

# ✅ Apply Time-Based Promotions
naya_df = naya_df.withColumn("promotion",
    when(col("day") == 1, "1st Day Special: 5% Off")
    .when(col("month") == 12, "15% Christmas Discount")
    .when(col("month") == 11, "Black Friday Deal: 30% Off")
    .when(col("day_of_week").isin([6, 7]), "Weekend Discount: 20% Off")
    .when(col("day_of_week") == 2, "Tuesday Special: 5% Off")
    .otherwise("No Special Discount"))

# ✅ Perform Market Basket Analysis (FP-Growth)
basket_items = naya_df.groupBy("Member_number", "Date").agg(collect_set("itemDescription").alias("items"))


# ✅ Apply Promotions Based on FP-Growth Rules
promotions_df = basket_items.withColumn("promotion",
    when(array_contains(col("items"), "Milk") & array_contains(col("items"), "Bread"), 
         "Combo Offer: Buy 1 Get 1 Free")
    .when(array_contains(col("items"), "Eggs") & array_contains(col("items"), "Butter"),
         "Breakfast Combo: 10% Off")
    .when(array_contains(col("items"), "Tea") & array_contains(col("items"), "Cookies"),
         "Tea-Time Special: 5% Off")
    .otherwise("No Special Discount"))

# ✅ Merge Time-Based & FP-Growth Promotions
final_promotion_df = promotions_df.join(naya_df, ["Member_number", "Date"], "left") \
                                  .select("Member_number", "Date", "items", "promotion")

# ✅ Write Streaming Output
(final_promotion_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("mergeSchema", "true")
    .option("checkpointLocation", "/FileStore/tables/stream_checkpoint")
    .start("/FileStore/tables/stream_output"))

# ✅ Display Results in Databricks
display(final_promotion_df)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, when, array_contains, year, month, dayofweek, dayofmonth
from pyspark.ml.fpm import FPGrowth

# ✅ Initialize Spark Session
spark = SparkSession.builder.appName("RealTime_Promotion_FPGrowth").getOrCreate()

# ✅ Define Streaming Data Schema
schema = StructType([
    StructField("Member_number", IntegerType(), True),
    StructField("Date", DateType(), True),
    StructField("itemDescription", StringType(), True)
])

# ✅ Read Real-Time Streaming Data (Simulating Transactional Data)
naya_df = (spark.readStream
           .format("csv")
           .option("header", "true")
           .schema(schema)
           .load("/FileStore/tables/stream_read/"))

# ✅ Extract Time Features
naya_df = (naya_df.withColumn("year", year(col("Date")))
                  .withColumn("month", month(col("Date")))
                  .withColumn("day", dayofmonth(col("Date")))
                  .withColumn("day_of_week", dayofweek(col("Date"))))

# ✅ Apply Time-Based Promotions
naya_df = naya_df.withColumn("promotion",
    when(col("day") == 1, "1st Day Special: 5% Off")
    .when(col("month") == 12, "15% Christmas Discount")
    .when(col("month") == 11, "Black Friday Deal: 30% Off")
    .when(col("day_of_week").isin([6, 7]), "Weekend Discount: 20% Off")
    .when(col("day_of_week") == 2, "Tuesday Special: 5% Off")
    .otherwise("No Special Discount"))

# ✅ Perform Market Basket Analysis (FP-Growth)
basket_items = naya_df.groupBy("Member_number", "Date").agg(collect_set("itemDescription").alias("items"))

# ✅ Apply FP-Growth Model for Association Rule Mining
fp_growth = FPGrowth(itemsCol="items", minSupport=0.003, minConfidence=0.05)
model = fp_growth.fit(basket_items)

# ✅ Extract Association Rules
rules = model.associationRules

# ✅ Apply Promotions Based on FP-Growth Rules
promotions_df = basket_items.withColumn("promotion",
    when(array_contains(col("items"), "Milk") & array_contains(col("items"), "Bread"), 
         "Combo Offer: Buy 1 Get 1 Free")
    .when(array_contains(col("items"), "Eggs") & array_contains(col("items"), "Butter"),
         "Breakfast Combo: 10% Off")
    .when(array_contains(col("items"), "Tea") & array_contains(col("items"), "Cookies"),
         "Tea-Time Special: 5% Off")
    .otherwise("No Special Discount"))

# ✅ Merge Time-Based & FP-Growth Promotions
final_promotion_df = promotions_df.join(naya_df, ["Member_number", "Date"], "left") \
                                  .select("Member_number", "Date", "items", "promotion")

# ✅ Write Streaming Output
(final_promotion_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("mergeSchema", "true")
    .option("checkpointLocation", "/FileStore/tables/stream_checkpoint")
    .start("/FileStore/tables/stream_output"))

# ✅ Display Results in Databricks
display(final_promotion_df)


# COMMAND ----------

