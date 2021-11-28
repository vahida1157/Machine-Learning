from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, StructType, IntegerType

# spark = SparkSession.builder.appName("test").getOrCreate()
#
# df = spark.read.csv("examples/src/main/resources/people.csv", header=True, sep=';')
# df.show()
# df.count()
# df.printSchema()
# df.select("name").show()
# df.select(["name", "job"]).show()
# df.filter(df['age'] > 31).show()
# spark = SparkSession.builder.appName("CTR").getOrCreate()

spark = SparkSession.builder.master("local[*]").appName("PySparkShell").config("spark.driver.memory",
                                                                               "40G").getOrCreate()

schema = StructType([
    StructField("id", StringType(), True),
    StructField("click", IntegerType(), True),
    StructField("hour", IntegerType(), True),
    StructField("C1", StringType(), True),
    StructField("banner_pos", StringType(), True),
    StructField("site_id", StringType(), True),
    StructField("site_domain", StringType(), True),
    StructField("site_category", StringType(), True),
    StructField("app_id", StringType(), True),
    StructField("app_domain", StringType(), True),
    StructField("app_category", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("device_ip", StringType(), True),
    StructField("device_model", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("device_conn_type", StringType(), True),
    StructField("C14", StringType(), True),
    StructField("C15", StringType(), True),
    StructField("C16", StringType(), True),
    StructField("C17", StringType(), True),
    StructField("C18", StringType(), True),
    StructField("C19", StringType(), True),
    StructField("C20", StringType(), True),
    StructField("C21", StringType(), True), ])
df = spark.read.csv("../HW4/train.csv", schema=schema, header=True)

df.printSchema()
df.count()
df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')
df = df.withColumnRenamed("click", "label")
print(df.columns)

df_train, df_test = df.randomSplit([0.7, 0.3], 42)
df_train.cache()
df_train.count()
df_test.cache()
df_test.count()
categorical = df_train.columns
categorical.remove('label')
print(categorical)

indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep") for c in categorical]

encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers],
                        outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers])

assembler = VectorAssembler(inputCols=encoder.getOutputCols(), outputCol="features")
stages = indexers + [encoder, assembler]

pipeline = Pipeline(stages=stages)
one_hot_encoder = pipeline.fit(df_train)

df_train_encoded = one_hot_encoder.transform(df_train)
df_train_encoded.show()
df_train_encoded = df_train_encoded.select(["label", "features"])
df_train_encoded.show()
df_train_encoded.cache()
df_train.unpersist()
df_test_encoded = one_hot_encoder.transform(df_test)
df_test_encoded = df_test_encoded.select(["label", "features"])
df_test_encoded.show()
df_test_encoded.cache()
df_test.unpersist()

classifier = LogisticRegression(maxIter=20, regParam=0.001, elasticNetParam=0.001)
lr_model = classifier.fit(df_train_encoded)
predictions = lr_model.transform(df_test_encoded)
predictions.cache()
predictions.show()

ev = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print(ev.evaluate(predictions))
