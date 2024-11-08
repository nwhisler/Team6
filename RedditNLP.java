package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;

import com.google.common.collect.Iterables;

import scala.Tuple2;

public class RedditNLP {
    private static final Pattern SPACE = Pattern.compile("\\s+");

    public static void main(String args[]) throws Exception {

        if (args.length < 1) {

            System.exit(1);

        }

        SparkSession spark = SparkSession
                .builder()
                .appName("RedditNLP")
                .master("local")
                .getOrCreate();

        // changed to load all the parquet files in a directory input path.
        String inputPath = "/spark-input/*.parquet";
        Dataset<Row> df = spark.read().parquet(inputPath);
        Dataset<Row> df_partial = df.select("id", "text", "date", "subreddit", "score", "language_score", "token_count");
        df_partial.cache();

        df_partial = df_partial
                .withColumn("year", functions.year(df_partial.col("date")))
                .withColumn("month", functions.month(df_partial.col("date")))
                .withColumn("day", functions.dayofmonth(df_partial.col("date")))
                .withColumn("hour", functions.hour(df_partial.col("date")));

        StringIndexer indexer = new StringIndexer()
                .setInputCol("subreddit")
                .setOutputCol("subredditIndex");
        Dataset<Row> indexed_df = indexer.fit(df_partial).transform(df_partial);
        indexed_df.cache();

        JavaRDD<Row> rows = indexed_df.select("id", "text", "year", "month", "day", "hour", "subredditIndex", "score", "language_score", "token_count").toJavaRDD();

        JavaPairRDD<String,String[]> tokens = rows.mapToPair(s -> new Tuple2(s.get(0),String.valueOf(s.get(1)).replaceAll("\\?\\!\\.\\,\\-","").split(" ")));

        JavaRDD<HashMap<String,Double>> tf_mapper = tokens.map(s -> {
            String document_id = s._1();
            String[] current_tokens = s._2();
            HashMap<String,Double> tf = new HashMap<>();

            try {
                for (String word : current_tokens) {
                    String key = word + " " + document_id;
                    if (tf.containsKey(key)) {
                        Double previousValue = tf.get(key);
                        tf.put(key, previousValue + 1.0);
                    } else {
                        tf.put(key, 1.0);
                    }
                }
            }
            catch(Exception e) {
                ;
            }

            return tf;

        });


        JavaRDD<Integer> corpus = tokens.map(s -> s._2().length);

        JavaPairRDD<Integer, Long> corpus_zipped = corpus.zipWithIndex();
        JavaPairRDD<HashMap<String,Double>, Long> tf_zipped = tf_mapper.zipWithIndex();

        JavaPairRDD<Long, Integer> corpus_zipped_reversed = corpus_zipped.mapToPair(s -> new Tuple2(s._2(), s._1()));
        JavaPairRDD<Long, HashMap<String,Double>> tf_zipped_reversed = tf_zipped.mapToPair(s -> new Tuple2(s._2(), s._1()));

        JavaPairRDD<Long, Tuple2<Integer, HashMap<String,Double>>> joined_RDD = corpus_zipped_reversed.join(tf_zipped_reversed);

        JavaPairRDD<Integer, HashMap<String,Double>> corpus_tf_result = joined_RDD.mapToPair(s -> new Tuple2(s._2()._1(), s._2()._2()));

        JavaRDD<HashMap<String,Double>> tf_result = corpus_tf_result.map(s -> {
            Integer corpus_size = s._1();
            HashMap<String,Double> map = s._2();
            Set<String> keys = map.keySet();

            for(String key : keys) {
                map.put(key, map.get(key)/corpus_size);
            }

            return map;
        });

//        Iterator<HashMap<String,Double>> result = tf_result.toLocalIterator();
//
//        while(result.hasNext()) {
//            HashMap<String,Double> map = result.next();
//            Set<String> keys = map.keySet();
//            for(String key : keys) {
//                System.out.println(key + " " + map.get(key));
//            }
//        }

        // Calculating IDF = log10(N/ni)
        // N = number of articles
        // ni = number of times term i appears within articles in corpus N
        long numCorpus = corpus_zipped.count();
        JavaPairRDD<String, String> tokenDocFlatMap = tokens.flatMapValues(s -> Arrays.asList(s).iterator()).mapToPair(s -> s.swap());
        JavaPairRDD<String, Iterable<String>> reducedTokenDocFlatMap = tokenDocFlatMap.groupByKey();
        JavaPairRDD<String, Double> IDF = reducedTokenDocFlatMap.mapToPair(s -> {
            int ni = Iterables.size(s._2());
            double idf = Math.log((double) numCorpus / ni);
            return new Tuple2<>(s._1(), idf);
        });

        // Calculating TFIDF = TFi * IDFi
        // TFi is the TF value per word
        // IDFi is the IDF value per word
        JavaPairRDD<String, Tuple2<String, Double>> TF = tf_result.flatMapToPair(s -> {
            List<Tuple2<String, Tuple2<String, Double>>> results = new ArrayList<>();
            for(String key : s.keySet()) {
                String token = SPACE.split(key)[0];
                String docId = SPACE.split(key)[1];
                results.add(new Tuple2<>(token, new Tuple2<>(docId, s.get(key))));
            }
            return results.iterator();
        });
        JavaPairRDD<String, Double> TFIDF = IDF.join(TF).mapToPair(s ->  new Tuple2<>(s._1() + " " + s._2()._2()._1(), s._2()._1() * s._2()._2()._2()));

        JavaPairRDD<String, Double> docTfidfSums = TFIDF
                .mapToPair(entry -> {
                    String[] parts = entry._1().split(" ");
                    String docID = parts[1];
                    return new Tuple2<>(docID, entry._2());
                })
                .reduceByKey(Double::sum);

        JavaRDD<Double> tfidf_sums = docTfidfSums.map(s -> s._2());

//        Dataset<Row> tfidf_df = spark.createDataFrame(tfidf_sums.rdd(), StructType.class).toDF("tfidf");
//        df_partial = df_partial.withColumn("tfidf", tfidf_df.col("tfidf"));
        JavaPairRDD<String, Row> rowsWithId = df_partial.select("id", "score", "year", "month", "day", "hour", "language_score", "token_count")
                .toJavaRDD()
                .mapToPair(row -> new Tuple2<>(row.getString(0), row));

        JavaPairRDD<String, Tuple2<Row, Double>> joinedRows = rowsWithId.join(docTfidfSums);

        JavaPairRDD<Double, double[]> feature_pairs = joinedRows.mapToPair(s -> {
            Row row = s._2()._1();
            double tfidf = s._2()._2();

            double label = (double) row.getLong(1);
            double year = (double) row.getInt(2);
            double month = (double) row.getInt(3);
            double day = (double) row.getInt(4);
            double hour = (double) row.getInt(5);
            double language_score = (double) row.getDouble(6);
            double token_count = (double) row.getLong(7);

            double[] features = {year, month, day, hour, tfidf, language_score, token_count};
            return new Tuple2<>(label, features);
        });

        JavaRDD<LabeledPoint> vectors = feature_pairs.map(s -> 
            new LabeledPoint(s._1(), Vectors.dense(s._2()))
        );

        Dataset<Row> labeledData = spark.createDataFrame(vectors, LabeledPoint.class);

        Dataset<Row>[] splits = labeledData.randomSplit(new double[]{0.8, 0.2}, 435);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        RandomForestRegressor rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(10)
                .setMaxBins(32);

        RandomForestRegressionModel model = rf.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        RegressionEvaluator evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse");
        
        double rmse = evaluator.evaluate(predictions);

        Dataset<Row> results = predictions.select("label", "prediction");

        JavaRDD<String> output = results.toJavaRDD().map(row -> {
            double actualScore = row.getDouble(0);
            double predictedScore = row.getDouble(1);
            return actualScore + "," + predictedScore;
        });
        System.out.println("RMSE for the model: " + rmse);

        output.saveAsTextFile("/ProjectOutput/");

        // Iterator<LabeledPoint> vector_result = vectors.toLocalIterator();

        // while(vector_result.hasNext()) {
        //     System.out.println(vector_result.next().toString());
        // }
        
        spark.stop();

    }

}
