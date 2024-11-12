package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.spark.sql.types.StructType;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
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
                .master("yarn")
                .getOrCreate();

        // changed to load all the parquet files in a directory input path.
        String inputPath = "/spark-input/*.parquet";
        Dataset<Row> df = spark.read().parquet(args[0]);
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

        JavaPairRDD<String,String[]> tokens = rows.mapToPair(s ->
                new Tuple2(s.get(0),String.valueOf(s.get(1)).replaceAll("[^a-zA-Z0-9]","").split(" ")));

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

        // Reads in a different parquet and only saves an RDD of the format (Token, Score)
        Dataset<Row> df2 = spark.read().parquet(args[1]);
        Dataset<Row> df_partial2 = df.select("text", "score");
        df_partial.cache();

        JavaRDD<Row> rows2 = df_partial2.select("text", "score").toJavaRDD();

        JavaPairRDD<String,Long> tokensScore = rows2.flatMapToPair(s -> {
            List<Tuple2<String, Long>> results = new ArrayList<>();
            for(String word : String.valueOf(s.get(0)).replaceAll("[^a-zA-Z0-9]","").split(" ")) {
                results.add(new Tuple2<>(word, s.getLong(1)));
            }
            return results.iterator();
        });

        // Combines this RDD into a single RDD that only has a single Token and the average score associated with that token
        JavaPairRDD<String, Tuple2<Integer, Long>> tokenCountScore = tokensScore.mapToPair(s -> new Tuple2<>(s._1(), new Tuple2<>(1,s._2())));
        JavaPairRDD<String, Double> tokenAveScore = tokenCountScore.reduceByKey((a, b) -> new Tuple2<>(a._1() + b._1(), a._2() + b._2()))
                .mapToPair(s -> new Tuple2<>(s._1(), ((double) s._2()._2())/((double) s._2()._1())));

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

        // Do the same thing as TFIDF except for TFIDF multiplied by Score
        JavaPairRDD<String, Double> TFIDFScore = TFIDF.mapToPair(s -> {
            String[] parts = s._1().split(" ");
            return new Tuple2<>(parts[0], new Tuple2<>(parts[1], s._2()));
        }).join(tokenAveScore).mapToPair(s -> new Tuple2<>(s._1() + " " + s._2()._1()._1(), s._2()._2()));
        // TFIDFScore.saveAsTextFile("CS435Project/TFIDFScore");
        JavaPairRDD<String, Double> docTfidfScoreSums = TFIDFScore.mapToPair(entry -> {
                    String[] parts = entry._1().split(" ");
                    String docID = parts[1];
                    return new Tuple2<>(docID, entry._2());
                }).reduceByKey(Double::sum);
        // docTfidfScoreSums.saveAsTextFile("CS435Project/docTfidfScoreSums");



        JavaPairRDD<String, Row> rowsWithId = df_partial.select("id", "score", "year", "month", "day", "hour", "language_score", "token_count")
                .toJavaRDD()
                .mapToPair(row -> new Tuple2<>(row.getString(0), row));

        JavaPairRDD<String, Tuple2<Tuple2<Row, Double>, Double>> joinedRows = rowsWithId.join(docTfidfSums).join(docTfidfScoreSums);

        JavaPairRDD<Double, double[]> feature_pairs = joinedRows.mapToPair(s -> {
            Row row = s._2()._1()._1();
            double tfidf = s._2()._1()._2();
            double tfidfScore = 0;
            if(row.getLong(7) != 0) {
                tfidfScore = s._2()._2()/ (double) row.getLong(7);
            }
   
            double label = (double) row.getLong(1);
            double year = (double) row.getInt(2);
            double month = (double) row.getInt(3);
            double day = (double) row.getInt(4);
            double hour = (double) row.getInt(5);
            double language_score = (double) row.getDouble(6);
            double token_count = (double) row.getLong(7);

            double[] features = {year, month, day, hour, tfidf, tfidfScore, language_score, token_count};
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

        //Baseline Predictions
        JavaRDD<Row> output_labels = indexed_df.select("score").toJavaRDD();
        JavaRDD<Integer> labels = output_labels.map(s -> Long.valueOf(s.getLong(0)).intValue());
        Integer labels_sum = labels.reduce((value1, value2) -> value1 + value2);
        Double label_count = Long.valueOf(labels.count()).doubleValue();
        double y_mean = labels_sum / label_count;
        JavaRDD<LabeledPoint> baseline_vectors = labels.map(s -> {
            double[] values = {0};
            return new LabeledPoint(s.doubleValue(), Vectors.dense(values));
        });
        Dataset<Row> baselineData = spark.createDataFrame(baseline_vectors, LabeledPoint.class);
        baselineData = baselineData.withColumn("prediction", functions.lit(y_mean));

        double baseline_rmse = evaluator.evaluate(baselineData);

        // Compaprison
        System.out.println("RMSE for the model: " + rmse);
        System.out.println("RMSE for baseline model: " + baseline_rmse);

//        output.saveAsTextFile("/ProjectOutput/");

        // Iterator<LabeledPoint> vector_result = vectors.toLocalIterator();

        // while(vector_result.hasNext()) {
        //     System.out.println(vector_result.next().toString());
        // }

        spark.stop();

    }

}
