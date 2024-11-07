package org.example;

import com.google.common.collect.Iterables;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.linalg.Vectors;

import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.StructType;

import java.util.*;
import java.util.regex.Pattern;

public class RedditNLP {
    private static final Pattern SPACE = Pattern.compile("\\s+");


    public static void main(String args[]) throws Exception {

        if(args.length < 1) {

            System.exit(1);

        }

        SparkSession spark = SparkSession
                .builder()
                .appName("RedditNLP")
                .master("local")
                .getOrCreate();

        Dataset<Row> df = spark.read().parquet(args[0]);
        Dataset<Row> df_partial = df.select("id", "text", "date", "subreddit", "score", "language_score", "token_count");

        df_partial = df_partial
                .withColumn("year", functions.year(df_partial.col("date")))
                .withColumn("month", functions.month(df_partial.col("date")))
                .withColumn("day", functions.dayofmonth(df_partial.col("date")))
                .withColumn("hour", functions.hour(df_partial.col("date")));

        // probably need to add org.apache.spark.ml to the pom.xml in the maven project, but afaik is the only way to easily index the subreddits.
        /**
         * <dependency>
         *      <groupId>org.apache.spark</groupId>
         *      <artifactId>spark-mllib_2.12</artifactId>
         *      <version>2.4.3</version>
         * </dependency>
         */
        StringIndexer indexer = new StringIndexer()
                .setInputCol("subreddit")
                .setOutputCol("subredditIndex");
        Dataset<Row> indexed_df = indexer.fit(df_partial).transform(df_partial);
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
        df_partial = df_partial.withColumn("tfidf", functions.lit(0.0));

        rows = df_partial.select("score", "year", "month", "day", "hour", "tfidf", "language_score", "token_count").toJavaRDD();
        JavaPairRDD<Double,double[]> feature_pairs = rows.mapToPair(s -> {
            Long value0 = (long) s.get(0);
            double label = value0.doubleValue();
            Integer value1 = (int) s.get(1);
            double year = value1.doubleValue();
            Integer value2 = (int) s.get(2);
            double month = value2.doubleValue();
            Integer value3 = (int) s.get(3);
            double day = value3.doubleValue();
            Integer value4 = (int) s.get(4);
            double hour = value4.doubleValue();
            double tfidf = (double) s.get(5);
            double language_score = (double) s.get(6);
            Long value7 = (long) s.get(7);
            double token_count = value7.doubleValue();

            double[] features = {year,month,day,hour,tfidf,language_score,token_count};

            return new Tuple2(Double.valueOf(label),features);

        });

        JavaRDD<LabeledPoint> vectors = feature_pairs.map(s -> {
            return new LabeledPoint(s._1().doubleValue(), Vectors.dense(s._2()));
        });

        Iterator<LabeledPoint> vector_result = vectors.toLocalIterator();

        while(vector_result.hasNext()) {
            System.out.println(vector_result.next().toString());
        }


        spark.stop();

    }

}