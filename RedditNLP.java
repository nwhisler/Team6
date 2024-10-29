package org.example;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.Dataset;

import java.util.*;

import org.apache.spark.sql.SparkSession;

public class RedditNLP {


    public static void main(String args[]) throws Exception {

        if(args.length < 1) {

            System.exit(1);

        }

        SparkSession spark = SparkSession
                .builder()
                .appName("RedditNLP").master("local")
                .getOrCreate();

        Dataset<Row> df = spark.read().parquet(args[0]);
        Dataset<Row> df_partial = df.select("id","text");
        JavaRDD<Row> rows = df_partial.toJavaRDD();
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

        Iterator<HashMap<String,Double>> result = tf_result.toLocalIterator();

        while(result.hasNext()) {
            HashMap<String,Double> map = result.next();
            Set<String> keys = map.keySet();
            for(String key : keys) {
                System.out.println(key + " " + map.get(key));
            }
        }


        spark.stop();

    }

}