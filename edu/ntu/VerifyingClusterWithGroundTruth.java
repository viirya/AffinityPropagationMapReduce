
package edu.ntu;

import java.io.*;
import java.util.*;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.io.compress.CompressionCodec;


public class VerifyingClusterWithGroundTruth {

    private static boolean compression = false;
    private static JobClient job_cli = null;
    

    public static class SimpleMapReduceBase extends MapReduceBase {
        JobConf job;
        String input_filename = null;


        @Override
        public void configure(JobConf job) {
            super.configure(job);
            this.job = job;

            input_filename = job.get("map.input.file");

            if (input_filename != null) {
                StringTokenizer tokenizer = tokenize(new Text(input_filename), "/");
                while (tokenizer.hasMoreTokens()) {
                    input_filename = tokenizer.nextToken();
                }

                tokenizer = tokenize(new Text(input_filename), ".");
                if (tokenizer.hasMoreTokens())
                    input_filename = tokenizer.nextToken();
            
                if (input_filename == null)
                    input_filename = "none";
            }
        }

        public StringTokenizer tokenize(String line, String pattern) {
            StringTokenizer tokenizer = new StringTokenizer(line, pattern);
            return tokenizer;
        } 

        public StringTokenizer tokenize(Text value, String pattern) {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line, pattern);
            return tokenizer;
        }
    }

    public static class LoadClusterAndGroundTruthMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws NumberFormatException, IOException {
 
            StringTokenizer tokenizer = tokenize(value, " \t");

            String image_id = null;
            String cluster_centroid = null;
            String output_value = null;

            if (tokenizer.countTokens() == 1) {
                /* load ground truth */
                image_id = value.toString();
                output_value = "ground_truth%" + input_filename; 
            } else if (tokenizer.countTokens() > 1) {
                /* load cluster data */
                image_id = tokenizer.nextToken();
                cluster_centroid = tokenizer.nextToken();

                output_value = cluster_centroid;
            }

            System.out.println(image_id + ":" + output_value);

            if (image_id != null && output_value != null)
                output.collect(new Text(image_id), new Text(output_value));
 
        }

    }
 
    public static class LoadClusterAndGroundTruthReducer extends SimpleMapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

            String cluster_centroid = "centroid";
            String ground_truth_class = "other";

            while (values.hasNext()) {
                Text value = values.next();

                System.out.println(value);

                StringTokenizer tokenizer = tokenize(value, "%");

                if (tokenizer.countTokens() > 1) {
                    /* ground truth */
                    tokenizer.nextToken();
                    ground_truth_class = tokenizer.nextToken();
                } else {
                   /* cluster */
                    cluster_centroid = tokenizer.nextToken();
                }

            }

            if (cluster_centroid != null && ground_truth_class != null)
                output.collect(new Text(cluster_centroid), new Text(ground_truth_class));

        }
    }
 

    public static class ScanClusterMemberMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws NumberFormatException, IOException {
 
            StringTokenizer tokenizer = tokenize(value, " \t");

            String cluster_centroid = tokenizer.nextToken();
            String member_type = tokenizer.nextToken();

            output.collect(new Text(cluster_centroid), new Text(member_type));


        }

    }

    public static class ScanClusterMemberReducer extends SimpleMapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {


            HashMap<String, Integer> member_type_map = new HashMap<String, Integer>();
            
            while (values.hasNext()) {

                String value = values.next().toString();
                int count = 0;

                if (value.equals("other"))
                    continue;

                if (member_type_map.containsKey(value))
                    count = member_type_map.get(value).intValue();
        
                member_type_map.put(value, new Integer(count + 1));
 
            }

            String max_type = "";
            int max_number = 0;
            int count_for_member = 0;
            for (Map.Entry<String, Integer> entry : member_type_map.entrySet()) {
                int member_number = member_type_map.get(entry.getKey()).intValue();
                if (member_number > max_number) {
                    max_type = entry.getKey();    
                    max_number = member_number;
                }
                count_for_member += member_number;
            }

            if (max_number > 0)
                output.collect(key, new Text(max_type + "\t" + String.valueOf((float)max_number / (float)count_for_member) + "\t" + String.valueOf(max_number) + "\t" + String.valueOf(count_for_member)));

        }
    }

        
    private static void setJobConfCompressed(JobConf job) {
        job.setBoolean("mapred.output.compress", true);
        job.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    }


    public static void main(String[] args) throws Exception {

        String input_path = null;
        String ground_truth_path = null;

        if (args.length < 2) {
            System.out.println("Usage: VerifyingClusterWithGroundTruth <input path> <grund truth path> [compress]");    
            System.exit(0);
        }

        input_path = args[0];
        ground_truth_path = args[1];

        if (args.length == 3 && args[2].equals("compress"))
            compression = true;

        job_cli = new JobClient();

        loadClusterAndGroundTruth(input_path, ground_truth_path);
        scanClusterMember();

    }
 
    public static void loadClusterAndGroundTruth(String input_path, String ground_truth_path) throws Exception {

        JobConf job_loaddata = new JobConf(new Configuration(), VerifyingClusterWithGroundTruth.class);
        job_loaddata.setJobName("LoadClusterAndGroundTruth");

        FileInputFormat.setInputPaths(job_loaddata, new Path(input_path + "/*.gz"), new Path(ground_truth_path + "/*"));
        FileOutputFormat.setOutputPath(job_loaddata, new Path("output/AF_data/verifying_temp"));

        job_loaddata.setOutputKeyClass(Text.class);
        job_loaddata.setOutputValueClass(Text.class);
        job_loaddata.setMapOutputKeyClass(Text.class);
        job_loaddata.setMapOutputValueClass(Text.class);
        job_loaddata.setMapperClass(LoadClusterAndGroundTruthMapper.class);
        job_loaddata.setReducerClass(LoadClusterAndGroundTruthReducer.class);
        job_loaddata.setNumMapTasks(38);
        job_loaddata.setNumReduceTasks(19);
        job_loaddata.setLong("dfs.block.size",134217728);

        if (compression)
            setJobConfCompressed(job_loaddata);

        try {
            job_cli.runJob(job_loaddata);
        } catch(Exception e){
            e.printStackTrace();
        }

    }

    public static void scanClusterMember() throws Exception {

        JobConf job_scancluster = new JobConf(new Configuration(), VerifyingClusterWithGroundTruth.class);
        job_scancluster.setJobName("ScanClusterMember");

        FileInputFormat.setInputPaths(job_scancluster, new Path("output/AF_data/verifying_temp"));
        FileOutputFormat.setOutputPath(job_scancluster, new Path("output/AF_data/verifying"));

        job_scancluster.setOutputKeyClass(Text.class);
        job_scancluster.setOutputValueClass(Text.class);
        job_scancluster.setMapOutputKeyClass(Text.class);
        job_scancluster.setMapOutputValueClass(Text.class);
        job_scancluster.setMapperClass(ScanClusterMemberMapper.class);
        job_scancluster.setReducerClass(ScanClusterMemberReducer.class);
        job_scancluster.setNumMapTasks(38);
        job_scancluster.setNumReduceTasks(19);
        job_scancluster.setLong("dfs.block.size",134217728);

        if (compression)
            setJobConfCompressed(job_scancluster);

        try {
            job_cli.runJob(job_scancluster);
        } catch(Exception e){
            e.printStackTrace();
        }

    } 
}


