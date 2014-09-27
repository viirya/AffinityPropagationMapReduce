package edu.ntu;

import java.io.*;
import java.util.*;
//import java.util.Map;
//import java.util.StringTokenizer;
//import java.util.HashMap;
//import java.util.ArrayList;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
//import org.apache.hadoop.mapred.MapReduceBase;
//import org.apache.hadoop.mapred.Mapper;
//import org.apache.hadoop.mapred.OutputCollector;
//import org.apache.hadoop.mapred.Reducer;
//import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.fs.*;
//import org.apache.hadoop.mapred.JobConf;
//import org.apache.hadoop.mapred.JobClient;
//import org.apache.hadoop.mapred.FileInputFormat;
//import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.io.compress.CompressionCodec; 

public class AffinityPropagationMapReduce {
    public static float damping_factor = 0.5f;
    public static float a_converge_threshold = 10.00f;
    public static float r_converge_threshold = 0.01f;
    public static float a_converge_portion = 0.01f;
    public static float r_converge_portion = 0.5f;

    private static boolean compression = false;

    public static class PreInitMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," (),\t");

            String img_id = tokenizer.nextToken();
            String img_k = tokenizer.nextToken();
            float sim = Float.parseFloat(tokenizer.nextToken());

            String similarity = Float.toString(sim);
            Text outputKey = new Text(img_id);
            context.write(outputKey, new Text(similarity));
        }     
    }

    public static class PreInitReducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String img_id = key.toString();
            
            float sim = 0.0f;
            int count = 0;
            for (Text val: values) {
                sim += Float.parseFloat(val.toString());
                count++;
            }

            Text outputKey = new Text(img_id);
            String similarity = Float.toString(sim) + " " + Integer.toString(count);
            context.write(outputKey, new Text(similarity));
        }
    }



    public static class PreSecondMapper extends Mapper<LongWritable, Text, IntWritable, Text>{
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," \t");
            String img_id = tokenizer.nextToken();
            float sim = Float.parseFloat(tokenizer.nextToken());
            int count = Integer.parseInt(tokenizer.nextToken());


            String similarity = Float.toString(sim) + " " + Integer.toString(count);
            IntWritable outputKey = new IntWritable(0);
            context.write(outputKey, new Text(similarity));

        }     
    }

    public static class PreSecondReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            
            double sum = 0.0f;
            double sumSq = 0.0f;
            int count = 0;

            for (Text val: values) {
                String line = val.toString();
                StringTokenizer tokenizer = new StringTokenizer(line," \t");
                double value = Double.parseDouble(tokenizer.nextToken());
                int nodeCount = Integer.parseInt(tokenizer.nextToken());

                sum += value;
                sumSq += value * value;
                count += nodeCount;
            }

            double mean = sum / count;
            double stddev = Math.sqrt(Math.abs(sumSq - mean * sum) / count);

            String outputString = "mean: " + Double.toString(mean) + " std: " + Double.toString(stddev); 
            context.write(key, new Text(outputString));

            try {                                    
                FileSystem fs;
                fs = FileSystem.get(context.getConfiguration());
                String path_str = context.getConfiguration().get("path");
                Path path_mean = new Path(path_str + "/mean");
                if(!fs.exists(path_mean)) {
                  DataOutputStream out = fs.create(path_mean);
                  out.writeDouble(mean);
                  out.close();
                }

                Path path_count = new Path(path_str + "/count");
                if(!fs.exists(path_count)) {
                  DataOutputStream out = fs.create(path_count);
                  out.writeInt(count);
                  out.close();
                }


            } catch(Exception e) {
                throw new IOException(e.getMessage());
            }
/*
            job.set("mean", Double.toString(mean));
            job.set("std", Double.toString(stddev));
            job.set("count", Integer.toString(count));
*/
        }
    }

    public static class PreThirdMapper extends Mapper<LongWritable, Text, FloatWritable, Text>{
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," (),\t");

            String img_i = tokenizer.nextToken();
            String img_k = tokenizer.nextToken();
            float sim = Float.parseFloat(tokenizer.nextToken());

            context.write(new FloatWritable(sim), new Text(Float.toString(sim)));
        }     
    }

    public static class PreThirdReducer extends Reducer<FloatWritable, Text, FloatWritable, Text> {

        public void reduce(FloatWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text val: values) {
                context.write(key, val);
            }
        }
    }

    public static class PreFourMapper extends Mapper<LongWritable, Text, LongWritable, Text> {

        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," (),\t");

            float fKey = Float.parseFloat(tokenizer.nextToken());
            float sim = Float.parseFloat(tokenizer.nextToken());

            int medianKey = Integer.parseInt(context.getConfiguration().get("median_key")); 
            //String outValue = "sim: " + Float.toString(sim) + " key: " + job.get("median_key");
            String outValue = Float.toString(sim);
            context.write(new LongWritable(0), new Text(outValue));
            //if (medianKey == key.get())
            //    context.write(new IntWritable(0), new Text(outValue));
        }     
    }

    public static class PreFourReducer extends Reducer<LongWritable, Text, IntWritable, Text> {

        public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int count = 0; 
            int medianKey = Integer.parseInt(context.getConfiguration().get("median_key"));
            for (Text val: values) {
                count++;
                float sim = Float.parseFloat(val.toString());
                if (count == medianKey) {
                    context.write(new IntWritable(0), new Text(Float.toString(sim)));
                    try {                                    
                        FileSystem fs;
                        fs = FileSystem.get(context.getConfiguration());
                        String path_str = context.getConfiguration().get("path");
                        Path path_median = new Path(path_str + "/median");
                        if(!fs.exists(path_median)) {
                          DataOutputStream out = fs.create(path_median);
                          out.writeFloat(sim);
                          out.close();
                        }
                    
                    } catch(Exception e) {
                        throw new IOException(e.getMessage());
                    }
                }
            }
        }
    }


    public static class InitMapper extends Mapper<LongWritable, Text, Text, Text> {


        private void output(String source, String target, float sim, Context context) throws NumberFormatException, IOException, InterruptedException {

            String similarity = "s" + " " + target + " " + Float.toString(sim);
            Text outputKey = new Text(source);
            context.write(outputKey, new Text(similarity));

            similarity = "s" + " " + source + " " + context.getConfiguration().get("median");
            context.write(outputKey, new Text(similarity));

            String responsibility = "r" + " " + target + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(responsibility));

            responsibility = "r" + " " + source + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(responsibility));

            String availability = "a" + " " + target + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(availability));

            availability = "a" + " " + source + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(availability));
 
        }

        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
	        String line = value.toString();
	        StringTokenizer tokenizer = new StringTokenizer(line," (),\t");
	        String img_i = tokenizer.nextToken();
	        String img_k = tokenizer.nextToken();
	        float sim = Float.parseFloat(tokenizer.nextToken());

            output(img_i, img_k, sim, context);
            output(img_k, img_i, sim, context);

            /*
            String similarity = similarity = "s" + " " + img_k + " " + Float.toString(sim);
            Text outputKey = new Text(img_i);
            context.write(outputKey, new Text(similarity));

            similarity = "s" + " " + img_i + " " + job.get("median");
            context.write(outputKey, new Text(similarity));

            String responsibility = "r" + " " + img_k + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(responsibility));

            responsibility = "r" + " " + img_i + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(responsibility));

            String availability = "a" + " " + img_k + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(availability));

            availability = "a" + " " + img_i + " " + Float.toString(0.0f);
            context.write(outputKey, new Text(availability));
            */
	    }	
    }

    public static class IterationMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," \t");
        
            String img_i = tokenizer.nextToken();
            String action = tokenizer.nextToken();
            String img_k = tokenizer.nextToken();
            float sim = Float.parseFloat(tokenizer.nextToken());
        
            Text output_key = new Text(img_i);
            context.write(output_key, new Text(String.format("%s %s %f", action, img_k, sim)));
        }
    }

    public static class CleanMapper extends Mapper<LongWritable, Text, Text, Text>{
	    public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line," \t");

            String img_i = tokenizer.nextToken();
            String action = tokenizer.nextToken();
            String img_k = tokenizer.nextToken();
            float sim = Float.parseFloat(tokenizer.nextToken());
            
            Text output_key = new Text(img_i);
            context.write(output_key, new Text(String.format("%s %s %f", action ,img_k ,sim)));
        }
    }


    public static class SA2RReducer extends Reducer<Text, Text, Text, Text> {

        private void checkConvergence(HashMap<String, Float> row_list_c, Context context) throws IOException, InterruptedException {

            int datacount = Integer.parseInt(context.getConfiguration().get("datacount"));
            if (((float)row_list_c.size() / (float)datacount) > a_converge_portion) {
                try {                    
                    FileSystem fs;                    
                    fs = FileSystem.get(context.getConfiguration());
                    String path_str = context.getConfiguration().get("path");
                    Path path = new Path(path_str);
                    if(!fs.exists(path)) {
                        DataOutputStream out = fs.create(path);
                        out.writeFloat(0.0f);
                        out.close();
                    }
                } catch(Exception e) {
                    throw new IOException(e.getMessage());                
                }
            }   
        }

	    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            String img_i = key.toString();
            String line;
            float sim;
            
            HashMap<String, Float> row_list_s = new HashMap<String, Float>();
            HashMap<String, Float> row_list_r = new HashMap<String, Float>();
            HashMap<String, Float> row_list_a = new HashMap<String, Float>();
            HashMap<String, Float> row_list_c = new HashMap<String, Float>();

            AffinityPropagationMapReduce.extractSRAList(values, row_list_s, row_list_r, row_list_a, row_list_c);

            if (img_i.equals("a_converged")) {
                checkConvergence(row_list_c, context);
                return;
            }
            
            String key_max = null;
            String key_second_max = null;
            //float val_max = Float.MIN_VALUE;
            //float val_second_max = Float.MIN_VALUE;
            float val_max = Float.NEGATIVE_INFINITY;
            float val_second_max = Float.NEGATIVE_INFINITY;
            float maxSimilarityOfExemplars = Float.NEGATIVE_INFINITY;
            float secondMaxSimilarityOfExemplars = Float.NEGATIVE_INFINITY;
            
            for (Map.Entry<String, Float> entry : row_list_s.entrySet()) {

                String cur_key = entry.getKey();
                float cur_a_plus_s = row_list_a.get(cur_key) + row_list_s.get(cur_key);
                if (cur_a_plus_s > val_max) {
            	    key_second_max = key_max;
            	    val_second_max = val_max;
              	    key_max = cur_key;
               	    val_max = cur_a_plus_s;
                    secondMaxSimilarityOfExemplars = maxSimilarityOfExemplars;
                    maxSimilarityOfExemplars = row_list_s.get(cur_key);
                } else if (cur_a_plus_s > val_second_max) {
              	    key_second_max = cur_key;
             	    val_second_max = cur_a_plus_s;
                    secondMaxSimilarityOfExemplars = row_list_s.get(cur_key);
                }
            }

            Text outputKey = new Text();
            boolean notConverged = true;
            float responsibility_difference = 0.0f;

            for (Map.Entry<String, Float> entry : row_list_s.entrySet()) {
                String cur_key = entry.getKey();
                float responsibility_i_k = 0.0f;

                if(!cur_key.equals(key_max)){
              	    responsibility_i_k = (1 - damping_factor) * row_list_r.get(cur_key) + damping_factor * (row_list_s.get(cur_key) - val_max);
                }
                else {
               	    responsibility_i_k = (1 - damping_factor) * row_list_r.get(cur_key) + damping_factor * (row_list_s.get(cur_key) - val_second_max);
                }
            
                if (cur_key.equals(img_i)) {
                    if(!cur_key.equals(key_max)) { 
                        responsibility_i_k = row_list_s.get(cur_key) - maxSimilarityOfExemplars;
                    }
                    else {
                        responsibility_i_k = row_list_s.get(cur_key) - secondMaxSimilarityOfExemplars;
                    }   
                }

                responsibility_difference += Math.abs((responsibility_i_k - row_list_r.get(cur_key)) / row_list_r.get(cur_key));
            
                /* isConverged */
                /*
                if(notConverged && Math.abs(responsibility_i_k-row_list_r.get(cur_key))/row_list_r.get(cur_key) > ConvergeThreshold){
                 	notConverged = false;
                    try {
            		    FileSystem fs;
            		    fs = FileSystem.get(job);
            		    String path_str = job.get("path");
            		    Path path = new Path(path_str);
            		    if(!fs.exists(path)) {
            		    	DataOutputStream out = fs.create(path);
            			    out.writeFloat(Math.abs(responsibility_i_k-row_list_r.get(cur_key))/row_list_r.get(cur_key));
            			    out.close();
            		    }
            		} catch(Exception e) {
            		    throw new IOException(e.getMessage());
            		}
            	}
                */
            
                outputKey.set(cur_key);
            
                String similarity = "s" + " " + img_i + " " + Float.toString(row_list_s.get(cur_key));
                context.write(outputKey, new Text(similarity));
                String responsibility = "r" + " " + img_i + " " + Float.toString(responsibility_i_k);
                context.write(outputKey,new Text(responsibility));
                String availability = "a" + " " + img_i + " " + Float.toString(row_list_a.get(cur_key));
                context.write(outputKey, new Text(availability));
            }

            /* check for convergence */
            float variance = responsibility_difference / (float)row_list_s.size();

            if(variance < r_converge_threshold) {
                String convergence = "c" + " " + img_i + " " + Float.toString(variance);
                context.write(new Text("r_converged"), new Text(convergence));
            }

            /*
            if(variance < ConvergeThreshold) {
                try {
            	    FileSystem fs;
            	    fs = FileSystem.get(job);
            	    String path_str = job.get("path");
            	    Path path = new Path(path_str);
            	    if(!fs.exists(path)) {
            	    	DataOutputStream out = fs.create(path);
            		    out.writeFloat(variance);
            		    out.close();
            	    }
            	} catch(Exception e) {
            	    throw new IOException(e.getMessage());
            	}
            }
            */

		}
	}

	public static class  R2AReducer extends Reducer<Text, Text, Text, Text>{

        private void checkConvergence(HashMap<String, Float> row_list_c, Context context) throws IOException, InterruptedException {
        
            int datacount = Integer.parseInt(context.getConfiguration().get("datacount"));
            if (((float)row_list_c.size() / (float)datacount) > r_converge_portion) {
                try {                    
                    FileSystem fs;                    
                    fs = FileSystem.get(context.getConfiguration());
                    String path_str = context.getConfiguration().get("path");
                    Path path = new Path(path_str);
                    if(!fs.exists(path)) {
                        DataOutputStream out = fs.create(path);
                        out.writeFloat(0.0f);
                        out.close();
                    }
                } catch(Exception e) {
                    throw new IOException(e.getMessage());                
                }
            }   
        }
  
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

		    String img_k = key.toString();
            //System.out.println("R2AReducer " + img_k);

		    HashMap<String, Float> s_list = new HashMap<String, Float>();
		    HashMap<String, Float> r_list = new HashMap<String, Float>();
		    HashMap<String, Float> a_list = new HashMap<String, Float>();
            HashMap<String, Float> c_list = new HashMap<String, Float>();

		    AffinityPropagationMapReduce.extractSRAList(values, s_list, r_list, a_list, c_list);

            if (img_k.equals("r_converged")) {
                checkConvergence(c_list, context);
                return;
            }

		    // generate output for S-list
		    for(Map.Entry<String, Float> entry : s_list.entrySet())
			    context.write(new Text(entry.getKey()), new Text(String.format("s %s %f", img_k, entry.getValue())));

		    // calculate r(k,k)+sum{r(i,k)} and generate output for R-list
            //System.out.println("reduce function");
		    //float sum = 0.0f;
            /*
            if (r_list.containsKey(img_k))
                System.out.println("key found");
            else
                System.out.println("key not found");
            */

            float sum = r_list.get(img_k);
		    for(Map.Entry<String, Float> entry : r_list.entrySet()) {
			    context.write(new Text(entry.getKey()), new Text(String.format("r %s %f", img_k, entry.getValue())));
			    if(!entry.getKey().equals(img_k))
			        sum += Math.max(0, entry.getValue());
		    }
            //System.out.println("sum is " + sum);

		    // calculate and output new values in A-list
		    boolean notConverged = true;

            float availability_difference = 0.0f;


		    for(Map.Entry<String, Float> entry : a_list.entrySet()) {
                float val = 0;

                if(entry.getKey().equals(img_k))
                    val = (1 - damping_factor) * entry.getValue() + damping_factor * (sum - r_list.get(img_k));
                else
                    val = (1 - damping_factor) * entry.getValue() + damping_factor * Math.min(0, (sum - Math.max(0, r_list.get(entry.getKey()))));

                availability_difference += Math.abs((val - entry.getValue()) / entry.getValue());

                /* isConverged */
                /*
                if(notConverged && Math.abs(val-entry.getValue())/entry.getValue() > ConvergeThreshold) {
                    notConverged = false;
                    try {
                        FileSystem fs;
                        fs = FileSystem.get(job);
                        String path_str = job.get("path");
                        Path path = new Path(path_str);
                        if(!fs.exists(path)) {
                            DataOutputStream out = fs.create(path);
                            out.writeFloat(Math.abs(val-entry.getValue())/entry.getValue());
                            out.close();
                        }
                    } catch(Exception e){
                        throw new IOException(e.getMessage());
                    }
                }
                */
                
                context.write(new Text(entry.getKey()), new Text(String.format("a %s %f", img_k, val)));
		    }


            float variance = availability_difference / (float)a_list.size();

            if(variance < a_converge_threshold) {
                String convergence = "c" + " " + img_k + " " + Float.toString(variance);
                context.write(new Text("a_converged"), new Text(convergence));           
            } 

            /*
            if(variance < ConvergeThreshold) {
               try {
                   FileSystem fs;
                   fs = FileSystem.get(job);
                   String path_str = job.get("path");
                   Path path = new Path(path_str);
                   if(!fs.exists(path)) {
                       DataOutputStream out = fs.create(path);
                       out.writeFloat(variance);
                       out.close();
                   }
               } catch(Exception e){
                   throw new IOException(e.getMessage());
               }
            }
            */


		}
	}

	public static class  CleanReducer extends Reducer<Text, Text, Text, Text>{
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

		    String img_i = key.toString();
		    HashMap<String, Float> s_list = new HashMap<String, Float>();
		    HashMap<String, Float> r_list = new HashMap<String, Float>();
		    HashMap<String, Float> a_list = new HashMap<String, Float>();
 		    HashMap<String, Float> c_list = new HashMap<String, Float>();
 
		    AffinityPropagationMapReduce.extractSRAList(values, s_list, r_list, a_list, c_list);

		    float max_sum_a_plus_r = Float.NEGATIVE_INFINITY;
		    String max_key = null;

		    for(Map.Entry<String, Float> entry : s_list.entrySet()) {
 			    String tmp_key = entry.getKey();
			    float sum = r_list.get(tmp_key) + a_list.get(tmp_key);
         		if (sum > max_sum_a_plus_r){
		    	    max_sum_a_plus_r = sum;
			        max_key = tmp_key;
			    }
		    }
		    //context.write(new IntWritable(img_i), new IntWritable(max_key));
            float val_a = 0.0f;
            float val_r = 0.0f;
            if (r_list.containsKey(max_key))
                val_r = r_list.get(max_key);
            if (a_list.containsKey(max_key))
                val_a = a_list.get(max_key);

		   context.write(new Text(img_i), new Text(String.format("%s %f %f %f", max_key, val_r, val_a, val_r+ val_a)));

		}
	}

	protected static void extractSRAList(Iterable<Text> values, HashMap<String, Float> row_list_s, HashMap<String, Float> row_list_r, HashMap<String, Float> row_list_a, HashMap<String, Float> row_list_c) {
	    String line;
	    String img_k;
	    float sim;

	    for (Text val: values) {
            line = val.toString();
            //System.out.println(line);
            StringTokenizer tokenizer = new StringTokenizer(line, " ");
            String action = tokenizer.nextToken();
            img_k = tokenizer.nextToken();
            sim = Float.parseFloat(tokenizer.nextToken());
            if (action.equals("s") == true) {
                if (!row_list_s.containsKey(img_k))
                    row_list_s.put(img_k, sim);
            } else if (action.equals("r") == true) {
                if (!row_list_r.containsKey(img_k))
                    row_list_r.put(img_k, sim);
            } else if (action.equals("a") == true) {
                if (!row_list_a.containsKey(img_k))
                    row_list_a.put(img_k, sim);
            } else if (action.equals("c") == true) {
                if (!row_list_c.containsKey(img_k))
                    row_list_c.put(img_k, sim);
            }
        }
	}

    private static void setJobConfCompressed(Configuration jobconf) {
        jobconf.setBoolean("mapred.output.compress", true);
        jobconf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    }


    private static void initComputeMeanStandDeviation(String inputpath) throws Exception {

        //JobConf job_preinit = new JobConf(AffinityPropagationMR.class);
        
        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        if (compression)
            setJobConfCompressed(conf);
        
        Job job_preinit = new Job(conf);
        job_preinit.setJarByClass(AffinityPropagationMapReduce.class);
        job_preinit.setJobName("preInitMapper");
        FileInputFormat.setInputPaths(job_preinit, new Path(inputpath + "/part*"));
        FileOutputFormat.setOutputPath(job_preinit, new Path("output/AF_data/pre"));
        job_preinit.setOutputKeyClass(Text.class);
        job_preinit.setOutputValueClass(Text.class);
        job_preinit.setMapperClass(PreInitMapper.class);
        job_preinit.setReducerClass(PreInitReducer.class);
        //job_preinit.setNumMapTasks(38);
        job_preinit.setNumReduceTasks(19);
        //job_preinit.setLong("dfs.block.size",134217728);

        
        try {
            job_preinit.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }
 
    }

    private static int computeMeanStandDeviation() throws Exception {

        //JobConf presecond_Job = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();                                                                                     
        conf.setLong("dfs.block.size",134217728);                                                                                     
        conf.set("path", "output/AF_data/second");
        if (compression)                                                                                                              
            setJobConfCompressed(conf);

        Job presecond_Job = new Job(conf);
        presecond_Job.setJarByClass(AffinityPropagationMapReduce.class);
        presecond_Job.setJobName("preSecondJob");
        FileInputFormat.setInputPaths(presecond_Job, new Path("output/AF_data/pre/part-*"));
        FileOutputFormat.setOutputPath(presecond_Job, new Path("output/AF_data/preResult"));
        presecond_Job.setOutputKeyClass(IntWritable.class);
        presecond_Job.setOutputValueClass(Text.class);
        presecond_Job.setMapperClass(PreSecondMapper.class);
        presecond_Job.setReducerClass(PreSecondReducer.class);
        //presecond_Job.setNumMapTasks(38);
        presecond_Job.setNumReduceTasks(1);
        //presecond_Job.set("path", "output/AF_data/second");
        //presecond_Job.setLong("dfs.block.size",134217728);
        
        int datacount = 0;
        
        try {
            //job_cli.runJob(presecond_Job);   
            
            presecond_Job.waitForCompletion(true);

            FileSystem fs;
            fs = FileSystem.get(conf);
            String path_str = "output/AF_data/second";
            Path path_count = new Path(path_str + "/count");
            if(fs.exists(path_count)) {
              DataInputStream in = fs.open(path_count);
              datacount = in.readInt();
              in.close();
            }
        } catch(Exception e){         
            e.printStackTrace();        
        }

        return datacount;
 
    }

    private static void sortInputData(String inputpath) throws Exception {

        //JobConf prethird_Job = new JobConf(AffinityPropagationMR.class);
        
        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        if (compression)
            setJobConfCompressed(conf);

        Job prethird_Job = new Job(conf);
        prethird_Job.setJarByClass(AffinityPropagationMapReduce.class);
        prethird_Job.setJobName("preThirdJob");
        FileInputFormat.setInputPaths(prethird_Job, new Path(inputpath + "/part*"));
        FileOutputFormat.setOutputPath(prethird_Job, new Path("output/AF_data/sort"));
        prethird_Job.setOutputKeyClass(FloatWritable.class);
        prethird_Job.setOutputValueClass(Text.class);
        prethird_Job.setMapperClass(PreThirdMapper.class);
        prethird_Job.setReducerClass(PreThirdReducer.class);
        //prethird_Job.setNumMapTasks(38);
        prethird_Job.setNumReduceTasks(1);
        //prethird_Job.setLong("dfs.block.size",134217728);

        try {
            prethird_Job.waitForCompletion(true);
        } catch(Exception e){         
            e.printStackTrace();        
        }
 
    }

    private static float getMedianOfInputData(int mediankey) throws Exception {

        //JobConf prejob_four = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("median_key", Integer.toString(mediankey));
        conf.set("path", "output/AF_data/median");
        if (compression)
            setJobConfCompressed(conf);

        Job prejob_four = new Job(conf);
        prejob_four.setJarByClass(AffinityPropagationMapReduce.class);
        prejob_four.setJobName("PreFourMapper");
        FileInputFormat.setInputPaths(prejob_four, new Path("output/AF_data/sort"));
        FileOutputFormat.setOutputPath(prejob_four, new Path("output/AF_data/median"));
        prejob_four.setOutputKeyClass(LongWritable.class);
        prejob_four.setOutputValueClass(Text.class);
        prejob_four.setMapperClass(PreFourMapper.class);
        prejob_four.setReducerClass(PreFourReducer.class);
        //prejob_four.setNumMapTasks(38);
        prejob_four.setNumReduceTasks(1);
        //prejob_four.setLong("dfs.block.size",134217728);
        //prejob_four.set("median_key", Integer.toString(mediankey));
        //prejob_four.set("path", "output/AF_data/median");

        float median = 0.0f;
		try {
            prejob_four.waitForCompletion(true);

            FileSystem fs;
            fs = FileSystem.get(conf);
            String path_str = "output/AF_data/median";
            Path path_median = new Path(path_str + "/median");
            if(fs.exists(path_median)) {
              DataInputStream in = fs.open(path_median);
              median = in.readFloat();
              in.close();
            }
		}
        catch(Exception e){
		    e.printStackTrace();
		}


        try {
            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path = new Path("output/AF_data/sort");
            if(fs.exists(path)){
                fs.delete(path, true);
            }
        } catch(Exception e){
            e.printStackTrace();
        }


        return median;
 
    }


    private static void initClustering(String inputpath, float median) throws Exception {

        //JobConf job_init = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("median", Float.toString(median));
        if (compression)
            setJobConfCompressed(conf);

        Job job_init = new Job(conf);
        job_init.setJarByClass(AffinityPropagationMapReduce.class);
        job_init.setJobName("initMapper");
        FileInputFormat.setInputPaths(job_init, new Path(inputpath + "/part*"));
        FileOutputFormat.setOutputPath(job_init, new Path("output/AF_data/SA2R/iter-0"));
        job_init.setOutputKeyClass(Text.class);
        job_init.setOutputValueClass(Text.class);
        job_init.setMapperClass(InitMapper.class);
        //job_init.setNumMapTasks(38);
        job_init.setNumReduceTasks(19);
        //job_init.setLong("dfs.block.size",134217728);
        //job_init.set("median", Float.toString(median));

		try {
            job_init.waitForCompletion(true);
		} catch(Exception e){
		    e.printStackTrace();
		}
 
    }

    private static boolean sa2rStep(int current_iteration, int datacount) throws Exception {

        //JobConf job_SA2R_conf = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("datacount", Integer.toString(datacount));
        conf.set("path", "output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration - 1) + "_isConverged");
        if (compression)
            setJobConfCompressed(conf);

        Job job_SA2R_conf = new Job(conf);
        job_SA2R_conf.setJarByClass(AffinityPropagationMapReduce.class);
        job_SA2R_conf.setJobName("SA2R-" + Integer.toString(current_iteration));
        FileInputFormat.setInputPaths(job_SA2R_conf, new Path("output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration) + "/part-*"));
        FileOutputFormat.setOutputPath(job_SA2R_conf, new Path("output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration)));
        job_SA2R_conf.setOutputKeyClass(Text.class);
        job_SA2R_conf.setOutputValueClass(Text.class);
        job_SA2R_conf.setMapperClass(IterationMapper.class);
        job_SA2R_conf.setReducerClass(SA2RReducer.class);
        //job_SA2R_conf.setNumMapTasks(38);
        job_SA2R_conf.setNumReduceTasks(19);   
        //job_SA2R_conf.set("path", "output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration - 1) + "_isConverged");
        //job_SA2R_conf.setLong("dfs.block.size",134217728);
        //job_SA2R_conf.set("datacount", Integer.toString(datacount));
        
        boolean converged = false;
        
        try {
            job_SA2R_conf.waitForCompletion(true);

            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path = new Path("output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration - 1)  + "_isConverged");
            if(current_iteration > 10 && fs.exists(path)) {
                converged = true;
                System.out.println("R2A iter " + Integer.toString(current_iteration - 1) + " is converged.");
            }
        } catch(Exception e){
            e.printStackTrace();
        }


        try {
            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path_SA2R = new Path("output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration - 1));
            if(fs.exists(path_SA2R)){
                fs.delete(path_SA2R, true);
            }
        } catch(Exception e){
            e.printStackTrace();
        }


        return converged;
 
    }


    private static boolean r2aStep(int current_iteration, int datacount) throws Exception {

        //JobConf job_R2A_conf = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("path", "output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration) + "_isConverged");
        conf.set("datacount", Integer.toString(datacount));
        if (compression)
            setJobConfCompressed(conf);

        Job job_R2A_conf = new Job(conf);
        job_R2A_conf.setJarByClass(AffinityPropagationMapReduce.class);
        job_R2A_conf.setJobName("R2A-" + Integer.toString(current_iteration));
        FileInputFormat.setInputPaths(job_R2A_conf, new Path("output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration) + "/part-*"));
        FileOutputFormat.setOutputPath(job_R2A_conf, new Path("output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration + 1)));
        job_R2A_conf.setOutputKeyClass(Text.class);
        job_R2A_conf.setOutputValueClass(Text.class);
        job_R2A_conf.setMapperClass(IterationMapper.class);
        job_R2A_conf.setReducerClass(R2AReducer.class);
        //job_R2A_conf.setNumMapTasks(38);
        job_R2A_conf.setNumReduceTasks(19);   
        //job_R2A_conf.set("path", "output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration) + "_isConverged");
        //job_R2A_conf.setLong("dfs.block.size",134217728);
        //job_R2A_conf.set("datacount", Integer.toString(datacount));
        
        boolean converged = false;
        
        try {
            job_R2A_conf.waitForCompletion(true);

            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path = new Path("output/AF_data/SA2R/" + "iter-" + Integer.toString(current_iteration) + "_isConverged");
            if(current_iteration > 10 && fs.exists(path)) {
                converged = true;
                System.out.println("SA2R iter " + Integer.toString(current_iteration) + " is converged.");
            }
        } catch(Exception e){
            e.printStackTrace();
        }
 
        try {
            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path_R2A = new Path("output/AF_data/R2A/" + "iter-" + Integer.toString(current_iteration - 1));  
            if(fs.exists(path_R2A)){
                fs.delete(path_R2A, true);
            }
        } catch(Exception e){
            e.printStackTrace();
        }

        return converged;

    }

    private static void writeOutput(int final_iteration) throws Exception {

 		//JobConf job_Clean_conf = new JobConf(AffinityPropagationMR.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        if (compression)
            setJobConfCompressed(conf);

        Job job_clean_conf = new Job(conf);
        job_clean_conf.setJarByClass(AffinityPropagationMapReduce.class);
		job_clean_conf.setJobName("Find exemplars");
		FileInputFormat.setInputPaths(job_clean_conf, new Path("output/AF_data/SA2R/" + "iter-" + Integer.toString(final_iteration + 1)));
		FileOutputFormat.setOutputPath(job_clean_conf, new Path("output/AF_data/Output/"));
		job_clean_conf.setOutputKeyClass(Text.class);
		job_clean_conf.setOutputValueClass(Text.class);
		job_clean_conf.setMapOutputKeyClass(Text.class);
		job_clean_conf.setMapOutputValueClass(Text.class);
		job_clean_conf.setMapperClass(CleanMapper.class);
		job_clean_conf.setReducerClass(CleanReducer.class);
		//job_clean_conf.setNumMapTasks(38);
		job_clean_conf.setNumReduceTasks(19);   
		//job_clean_conf.setLong("dfs.block.size",134217728);


		try {
            job_clean_conf.waitForCompletion(true);
		} catch(Exception e){
		    e.printStackTrace();
		}
 
    }

	public static void main(String[] args) throws Exception{


        String inputpath = null;
        if (args.length > 0)
            inputpath = args[0];        

        if (args.length > 1 && args[1].equals("compress"))
            compression = true;            

        boolean smallClusterSize = false; 
        if (args.length > 2 && args[2].equals("s"))
            smallClusterSize = true;      
    
        int iteration_time = 50;             


        initComputeMeanStandDeviation(inputpath);
        int datacount = computeMeanStandDeviation();

        sortInputData(inputpath);

        int mediankey = 0;

        if (datacount % 2 == 1)
            mediankey = (datacount + 1) / 2;
        else
            mediankey = datacount / 2;

        if (smallClusterSize)
            mediankey = 1;

        float median = getMedianOfInputData(mediankey);

 
        initClustering(inputpath, median);
 
        int current_iteration = 0;

        for(int i = 0; i < iteration_time; i++) {
            current_iteration = i;

            boolean r2a_converged = sa2rStep(current_iteration, datacount);
            boolean sa2r_converged =  r2aStep(current_iteration, datacount);

            if (sa2r_converged && r2a_converged)
                break;

        }


        writeOutput(current_iteration);

	}

}
