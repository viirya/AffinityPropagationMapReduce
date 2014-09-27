
OUTPUT_DIR=bin
OUTPUT_JAR=build/AP.jar
SRC = edu/ntu/*.java
JAVA_DEP = /usr/lib/hadoop/hadoop-0.18.3-6cloudera0.3.0-core.jar:.

all: ${SRC}
	rm -rf ${OUTPUT_DIR}
	mkdir ${OUTPUT_DIR}
	javac -classpath ${JAVA_DEP} ${SRC} -d ${OUTPUT_DIR}
	jar -cfv ${OUTPUT_JAR} -C ${OUTPUT_DIR} .

clean:
	hadoop dfs -rmr output/AF_data/*

verify-clean:
	hadoop dfs -rmr output/AF_data/verifying_temp
	hadoop dfs -rmr output/AF_data/verifying
    
run:
	hadoop jar ${OUTPUT_JAR} edu.ntu.AffinityPropagationMR data/flickr550/graph/flickr550.graph.full_size_HA_1M_vw_by_flicrk11k/threshold_0.04 compress

#data/flickr550/graph/flickr550_psedoobj_normalized/threshold0.01 compress

#data/flickr550/graph/flickr550.graph.full_size_HA_1M_vw_by_flicrk11k/threshold_0.04 compress

#data/flickr550/graph/flickr550_psedoobj_normalized/threshold0.005 compress

#output/graph_backup/flickr550.full_size_HA_1M_vw_by_flicrk11k/threshold_0.002 compress

verify:
	hadoop jar ${OUTPUT_JAR} edu.ntu.VerifyingClusterWithGroundTruth output/AF_data/Output data/flickr550/ground_truth compress

#output/graph_backup/flickr550.full_size_HA_1M_vw_by_flicrk11k/threshold_0.04 compress

#data/flickr550.graph.full_size_HA_1M_vw_by_flicrk11k.threshold0.04 compress

#output/graph_backup/flickr550.full_size_HA_1M_vw_by_flicrk11k/threshold_0.002
