# Topic modeling using twitter data

This repo contains necessary tools for creating a topic model based on twitter data. 

 - Docker-compose scripts for a local mongodb and spark cluster 
 - Code to load live twitter data to the mongodb instance `main.py` 
 - Preprocess the data using notebook in the spark local instance using pyspark `preprocess` 
 - Two different types of topic models: `etm` and `ctm`. Read about them in https://github.com/adjidieng/ETM and https://github.com/MilaNLProc/contextualized-topic-models respectively. 
 
    