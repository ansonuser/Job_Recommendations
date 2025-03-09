### Introduction

This is a toy example of job recommendation.

### Data Source
1. Dice
2. Themuse 

### DB
Elasticsearch: 
1. Data has different features
2. Consider frequency of job updating is not so often. Word->doc and standardized token design provide agility to handle. 

### WorkFlow
                               modeling
crawling -> matching -> database --->  filter
                | filter      ^  
                V             | feedback
            notification -> application

### Details
It's an unsupervised task at beginning(cold start). Match the resume to each job description.  

metric:
- Semantic similarity: Computes text similarity between resume & JD (word embeding)
- TF-IDF : Keywords (Skill)

Until enough label, using PEFT to fine-tune LLM for binary classification. It can be combined with similarity too.

### Requirements
1. Docker
2. Elasticsearch

```
docker network create elastic
# pull
docker run -d -p 9200:9200 -e "discovery.type=single-node" --net elastic --name elasticsearch docker.elastic.co/elasticsearch/elasticsearch:8.6.0

# set password
docker exec -it elasticsearch bin/elasticsearch-reset-password -u elastic -i

```
Vis Tool (Optional)
```
# generate token to the user (It will be used when you first log in kibana)
docker exec -it elasticsearch bin/elasticsearch-create-enrollment-token -s kibana

docker run -d --name kibana `
    --net elastic `
    -p 5601:5601 `
    docker.elastic.co/kibana/kibana:8.5.3


```



Test if your elasticsearch function normally:
```
import requests
r = requests.get("https://localhost:$PORT_ID/", auth=("elastic", $PASSWORD), verify=False)
# %%
```

For efficieny, make sure timestamp is mapped to date. If not, please update it as follows:

$ PUT jobs_db/_mapping
```
{
  "properties": {
    "timestamp": { "type": "date" }
  }
}
```

Rank the positions from your database
```
python predction/predict.py
```