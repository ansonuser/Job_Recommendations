## Introduction

- [Application Tracking System](## Application Tracking System):

A Simple version of ATS, see how you resume match the position.  


- Language Model Recommendation:

A toy example of job recommendation. Using Mistral-7B in 8bits quantization takes around 7~8G VRAM for summary and classification, BGE-M3 as emedding model for semantic similarity. Combine two models to rank how positions match your resume. 

## Application Tracking System

```
cd $PATH_TO_FLASK

python app.py
```

## Language Model Recommendation

## Data Source
1. Dice
2. Themuse 

## DB
Elasticsearch: 
1. Data has different features
2. Consider frequency of job updating is not so often. Word->doc and standardized token design provide agility to handle. 

## WorkFlow

| Step          | Process      |
|--------------|-------------|
|              |             |
| Crawling     | → Matching  |
| Matching     | → Database  |
| Database     | → Filter(Modeling)|
| Filter       | → Notification  |
| Notification |  → Application  |
| Application   | → Database(Feedback) |


## Details
It's an unsupervised task at beginning(cold start). Match the resume to each job description.  

Ranking:
- Semantic similarity: Computes text similarity between resume & JD (word embeding)
- Logit : LLM prediction 

Until enough label, using PEFT to fine-tune LLM for binary classification. It can be combined with similarity too.

## Requirements
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

```
PUT jobs_db/_mapping
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



## License
MIT license