from elasticsearch import Elasticsearch
import datetime  
import sys
import os
sys.path.append(os.getcwd() + "\\..")
from utils.helper import CFG, Logger
from utils.dataset import Job, Resume
import yaml 

class DataStream:
    def __init__(self):
        self.es = Elasticsearch(
            ["https://localhost:9200"],
            basic_auth = ("elastic", str(CFG['elasticsearch']['Password'])),
            verify_certs=False
            )  
        self.index_name = "jobs_db"
        self.resume_path = os.getcwd(".") + "\\..\\configs\\resume.yaml"
    def query_job_by_time(self, days=7):
        now = datetime.datetime.now(datetime.timezone.utc)
        days_ago = now - datetime.timedelta(days=days)
        start_date = days_ago.strftime("%Y-%m-%d %H:%M:%S")
        end_date = now.strftime("%Y-%m-%d %H:%M:%S")
        query = {
            "size": 50, 
            "query": {
                "range": {
                    "Timestamp": {  # Replace with your actual timestamp field name
                        "gte": start_date,
                        "lte": end_date,
                        "format": "yyyy-MM-dd HH:mm:ss"
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=query)  # Adjust 'size' for more results
        return response
    
    def close_elastic(self):
        del self.es

    def send_data(self):
        response = self.query_job_by_time()
        feature_names = ["Title", "Overview", "Top Skills", "Description", "Company Name"]
        jobs = [ Job(**{k:response["hits"]["hits"][i]["_source"][k] for k in feature_names}) for i in range(len(response["hits"]["hits"]))]
        with open(self.resume_path, "r") as f:
            resume = yaml.safe_load(f)
        resume = Resume(**resume)
        return resume, jobs


if __name__ == "__main__":
    ds = DataStream()
    resume, jobs = ds.send_data()

