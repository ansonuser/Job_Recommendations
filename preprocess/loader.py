from elasticsearch import Elasticsearch
import datetime  
import sys
import os
sys.path.append(os.getcwd() + os.sep +"..")
from utils.helper import CFG, Logger
from utils.dataset import Job, Resume
import yaml 
from typing import List, Tuple
class DataStream:
    def __init__(self, qsize=10, index_name="jobs_db"):
        """
        Arguments:
            qsize: int 
                number of row data being selected
        """
        self.es = Elasticsearch(
            ["https://localhost:9200"],
            basic_auth = ("elastic", str(CFG['elasticsearch']['Password'])),
            verify_certs=False
            )  
        self.query_size = qsize 
        self.index_name = index_name
        self.resume_path = os.getcwd() + f"{os.sep}..{os.sep}configs{os.sep}resume.yaml"
        self.mapping = None
        self.feature_names = None
        self.set_features()

    def set_features(self):
        feature_names = ["Location", "Title", "Overview", "Top Skills", "Description", "Company Name", "Link", "Payment", "Contract Type", "Source"]
        feature_names +=  ["posted_at", "location", "title", "name",  "short_name", "short_title"]
        self.feature_names = feature_names
        self.mapping =  {
            "posted_at" : "Post Date",
            "location": "Location",
            "title": "Title",
            "name": "Company_Name"
            }


    def query_job_by_time(self, days=7):
        now = datetime.datetime.now(datetime.timezone.utc)
        days_ago = now - datetime.timedelta(days=days)
        start_date = days_ago.strftime("%Y-%m-%d %H:%M:%S")
        end_date = (now+ datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S") 
        print("start date:", start_date)
        print("end date:", end_date)
        query = {
            "size": self.query_size, 
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

    def send_data(self, d=7):#Tuple(Resume, List):
        """
        1. Get resume
        2. Get job released in last 7 days
        """
        response = self.query_job_by_time(days=d)
       
        jobs = [ ]
        for i in range(len(response["hits"]["hits"])):
            try:
                filtered_features = {}
                for f in self.feature_names: 
                    if f in response["hits"]["hits"][i]["_source"]:
                        if f in self.mapping:
                            aligned_f = self.mapping[f]
                            filtered_features[aligned_f] = response["hits"]["hits"][i]["_source"][f]
                        else:
                            filtered_features[f] = response["hits"]["hits"][i]["_source"][f]

                job = Job(**filtered_features)
                jobs.append(job)
            except Exception as e:
                print(str(e))
                print("Error at:", response["hits"]["hits"][i]["_source"])
             
        with open(self.resume_path, "r") as f:
            resume = yaml.safe_load(f)
        resume = Resume(**resume)
        print("Total jobs:", len(jobs))
        return resume, jobs


if __name__ == "__main__":
    ds = DataStream()
    resume, jobs = ds.send_data()

