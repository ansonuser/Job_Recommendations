from sentence_transformers import SentenceTransformer
from utils.dataset import Resume, Job
from typing import List
import torch


class Matcher:
    model = SentenceTransformer("BAAI/bge-m3")

    def __init__(self, cv:Resume, jobs:List[Job]):
        self.cv = cv 
        self.jobs = []
        self.jobs += jobs
        self.cv_embeddings = None        
        self.results = []
        self.sep = "\n"
    def encode(self, data:str, cv=False)->torch.Tensor:
        if cv:
            self.cv_embeddings = self.model.encode(data, convert_to_tensor=True)
            return self.cv_embeddings
        else:
            return self.model.encode(data, convert_to_tensor=True)

    def sim(self):
        if self.cv_embeddings is None:
            data = [self.cv.Level, self.cv.Skills, self.cv.Experience, self.cv.Fields, self.cv.Task, self.cv.Other]
            data = self.sep.join(data)
            self.encode(data, cv=True)
        cur = []
        for job in self.jobs:
            data = [job.Title, job.Top_Skills]#, job.description]
            data = self.sep.join(data)
            score = torch.dot(self.cv_embeddings, self.encode(data))
            cur.append((job["Compnay_Name"] + "," + job["Title"], score.item()))
        cur.sort(key=lambda x:x[1])
        return cur
        
        
            
        