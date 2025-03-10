from sentence_transformers import SentenceTransformer
from utils.dataset import Resume, Job
from typing import List
import torch


class Matcher:
    model = SentenceTransformer("BAAI/bge-m3")

    def __init__(self, cv:Resume=None, jobs:List[Job]=[]):
        self.cv = cv 
        self.jobs = []
        self.jobs += jobs
        self.cv_embeddings = None        
        self.results = []
        self.sep = "\n"
    
    def set_data(self, cv:Resume=None, jobs:List[Job]=[]):
        self.cv = cv
        self.jobs = jobs
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
            cur.append((job.Company_Name + "," + job.Title, job.Link, score.item()))
        order = sorted(cur, key=lambda x:x[2], reverse=True) 
        return [(c[0], c[1], o) for c,o in zip(cur, order)]
        
        
            
        