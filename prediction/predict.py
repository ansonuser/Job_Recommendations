import datetime  
import sys
import os
sys.path.append(os.getcwd() + "\\..")
from utils.helper import CFG, Logger
from preprocess.loader import DataStream
from nlp_functions.match_func import Matcher
from nlp_functions.llm_utils import Agent
from collections import defaultdict
import pandas as pd
import datetime
import gc
import torch
import time
import numpy as np
import pdb

class Predictor:
    def __init__(self, qsize=2):
        self.supervisor = Agent(model_path=CFG["llm"]["mistral"])
        self.unsupervisor = Matcher()
        self.stream = DataStream(qsize=qsize)
        self.max_rank = 20
        self.max_token_size = 250 
    def recommend(self, mode="AM"):
        """
        mode: 'A', 'M', 'AM'
        """
        ranks = None
        flags = [False]*2
        if "M" in mode:
            self.unsupervisor.set_data(*self.stream.send_data())    
            similarity = self.unsupervisor.sim()
            flags[0] = True
        self.unsupervisor.model.to("cpu")
        self.unsupervisor = None
        print("Before delete model:", torch.cuda.memory_reserved()/10**9)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        print("After delete model:", torch.cuda.memory_reserved()/10**9)
        if "A" in mode:
            self.supervisor.load_models()
            print("After loading model:", torch.cuda.memory_reserved()/10**9)
   
            resume, jobs = self.stream.send_data()
            self.supervisor.resume = resume.form()[:self.max_token_size]
            self.supervisor.model.eval()
            res = []
            scores = []
            for job in jobs:
                job_id = job.Company_Name + "," + job.Title
                score = self.supervisor.do_grade([job.form()])
                res.append((job_id, job.Link))
                scores.append(score[0])
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 
            ranks = np.argsort(scores)
            res = [(c[0], c[1], r) for c,r in zip(res, ranks)]
            flags[1] = True


        if "AM" == mode:
            link_map = {}
            rank_dict = defaultdict(int)
            n = len(res)
            for i in range(n):
                key = link = None
                if flags[0]:
                    key, link = similarity[i][:2]
                    rank_dict[key] += similarity[i][2]/2
                if flags[1]:
                    key, link = res[i][:2] 
                    rank_dict[key] += res[i][2]/2
                link_map[key] = link

            ranks = [(k, link_map[k], v) for k,v in rank_dict.items()]
            ranks.sort(key = lambda x:x[2])
        else:
            ranks = res
        ranks = ranks[:self.max_rank]
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%%S")
        pd.DataFrame(ranks, columns=["Job ID", "Link", "Rank"]).to_csv(f"job_rank_{time_str}.csv")
        return ranks
    
if __name__ == "__main__":
    predictor = Predictor(qsize=50)
    ranks = predictor.recommend()
    print(ranks)