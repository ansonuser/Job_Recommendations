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

class Predictor:
    def __init__(self):
        self.supervisor = Agent(model_path=CFG["llm"]["mistral"])
        assert self.supervisor.model_path is not None
        self.unsupervisor = Matcher()
        self.stream = DataStream()
        self.max_size = 20
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

        if "A" in mode:
            self.supervisor.load_models()
            resume, jobs = self.stream.send_data()
            self.supervisor.resume = resume
            res = []
            for job in jobs:
                job_id = job["Compnay_Name"] + "," + job["Title"]
                score = self.supervisor.do_grade(job.form())
                res.append((job_id, job["Link"], score))
            order = sorted(res, key=lambda x:x[2], reverse=True) 
            res = [(c[0], c[1], o) for c,o in zip(res, order)]
            flags[1] = True


        if "AM" == mode:
            link_map = {}
            rank_dict = defaultdict(0)
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
        ranks = ranks[:self.max_size]
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%%S")
        pd.DataFrame(ranks, columns=["Job ID", "Link", "Rank"]).to_csv(f"job_rank_{time_str}.csv")
        return ranks
    
if __name__ == "__main__":
    predictor = Predictor()
    ranks = predictor.recommend()
    print(ranks)