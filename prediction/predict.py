import datetime  
import sys
import os
sys.path.append(os.getcwd() + "\\..")
from utils.helper import CFG, Logger
from preprocess.loader import DataStream
from nlp_functions.match_func import Matcher
from nlp_functions.llm_utils import Agent
from collections import defaultdict

class Predictor:
    def __init__(self):
        self.supervisor = Agent()
        self.unsupervisor = Matcher()
        self.stream = DataStream()
        
    def recommend(self, mode="AM"):
        """
        mode: 'A', 'M', 'AM'
        """
        ranks = None
        flags = [False]*2
        if "M" in mode:
            matcher = self.unsupervisor(self.stream.send_data())    
            similarity = matcher.sim()
            flags[0] = True

        if "A" in mode:
            self.supervisor.load_models()
            resume, jobs = self.stream.send_data()
            self.supervisor.resume = resume
            res = []
            for job in jobs:
                job_id = job["Compnay_Name"] + "," + job["Title"]
                score = self.supervisor.do_grade(job.form())
                res.append((job_id, score))
                res.sort(key=lambda x:x[1])
            flags[1] = True

        if "AM" == mode:
            rank_dict = defaultdict(0)
            n = len(res)
            for i in range(n):
                if flags[0]:
                    rank_dict[similarity[i][0]] += similarity[i][0]/2
                if flags[1]:
                    rank_dict[res[i][0]] += res[i][0]/2

            ranks = [(k, v) for k,v in rank_dict.items()]
            ranks.sort(key = lambda x:x[1])
        return ranks
    
if __name__ == "__main__":
    predictor = Predictor()
    ranks = predictor.recommend()
    print(ranks)