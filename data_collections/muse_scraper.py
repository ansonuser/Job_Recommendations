from bs4 import BeautifulSoup
import re
import sys
import os
sys.path.append(os.getcwd() + "\\..")
from utils.helper import HEADERS
import requests
import copy
from typing import List, Tuple
from collections import deque
import time
import random
from utils.helper import CFG, Logger
from elasticsearch import Elasticsearch


class TheMuse:
    """
    Wrapper for crawling data from  https://www.themuse.com/
    """
    def __init__(self, limit_per:int=10, last_days:int=14, remote:bool=True, min_return:int=None):
        self.domain_name = "https://www.themuse.com/"
        self.limit_per = limit_per
        self.last_days = last_days
        self.remote = remote
        self.features = ["title", "company@name$short_name", "short_title", "posted_at"]
        if not min_return:
            self.min_return = limit_per
        else:
            self.min_return = min_return
        self.levels = ["mid", "senior"]
        self.es = Elasticsearch(
            [f"https://localhost:{CFG['elasticsearch']['Port']}"],
            basic_auth = ("elastic", str(CFG['elasticsearch']['Password'])),
            verify_certs=False
        )
        # job_id : link
    def get_url(self, keywords, limit_per, last_days, remote=True)->Tuple[str, dict]:
        """
        Generate url for searching
        
        Parameters
        --------------------
        keywords: List
            Your desired positions

        limit_per: int
            The maximum return number per request

        last_days: int
            Only search data no earlier than last_days before
         
        remote: bool
            Request for remote or not

        return: [url:str, info:dict]
            url:
              for the searching page.
            Ex: 
            "https://www.themuse.com/api/search-renderer/jobs?ctsEnabled=true&query=Data&level=,mid&posted_date_range=last_7d&limit=1&timeout=5000"
        
            info:
              information for positions
              keys: location, company name(short name), title(short title), post time, query time, job description
            Note: job_id ($company_name,$title) as database index  
        
        """
        if remote:
            infix = "&remote_work_location_type=virtual"
        else:
            infix = ""
        search_words = "+".join(keywords)
        levels = ",".join(self.levels)
        url = f"{self.domain_name}api/search-renderer/jobs?ctsEnabled=true&query={search_words}&level=,{levels}{infix}&posted_date_range=last_{last_days}d&limit={limit_per}&timeout=5000"
        
        return url
    
    def get_info(self, cur_hit)->Tuple[str, dict]:
        """
        Extract important keys from input
        """
        res = {}
        if cur_hit["locations"]:
            loc = cur_hit["locations"][0]["country"] + "," + cur_hit["locations"][0]["address"]
        else:
            loc = ""
        res["location"] = loc
        for feature in self.features:
            layers = feature.split("@")
            tmp = copy.copy(cur_hit)
            for l in layers:
                if "$" in l: # "union" operation 
                    keys = l.split("$")
                    for key in keys:
                        res[key] = tmp[key]
                else:
                    tmp = tmp[l]
                    if not isinstance(tmp, dict):
                        res[l] = tmp
                        break

        suffix = res["short_name"] + "/" + res["short_title"]
        return (self.domain_name + "jobs/" + suffix, res)
  
    

    def search(self, keywords:List, configs:dict=None, default=True):
        """
        First searching for the keywords. If configs provided, use the settings from configs.
        Otherwise, use default settings.
        """
        if default:
            url = self.get_url(keywords, self.limit_per, self.last_days, self.remote)
        else:
            url = self.get_url(keywords, configs["limit_per"], configs["last_days"], configs["remote"])
      
        resp_json = self.do_request(url)  
        # truncated by min constraint
        total_num = min(self.limit_per, self.min_return)
        url_queue = deque(maxlen=total_num)

       
        for hit in resp_json["hits"][:total_num]:
            try:
                url_queue.appendleft(self.get_info(hit["hit"]))
            except Exception as e:
                print(str(e))
                print("Error when add info to queue")
                continue
        Logger.info(f"[{str(self)}] ✅ Stored {len(url_queue)} job links.")
        res = []
        while url_queue:
            link, info_dict = url_queue.pop()
            jd = self.get_detail(link)
            info_dict["Description"] = jd
            job_id = info_dict["name"] + "," + info_dict["title"]
            info_dict["Labels"] = [] # cold start
            info_dict["Source"] = str(self)
            info_dict["Timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.es.index(index="jobs_db", id=job_id, document=info_dict)
            Logger.info(f"[{str(self)}] Insert {job_id} to jobs_db.")
            res.append(info_dict)
            sleep_time = random.uniform(1, 3)
            time.sleep(sleep_time)
        return res
    
    def close_elastic(self):
        del self.es

    @classmethod
    def do_request(cls, url, json=True):
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            raise Exception(f"Connection Error:{resp.status_code}")
        if json:
            return resp.json()
        else:
            return resp.text
        
    def __str__(self):
        return "TheMuse"
        
    def get_detail(self, url)->str:
        """
        Click the link and extract content from JD page.
        """
        try:
            detail =  self.do_request(url, json=False)
            soup = BeautifulSoup(detail, "html.parser")
            jds = soup.find_all("article",class_=re.compile("JobIndividualBody_jobBodyDescription"))
            jd = jds[0].find_all(recursive=False)[-1]
            for br in jd.find_all("br"):
                br.replace_with("\n")

            for ul in jd.find_all("ul"):
                items = [li.get_text(strip=True) for li in ul.find_all("li")]
                ul.replace_with("\n".join(f"- {item}" for item in items))

            clean_text = jd.get_text(separator="\n", strip=True)
            return clean_text
        except:
            return ""
 

if __name__ == "__main__":
    scraper = TheMuse(limit_per=50)
    keywords = ["Data", "Machine", "Scientist"]
    jobs = scraper.search(keywords)
    scraper.close_elastic()