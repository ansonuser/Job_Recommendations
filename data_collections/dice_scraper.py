from selenium import webdriver
import os
import sys
sys.path.append(os.getcwd() + "\\..")
from utils.helper import options, CFG, Logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import time
from typing import List
from elasticsearch import Elasticsearch

class Dice:
    """"
    Wrapper for crawling data from  https://www.dice.com
    """
    def __init__(self):
        self.driver = webdriver.Chrome(options=options)
        self.domain_name = "https://www.dice.com"
        self.wait = WebDriverWait(self.driver, 10)
        self.es = Elasticsearch(
            [f"https://localhost:{CFG['elasticsearch']['Port']}"],
            basic_auth = ("elastic", str(CFG['elasticsearch']['Password'])),
            verify_certs=False
        )

    def quit(self):
        self.driver.quit()
        
    def run(self, keywords:List, page_size:int=10)->List[dict]:
        """
        Get jobs' infomation

        Parameters
        --------------------
        keywords: List
            Keywords for desired position

        page_size: int
            The maximum amount of positions per request

        return : list
            List of positions info
            keys:  
                location, title, post date, company name, link of position, payment, contract type, top skills, description, query time
            Note: job_id: $company_name,$title
        """
        keywords = "%20".join(keywords) #sep by space 
        job_links = self.build_link_queue(keywords, page_size)
        data = []
        for job in job_links:
            data.append(self.visit_queue(job))
        
        return data

    def build_link_queue(self, keywords, page_size=10):
  
        job_links = []  
       
        url = f"{self.domain_name}/jobs?q={keywords}&countryCode=US&&page=1&pageSize={page_size}&filters.postedDate=SEVEN&filters.workplaceTypes=Remote&filters.employmentType=FULLTIME&language=en"
        self.driver.get(url)
        try:
            job_elements = self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.card-title-link")))
            for job_element in job_elements:
                job_title = job_element.text.strip()
                # Extract job ID from the "a" tag's id attribute
                job_id = job_element.get_attribute("id")
                if job_id:
                    job_url = f"{self.domain_name}/job-detail/{job_id}"
                    job_links.append({"title": job_title, "url": job_url})

            Logger.info(f"[{str(self)}] ✅ Stored {len(job_links)} job links.")
        except Exception as e:
            print("❌ Error:", str(e))
        return job_links
    
    def __str__(self):
        return "Dice"
    
    def visit_queue(self, job):
        job_data = None    
        try:
            self.driver.get(job["url"])
            
            time.sleep(random.uniform(3, 5))  
           
            try:
                job_loc = self.driver.find_element(By.XPATH, "//*[@data-cy='location']").text.strip()
            except:
                job_loc = "No Location"
            # Extract Company Name
            try:
                company_name = self.driver.find_element(By.XPATH, "//*[@data-cy='companyNameLink']").text.strip()
            except:
                company_name = "No Company Info"

            # Extract Post Date
            try:
                post_date = self.driver.find_element(By.XPATH, "//*[@data-cy='postedDate']").text.strip()
            except:
                post_date = "No Post Date"

            try:
                overview_element = self.driver.find_element(By.XPATH, "//*[contains(@class, 'job-overview_jobDetails')]")
                overview = overview_element.text.strip()
            except:
                overview = "No Overview Found"

            # Extract Payment (Salary)
            try:
                payment_element = self.driver.find_element(By.XPATH, "//*[contains(@data-cy, 'payDetails')]")
                payment = payment_element.text.strip()
            except:
                payment = "No Payment Info"

            # Extract Contract Type
            try:
                contract_elements = self.driver.find_elements(By.XPATH, "//*[contains(@data-cy, 'employmentDetails')]//span")
                contract_type = ", ".join([el.text.strip() for el in contract_elements])
            except:
                contract_type = "No Contract Info"

            try:
                skills_elements = self.driver.find_elements(By.XPATH, "//*[contains(@data-cy, 'skillsList')]//span")
                top_skills = [skill.text.strip() for skill in skills_elements[:10]]  # Extract first 5 skills only
            except:
                top_skills = ["No Skills Found"]

            # Extract Full Job Description (Corrected XPath)
            try:
                job_description_element = self.driver.find_element(By.XPATH, "//*[contains(@class, 'job-details_jobDetails')]")
                job_description = job_description_element.text.strip()
            except:
                job_description = "No Description Found"

            job["title"] = job["title"].lower() 
            company_name = company_name.lower()
            job_data = {
                "Title": job["title"],
                "Location": job_loc,
                "Post Date": post_date,
                "Company Name": company_name,
                "Link": job["url"],
                "Overview": overview,
                "Payment": payment,
                "Contract Type": contract_type,
                "Top Skills": ", ".join(top_skills),
                "Description": job_description,
                "Source": str(self),
                "Labels": [],  # cold start
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            job_id = job_data["Company Name"] + "," + job_data["Title"]
            self.es.index(index="jobs_db", id=job_id, document=job_data)
            Logger.info(f"[{str(self)}] Insert {job_id} to jobs_db.")
        except Exception as e:
            print(f"⚠️ Skipping job {job_data['title']} due to error:", str(e))
        return job_data
    def close_elastic(self):
        del self.es

        
if __name__ == "__main__":
    keywords = ["data", "scientitst", "machine", "learning"]
    size = 50
    dice = Dice()
    dice.run(keywords, size)
