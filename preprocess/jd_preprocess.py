import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

class JobATS:
    def __init__(self):
        # Initialize database of jobs and applicants
        self.jobs = pd.DataFrame()
        self.applicants = pd.DataFrame()
        self.applications = pd.DataFrame()
        
    def add_job(self, job_id, title, company, location, department, description, 
                skills_required, experience_required, date_posted):
        """Add a new job to the database"""
        new_job = {
            'job_id': job_id,
            'title': title,
            'company': company,
            'location': location,
            'department': department,
            'description': description,
            'skills_required': skills_required,
            'experience_required': experience_required,
            'date_posted': date_posted,
            'status': 'Open'
        }
        
        self.jobs = pd.concat([self.jobs, pd.DataFrame([new_job])], ignore_index=True)
        print(f"Added job: {title} with ID: {job_id}")
        
    def add_applicant(self, applicant_id, name, email, phone, resume_text, skills, experience):
        """Add a new applicant to the database"""
        new_applicant = {
            'applicant_id': applicant_id,
            'name': name,
            'email': email,
            'phone': phone,
            'resume_text': resume_text,
            'skills': skills,
            'experience': experience,
            'date_registered': datetime.now().strftime("%Y-%m-%d")
        }
        
        self.applicants = pd.concat([self.applicants, pd.DataFrame([new_applicant])], ignore_index=True)
        print(f"Added applicant: {name} with ID: {applicant_id}")
    
    def apply_to_job(self, applicant_id, job_id, cover_letter=None):
        """Record an application from an applicant to a job"""
        if not self._job_exists(job_id):
            return f"Error: Job ID {job_id} does not exist"
        
        if not self._applicant_exists(applicant_id):
            return f"Error: Applicant ID {applicant_id} does not exist"
        
        new_application = {
            'application_id': f"APP-{len(self.applications) + 1}",
            'applicant_id': applicant_id,
            'job_id': job_id,
            'date_applied': datetime.now().strftime("%Y-%m-%d"),
            'status': 'Applied',
            'cover_letter': cover_letter,
            'match_score': self._calculate_match_score(applicant_id, job_id)
        }
        
        self.applications = pd.concat([self.applications, pd.DataFrame([new_application])], ignore_index=True)
        return f"Application submitted for applicant {applicant_id} to job {job_id}"
    
    def update_application_status(self, application_id, new_status):
        """Update the status of an application"""
        if application_id not in self.applications['application_id'].values:
            return f"Error: Application ID {application_id} does not exist"
        
        valid_statuses = ['Applied', 'Screening', 'Interview', 'Technical Test', 'Offer', 'Hired', 'Rejected']
        if new_status not in valid_statuses:
            return f"Error: Status must be one of {valid_statuses}"
        
        idx = self.applications.index[self.applications['application_id'] == application_id].tolist()[0]
        self.applications.at[idx, 'status'] = new_status
        return f"Updated application {application_id} status to {new_status}"
    
    def get_job_applicants(self, job_id, sort_by_match=True):
        """Get all applicants for a specific job"""
        if not self._job_exists(job_id):
            return f"Error: Job ID {job_id} does not exist"
        
        job_applications = self.applications[self.applications['job_id'] == job_id]
        if job_applications.empty:
            return "No applications found for this job"
        
        result = pd.merge(job_applications, self.applicants, on='applicant_id')
        
        if sort_by_match:
            result = result.sort_values(by='match_score', ascending=False)
            
        return result
    
    def get_applicant_jobs(self, applicant_id):
        """Get all jobs that an applicant has applied to"""
        if not self._applicant_exists(applicant_id):
            return f"Error: Applicant ID {applicant_id} does not exist"
        
        applicant_applications = self.applications[self.applications['applicant_id'] == applicant_id]
        if applicant_applications.empty:
            return "No applications found for this applicant"
        
        result = pd.merge(applicant_applications, self.jobs, on='job_id')
        return result
    
    def recommend_jobs(self, applicant_id, top_n=5):
        """Recommend jobs for an applicant based on their skills and experience"""
        if not self._applicant_exists(applicant_id):
            return f"Error: Applicant ID {applicant_id} does not exist"
        
        # Get applicant details
        applicant = self.applicants[self.applicants['applicant_id'] == applicant_id].iloc[0]
        applicant_skills = applicant['skills']
        applicant_experience = applicant['experience']
        
        # Get all open jobs
        open_jobs = self.jobs[self.jobs['status'] == 'Open']
        if open_jobs.empty:
            return "No open jobs available for recommendation"
        
        # Jobs the applicant has already applied to
        applied_jobs = self.applications[self.applications['applicant_id'] == applicant_id]['job_id'].tolist()
        
        # Filter out jobs already applied to
        open_jobs = open_jobs[~open_jobs['job_id'].isin(applied_jobs)]
        if open_jobs.empty:
            return "You have already applied to all available jobs"
        
        # Calculate match scores for each job
        match_scores = []
        for _, job in open_jobs.iterrows():
            score = self._calculate_match_score_detailed(applicant, job)
            match_scores.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'match_score': score,
                'skills_match': self._skill_match_percentage(applicant_skills, job['skills_required']),
                'experience_match': self._experience_match_percentage(applicant_experience, job['experience_required'])
            })
        
        # Sort by match score and return top N
        match_scores_df = pd.DataFrame(match_scores)
        return match_scores_df.sort_values(by='match_score', ascending=False).head(top_n)
    
    def recommend_candidates(self, job_id, top_n=5):
        """Recommend candidates for a job based on their skills and experience"""
        if not self._job_exists(job_id):
            return f"Error: Job ID {job_id} does not exist"
        
        # Get job details
        job = self.jobs[self.jobs['job_id'] == job_id].iloc[0]
        
        # Calculate match scores for each applicant
        match_scores = []
        for _, applicant in self.applicants.iterrows():
            score = self._calculate_match_score_detailed(applicant, job)
            match_scores.append({
                'applicant_id': applicant['applicant_id'],
                'name': applicant['name'],
                'match_score': score,
                'skills_match': self._skill_match_percentage(applicant['skills'], job['skills_required']),
                'experience_match': self._experience_match_percentage(applicant['experience'], job['experience_required'])
            })
        
        # Sort by match score and return top N
        match_scores_df = pd.DataFrame(match_scores)
        return match_scores_df.sort_values(by='match_score', ascending=False).head(top_n)
    
    def search_jobs(self, keywords, location=None, department=None):
        """Search jobs based on keywords, location, and department"""
        if self.jobs.empty:
            return "No jobs in the database"
        
        # Create a copy of jobs
        result = self.jobs.copy()
        
        # Filter by location if provided
        if location:
            result = result[result['location'].str.contains(location, case=False, na=False)]
        
        # Filter by department if provided
        if department:
            result = result[result['department'].str.contains(department, case=False, na=False)]
        
        # Filter by keywords in title and description
        if keywords:
            keyword_filter = result['title'].str.contains(keywords, case=False, na=False) | \
                            result['description'].str.contains(keywords, case=False, na=False) | \
                            result['skills_required'].str.contains(keywords, case=False, na=False)
            result = result[keyword_filter]
        
        if result.empty:
            return "No jobs match your search criteria"
            
        return result
    
    def _calculate_match_score(self, applicant_id, job_id):
        """Calculate the match score between an applicant and a job (simple version)"""
        applicant = self.applicants[self.applicants['applicant_id'] == applicant_id].iloc[0]
        job = self.jobs[self.jobs['job_id'] == job_id].iloc[0]
        
        return self._calculate_match_score_detailed(applicant, job)
    
    def _calculate_match_score_detailed(self, applicant, job):
        """Calculate a detailed match score between an applicant and a job"""
        # Calculate match for skills (60% of total score)
        skills_match = self._skill_match_percentage(applicant['skills'], job['skills_required'])
        
        # Calculate match for experience (40% of total score)
        experience_match = self._experience_match_percentage(applicant['experience'], job['experience_required'])
        
        # Combine scores (weighted)
        total_score = (skills_match * 0.6) + (experience_match * 0.4)
        
        return round(total_score, 2)
    
    def _skill_match_percentage(self, applicant_skills, job_skills):
        """Calculate the percentage of job skills that match with applicant skills"""
        if not isinstance(applicant_skills, list):
            applicant_skills = applicant_skills.split(',')
        if not isinstance(job_skills, list):
            job_skills = job_skills.split(',')
            
        applicant_skills = [skill.strip().lower() for skill in applicant_skills]
        job_skills = [skill.strip().lower() for skill in job_skills]
        
        # Count matching skills
        matching_skills = sum(1 for skill in job_skills if any(s in skill for s in applicant_skills) or 
                             any(skill in s for s in applicant_skills))
        
        # Calculate percentage match
        if not job_skills:
            return 100  # No skills required for the job
        
        return (matching_skills / len(job_skills)) * 100
    
    def _experience_match_percentage(self, applicant_experience, job_required_experience):
        """Calculate how well the applicant's experience matches the job requirements"""
        try:
            # Extract years of experience
            applicant_exp = float(applicant_experience.split()[0])
            required_exp = float(job_required_experience.split()[0])
            
            if applicant_exp >= required_exp:
                return 100
            else:
                # Partial match if applicant has some experience but less than required
                return (applicant_exp / required_exp) * 100
        except (ValueError, AttributeError):
            # Default to 50% if we can't parse the experience values
            return 50
    
    def _job_exists(self, job_id):
        """Check if a job exists in the database"""
        return job_id in self.jobs['job_id'].values
    
    def _applicant_exists(self, applicant_id):
        """Check if an applicant exists in the database"""
        return applicant_id in self.applicants['applicant_id'].values
    
    def generate_job_report(self, job_id):
        """Generate a report for a specific job"""
        if not self._job_exists(job_id):
            return f"Error: Job ID {job_id} does not exist"
        
        job = self.jobs[self.jobs['job_id'] == job_id].iloc[0]
        applications = self.applications[self.applications['job_id'] == job_id]
        
        report = {
            'job_title': job['title'],
            'job_id': job_id,
            'date_posted': job['date_posted'],
            'total_applications': len(applications),
            'application_status_counts': applications['status'].value_counts().to_dict(),
            'average_match_score': applications['match_score'].mean() if not applications.empty else 0,
            'top_candidates': self.get_job_applicants(job_id, sort_by_match=True).head(5) if not applications.empty else "No applicants yet"
        }
        
        return report
    
    def generate_applicant_report(self, applicant_id):
        """Generate a report for a specific applicant"""
        if not self._applicant_exists(applicant_id):
            return f"Error: Applicant ID {applicant_id} does not exist"
        
        applicant = self.applicants[self.applicants['applicant_id'] == applicant_id].iloc[0]
        applications = self.applications[self.applications['applicant_id'] == applicant_id]
        
        report = {
            'applicant_name': applicant['name'],
            'applicant_id': applicant_id,
            'date_registered': applicant['date_registered'],
            'total_applications': len(applications),
            'application_status_counts': applications['status'].value_counts().to_dict() if not applications.empty else {},
            'average_match_score': applications['match_score'].mean() if not applications.empty else 0,
            'recommended_jobs': self.recommend_jobs(applicant_id, top_n=3)
        }
        
        return report

# Example usage based on the job listings in the image
def populate_sample_data():
    ats = JobATS()
    
    # Add jobs from the image
    ats.add_job(
        job_id="790301291204",
        title="R2R Application Engineer (AI machine learning) || Senior - (E2B)",
        company="Applied Global Services",
        location="Hsinchu, TWN",
        department="Engineering",
        description="The Automation Products Group in Applied Global Services is seeking a next-generation Run-to-Run/Advanced Process Control (R2R/APC) application development engineer who can integrate advanced process control (APC) with artificial intelligence/machine learning (AI/ML) technologies. This role is primarily responsible for defining, designing, developing, and deploying high-performance R2R controllers and AI/ML models, integrating them into customer process environments with a strong focus on Virtual Metrology, Neural Networks, and Data Analytics applications.",
        skills_required=["Technical Presentations", "Industrial Software", "Feature Engineering", "Machine Learning", "Deep Learning", "Statistical Process Control", "Python", "System Integration"],
        experience_required="5-10 years",
        date_posted="60 days ago"
    )
    
    ats.add_job(
        job_id="E3-123456",
        title="Algorithm Developer III - (E3)",
        company="Applied Materials",
        location="Hsinchu, TWN",
        department="Engineering",
        description="Design and develop new R2R controllers and Virtual Metrology modules, leveraging Neural Networks and machine learning models to optimize process control and improve manufacturing quality and yield. Integrate AI/ML algorithms (e.g., Python + TensorFlow / PyTorch / Scikit-learn) with data analytics workflows within Applied Materials' SmartFactory framework.",
        skills_required=["Computer Science", "Linux", "Research", "Algorithms", "Machine Learning", "Python", "Data Processing"],
        experience_required="5+ years",
        date_posted="15 days ago"
    )
    
    ats.add_job(
        job_id="E2A-789012",
        title="MES Software Engineer II - (E2A)",
        company="Applied Materials",
        location="Hsinchu, TWN",
        department="Engineering",
        description="Develop and maintain MES (Manufacturing Execution System) components. Work with databases and integration systems to ensure smooth manufacturing operations.",
        skills_required=["C++", "Git", "SQL", "Programming"],
        experience_required="3-5 years",
        date_posted="15 days ago"
    )
    
    ats.add_job(
        job_id="E2A-345678",
        title="MES UI Software Engineer II - (E2A)",
        company="Applied Materials",
        location="Hsinchu, TWN",
        department="Engineering",
        description="Develop user interfaces for Manufacturing Execution Systems. Create intuitive and efficient UI components for factory floor operators and engineers.",
        skills_required=["SQL", "Programming", "C++", "Git", "Python"],
        experience_required="3-5 years",
        date_posted="15 days ago"
    )
    
    ats.add_job(
        job_id="E3-567890",
        title="Algorithm Developer III (SemVision ADC) - (E3)",
        company="Applied Materials",
        location="Hsinchu, TWN",
        department="Engineering",
        description="Develop algorithms for semiconductor inspection and metrology systems. Implement machine vision and data analysis algorithms for defect detection.",
        skills_required=["Algorithms", "Machine Learning", "Image Processing", "Python", "C++"],
        experience_required="5+ years",
        date_posted="10 days ago"
    )
    
    # Add some sample applicants
    ats.add_applicant(
        applicant_id="APP001",
        name="John Smith",
        email="john.smith@example.com",
        phone="555-123-4567",
        resume_text="Experienced ML engineer with background in semiconductor industry. Proficient in Python, TensorFlow, and statistical analysis.",
        skills=["Python", "Machine Learning", "TensorFlow", "Statistical Process Control", "Feature Engineering", "Neural Networks"],
        experience="6 years"
    )
    
    ats.add_applicant(
        applicant_id="APP002",
        name="Jane Doe",
        email="jane.doe@example.com",
        phone="555-987-6543",
        resume_text="Software engineer specializing in MES systems. Strong SQL and database skills with C++ and Python programming experience.",
        skills=["SQL", "C++", "Python", "Git", "Database Design", "MES"],
        experience="4 years"
    )
    
    ats.add_applicant(
        applicant_id="APP003",
        name="Robert Chen",
        email="robert.chen@example.com",
        phone="555-456-7890",
        resume_text="Algorithm developer with PhD in Computer Science. Expertise in machine vision, image processing, and deep learning applications.",
        skills=["Algorithms", "Machine Learning", "Image Processing", "Python", "C++", "Research", "Deep Learning"],
        experience="7 years"
    )
    
    # Submit some applications
    ats.apply_to_job("APP001", "790301291204")
    ats.apply_to_job("APP001", "E3-123456")
    ats.apply_to_job("APP002", "E2A-789012")
    ats.apply_to_job("APP002", "E2A-345678")
    ats.apply_to_job("APP003", "E3-567890")
    
    return ats

# Example of running the system
if __name__ == "__main__":
    ats = populate_sample_data()
    
    # Show job recommendations for an applicant
    print("\nJob recommendations for Wang:")
    print(ats.recommend_jobs("APP003"))
    
    # Show candidate recommendations for a job
    print("\nCandidate recommendations for R2R Application Engineer position:")
    print(ats.recommend_candidates("790301291204"))
    
    # Generate job report
    print("\nJob Report for Algorithm Developer III:")
    print(ats.generate_job_report("E3-123456"))
    
    # Search for jobs with specific keywords
    print("\nSearch results for 'Machine Learning' jobs:")
    print(ats.search_jobs("Machine Learning"))