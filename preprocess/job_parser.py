import PyPDF2
import spacy
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string
from Levenshtein import distance

class JobParser:
    """
    ATS, key words extraction.
    Tricks:
    1. Lemmatizer: en_core_web_sm
    2. Rule Based: stop words
    3. Keywords
    """
    def __init__(self):
        # Download necessary NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        # Load NLP model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Prepare skill detection
        self.skill_patterns = self._load_skill_patterns()
        
        # Common resume sections
        self.resume_sections = [
            'education', 'experience', 'work experience', 
            'skills', 'technical skills', 'projects', 'certifications',
            'achievements', 'publications', 'languages'
        ]
        
    def _load_skill_patterns(self):
        """Load common technical and soft skills for pattern matching"""
        # Technical skills
        technical_skills = [
            # Programming languages
            'python', 'java', 'javascript', 'c\\+\\+', 'c#', 'ruby', 'go', 'r', 'php', 'swift',
            'typescript', 'kotlin', 'rust', 'scala', 'perl', 'sql', 'bash', 'powershell', 'matlab',
            # Web technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'asp.net', 'express', 'jquery', 'bootstrap', 'laravel', 'rails',
            # Data science/ML
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'scipy', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
            'artificial intelligence', 'data mining', 'big data', 'data visualization',
            # Cloud/DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform',
            'ansible', 'puppet', 'chef', 'devops', 'sre', 'cloud',
            # Databases
            'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'sqlite', 'redis', 
            'cassandra', 'elasticsearch', 'dynamodb', 'nosql', 'rdbms',
            # Engineering/Manufacturing specific skills from job listings
            'r2r', 'apc', 'advanced process control', 'virtual metrology', 'neural networks',
            'semiconductor', 'manufacturing', 'statistical process control', 'spc', 'mes',
            'fault detection', 'fdc', 'equipment integration', 'metrology', 'yield improvement',
            'image processing', 'multivariate analysis', 'smartfactory', 'dispatching',
            # Tools and frameworks
            'git', 'svn', 'jira', 'confluence', 'slack', 'trello', 'agile', 'scrum',
            'kanban', 'waterfall', 'figma', 'adobe', 'photoshop', 'illustrator',
        ]

        # Soft skills
        soft_skills = [
            'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
            'decision making', 'time management', 'adaptability', 'flexibility', 'creativity',
            'organization', 'stress management', 'conflict resolution', 'emotional intelligence',
            'negotiation', 'persuasion', 'presentation', 'public speaking', 'writing',
            'collaboration', 'interpersonal', 'customer service', 'project management',
            'mentoring', 'coaching', 'analytical', 'attention to detail', 'multitasking',
            'research', 'planning', 'strategic thinking', 'innovation', 'self-motivation',
            'work ethic', 'professionalism', 'confidentiality', 'integrity', 'ethics'
        ]
        
        return technical_skills + soft_skills
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        return text
    
    def extract_skills(self, text):
        """Extract skills from text using pattern matching and NLP"""
        skills = []
        
        # Pattern matching for known skills
        for skill in self.skill_patterns:
            pattern = r'\b' + skill + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(skill)
        
        # NLP-based extraction for additional skills
        doc = self.nlp(text)
        
        # Extract noun phrases as potential skills
        for chunk in doc.noun_chunks:
            # Filter out common non-skill noun phrases
            if (len(chunk.text.split()) <= 3 and  # Limit to 3 words max
                chunk.text.lower() not in stopwords.words('english') and
                not any(char in string.punctuation for char in chunk.text)):
                for pattern in self.skill_patterns:
                    d = distance(pattern, chunk.text.lower())
                    score = max(0, 1-d/max(len(pattern), len(chunk.text))) 
                    if score > 0.6:
                        skills.append(chunk.text.lower())
                        break
        # Remove duplicates while preserving order
        unique_skills = []
        for skill in skills:
            if skill not in unique_skills:
                unique_skills.append(skill)
        
        return unique_skills
    
    def extract_keywords(self, text, top_n=20):
        """Extract the most important keywords from the resume"""
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords relevant to resumes
        custom_stopwords = [
            'location', 'resume', 'curriculum', 'vitae', 'cv', 'name', 'email', 'phone', 'address',
            'linkedin', 'github', 'website', 'objective', 'summary', 'profile', 'contact',
            'references', 'available', 'upon', 'request', 'page', 'january', 'february', 
            'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 
            'november', 'december'
        ]
        stop_words.update(custom_stopwords)
        
        # Create a CountVectorizer to extract single, bi-gram and tri-gram terms
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words=list(stop_words),
            ngram_range=(1, 3),
            max_features=top_n
        ) # BoW
        
        # Apply the vectorizer
        try:
            X = vectorizer.fit_transform([text])
            
            # Get feature names and their counts
            count_values = X.toarray()[0]
            feature_names = vectorizer.get_feature_names_out()
            
            # Create a DataFrame with terms and their counts
            keywords_df = pd.DataFrame({
                'keyword': feature_names,
                'count': count_values
            })
            
            # Sort by count (frequency) in descending order
            keywords_df = keywords_df.sort_values(by='count', ascending=False)
            
            return keywords_df.head(top_n)
        except ValueError:
            # Handle case where no features are extracted
            return pd.DataFrame(columns=['keyword', 'count'])
    
    def extract_sections(self, text):
        """Attempt to extract different sections of the resume"""
        sections = {}
        
        # Identify potential section headers
        lines = text.split('\n')
        current_section = None
        section_content = []
        title_line = None
        for i, line in enumerate(lines):
  
            line_lower = line.lower().strip()
            
            # Check if this line might be a section header
            if (line_lower in self.resume_sections or
                any(section in line_lower for section in self.resume_sections)):
                
                # Save the previous section if it exists
                if current_section and section_content:
                    sections[current_section] = ((title_line, i),'\n'.join(section_content))
                
                # Start a new section
                current_section = line.strip()
                title_line = i
                section_content = []
            else:
                # Add to current section content
                if current_section:
                    section_content.append(line)
        
        # Add the last section
        if current_section and section_content:
            sections[current_section] = ((title_line, len(lines)),'\n'.join(section_content))
        
        return sections
    
    def get_education(self, text, interval=None):
        """Extract education information"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'bs', 'ms', 'ba', 'ma', 'mba',
            'associate', 'degree', 'university', 'college', 'institute', 'school',
            'gpa', 'major', 'minor', 'concentration', 'graduated', 'program'
        ]
        
        education_info = []  
        
        # If we found a dedicated education section, use that
        if interval is not None:
            # Process education section
            lines = text.splitlines()
            lines = lines[interval[0]:interval[1]]
            current_entry = []
            
            for line in lines:
                if line.strip():  # If not empty line
                    current_entry.append(line.strip())
                elif current_entry:  # Empty line and we have content
                    education_info.append(' '.join(current_entry))
                    current_entry = []
            
            # Add final entry if it exists
            if current_entry:
                education_info.append(' '.join(current_entry))
        else:
            # Try to find education information throughout the document
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in education_keywords):
                    education_info.append(sentence)
        
        return education_info
    
    def get_experience_years_jd(self, text):
        """Estimate years of experience from job description"""
        # Look for patterns like "X years of experience" or "X+ years"
        experience_patterns = [
            r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)\+?\s+years?',
            r'(\d+)\+?\s+years?\s+(?:in\s+)(?:the\s+)?(?:field|industry)'
        ]
        
        max_years = 0
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    # Not a valid integer
                    pass
        
        # If we couldn't find years directly, try to estimate from job history
        if max_years == 0:
            # Extract date ranges from text
            date_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+(\d{4})[\s,.-]+(?:to|-)[\s,.-]+(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+(\d{4})|present|current)'
            date_matches = re.findall(date_pattern, text.lower())
            
            if date_matches:
                # Calculate years from each job and sum them up
                total_years = 0
                current_year = 2025  # Using current year
                
                for date_range in date_matches:
                    start_year = int(date_range[0])
                    if date_range[1]:  # If there's an end year
                        end_year = int(date_range[1])
                    else:  # If "present" or "current"
                        end_year = current_year
                    
                    total_years += (end_year - start_year)
                
                max_years = total_years
        
        return max_years
    
    def get_experience_years_resume(self, text, interval=None):
        from dateutil import parser
        from datetime import datetime
        lines = text.splitlines()
        if interval is not None:
            lines = lines[interval[0]:interval[1]]
        results = []

        # Regex for MM/YYYY - MM/YYYY or present
        date_pattern = re.compile(r'(?P<start>\d{1,2}/\d{4})\s*[–\-to]+\s*(?P<end>\d{1,2}/\d{4}|present|now)', re.IGNORECASE)
        total_months = 0
        for line in lines:
            match = date_pattern.search(line)
            if match:
                start_str, end_str = match.group("start"), match.group("end")
                try:
                    start = parser.parse(start_str, fuzzy=True)
                    end = datetime.now() if end_str.lower() in ['present', 'now'] else parser.parse(end_str, fuzzy=True)
                    months = (end.year - start.year) * 12 + (end.month - start.month)
                    total_months += max(months, 0)
                    title = line[:match.start()].strip()
                    results.append({
                        "title": title,
                        "start": start.date(),
                        "end": end.date(),
                        "period": months,
                    })
                except Exception as e:
                    print(f"⚠️ Error parsing dates in line: {line}\n{e}")
        total_years = round(total_months / 12, 2)
        return {"YOE":total_years, "Titles":results}
        
    def parse_info(self, text=None, pdf_path=None, pdf=True):
        """Parse information and extract all information"""
        if pdf:
        # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return {"error": "Could not extract text from PDF"}
        else:
            assert text is not None
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract skills
        skills = self.extract_skills(processed_text)
        
        # Extract keywords
        keywords_df = self.extract_keywords(processed_text)
        keywords = keywords_df['keyword'].tolist()

        # Extract sections
        sections = self.extract_sections(text)
        
        exp_interval = None
        edu_interval = None
        for section_k, section_v in sections.items():

            section_k = section_k.lower()
            if "experience" in section_k:
                exp_interval = [section_v[0][0], section_v[0][1]]
            if "education" in section_k:
                edu_interval = [section_v[0][0], section_v[0][1]]

      
        # Extract education
        if pdf:
            education = self.get_education(text, edu_interval)
        
        # Estimate years of experience
        if pdf:
            experience_years = self.get_experience_years_resume(text, exp_interval)
        else:
            experience_years = self.get_experience_years_jd(processed_text)
        
       
        
        # Get the most likely job titles based on the content
        if pdf:
            potential_job_titles = self.extract_job_titles(text)
        
        # Compile the results
        if pdf:
            result = {
                "skills": skills,
                "keywords": keywords,
                "education": education,
                "experience_years": experience_years,
                "sections": list(sections.keys()),
                "potential_job_titles": potential_job_titles,
                "full_text": text  # Include the original text for reference
            }
        else:
            result = {
                "skills": skills,
                "keywords": keywords,
                "experience_years": experience_years,
                "sections": list(sections.keys()),
                "full_text": text
            }
        
        return result
    
    def extract_job_titles(self, text):
        """Extract potential job titles from resume"""
        # Common job title prefixes
        job_prefixes = [
            "senior", "lead", "principal", "staff", "chief", "head", 
            "junior", "associate", "assistant", "director"
        ]
        
        # Common job domains in tech/engineering
        job_domains = [
            "engineer", "developer", "architect", "analyst", "scientist", 
            "manager", "administrator", "specialist", "consultant", "technician"
        ]
        
        # Common job specialties
        job_specialties = [
            "software", "hardware", "systems", "application", "data", "network",
            "ai", "ml", "machine learning", "frontend", "backend", "full stack",
            "devops", "qa", "test", "research", "security", "cloud", "database",
            "web", "mobile", "ios", "android", "embedded", "process", "product",
            "r2r", "apc", "algorithm", "metrology"
        ]
        
        # Try to find job titles in the text
        job_titles = []
        
        # Using pattern matching
        for prefix in [""] + job_prefixes:
            for domain in job_domains:
                for specialty in [""] + job_specialties:
                    if prefix and specialty:
                        pattern = fr'\b{prefix}\s+{specialty}\s+{domain}\b'
                    elif prefix:
                        pattern = fr'\b{prefix}\s+{domain}\b'
                    elif specialty:
                        pattern = fr'\b{specialty}\s+{domain}\b'
                    else:
                        pattern = fr'\b{domain}\b'
                    
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        job_titles.append(match)
        
        # Use NLP to extract job titles
        doc = self.nlp(text)
        
        # Look for job title patterns in the first few paragraphs
        first_part = ' '.join(text.split('\n')[:10])
        job_doc = self.nlp(first_part)
        
        for ent in job_doc.ents:
            if ent.label_ == "PERSON":
                # Check if there's a job title near this entity
                for token in job_doc:
                    if (token.i > ent.end and 
                        any(domain in token.text.lower() for domain in job_domains)):
                        job_context = job_doc[ent.end:token.i+5].text
                        if job_context not in job_titles:
                            job_titles.append(job_context)
        
        # Remove duplicates and sort by length (shorter titles are typically more accurate)
        unique_job_titles = []
        for title in sorted(job_titles, key=len):
            title = title.strip()
            # Check if this title is not a subset of already found titles
            if title and not any(title in t for t in unique_job_titles):
                unique_job_titles.append(title)
        
        return unique_job_titles[:5]  # Return top 5 potential job titles


def extrack_keywords_from_jd(jd):
    parser = JobParser()
    results = parser.parse_info(text=jd, pdf=False)
    return results

# Example usage
def extract_skills_from_resume(pdf_path):
    """Wrapper function to extract skills and keywords from a resume"""
    parser = JobParser()
    results = parser.parse_info(pdf_path=pdf_path, pdf=True)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return None
    
    print(f"Found {len(results['skills'])} skills:")
    for skill in results['skills']:
        print(f"- {skill}")
    
    print(f"\nTop keywords:")
    for keyword in results['keywords'][:10]:
        print(f"- {keyword}")
    
    print(f"\nEstimated experience: {results['experience_years']} years")
    
    print(f"\nPotential job titles:")
    for title in results['potential_job_titles']:
        print(f"- {title}")
    
    return results

# Integration with JobATS system
def add_applicant_from_resume(ats, pdf_path, applicant_name, email, phone):
    """Add an applicant to the ATS system from resume PDF"""
    parser = JobParser()
    resume_data = parser.parse_resume(pdf_path)
    
    if "error" in resume_data:
        return f"Error processing resume: {resume_data['error']}"
    
    # Generate a unique applicant ID
    applicant_id = f"APP{len(ats.applicants) + 1:03d}"
    
    # Add the applicant to the ATS
    ats.add_applicant(
        applicant_id=applicant_id,
        name=applicant_name,
        email=email,
        phone=phone,
        resume_text=resume_data["full_text"],
        skills=resume_data["skills"],
        experience=f"{resume_data['experience_years']} years"
    )
    
    # Generate job recommendations for the new applicant
    recommendations = ats.recommend_jobs(applicant_id)
    
    return {
        "applicant_id": applicant_id,
        "extracted_skills": resume_data["skills"],
        "experience_years": resume_data["experience_years"],
        "potential_job_titles": resume_data["potential_job_titles"],
        "job_recommendations": recommendations
    }

# Example of using the resume parser with the JobATS system
if __name__ == "__main__":
    # Initialize the ATS system with sample data
    # import sys
    # import os
    # sys.path.append(os.getcwd()+os.sep+"..")
    from jd_preprocess import populate_sample_data
    ats = populate_sample_data()
    
    # Parse a resume and add to the system
    pdf_path = "example_resume.pdf"
    result = add_applicant_from_resume(
        ats=ats,
        pdf_path=pdf_path,
        applicant_name="Alex Johnson",
        email="alex.johnson@example.com",
        phone="555-789-0123"
    )
    
    print("\nNew applicant added to the system:")
    print(f"Applicant ID: {result['applicant_id']}")
    print(f"Extracted skills: {', '.join(result['extracted_skills'][:10])}...")
    print(f"Experience: {result['experience_years']} years")
    print("\nRecommended jobs:")
    print(result['job_recommendations'])