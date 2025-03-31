from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import datetime
import os
from markupsafe import Markup, escape
from flask import send_from_directory
import sys
sys.path.append(os.getcwd()+f"{os.sep}..")
from preprocess.job_parser import extract_skills_from_resume
from Levenshtein import distance

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.jinja_env.add_extension('jinja2.ext.do')
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
clearall = False
# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    skills = db.relationship('UserSkill', backref='user', lazy=True)
    applications = db.relationship('Application', backref='applicant', lazy=True)

class UserSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)

class Skill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    users = db.relationship('UserSkill', backref='skill', lazy=True)
    jobs = db.relationship('JobSkill', backref='skill', lazy=True)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200))
    description = db.Column(db.Text)
    department = db.Column(db.String(100))
    job_id = db.Column(db.String(100), unique=True)
    is_full_time = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    skills = db.relationship('JobSkill', backref='job', lazy=True)
    applications = db.relationship('Application', backref='job', lazy=True)

class JobSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    applied_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    match_percentage = db.Column(db.Float, default=0)
    status = db.Column(db.String(50), default="Applied")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    jobs = Job.query.order_by(Job.created_at.desc()).limit(20).all()

    return render_template('index.html', jobs=jobs, now=datetime.datetime.now())

@app.template_filter('nl2br')
def nl2br_filter(s):
    if s is None:
        return ''
    # Escape HTML to avoid injection, then replace newlines with <br>
    return Markup(escape(s).replace('\n', '<br>'))

@app.route('/job/<int:job_id>')
def job_details(job_id):
    job = Job.query.get_or_404(job_id)
    skills = set()
    for js in job.skills:
        skills.add(js.skill)

    skills = list(skills)
    match_percentage = 0
    if current_user.is_authenticated:
        user_skill_ids = [us.skill_id for us in current_user.skills]
        job_skill_ids = [js.skill_id for js in job.skills]
        job_skill_ids = set(job_skill_ids)
        if job_skill_ids:  # Avoid division by zero
            match_percentage = len(set(user_skill_ids).intersection(job_skill_ids)) / len(job_skill_ids) * 100
    return render_template('job_details.html', job=job, skills=skills, match_percentage=match_percentage)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Username already exists')
        elif email_exists:
            flash('Email already exists')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('index'))
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/resumes/<filename>')
@login_required  # Optional: protect access
def serve_resume(filename):
    resume_folder = os.path.join(app.root_path, 'resumes')
    print("Serving from:", resume_folder)
    return send_from_directory(resume_folder, filename)

@app.route('/profile')
@login_required
def profile():

    def match(pattern, user_skills):
        for skill in user_skills:
            d = distance(pattern, skill)
            score = max(0, 1-d/max(len(pattern), len(skill))) 
            if score >= 0.7:
                return True
        return False
    resume_url = url_for('serve_resume', filename=f'{current_user.id}_resume.pdf') if os.path.exists(f'resumes/{current_user.id}_resume.pdf') else None
    resume_path = os.path.join(app.root_path, f'resumes/{current_user.id}_resume.pdf')
    resume_ats = extract_skills_from_resume(resume_path)
    skills = []
    matched_skills = []
    skills = Skill.query.all()

    if len(skills) > 0 and resume_ats is not None:
        matched_skills = [skill for skill in skills if match(skill.name.lower(), resume_ats["skills"])]
    else:
        matched_skills = []
    # Update user's skills
    UserSkill.query.filter_by(user_id=current_user.id).delete()
    for skill_id in matched_skills:
        skill_id = int(skill_id.id)
        user_skill = UserSkill(user_id=current_user.id, skill_id=skill_id)
        db.session.add(user_skill)
    db.session.commit()
    user_skill_ids = [us.skill_id for us in current_user.skills]
    user_skills = [s for s in skills if s.id in user_skill_ids]
    applications = Application.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', user=current_user, user_skills=user_skills, skills=skills, applications=applications,  resume_url=resume_url)

@app.route('/apply/<int:job_id>', methods=['POST'])
@login_required
def apply_job(job_id):
    job = Job.query.get_or_404(job_id)
    
    # Check if already applied
    existing_application = Application.query.filter_by(user_id=current_user.id, job_id=job.id).first()
    if existing_application:
        flash('You have already applied for this job')
        return redirect(url_for('job_details', job_id=job.id))
    
    # Calculate match percentage
    user_skill_ids = [us.skill_id for us in current_user.skills]
    job_skill_ids = [js.skill_id for js in job.skills]
    
    match_percentage = 0
    if job_skill_ids:  # Avoid division by zero
        match_percentage = len(set(user_skill_ids).intersection(set(job_skill_ids))) / len(job_skill_ids) * 100
    
    # Create application
    application = Application(
        user_id=current_user.id,
        job_id=job.id,
        match_percentage=match_percentage
    )
    db.session.add(application)
    db.session.commit()
    
    flash('Application submitted successfully')
    return redirect(url_for('job_details', job_id=job.id))

@app.route('/search')
def search():
    query = request.args.get('q', '')
    location = request.args.get('location', '')
    
    jobs = Job.query
    
    if query:
        jobs = jobs.filter(Job.title.contains(query) | Job.description.contains(query))
    
    if location:
        jobs = jobs.filter(Job.location.contains(location))
    
    jobs = jobs.order_by(Job.created_at.desc()).all()
    
    return render_template('search_results.html', jobs=jobs, query=query, location=location)

@app.route('/admin')
@login_required
def admin():
    # Simple admin check - in a real app, use proper role-based access control
    if current_user.username != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    jobs = Job.query.all()
    return render_template('admin.html', jobs=jobs)


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files.get('resume')
    if file and file.filename.endswith('.pdf'):
        filename = f"{current_user.id}_resume.pdf"
        if not os.path.exists('resumes'):
            os.makedirs('resumes')
        save_path = os.path.join('resumes', filename)
        file.save(save_path)
        flash("Resume uploaded successfully.")
    else:
        flash("Only PDF files are allowed.")
    return redirect(url_for('profile'))


@app.route('/admin/add_job', methods=['GET', 'POST'])
@login_required
def add_job():
    if current_user.username != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        company = request.form.get('company')
        location = request.form.get('location')
        department = request.form.get('department')
        description = request.form.get('description')
        job_id = request.form.get('job_id')
        is_full_time = 'is_full_time' in request.form
        
        job = Job(
            title=title,
            company=company,
            location=location,
            department=department,
            description=description,
            job_id=job_id,
            is_full_time=is_full_time
        )
        db.session.add(job)
        db.session.commit()
        
        # Add skills to job
        skill_ids = request.form.getlist('skills')
        for skill_id in skill_ids:
            job_skill = JobSkill(job_id=job.id, skill_id=int(skill_id))
            db.session.add(job_skill)
        db.session.commit()
        
        flash('Job added successfully')
        return redirect(url_for('admin'))
    
    skills = Skill.query.all()
    return render_template('add_job.html', skills=skills)

# Create database tables
with app.app_context():
    db.create_all()
    db.session.commit()
    if clearall:
        Job.query.delete()
        JobSkill.query.delete()
        Skill.query.delete()
        User.query.delete()
        UserSkill.query.delete()
        Application.query.delete()
    db.session.commit()
    # Add initial skills if they don't exist
    initial_skills = [
        "Technical Presentations", "Industrial Software", "Feature Engineering",
        "Machine Learning", "Python", "Research", "Algorithms", "Image Processing",
        "Analysis", "Deep Learning", "C++", "Computer Science", "Linux",
        "Data Analysis", "Statistical Process Control", "System Integration"
    ]
    
    for skill_name in initial_skills:
        if not Skill.query.filter_by(name=skill_name).first():
            skill = Skill(name=skill_name)
            db.session.add(skill)
        
    
    # Add admin user if it doesn't exist
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            password=generate_password_hash('admin123', method='pbkdf2:sha256')
        )
        db.session.add(admin)
    
    # Add sample jobs if none exist
    if Job.query.count() == 0:
        sample_jobs = [
            {
                'title': 'R2R Application Engineer (AI machine learning) II Senior - (E2B)',
                'company': 'Apple',
                'location': 'Hsinchu,TWN',
                'department': 'Engineering',
                'job_id': '79630129424',
                'description': """The Automation Products Group in Applied Global Services is seeking a next-generation Run-to-Run/Advanced Process Control (R2R/APC) application development engineer who can integrate advanced process control (APC) with artificial intelligence/machine learning (AI/ML) technologies. This role is primarily responsible for defining, designing, developing, testing, and deploying high-performance R2R controllers and AI/ML models, integrating them into customer process environments with a strong focus on Virtual Metrology, Neural Networks, and Data Analytics applications. You will collaborate closely with R2R solution architects and project managers to deliver turnkey solutions while continuously optimizing and exploring new technologies.

Additional responsibilities include involvement in presales and deployment activities. These responsibilities are as follows:

• R2R Controller & AI/ML Model Design and Development
○ Design and develop new R2R controllers and Virtual Metrology modules, leveraging Neural Networks and machine learning models to optimize process control and improve manufacturing quality and yield.
○ Integrate AI/ML algorithms (e.g., Python + TensorFlow / PyTorch / Scikit-learn) with data analytics workflows within Applied Materials' SmartFactory framework or other advanced infrastructures.

• Virtual Metrology and Advanced Data Analytics
○ Apply AI/ML models to process and sensor data for Virtual Metrology, quality prediction, yield improvement, and root cause analysis.
○ Utilize advanced data analytics methods, such as multivariate analysis, Statistical Process Control (SPC), and Fault Detection & Classification (FDC), to help customers optimize their manufacturing processes.""",
                'is_full_time': True,
                'skills': ['Technical Presentations', 'Industrial Software', 'Feature Engineering', 'Machine Learning']
            },
            {
                'title': 'Algorithm Developer (SemVision)',
                'company': 'Google',
                'location': 'Hsinchu,TWN',
                'department': 'Engineering',
                'job_id': '79630129425',
                'description': 'Algorithm Developer position for semiconductor equipment...',
                'is_full_time': True,
                'skills': ['Python', 'Research', 'Algorithms', 'Machine Learning', 'Image Processing', 'Analysis', 'Deep Learning', 'C++']
            },
            {
                'title': 'Algorithm Developer III - (E3)',
                'company': 'Meta',
                'location': 'Hsinchu,TWN',
                'department': 'Engineering',
                'job_id': '79630129426',
                'description': 'Senior Algorithm Developer position...',
                'is_full_time': True,
                'skills': ['Computer Science', 'Linux', 'Research', 'Algorithms', 'Machine Learning', 'Image Processing', 'Data Analysis']
            },
            {
                'title': 'MES Software Engineer II - (E2A)',
                'company': 'TSMC',
                'location': 'Hsinchu,TWN',
                'department': 'Engineering',
                'job_id': '79630129427',
                'description': 'Manufacturing Execution System engineer...',
                'is_full_time': True,
                'skills': ['System Integration', 'Industrial Software', 'Python']
            }
        ]
        
        for job_data in sample_jobs:
            job = Job(
                title=job_data['title'],
                company=job_data['company'],
                location=job_data['location'],
                department=job_data['department'],
                job_id=job_data['job_id'],
                description=job_data['description'],
                is_full_time=job_data['is_full_time']
            )
            db.session.add(job)
            db.session.commit()
            
            # Add skills to job
            for skill_name in job_data['skills']:
                skill = Skill.query.filter_by(name=skill_name).first()
                if skill:
                    print("add jobskill:", skill)
                    job_skill = JobSkill(job_id=job.id, skill_id=skill.id)
                    db.session.add(job_skill)
        
    db.session.commit()

if __name__ == '__main__':

    app.run(debug=True)