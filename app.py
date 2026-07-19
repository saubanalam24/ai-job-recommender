import os
from flask import Flask, render_template, request, redirect, url_for
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from core.agent_workflow import run_agent_pipeline
from core.models import db, User, History
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from PyPDF2 import PdfReader

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super_secret_dev_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()
print("Loading AI Model and Vector Database... please wait a few seconds...")
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="job_postings")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
def index():
    total_jobs_in_db = collection.count()
    
    if request.method == 'POST':
        resume_text = request.form.get('resume_text', '')
        resume_file = request.files.get('resume_file')
        
        if resume_file and resume_file.filename.endswith('.pdf'):
            reader = PdfReader(resume_file)
            extracted_text = ""
            for page in reader.pages:
                if page.extract_text():
                    extracted_text += page.extract_text() + "\n"
            if extracted_text.strip():
                resume_text = extracted_text
        
        if not resume_text.strip():
            return render_template('index.html', total_jobs=total_jobs_in_db)
        
        resume_vector = model.encode([resume_text]).tolist()
        
        results = collection.query(
            query_embeddings=resume_vector,
            n_results=20
        )
        
        jobs = []
        seen_titles = set()
        
        if results['distances'] and len(results['distances']) > 0:
            for i in range(len(results['ids'][0])):
                raw_distance = results['distances'][0][i]
                
                similarity = 1.0 - (raw_distance / 2.0)
                match_score = round(max(0.0, similarity) * 100, 2)
                
                if match_score >= 30.0:
                    job_title = results['metadatas'][0][i]['job_title']
                    
                    if job_title not in seen_titles:
                        jobs.append({
                            'job_title': job_title,
                            'job_description': results['metadatas'][0][i]['job_description'],
                            'match_score': match_score
                        })
                        seen_titles.add(job_title)
        
        agent_analysis = ""
        if jobs:
            top_job = jobs[0]['job_description']
            raw_analysis = run_agent_pipeline(resume_text, top_job)
            
            if isinstance(raw_analysis, tuple):
                agent_analysis = str(raw_analysis[0])
            else:
                agent_analysis = str(raw_analysis)
            
            if current_user.is_authenticated:
                new_history = History(
                    user_id=current_user.id,
                    resume_text=resume_text,
                    agent_analysis=agent_analysis
                )
                db.session.add(new_history)
                db.session.commit()
            
        return render_template('index.html', jobs=jobs, agent_analysis=agent_analysis, total_jobs=total_jobs_in_db, search_completed=True)
        
    return render_template('index.html', total_jobs=total_jobs_in_db)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            error = "Passwords do not match."
            return render_template('register.html', error=error)
            
        if len(password) < 8:
            error = "Password must be at least 8 characters long."
            return render_template('register.html', error=error)
            
        if User.query.filter_by(username=username).first():
            error = "Username already exists."
            return render_template('register.html', error=error)
            
        if User.query.filter_by(email=email).first():
            error = "Email already registered."
            return render_template('register.html', error=error)
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        login_input = request.form.get('login_input')
        password = request.form.get('password')
        
        user = User.query.filter((User.username == login_input) | (User.email == login_input)).first()
        
        if not user:
            error = "User not registered."
        elif not bcrypt.check_password_hash(user.password, password):
            error = "Incorrect password."
        else:
            login_user(user)
            return redirect(url_for('dashboard'))
            
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    histories = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('dashboard.html', histories=histories)

if __name__ == '__main__':
    app.run(debug=True)