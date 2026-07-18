import os
from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from core.agent_workflow import run_agent_pipeline

load_dotenv()

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="job_postings")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_text = request.form.get('resume_text', '')
        
        if not resume_text.strip():
            return render_template('index.html')
        
        resume_vector = model.encode([resume_text]).tolist()
        
        results = collection.query(
            query_embeddings=resume_vector,
            n_results=3
        )
        
        jobs = []
        if results['distances'] and len(results['distances']) > 0:
            for i in range(len(results['ids'][0])):
                raw_distance = results['distances'][0][i]
                match_score = round(max(0.0, (1.0 - raw_distance)) * 100, 2)
                
                jobs.append({
                    'job_title': results['metadatas'][0][i]['job_title'],
                    'job_description': results['metadatas'][0][i]['job_description'],
                    'match_score': match_score
                })
        
        agent_analysis = ""
        if jobs:
            top_job = jobs[0]['job_description']
            agent_analysis = run_agent_pipeline(resume_text, top_job)
            
        return render_template('index.html', jobs=jobs, agent_analysis=agent_analysis)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)