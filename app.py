from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from core.agent_workflow import run_agent_pipeline

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('data/cleaned_jobs.csv').head(5000)
job_texts = df['combined_text'].tolist()
job_embeddings = model.encode(job_texts, convert_to_tensor=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    data = request.json
    resume_text = data.get('resume', '')
    
    if not resume_text:
        return jsonify({"error": "No resume text provided"}), 400
        
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    scores = util.cos_sim(resume_emb, job_embeddings)[0]
    top_k = torch.topk(scores, k=5)
    
    matches = []
    for score, idx in zip(top_k[0], top_k[1]):
        job_idx = idx.item()
        matches.append({
            "title": df.iloc[job_idx]['job_title'],
            "description": df.iloc[job_idx]['cleaned_description'],
            "match_pct": round(score.item() * 100, 2)
        })
    
    best_job_desc = matches[0]['description']
    gaps, roadmap = run_agent_pipeline(resume_text, best_job_desc)
    
    return jsonify({
        "top_matches": matches,
        "gap_analysis": gaps,
        "roadmap": roadmap
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)