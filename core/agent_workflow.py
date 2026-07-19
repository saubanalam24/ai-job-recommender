import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.environ.get("GROQ_API_KEY")
)

recruiter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical recruiter analyzing a candidate for a specific job role. Identify clear skill gaps, missing technologies, and experience mismatches between the resume and the job description."),
    ("user", "Candidate Resume:\n{resume}\n\nTarget Job Description:\n{job_desc}")
])

coach_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strategic career coach. Take the provided skill gap analysis and create a highly customized, step-by-step learning roadmap for the candidate to bridge these gaps quickly."),
    ("user", "Skill Gap Analysis:\n{gap_analysis}")
])

recruiter_chain = recruiter_prompt | llm
coach_chain = coach_prompt | llm

def run_agent_pipeline(resume_text, target_job_text):
    print("running recruiter agent...")
    recruiter_response = recruiter_chain.invoke({
        "resume": resume_text,
        "job_desc": target_job_text
    })
    gap_analysis = recruiter_response.content
    
    print("running coach agent...")
    coach_response = coach_chain.invoke({
        "gap_analysis": gap_analysis
    })
    roadmap = coach_response.content
    
    return gap_analysis, roadmap

if __name__ == "__main__":
    sample_resume = "Third-year BS student at IIT Jodhpur. Skills in C++, Python, Flask, Machine Learning."
    sample_job = "Looking for an AI Intern with experience in PyTorch, Large Language Models, and Recommendation Systems."
    
    gaps, plan = run_agent_pipeline(sample_resume, sample_job)
    print("\n=== gaps ===\n", gaps)
    print("\n=== plan ===\n", plan)