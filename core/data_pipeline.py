import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
import os

print("1. Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("2. Setting up ChromaDB Persistent Storage...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="job_postings")

print("3. Reading and Cleaning Data...")
df = pd.read_excel('data/raw_jobs.xlsx')

df = df.rename(columns={
    'title': 'job_title',
    'tagsAndSkills': 'skills',
    'jobDescription': 'job_description'
})

columns_to_keep = ['job_title', 'skills', 'job_description']
df = df[columns_to_keep].dropna()

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

df['cleaned_description'] = df['job_description'].apply(clean_text)
df['cleaned_skills'] = df['skills'].apply(clean_text)
# We combine these so the AI can read everything as one solid block of context
df['combined_text'] = df['job_title'] + " " + df['cleaned_skills'] + " " + df['cleaned_description']


df = df.head(1000)

print("4. Generating Vectors (This might take a minute)...")
documents = df['combined_text'].tolist()
ids = [f"job_{i}" for i in range(len(df))]


metadatas = [{"job_title": title, "job_description": desc} for title, desc in zip(df['job_title'], df['job_description'])]

embeddings = model.encode(documents).tolist()

print("5. Saving to Database...")
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("ETL Pipeline Success! Jobs are now permanently stored in ChromaDB.")