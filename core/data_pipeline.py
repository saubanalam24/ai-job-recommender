import pandas as pd
import re

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

df['combined_text'] = df['job_title'] + " " + df['cleaned_skills'] + " " + df['cleaned_description']

df.head(5000).to_csv('data/cleaned_jobs.csv', index=False)
print("Data pipeline finished successfully!")