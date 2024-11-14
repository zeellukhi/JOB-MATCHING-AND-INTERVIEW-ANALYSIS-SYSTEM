# -------------------------------------------------------------------------
# ----------------1. Imports and Environment Setup-------------------------
# -------------------------------------------------------------------------

import os
import json
import re
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from docx import Document
import fitz  # PyMuPDF for PDF extraction
import google.generativeai as genai  # For generative AI model


# -------------------------------------------------------------------------
# --------------------2. Vector Database Structure-------------------------
# -------------------------------------------------------------------------
# Pinecone Vector Database Setup:
# - Index Name: 'job-matching'
# - Embedding Model: 'all-MiniLM-L6-v2' with 384-dimensional embeddings.
# - Metadata fields include:
#     * title: Name of the job file
#     * content: Full job description text
#     * min_experience: Minimum required experience (in years)
#     * skills: Extracted relevant skills from the job description
#     * education: Extracted education information
#     * certifications: Extracted certifications
# -------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client with API key from environment variable
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Configure Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embedding model for vector encoding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define index parameters for Pinecone
index_name = "job-matching"
dimension = 384  # Embedding dimension based on 'all-MiniLM-L6-v2' model

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the existing or newly created Pinecone index
index = pc.Index(index_name)

# Define the vector structure for metadata storage
VECTOR_STRUCTURE = {
    "title": "",  # Job file name
    "content": "",  # Full job description
    "min_experience": 0,  # Minimum required experience in years
    "skills": [],  # Extracted skills from job description
    "education": [],  # Extracted education information
    "certifications": []  # Extracted certifications
}

#-------------------------------------------------------------------------
# -----------------------3. Helper Functions -----------------------------
#-------------------------------------------------------------------------


# Function to extract content from a resume file (PDF or DOCX)
def extract_resume_content(resume_file):
    file_extension = resume_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(resume_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(resume_file)
    return ""

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    content = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        content += page.get_text("text")
    return content

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_file):
    document = Document(docx_file)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

# Function to extract job description and structured information from a file
def extract_job_description(job_file):
    content = job_file.read().decode("utf-8") if job_file.name.endswith('.txt') else json.load(job_file).get('description', '')
    min_experience = extract_minimum_experience(content)
    job_info = extract_job_info(content)
    return content, min_experience, job_info

# Function to extract minimum experience required from text
def extract_minimum_experience(text):
    match = re.search(r'(\d+)\s*[\+]? years?', text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

# Function to calculate total years of experience mentioned in resume content
def calculate_years_of_experience(content):
    years = re.findall(r'(\d+)\s*[\+]? years?', content, re.IGNORECASE)
    return sum(int(year) for year in years)

# Reusable function for extracting structured information
def extract_structured_info(content):
    """Extracts skills, education, and certifications from the given content."""
    skills = re.findall(
        r'\b(?:Python|Java|SQL|HTML|CSS|JavaScript|C\+\+|C#|PHP|R|TypeScript|'
        r'Kali Linux|Wireshark|Metasploit|NumPy|Pandas|Matplotlib|TensorFlow|'
        r'Keras|Scikit-learn|PyTorch|Spark|Hadoop|AWS|Azure|Google Cloud|'
        r'Docker|Kubernetes|Linux|Unix|Bash|PowerShell|Jenkins|Git|'
        r'React|Angular|Vue|Django|Flask|REST API|GraphQL|MySQL|PostgreSQL|'
        r'MongoDB|Oracle|BigQuery|Tableau|Power BI|Figma|Sketch)\b',
        content, re.IGNORECASE
    )
    education = re.findall(
        r'\b(?:B\.Tech|Bachelor|Master|BSc|MSc|PhD|MBA|Associate|Diploma|'
        r'Certificate|High School|University|College|Engineering|Computer Science|'
        r'Information Technology|Business Administration|Data Science|Physics|Mathematics|'
        r'Statistics|Cybersecurity)\b',
        content, re.IGNORECASE
    )
    certifications = re.findall(
        r'\b(?:IBM|EC-Council|Certified|Ethical Hacking|AWS Certified|Azure Certified|'
        r'Google Cloud Certified|PMP|Scrum Master|CompTIA|CISSP|CCNA|CCNP|'
        r'CEH|CISM|CISA|OCSP|Security\+|Network\+|ITIL|Prince2|Microsoft Certified)\b',
        content, re.IGNORECASE
    )
    return {
        "skills": skills,
        "education": education,
        "certifications": certifications
    }

# Function to extract structured information (skills, education, certifications) from content
def extract_resume_info(content):
    return extract_structured_info(content)

def extract_job_info(content):
    return extract_structured_info(content)

# Function to match skills, education, and certifications between resume and job
def matched_skills(resume_info, job_info):
    return set(resume_info["skills"]) & set(job_info["skills"])

def matched_education(resume_info, job_info):
    return set(resume_info["education"]) & set(job_info["education"])

def matched_certifications(resume_info, job_info):
    return set(resume_info["certifications"]) & set(job_info["certifications"])

#-------------------------------------------------------------------------
#-------------------------4. Logic Functions -----------------------------
#-------------------------------------------------------------------------


# Function to store job descriptions in Pinecone database, with duplicate check
def store_job_descriptions(job_descriptions):
    """Stores job descriptions in Pinecone, embedding text and adding metadata."""
    for job_file in job_descriptions:
        job_content, min_experience, job_info = extract_job_description(job_file)
        embedding = embedding_model.encode(job_content)
        
        # Construct metadata using defined structure
        metadata = VECTOR_STRUCTURE.copy()
        metadata.update({
            "title": job_file.name,
            "content": job_content,
            "min_experience": min_experience,
            "skills": job_info["skills"],
            "education": job_info["education"],
            "certifications": job_info["certifications"]
        })
        
        # Check if the vector already exists
        existing_vector = index.fetch(ids=[job_file.name])
        
        if existing_vector and job_file.name in existing_vector["vectors"]:
            # If vector exists, delete it
            index.delete(ids=[job_file.name])
            # st.info(f"Existing vector for '{job_file.name}' found and deleted.")
        
        # Upsert (add or update) the vector in Pinecone
        index.upsert(vectors=[(job_file.name, embedding, metadata)])
        # st.success(f"Job description '{job_file.name}' successfully added to the vector database.")
    st.success("Job descriptions successfully added to the vector database.")

# Function to summarize text content
def summarize_text(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(f"Summarize the following text: {text}")
    return response.text

# Function to retrieve top matching jobs based on resume embedding
def retrieve_matching_jobs(resume_content, candidate_experience):
    """Retrieves top-matching jobs from Pinecone based on resume content."""
    # RAG Step 1: **Retrieval**
    resume_embedding = embedding_model.encode(resume_content).tolist()
    results = index.query(vector=resume_embedding, top_k=5, include_metadata=True)
    matching_jobs = [
        res for res in results.matches if candidate_experience >= res.metadata["min_experience"]
    ]
    return matching_jobs

# Function to generate justification using RAG approach
def generate_justification_with_rag(matching_job, resume_info, candidate_experience):
    # RAG Step 2: **Augmented Generation**
    job_info = {
        "skills": matching_job.metadata.get("skills", []),
        "education": matching_job.metadata.get("education", []),
        "certifications": matching_job.metadata.get("certifications", [])
    }
    
    match_skills = matched_skills(resume_info, job_info)
    match_education = matched_education(resume_info, job_info)
    match_certifications = matched_certifications(resume_info, job_info)
    
    context_snippets = [
        f"Job Description: {matching_job.metadata['content']}",
        f"Candidate Skills: {', '.join(resume_info['skills'])}",
        f"Candidate Experience: {candidate_experience} years",
        f"Matched Skills: {', '.join(match_skills)}",
        f"Matched Education: {', '.join(match_education)}",
        f"Matched Certifications: {', '.join(match_certifications)}",
    ]
    
    prompt = f"""
    Using the following context, provide a justification on why the candidate is a good fit:
    
    {context_snippets[0]}
    {context_snippets[1]}
    {context_snippets[2]}
    {context_snippets[3]}
    {context_snippets[4]}
    {context_snippets[5]}
    
    Answer in brief in a structured way, addressing skill match, experience fit, education relevance, and technology alignment.
    """
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


#-------------------------------------------------------------------------
#-------------------------5. Streamlit UI Code ---------------------------
#-------------------------------------------------------------------------


# Streamlit UI for uploading job descriptions and resumes
st.title("Profile-Job Matching Analytics & Justification")

# File upload components for job descriptions and resume
uploaded_job_descriptions = st.file_uploader("Upload Job Descriptions (TXT or JSON)", accept_multiple_files=True, type=['txt', 'json'])
uploaded_resume = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'])

# Main logic for processing input files and displaying results
if uploaded_resume and uploaded_job_descriptions:

    # Store job descriptions in Pinecone database
    store_job_descriptions(uploaded_job_descriptions)
    
    # Extract resume content and calculate candidate's experience
    resume_content = extract_resume_content(uploaded_resume)
    candidate_experience = calculate_years_of_experience(resume_content)
    resume_info = extract_resume_info(resume_content)
    
    # Retrieve top matching jobs
    matching_jobs = retrieve_matching_jobs(resume_content, candidate_experience)

    # Display the best match and generate justification if any matching job is found
    if matching_jobs:
        best_match = matching_jobs[0]
        job_summary = summarize_text(best_match.metadata['content'])
        job_info = {
            "skills": best_match.metadata["skills"],
            "education": best_match.metadata["education"],
            "certifications": best_match.metadata["certifications"]
        }
        
        justification = generate_justification_with_rag(best_match, resume_info, candidate_experience)
        
        st.subheader(f"**Best Matching Job Description:** {best_match.metadata['title']}")
        st.write(f"**Similarity Score:** {best_match.score:.2f}")
        st.write(f"**Job Description Summary:** {job_summary}")
        st.write(f"**Minimum Experience Required:** {best_match.metadata['min_experience']} years")
        st.write(f"**Candidate's Experience:** {candidate_experience} years")
        
        st.subheader("Match Analytics & Justification:")
        st.write("**Matched Skills:**", ', '.join(matched_skills(resume_info, job_info)))
        st.write("**Matched Education:**", ', '.join(matched_education(resume_info, job_info)))
        st.write("**Matched Certifications:**", ', '.join(matched_certifications(resume_info, job_info)))
        st.write(justification)
    else:
        st.warning("No matching jobs found for the candidate’s experience level.")

# Display a message if no files are uploaded
elif not uploaded_resume and not uploaded_job_descriptions:
    st.info("Upload resume and job descriptions to find the best matching job.")

# Display a warning if only one file is uploaded
else:
    st.warning("Please upload both resume and job descriptions to proceed.")

# -------------------------------------------------------------------------
# ----------------6. Conclusion and Next Steps-----------------------------
# -------------------------------------------------------------------------

# Conclusion:
# In this project, we have successfully implemented a Profile-Job Matching system that leverages Pinecone for vector storage
# and retrieval, Google Generative AI for text summarization and justification generation, and Streamlit for the user interface.
# The system allows users to upload job descriptions and resumes, match candidates to the best-fitting jobs, and generate a
# justification for the match based on skills, experience, education, and certifications.

# Next Steps:
# - Enhance the matching algorithm with additional criteria such as location, salary range, and job type.
# - Implement a feedback loop to improve job matching accuracy over time.
# - Integrate with additional AI models for more advanced analysis and insights.
# - Optimize the system for scalability and real-time performance.
# - Conduct user testing and gather feedback for further improvements.
# - Explore additional use cases and applications for the Profile-Job Matching system.

# -----------------------------------------------------------------------
# ------------------------- 7. References -------------------------------
# -----------------------------------------------------------------------

# - Pinecone Documentation: https://www.pinecone.io/docs/
# - Sentence Transformers Documentation: https://www.sbert.net/
# - Google Generative AI Documentation: https://cloud.google.com/ai-platform/generators/docs
# - Streamlit Documentation: https://docs.streamlit.io/
# - PyMuPDF Documentation: https://pymupdf.readthedocs.io/en/latest/
# - Docx Documentation: https://python-docx.readthedocs.io/en/latest/
# - dotenv Documentation: https://pypi.org/project/python-dotenv/

# -------------------------------------------------------------------------