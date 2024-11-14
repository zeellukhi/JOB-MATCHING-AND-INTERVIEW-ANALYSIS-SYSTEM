# -------------------------------------------------------------------------
# ----------------1. Imports and Environment Setup-------------------------
# -------------------------------------------------------------------------

import streamlit as st
import ffmpeg
import tempfile
import os
import google.generativeai as genai
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import re

# Load environment variables for API keys
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Hugging Face ASR (Automatic Speech Recognition) pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Initialize semantic similarity model for analyzing active listening
similarity_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

#------------------------------------------------------------------------
# -------------------------2. Helper Functions --------------------------
#------------------------------------------------------------------------

# Function to extract audio from a video file
def extract_audio(video_path, audio_output_path):
    #Extracts audio from a video file using ffmpeg.
    try:
        ffmpeg.input(video_path).output(audio_output_path).run(overwrite_output=True)
        return audio_output_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio into text using the ASR model
def transcribe_audio(audio_output_path):
    #Transcribes the given audio file into text using an ASR pipeline.
    
    transcription = asr_pipeline(audio_output_path, return_timestamps=True)
    return transcription['text']

# Function to extract likely questions from the transcription text
def extract_likely_questions(transcription):
    #Extracts potential questions from the transcribed text using regular expressions.
    
    question_pattern = r'(\b(?:What|How|Why|Where|When|Tell me about|Can you explain|Could you tell me|Do you think|Would you|Would you mind)\b[^?.!]*\?)'
    likely_questions = re.findall(question_pattern, transcription)
    likely_questions = [q.strip() for q in likely_questions if len(q.split()) > 3]  # Filter out very short questions
    return likely_questions

# Function to extract likely answers from the transcription text based on questions' proximity
def extract_likely_answers(transcription, likely_questions):
    #Extracts answers from the transcription text based on the proximity to the questions.
    
    sentences = transcription.split('.')
    answers = {}
    for question in likely_questions:
        question_index = transcription.find(question)
        answer_start_index = question_index + len(question)
        answer_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 3][:5]
        answers[question] = ' '.join(answer_sentences)
    return answers


#----------------------------------------------------------------------------
# ---------------------------- 3. Analyze Function --------------------------
#----------------------------------------------------------------------------

# ---------------------------  Note  ----------------------------------------
# Have made an attemp to give a function to analyze the communication style, 
# active listening, and engagement of the candidate based on the interview transcript.
# the obtain result might not be accurate as the model used is not trained on the data
# Further ehancement can be made by training the model on the data to get more accurate result
# --------------------------------------------------------------------------

# ---------------------------  Note  ----------------------------------------
# focus on approch made to try to give sum contex to the prompt to solve the problem effectively
# --------------------------------------------------------------------------

# ---------------------------  Note  ----------------------------------------
# The function to analyze the communication style, active listening, and engagement of the 
# candidate based on the interview transcript.
# The function uses sentiment analysis to evaluate the communication style, semantic similarity 
# to assess active listening, and keyword extraction to analyze engagement.
# The function returns the average sentiment score, communication style (positive or negative), 
# active listening score, and engagement score.
# --------------------------------------------------------------------------



# Function to analyze the communication style based on sentiment analysis
def analyze_communication_style(transcription):
    #Analyzes the communication style of the candidate based on sentiment analysis.
    
    likely_questions = extract_likely_questions(transcription)
    answers = extract_likely_answers(transcription, likely_questions)
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    sentiments = [sentiment_analyzer(answer)[0]['score'] for answer in answers.values()]
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    communication_style = "Positive" if average_sentiment > 0.5 else "Negative"
    return average_sentiment, communication_style

# Function to analyze active listening based on semantic similarity between questions and answers
def analyze_active_listening(transcription):
    #Analyzes the candidate's active listening ability based on the semantic similarity between questions and answers.
    
    likely_questions = extract_likely_questions(transcription)
    answers = extract_likely_answers(transcription, likely_questions)
    
    question_embeddings = similarity_model.encode(likely_questions, convert_to_tensor=True)
    answer_embeddings = similarity_model.encode(list(answers.values()), convert_to_tensor=True)
    
    similarities = [util.cos_sim(question_embedding, answer_embedding).max().item() for answer_embedding in answer_embeddings for question_embedding in question_embeddings]
    active_listening_score = sum(similarities) / len(similarities) if similarities else 0
    return active_listening_score

# Function to analyze engagement based on answer length and common engagement cues
def analyze_engagement(transcription):
    #Analyzes the candidate's engagement based on answer length and the presence of engagement cues in the transcription.
    
    likely_questions = extract_likely_questions(transcription)
    answers = extract_likely_answers(transcription, likely_questions)
    
    engagement_score = sum(1 for answer in answers.values() if len(answer.split()) > 10)
    
    engagement_keywords = ["I see", "I understand", "Absolutely", "Right", "Interesting", "Good point"]
    engagement_score += sum(1 for keyword in engagement_keywords if keyword.lower() in transcription.lower())
    
    return engagement_score

#-------------------------------------------------------------------------------------
# ------------------------4. Generative Ai to Analyze via Prompt ---------------------
#-------------------------------------------------------------------------------------

# Function to summarize and analyze the interview transcript using Google Generative AI
def analyze_transcript_with_scores(transcription, communication_score, active_listening_score, engagement_score):
    """
    Summarizes and analyzes the interview transcript using Google Generative AI based on communication, active listening, and engagement scores.
    Args:
        transcription (str): Transcribed text from audio.
        communication_score (float): Candidate's communication score.
        active_listening_score (float): Candidate's active listening score.
        engagement_score (int): Candidate's engagement score.
    Returns:
        str: Summary and analysis of the interview.
    """
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    
    prompt = f"""
    You are an expert interview analysis assistant. Based on the following interview transcript, provide a summary of the candidate's responses and analyze the candidate's communication style, active listening, and engagement with the interviewer. Use the following scores to guide your analysis, but do not mention them in your final response:

    - Communication Score: {communication_score}
    - Active Listening Score: {active_listening_score}
    - Engagement Score: {engagement_score}

    Transcript: {transcription}

    ### Analysis Instructions:
    - **Communication Style**: Evaluate the clarity, tone, and depth of the candidate's communication, considering how well they express ideas and engage the interviewer.
    - **Active Listening**: Assess the candidate’s ability to listen, understand, and respond effectively to questions. Consider the relevance and coherence of their answers.
    - **Engagement**: Analyze the level of interaction between the candidate and the interviewer. Look for signs of attentiveness, rapport, and conversational cues.

    ### Output Format:
    - **Summary of Responses**: Provide a concise summary of the candidate's answers, highlighting key points and insights.
    - **Communication Style**: Discuss the candidate’s tone, clarity, and communication skills, emphasizing how effectively they convey their thoughts and connect with the interviewer.
    - **Active Listening**: Describe the candidate's active listening abilities, highlighting their relevance to the conversation, how well they understand and respond to questions, and their attentiveness.
    - **Engagement with Interviewer**: Provide an assessment of the candidate’s level of engagement with the interviewer, considering factors like attentiveness, rapport, and the quality of the interaction.

    Provide a qualitative assessment of each trait, focusing on the candidate’s behavior and responses, without referencing the numeric scores directly.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No response generated."

#---------------------------------------------------------------------------
# ---------------------------- 5. Streamlit UI Code ------------------------
#---------------------------------------------------------------------------

# Streamlit UI for video file upload and analysis
st.title("Interview Analysis Module")

# Upload video file (mp4, avi, mov, mkv)
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_file.read())
        video_path = temp_video_file.name
    
    # Extract audio from video
    audio_output_path = f"{temp_video_file.name}.mp3"
    extract_audio(video_path, audio_output_path)
    
    # Transcribe the audio to text
    transcription = transcribe_audio(audio_output_path)
    
    # Analyze communication style
    communication_score, _ = analyze_communication_style(transcription)
    
    # Analyze active listening
    active_listening_score = analyze_active_listening(transcription)
    
    # Analyze engagement
    engagement_score = analyze_engagement(transcription)
    
    # Generate summary and analysis using Google Generative AI
    analysis_output = analyze_transcript_with_scores(transcription, communication_score, active_listening_score, engagement_score)
    
    # Display results in Streamlit UI
    st.subheader("Transcription:")
    st.write(transcription)

    st.subheader("Analysis Summary:")
    st.write(analysis_output)


# -----------------------------------------------------------------------
# ------------------------- 6. Conclusion -------------------------------
# -----------------------------------------------------------------------

# The above code is a simple implementation of an interview analysis module that transcribes audio from a video file, 
# analyzes the candidate's communication style, active listening, and engagement based on the interview transcript, and 
# generates a summary and analysis using Google Generative AI. The analysis can be further enhanced by training the 
# models on specific interview data to improve accuracy and relevance. The code can be extended to include additional 
# features such as sentiment analysis, keyword extraction, and summarization to provide more insights into the candidate's 
# performance during the interview. The code can also be integrated into a larger application or platform for automated 
# interview analysis and feedback generation.

# -----------------------------------------------------------------------
# ------------------------- 7. Next Step -------------------------------
# -----------------------------------------------------------------------

# 1. Train the models on specific interview data to improve accuracy and relevance.
# 2. Include additional features such as sentiment analysis, keyword extraction, and summarization to provide more insights.
# 3. Integrate the code into a larger application or platform for automated interview analysis and feedback generation.
# 4. Optimize the code for performance and scalability to handle large volumes of interview data efficiently.
# 5. Explore the use of machine learning models for more advanced analysis and insights

# -----------------------------------------------------------------------
# ------------------------- 8. References -------------------------------
# -----------------------------------------------------------------------

# References:
# - Google Generative AI: https://cloud.google.com/generative-ai
# - Hugging Face Transformers: https://huggingface.co/transformers/
# - Sentence Transformers: https://www.sbert.net/
# - Streamlit Documentation:https://docs.streamlit.io/
# - ffmpeg Documentation: https://ffmpeg.org/documentation.html
# - OpenAI ASR Pipeline: https://huggingface.co/openai/whisper-base
# - DistilBERT Sentiment Analysis: https://huggingface.co/transformers/model_doc/distilbert.html
# - Paraphrase-MiniLM-L6-v2: https://www.sbert.net/docs/pretrained_models.html
# - dotenv Documentation: https://pypi.org/project/python-dotenv/

# -----------------------------------------------------------------------