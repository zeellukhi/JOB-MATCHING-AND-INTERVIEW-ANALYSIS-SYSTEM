# JOB-MATCHING-AND-INTERVIEW-ANALYSIS-SYSTEM
An AI-Powered Solution for Enhancing Recruitment Efficiency  through Job-Resume Matching and Candidate Interview Analysis 

# Job Matching and Interview Analysis

This project provides tools for job matching and interview analysis using machine learning, natural language and generative AI processing techniques.

## Features

### Job Matching
- **Resume Parsing**: Extracts text from PDF and DOCX resumes.
- **Job Description Parsing**: Extracts job requirements from text and JSON files.
- **Experience Calculation**: Calculates years of experience from resume content.
- **Skill, Education, and Certification Matching**: Matches resume information with job requirements.
- **Justification Generation**: Generates a justification for job matching using a generative AI model.

### Interview Analysis
- **Audio Extraction**: Extracts audio from video files.
- **Transcription**: Transcribes audio to text using an ASR model.
- **Question and Answer Extraction**: Identifies likely questions and answers from the transcription.
- **Communication Style Analysis**: Analyzes the communication style using sentiment analysis.
- **Active Listening Analysis**: Analyzes active listening skills using semantic similarity.
- **Engagement Analysis**: Analyzes engagement based on answer length and interaction cues.
- **Summary and Analysis Generation**: Generates a summary and analysis of the interview using a generative AI model.

## Installation

1. Clone the repository/Download the Repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the following environment variables:
        ```env
        GOOGLE_API_KEY="your_google_api_key"
        PINECONE_API_KEY="your_pinecone_api_key"
        ```

## Usage

### Job Matching

1. Run the Streamlit app:
    ```sh
    streamlit run Job_Matching.py
    ```

2. Upload job descriptions and resumes through the Streamlit interface.
Use jd from Data->JS
Use Resume from Data->Resume

### Interview Analysis

1. Run the Streamlit app:
    ```sh
    streamlit run Interview_Analysis.py
    ```

2. Upload a video file through the Streamlit interface.
Use Video From Data->Job Interview

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License.
