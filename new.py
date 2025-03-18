import os
import time
import shutil
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini
from agno.utils.log import logger
import pyttsx3
from gtts import gTTS  
import boto3
import requests

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Create FastAPI app
app = FastAPI()

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Validate AWS credentials
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    raise ValueError("Error: AWS credentials or S3 bucket name not set in environment variables.")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

transcribe_client = boto3.client(
    "transcribe",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class Ratings(BaseModel):
    clarity: float
    structure: float
    confidence: float
    relevance: float
    communication: float
    overall_rating: float

class Feedback(BaseModel):
    strengths: str
    improvements: str
    suggestions: str

class InterviewAnalysis(BaseModel):
    transcript: str
    ratings: Ratings
    feedback: Feedback
    candidate_response: str

model = Gemini(id="gemini-2.0-flash-exp")
agent = Agent(
    model=model,
    markdown=True,
    response_model=InterviewAnalysis,
    structured_outputs=True,
)

def text_to_speech(text: str) -> str:
    """Converts chatbot response to speech using gTTS."""
    tts = gTTS(text=text, lang='en')
    output_path = ("output.mp3")
    tts.save(output_path)
    return output_path

def save_audio_file(audio: UploadFile) -> str:
    """Saves uploaded audio to a temp folder and returns the file path."""
    filename = f"audio_{int(time.time())}.wav"
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    return file_path, filename

def upload_to_s3(file_path: str, filename: str):
    """Uploads file to S3."""
    if not file_path or not filename:
        raise ValueError("File path or filename is missing")
    s3_client.upload_file(file_path, S3_BUCKET_NAME, filename)

def get_transcription_result(job_name: str) -> str:
    """Waits for transcription to complete and returns the transcript."""
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            transcript_url = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            break
        elif status == "FAILED":
            raise HTTPException(status_code=500, detail="Transcription job failed.")
        time.sleep(5)

    transcript_response = requests.get(transcript_url)
    transcript_data = transcript_response.json()
    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]
    return transcript_text

@app.post("/upload-audio/")
def upload_audio(audio: UploadFile = File(...)):
    """Endpoint to upload an audio file."""
    file_path, filename = save_audio_file(audio)
    upload_to_s3(file_path, filename)
    job_name = filename + ".txt"
    return {"message": "Audio uploaded successfully. Transcription in progress.", "job_name": job_name}

import asyncio

@app.post("/analyze-interview/")
async def analyze_interview(job_name: str, candidate_name: str, college: str, branch: str):
    """Processes the transcribed interview and returns feedback."""
    transcript = get_transcription_result(job_name)
    
    prompt = f"""
    You are an AI mock interview coach designed to evaluate the candidate's response using
    **Candidate Details:**
    - Name: {candidate_name}
    - College: {college}
    - Branch: {branch}
    and the **transcript**:
    "{transcript}"
    
    Evaluate based on clarity, structure, confidence, relevance, and communication.
    remember use the name , college, branch details from the candidate details becatue the transcript some ime do not give correct results
    ### **Your Response Format (JSON Output):**
    {{
        "transcript": "Generated transcript of the candidate’s response",
        "ratings": {{
            "clarity": "Score out of 10",
            "structure": "Score out of 10",
            "confidence": "Score out of 10",
            "relevance": "Score out of 10",
            "communication": "Score out of 10",
            "overall_rating": "Average of all scores"
        }},
        "feedback": {{
            "strengths": "Highlight strong points in the candidate’s response.",
            "improvements": "Identify areas for improvement.",
            "suggestions": "Provide actionable advice for better responses."
        }},
        "candidate_response": "Assume you are a candidate attending a job interview and answer the question 'Tell me about yourself.' use the content given by the user as an input context to agent"
    }}
    """
    
    try:
        # No need for `await` if `agent.run(prompt)` is synchronous
        response = agent.run(prompt)  

        logger.info(f"AI Response: {response}")  
        return response.content.ratings, response.content.feedback, text_to_speech(str(response.content.candidate_response))
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail="Error generating interview analysis.")
