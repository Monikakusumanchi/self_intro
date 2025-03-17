import os
import time
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini
from agno.utils.log import logger

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set in environment variables.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Create FastAPI app
app = FastAPI()

# Directory for storing temporary audio files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define Pydantic models
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

# Initialize Agent
model = Gemini(id="gemini-2.0-flash-exp")
agent = Agent(
    model=model,
    markdown=True,
    response_model=InterviewAnalysis,
    structured_outputs=True,
)

def save_audio_file(audio: UploadFile) -> str:
    """Saves uploaded audio to a temp folder and returns the file path."""
    filename = f"audio_{int(time.time())}.wav"
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    return file_path

@app.post("/upload-audio/")
def upload_audio(audio: UploadFile = File(...)):
    """Endpoint to upload an audio file."""
    if not audio.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    file_path = save_audio_file(audio)
    return {"message": "Audio uploaded successfully", "file_path": file_path}

@app.post("/analyze-interview/")
def analyze_interview(file_path: str):
    """Processes the uploaded audio file and returns interview feedback."""
    # if not Path(file_path).exists():
    #     raise HTTPException(status_code=404, detail="Audio file not found.")
    
    # try:
    #     with open(file_path, "rb") as f:
    #         audio_bytes = f.read()
        
    #     audio = Audio(content=audio_bytes, mime_type="audio/wav")
    # except Exception as e:
    #     logger.error(f"Error loading audio file: {e}")
    #     raise HTTPException(status_code=500, detail="Error processing the audio file.")
    trans = """
    Hi. Hi, sir. Good afternoon. Uh, Nikita, tell me about yourself first. First of all, thanks to Teki for Software Solutions for giving this opportunity to introduce myself. My name is D Nikita. I was born and brought up in Hyderabad, coming to my education qualification. Right now, I am pursuing BCom final year in uh in Siddharthare College. I have completed my intermediate from Shiari Junior College. I have done my SSE from Holy Cross High School in my family. We are 5 members, including me, my father, mother, my elder brother, me and my grandmother. My father is a salesman. My mother is a homemaker. My my elder brother is um uh sorry. My elder brother is, um, right now he's searching for a job. My goals, uh, my goal is to get a job in MNC company and become a a financially independent person. My hobbies are My hobbies are Listening music Uh, we. That's it. What is, uh, what are your strengths and weakness? My strength is my family and my friends. My weakness is my father. What father, your father is your weakness also, uh, he has health issues, that's why, uh. OK. Oh, OK. You said your hobbies is to listening music. Yeah, uh, what type of genre, uh, you prefer to listen to music, melody songs, melody songs, yeah. Can you sing a song? See right now OK. Uh, just tell me about your, uh, your role model. Do you have any role model? I just follow actor, TV actor. Uh, he's good looking as a, uh, he's a, he came from a low, lower family and he become Uh, that's it. OK. Uh, I will ask you one question. Uh, a man stands on one side of the river and his dog is on, uh, other side of the river, and he called his dog and uh immediately crosses the river, uh, without, uh, getting wet. How is it possible? There is no bridge or there is no boat. How did the dog, uh, go to him? Without getting wet. He came, uh, he, he, he didn't go. He just stayed at that side. from riverside, he can go. He, he, he didn't go anywhere, he just stand on his side, and he called his dog. The dog just Went to uh went to the other side without getting wet, there's no board, there's no bridge. Is there any other way to go, sir? There's no any other way. From coastal side there's no other way OK. OK, think about it. Uh, I will ask you another question. Uh, if you want to be an animal, which one will you choose? Peacock. I'm, I'm asking animal, animal, uh. Dear, I like white deer. Because they look very cute and. Oh it's so cute. OK. OK, did you get the answer? No, sir. I didn't get it. OK. Uh, how's the workshop? It is very nice. OK. Uh, can you differentiate yourself before workshop and after workshop before workshop, I don't know about any anything about IT. After you coming, I know about basics of IT because I, I did not even know about the technical job and non-technical job. From I came here only. I get to know that. Is it uh Uh, can you scale it, uh, from 10 points? How, uh, how many points will you give? For the workshop. 9.5. OK. OK, thank you. Thank you.
    """
    prompt = f"""
    You are an AI mock interview coach designed to evaluate the candidate's response. using the transcript {trans} 
    Evaluate the response based on clarity, structure, confidence, relevance, and communication.
    
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
        "candidate_response": "Assume you are a candidate attending a job interview and answer the question 'Tell me about yourself.'"
    }}
    """
    
    try:
        response = agent.run(prompt)  # Correct way to fetch AI response
        logger.info(f"AI Response: {response}")  # Debugging
        return  response.content.ratings, response.content.feedback, response.content.candidate_response
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail="Error generating interview analysis.")