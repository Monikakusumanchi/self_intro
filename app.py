import os
import time
import gradio as gr
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini
from agno.utils.log import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set in environment variables.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Create a temp directory if it doesn't exist
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define Pydantic model for structured output
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
model = Gemini(id="gemini-2.0-flash-001")
agent = Agent(
    model=model,
    markdown=True,
    response_model=InterviewAnalysis,
    structured_outputs=True,
)
# Temporary variable to store the last recorded audio file path
temp_audio_file = None

import shutil

def save_audio(audio_path):
    """Saves the recorded/uploaded audio in the temp folder."""
    global temp_audio_file
    if not audio_path:
        return "⚠️ No audio recorded or uploaded."

    # Generate a unique filename using timestamp
    filename = f"audio_{int(time.time())}.wav"
    temp_audio_file = os.path.join(TEMP_DIR, filename)
    
    # Move the file using shutil to avoid cross-device issues
    shutil.move(audio_path, temp_audio_file)
    
    return f"✅ Audio saved: {temp_audio_file}"


def analyze_interview():
    """Processes the last recorded audio file."""
    global temp_audio_file

    if not temp_audio_file or not Path(temp_audio_file).exists():
        return "⚠️ No valid audio file found.", "", ""

    logger.info(f"Processing audio file: {temp_audio_file}")

    try:
        # Read the audio file as bytes
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()

        # Pass raw bytes to Agno's Audio class
        audio = Audio(content=audio_bytes, mime_type="audio/wav")  # Adjust mime_type if needed

    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        return "⚠️ Error loading the audio file.", "❌ No suggestions available.", "❌ Could not generate a response."
    trans = """
    Hi. Hi, sir. Good afternoon. Uh, Nikita, tell me about yourself first. First of all, thanks to Teki for Software Solutions for giving this opportunity to introduce myself. My name is D Nikita. I was born and brought up in Hyderabad, coming to my education qualification. Right now, I am pursuing BCom final year in uh in Siddharthare College. I have completed my intermediate from Shiari Junior College. I have done my SSE from Holy Cross High School in my family. We are 5 members, including me, my father, mother, my elder brother, me and my grandmother. My father is a salesman. My mother is a homemaker. My my elder brother is um uh sorry. My elder brother is, um, right now he's searching for a job. My goals, uh, my goal is to get a job in MNC company and become a a financially independent person. My hobbies are My hobbies are Listening music Uh, we. That's it. What is, uh, what are your strengths and weakness? My strength is my family and my friends. My weakness is my father. What father, your father is your weakness also, uh, he has health issues, that's why, uh. OK. Oh, OK. You said your hobbies is to listening music. Yeah, uh, what type of genre, uh, you prefer to listen to music, melody songs, melody songs, yeah. Can you sing a song? See right now OK. Uh, just tell me about your, uh, your role model. Do you have any role model? I just follow actor, TV actor. Uh, he's good looking as a, uh, he's a, he came from a low, lower family and he become Uh, that's it. OK. Uh, I will ask you one question. Uh, a man stands on one side of the river and his dog is on, uh, other side of the river, and he called his dog and uh immediately crosses the river, uh, without, uh, getting wet. How is it possible? There is no bridge or there is no boat. How did the dog, uh, go to him? Without getting wet. He came, uh, he, he, he didn't go. He just stayed at that side. from riverside, he can go. He, he, he didn't go anywhere, he just stand on his side, and he called his dog. The dog just Went to uh went to the other side without getting wet, there's no board, there's no bridge. Is there any other way to go, sir? There's no any other way. From coastal side there's no other way OK. OK, think about it. Uh, I will ask you another question. Uh, if you want to be an animal, which one will you choose? Peacock. I'm, I'm asking animal, animal, uh. Dear, I like white deer. Because they look very cute and. Oh it's so cute. OK. OK, did you get the answer? No, sir. I didn't get it. OK. Uh, how's the workshop? It is very nice. OK. Uh, can you differentiate yourself before workshop and after workshop before workshop, I don't know about any anything about IT. After you coming, I know about basics of IT because I, I did not even know about the technical job and non-technical job. From I came here only. I get to know that. Is it uh Uh, can you scale it, uh, from 10 points? How, uh, how many points will you give? For the workshop. 9.5. OK. OK, thank you. Thank you.
    """
    prompt = f"""
    You are an AI mock interview coach designed to help freshers practice the question: "Tell me about yourself." 
    Your task is to evaluate the candidate's response based on the provided transcript {trans} and generate feedback.

    ### **Evaluation Criteria:**
    1️⃣ **Clarity (1-10):** Is the response clear and structured?  
    2️⃣ **Structure (1-10):** Does it follow a logical flow (Intro → Background → Skills → Strengths → Conclusion)?  
    3️⃣ **Confidence (1-10):** Does the candidate sound confident and engaging?  
    4️⃣ **Relevance (1-10):** Are they highlighting important aspects (education, skills, experience)?  
    5️⃣ **Communication (1-10):** How well are they articulating their thoughts?  
    6️⃣ **Overall Rating (1-10):** Average score based on the above aspects.  

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
        "candidate_response": "Assume you are a candidate attending a job interview, and answer the question 'Tell me about yourself.' Use the information spoken by the candidate."
    }}
    """

    try:
        # response = agent.print_response(prompt, stream=True)
        response = agent.run(prompt)  # Correct way to fetch AI response
        logger.info(f"AI Response: {response}")  # Debugging
      
        if response:
            return  response.content.ratings, response.content.feedback, response.content.candidate_response
        else:
            return "⚠️ Transcript generation failed.", "❌ No suggestions available.", "❌ Could not generate a response."
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return "⚠️ Error processing the audio file.", "❌ No suggestions available.", "❌ Could not generate a response."

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 AI Interview Analyzer")
    gr.Markdown("Record or upload your response. Click **Stop** to save the audio, then **Submit** to analyze.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎙️ Record or Upload Audio")

    stop_btn = gr.Button("🛑 Stop & Save Audio")
    submit_btn = gr.Button("📤 Submit & Analyze")

    status_output = gr.Textbox(label="ℹ️ Status")
    transcript_output = gr.Textbox(label="📝 Generated Transcript")
    suggestions_output = gr.Textbox(label="📢 Suggestions for Improvement")
    candidate_response_output = gr.Textbox(label="🤖 AI Candidate Response")

    stop_btn.click(
        save_audio,
        inputs=[audio_input],
        outputs=[status_output]
    )

    submit_btn.click(
        analyze_interview,
        outputs=[transcript_output, suggestions_output, candidate_response_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
