import gradio as gr
import requests
import os
import time

UPLOAD_URL = "http://127.0.0.1:8000/upload-audio/"
ANALYZE_URL = "http://127.0.0.1:8000/analyze-interview/"

def process_interview(name, college, branch, audio_file):
    """Uploads the audio and analyzes the interview."""
    if not audio_file:
        return "Please record your response.", None
    
    files = {"audio": open(audio_file, "rb")}
    response = requests.post(UPLOAD_URL, files=files)
    if response.status_code != 200:
        return "Error uploading audio.", None
    time.sleep(5) 
    job_name = response.json().get("job_name")
    print(job_name)

    # Wait for transcription
    
    # payload = {"job_name": job_name, "candidate_name": name, "college": college, "branch": branch}
    # analysis_response = requests.post(ANALYZE_URL, json=payload)
    analysis_response = requests.post(
    ANALYZE_URL, 
    params={  # Use `params` instead of `json`
        "job_name": job_name,
        "candidate_name": name,
        "college": college,
        "branch": branch
    }
    )
    if analysis_response.status_code != 200:
        print("Response:", analysis_response.text)  # This will show FastAPI's error message
        return "Error analyzing interview.", None
    
    result = analysis_response.json()
    ratings = result[0]
    feedback = result[1]
    audio_output = result[2]
    
    output_text = f"""
    **Interview Analysis:**
    - Clarity: {ratings['clarity']}/10
    - Structure: {ratings['structure']}/10
    - Confidence: {ratings['confidence']}/10
    - Relevance: {ratings['relevance']}/10
    - Communication: {ratings['communication']}/10
    - Overall Rating: {ratings['overall_rating']}/10
    
    **Feedback:**
    - Strengths: {feedback['strengths']}
    - Areas for Improvement: {feedback['improvements']}
    - Suggestions: {feedback['suggestions']}
    """
    
    return output_text, audio_output


gui = gr.Blocks(theme='NoCrypt/miku')
with gui:
    gr.Markdown("## Mock Interview")
    gr.Markdown("**Question: Tell me about yourself**")
    
    with gr.Row():
        name = gr.Textbox(label="Candidate Name")
        college = gr.Textbox(label="College")
        branch = gr.Textbox(label="Branch")
    
    audio = gr.Audio(sources="microphone", type="filepath", label="Record your response")
    
    analyze_button = gr.Button("Submit Response")
    output_text = gr.Textbox(label="Interview Analysis")
    audio_output = gr.Audio(label="AI Response Audio")
    
    analyze_button.click(process_interview, inputs=[name, college, branch, audio], outputs=[output_text, audio_output])

gui.launch()
