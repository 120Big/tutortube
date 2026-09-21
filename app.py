from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="TutorTube API")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VideoRequest(BaseModel):
    youtube_url: str
    transcript: str


@app.get("/")
def home():
    return {"message": "TutorTube API is running"}


@app.post("/learn")
def create_lesson(data: VideoRequest):

    if not data.transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Transcript is empty."
        )

    prompt = f"""
You are TutorTube, a patient teacher teaching a complete beginner.

Use ONLY the information contained in the transcript.

Create a structured lesson containing:

1. Lesson title
2. What the learner will learn
3. Step-by-step explanation
4. Important points
5. A simple practice task
6. 3 short quiz questions with answers

Make the explanation simple, practical, and beginner-friendly.

TRANSCRIPT:
{data.transcript}
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        return {
            "video_id": data.youtube_url,
            "lesson": response.output_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI generation failed: {str(e)}"
        )
