from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re

app = FastAPI(title="TutorTube API")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VideoRequest(BaseModel):
    youtube_url: str


@app.get("/")
def home():
    return {"message": "TutorTube API is running"}


@app.post("/learn")
def create_lesson(data: VideoRequest):

    # Extract YouTube video ID
    match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([^&?/]+)",
        data.youtube_url
    )

    if not match:
        return {"error": "Invalid YouTube URL"}

    video_id = match.group(1)

    # Get transcript
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    text = " ".join(
        snippet.text for snippet in transcript
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
{text}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return {
        "video_id": video_id,
        "lesson": response.output_text
    }
