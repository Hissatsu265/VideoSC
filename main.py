from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from cv_match.services import process_video, ask_question
from services import process_video, ask_question
app = FastAPI()

class VideoRequest(BaseModel):
    url: str
    choice:str

class QuestionRequest(BaseModel):
    question: str

@app.post("/process_video/")
async def process_video_endpoint(request: VideoRequest):
    try:
        return process_video(request.url,request.choice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask_question/")
async def ask_question_endpoint(request: QuestionRequest):
    try:
        return ask_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
