from fastapi import FastAPI
from pydantic import BaseModel
from model import generate_summary
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(data: TextInput):
    summary = generate_summary(data.text)
    return {"summary": summary}

@app.get("/")
def home():
    return {"status": "Summarizer API Running"}
