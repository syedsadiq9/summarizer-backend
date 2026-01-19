from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI()

# Load model (fast + stable)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Summarizer API Running"}

@app.post("/summarize")
def summarize(data: TextInput):
    result = summarizer(data.text, max_length=150, min_length=60, do_sample=False)
    return {"summary": result[0]["summary_text"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
