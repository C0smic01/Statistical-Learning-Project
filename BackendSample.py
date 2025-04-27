# app.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model_path = "./emotion_model"
if os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
else:
    # Fallback to pretrained model if fine-tuned not available
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Create pipeline
emotion_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

id2label = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    if not text:
        return {"error": "No text provided"}
    
    # Emotion prediction
    results = emotion_classifier(text)
    
    # Result formatting
    emotions = []
    for result in results[0]:
        emotion = {
            "label": result["label"],
            "score": round(result["score"], 4)
        }
        emotions.append(emotion)
    
    # Sort emotions by score
    emotions.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "text": text,
        "emotions": emotions,
        "top_emotion": emotions[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)