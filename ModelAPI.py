from fastapi import FastAPI
from transformers import pipeline
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

def get_classifier():
    try:
        classifier = pipeline("text-classification", model="./emotion_model", tokenizer="./emotion_model", return_all_scores=True)
    except Exception as e:
        print(f"Error loading model: {e}")
    return classifier

@app.post("/predict")
async def predict(request: TextRequest):
    classifier = get_classifier()
    result = classifier(request.text)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)