from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TextRequest(BaseModel):
    text: str

classifier = None

def get_classifier():
    global classifier
    if classifier is None:
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