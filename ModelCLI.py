#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TARGET_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict_emotion(text, model_path="./emotion_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    emotion = TARGET_EMOTIONS[predicted_class]
    confidence = predictions[0][predicted_class].item()
    
    return emotion, confidence, predictions[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from text')
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model_path', type=str, default='./emotion_model', help='Path to model')
    parser.add_argument('--verbose', action='store_true', help='Show detailed probabilities')
    
    args = parser.parse_args()
    
    emotion, confidence, all_probs = predict_emotion(args.text, args.model_path)
    
    print(f"Text: '{args.text}'")
    print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
    
    if args.verbose:
        print("All probabilities:")
        for i, emo in enumerate(TARGET_EMOTIONS):
            print(f"  {emo}: {all_probs[i]:.4f}")
        
if __name__ == "__main__":
    main()