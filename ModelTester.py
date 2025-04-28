import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TARGET_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def predict_emotion(text, model_path="./emotion_model"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    # Get emotion label and confidence
    emotion = TARGET_EMOTIONS[predicted_class]
    confidence = predictions[0][predicted_class].item()
    
    return emotion, confidence, predictions[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--model_path', type=str, default='./emotion_model', help='Path to model')
    
    args = parser.parse_args()
    
    if args.text:
        emotion, confidence, all_probs = predict_emotion(args.text, args.model_path)
        print(f"Text: '{args.text}'")
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
        print("All probabilities:")
        for i, emo in enumerate(TARGET_EMOTIONS):
            print(f"  {emo}: {all_probs[i]:.4f}")
    else:
        print("Please provide text using the --text argument")
        
if __name__ == "__main__":
    main()