import { ClassificationResult, EmotionType, HuggingFaceResponse } from '../types';

export const analyzeText = async (text: string): Promise<ClassificationResult> => {
  if (!text.trim()) {
    return {
      emotions: {
        anger: 0,
        disgust: 0,
        fear: 0,
        joy: 0,
        neutral: 1,
        sadness: 0,
        surprise: 0
      },
      dominantEmotion: 'neutral'
    };
  }

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error('Failed to analyze text. The model service might be unavailable.');
    }

    const data = await response.json();
    if (!Array.isArray(data) || !Array.isArray(data[0])) {
      throw new Error('Invalid response format from the model');
    }

    const emotions: Record<EmotionType, number> = {
      anger: 0,
      disgust: 0,
      fear: 0,
      joy: 0,
      neutral: 0,
      sadness: 0,
      surprise: 0
    };

    // Parse the response and map it to our emotion object
    const results = data[0] as HuggingFaceResponse[];
    results.forEach(result => {
      emotions[result.label as EmotionType] = result.score;
    });

    // Find the dominant emotion
    const dominantEmotion = Object.entries(emotions).reduce(
      (max, [emotion, score]) => (score > emotions[max as EmotionType] ? emotion : max),
      'neutral'
    ) as EmotionType;

    return {
      emotions,
      dominantEmotion
    };
  } catch (error) {
    console.error('Error analyzing text:', error);
    throw error;
  }
};