import { ClassificationResult, EmotionType, HuggingFaceResponse } from '../types';

export const analyzeText = async (text: string): Promise<ClassificationResult> => {
  if (!text.trim()) {
    return {
      emotions: {
        joy: 0,
        sadness: 0,
        anger: 0,
        fear: 0,
        love: 0,
        surprise: 0
      },
      dominantEmotion: 'joy' // Default to joy when empty
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
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('Invalid response format from the model');
    }

    // Initialize the emotions object with all zeros
    const emotions: Record<EmotionType, number> = {
      anger: 0,
      fear: 0,
      joy: 0,
      sadness: 0,
      love: 0,
      surprise: 0
    };

    // The API returns an array with one item, which is an array of emotion objects
    const results = Array.isArray(data[0]) ? data[0] : data;
    
    // Parse the response and map it to our emotion object
    results.forEach((result: HuggingFaceResponse) => {
      if (result.label in emotions) {
        emotions[result.label as EmotionType] = result.score;
      }
    });

    // Find the dominant emotion with proper type safety
    const dominantEmotion = Object.entries(emotions).reduce<EmotionType>(
      (max, [emotion, score]) => {
        const typedEmotion = emotion as EmotionType;
        return score > emotions[max] ? typedEmotion : max;
      },
      'joy'
    );

    return {
      emotions,
      dominantEmotion
    };
  } catch (error) {
    console.error('Error analyzing text:', error);
    throw error;
  }
};