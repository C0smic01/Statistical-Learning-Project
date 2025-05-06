export interface EmotionScore {
  emotion: string;
  score: number;
}

export type EmotionType = 'joy' | 'sadness' | 'anger' | 'fear' | 'love' | 'surprise';

export interface ClassificationResult {
  emotions: Record<EmotionType, number>;
  dominantEmotion: EmotionType;
}

export interface HuggingFaceResponse {
  label: string;
  score: number;
}

export interface HistoryEntry {
  id: string;
  timestamp: number;
  input: string;
  result: ClassificationResult;
}