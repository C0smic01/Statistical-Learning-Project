import React from 'react';
import { EmotionType } from '../types';

interface EmotionBarProps {
  emotion: EmotionType;
  score: number;
  isHighest: boolean;
}

const EmotionBar: React.FC<EmotionBarProps> = ({ emotion, score, isHighest }) => {
  const getEmotionColor = (emotion: EmotionType): string => {
    const colors: Record<EmotionType, string> = {
      joy: 'bg-yellow-400',
      sadness: 'bg-blue-400',
      anger: 'bg-red-400',
      fear: 'bg-purple-400',
      disgust: 'bg-emerald-400',
      surprise: 'bg-cyan-400',
      neutral: 'bg-gray-400',
    };
    return colors[emotion];
  };

  const getEmotionLabel = (emotion: EmotionType): string => {
    return emotion.charAt(0).toUpperCase() + emotion.slice(1);
  };

  return (
    <div className="flex items-center mb-2 group">
      <div className="w-20 text-sm font-medium mr-2 text-gray-300">{getEmotionLabel(emotion)}</div>
      <div className="flex-1 bg-[#0F2037] rounded-full h-6 overflow-hidden">
        <div
          className={`h-full ${getEmotionColor(emotion)} transition-all duration-500 ease-out ${
            isHighest ? 'animate-pulse' : ''
          }`}
          style={{ width: `${Math.max(score * 100, 2)}%` }}
        ></div>
      </div>
      <div className="w-12 text-right text-sm ml-2 text-gray-300">{(score * 100).toFixed(0)}%</div>
    </div>
  );
};

export default EmotionBar;