import React from 'react';
import { ClassificationResult, EmotionType } from '../types';
import EmotionBar from './EmotionBar';

interface EmotionVisualizationProps {
  result: ClassificationResult | null;
}

const EmotionVisualization: React.FC<EmotionVisualizationProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="bg-[#162B45] p-5 rounded-lg border border-[#1E293B]">
        <p className="text-center text-gray-400">Enter text to see emotion analysis</p>
      </div>
    );
  }

  // Sort emotions by score in descending order
  const sortedEmotions = Object.entries(result.emotions)
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    .map(([emotion, score]) => ({
      emotion: emotion as EmotionType,
      score,
    }));

  return (
    <div className="bg-[#162B45] p-5 rounded-lg border border-[#1E293B] transition-all duration-300">
      <h3 className="font-medium text-lg mb-4 text-white">Emotion Analysis</h3>
      
      <div className="mb-6">
        {sortedEmotions.map((item) => (
          <EmotionBar
            key={item.emotion}
            emotion={item.emotion}
            score={item.score}
            isHighest={item.emotion === result.dominantEmotion}
          />
        ))}
      </div>
      
      <div className="text-center p-3 bg-[#0F2037] rounded-md border border-[#1E293B]">
        <p className="text-sm text-gray-400">Dominant emotion:</p>
        <p className="text-xl font-semibold mt-1 capitalize text-[#00D8FF]">{result.dominantEmotion}</p>
      </div>
    </div>
  );
};

export default EmotionVisualization