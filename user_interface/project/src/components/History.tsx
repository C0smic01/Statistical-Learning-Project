import React from 'react';
import { HistoryEntry } from '../types';
import { Clock, X } from 'lucide-react';
import EmotionVisualization from './EmotionVisualization';

interface HistoryProps {
  history: HistoryEntry[];
  onClose: () => void;
  onSelectEntry: (entry: HistoryEntry) => void;
}

const History: React.FC<HistoryProps> = ({ history, onClose, onSelectEntry }) => {
  return (
    <>
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity duration-300"
        onClick={onClose}
      />
      <div 
        className="fixed inset-y-0 right-0 w-96 bg-[#0F2037] border-l border-[#1E293B] shadow-xl 
                 transform translate-x-0 transition-transform duration-300 ease-in-out z-50 
                 flex flex-col animate-slide-in"
      >
        <div className="p-4 border-b border-[#1E293B] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="text-[#00D8FF]" size={20} />
            <h2 className="text-lg font-medium text-white">History</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-2 rounded-full
                     hover:bg-[#162B45]"
          >
            <X size={20} />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          {history.length === 0 ? (
            <div className="p-4 text-center text-gray-400">
              No history yet
            </div>
          ) : (
            <div className="divide-y divide-[#1E293B]">
              {history.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() => onSelectEntry(entry)}
                  className="w-full p-4 text-left hover:bg-[#162B45] transition-colors group"
                >
                  <p className="text-sm text-gray-400 mb-1">
                    {new Date(entry.timestamp).toLocaleString()}
                  </p>
                  <p className="text-white line-clamp-2 group-hover:text-[#00D8FF] transition-colors">
                    {entry.input}
                  </p>
                  <p className="text-sm text-emerald-400 mt-1">
                    Dominant: {entry.result.dominantEmotion}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default History;