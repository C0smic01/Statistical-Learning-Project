import React from 'react';
import { HistoryEntry } from '../types';
import { Clock, X, Trash2, Loader2 } from 'lucide-react';
import EmotionVisualization from './EmotionVisualization';

interface HistoryProps {
  history: HistoryEntry[];
  onClose: () => void;
  onSelectEntry: (entry: HistoryEntry) => void;
  onDeleteEntry: (id: string) => void;
  deletingIds: Set<string>;
  isClosing?: boolean;
}

const History: React.FC<HistoryProps> = ({ 
  history, 
  onClose, 
  onSelectEntry, 
  onDeleteEntry, 
  deletingIds,
  isClosing = false 
}) => {
  return (
    <>
      <div 
        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-40 ${
          isClosing ? 'animate-fade-out' : 'animate-fade-in'
        }`}
        onClick={onClose}
      />
      <div 
        className={`fixed inset-y-0 right-0 w-96 bg-[#0F2037] border-l border-[#1E293B] shadow-xl 
                 z-50 flex flex-col ${
                   isClosing ? 'animate-slide-out' : 'animate-slide-in'
                 }`}
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
              {history.map((entry) => {
                console.log('Rendering history entry with ID:', entry.id);
                return (
                  <div
                    key={entry.id}
                    className={`group relative hover:bg-[#162B45] transition-all duration-300 ${
                      deletingIds.has(entry.id) ? 'opacity-50' : ''
                    }`}
                  >
                    <button
                      onClick={() => onSelectEntry(entry)}
                      className="w-full p-4 text-left"
                      disabled={deletingIds.has(entry.id)}
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
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        console.log('Delete button clicked for entry ID:', entry.id);
                        onDeleteEntry(entry.id);
                      }}
                      disabled={deletingIds.has(entry.id)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-2 opacity-0 group-hover:opacity-100
                               text-gray-400 hover:text-red-400 transition-all duration-200 rounded-full
                               hover:bg-red-400/10 disabled:cursor-not-allowed"
                      title="Delete entry"
                    >
                      {deletingIds.has(entry.id) ? (
                        <Loader2 size={18} className="animate-spin" />
                      ) : (
                        <Trash2 size={18} />
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default History;