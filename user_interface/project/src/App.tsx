import React, { useState, useEffect } from 'react';
import TextInput from './components/TextInput';
import EmotionVisualization from './components/EmotionVisualization';
import History from './components/History';
import { ClassificationResult, HistoryEntry } from './types';
import { analyzeText } from './services/classificationService';
import { BarChart3, Brain, Send, History as HistoryIcon } from 'lucide-react';
import { supabase } from './lib/supabase';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const { data, error } = await supabase
        .from('history')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      const historyEntries: HistoryEntry[] = data.map(entry => ({
        id: entry.id,
        timestamp: new Date(entry.created_at).getTime(),
        input: entry.input_text,
        result: {
          emotions: entry.emotions,
          dominantEmotion: entry.dominant_emotion
        }
      }));

      setHistory(historyEntries);
    } catch (err) {
      console.error('Error loading history:', err);
    }
  };

  const handleTextChange = (newText: string) => {
    setText(newText);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    try {
      setIsAnalyzing(true);
      setError(null);
      const result = await analyzeText(text);
      setResult(result);
      
      const { error: saveError } = await supabase
        .from('history')
        .insert({
          input_text: text,
          emotions: result.emotions,
          dominant_emotion: result.dominantEmotion
        });

      if (saveError) throw saveError;

      await loadHistory();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze text. Please try again later.';
      setError(errorMessage);
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleHistoryEntrySelect = (entry: HistoryEntry) => {
    setText(entry.input);
    setResult(entry.result);
    setIsHistoryOpen(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0A1628] to-[#1E293B] flex flex-col items-center py-10 px-4">
      <header className="mb-10 text-center w-full max-w-xl">
        <div className="flex items-center justify-center mb-2">
          <Brain className="text-[#00D8FF] mr-2" size={30} />
          <h1 className="text-3xl font-bold text-white">Emotion Analyzer</h1>
        </div>
        <p className="text-gray-400 max-w-md mx-auto">
          Enter text below to analyze its emotional content
        </p>
      </header>

      <button
        onClick={() => setIsHistoryOpen(true)}
        className="fixed top-4 right-4 z-30 bg-[#162B45] p-2 rounded-full hover:bg-[#1E293B] 
                 transition-colors border border-[#1E293B] hover:border-[#00D8FF] group"
        title="View History"
      >
        <HistoryIcon size={24} className="text-gray-400 group-hover:text-[#00D8FF] transition-colors" />
      </button>

      <main className="w-full max-w-xl bg-[#0F2037] rounded-xl shadow-xl overflow-hidden border border-[#1E293B]">
        <div className="p-6">
          <div className="space-y-4">
            <TextInput maxLength={200} onTextChange={handleTextChange} value={text} />
            
            {error && (
              <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg border border-red-900/30">
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="w-full py-2 px-4 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 
                       disabled:bg-emerald-500/50 disabled:cursor-not-allowed transition-colors
                       flex items-center justify-center gap-2 font-medium"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-5 h-5 border-2 border-t-white border-white/30 rounded-full animate-spin"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Send size={18} />
                  Analyze Text
                </>
              )}
            </button>
          </div>
          
          <div className="my-6 flex items-center">
            <div className="flex-1 h-px bg-[#1E293B]"></div>
            <div className="px-3">
              <BarChart3 className="text-[#00D8FF]" size={24} />
            </div>
            <div className="flex-1 h-px bg-[#1E293B]"></div>
          </div>
          
          <EmotionVisualization result={result} />
        </div>
      </main>
      
      <footer className="mt-8 text-center text-gray-500 text-sm">
        <p>Using Hugging Face emotion-english-distilroberta-base model</p>
      </footer>

      {isHistoryOpen && (
        <History
          history={history}
          onClose={() => setIsHistoryOpen(false)}
          onSelectEntry={handleHistoryEntrySelect}
        />
      )}
    </div>
  );
}

export default App;