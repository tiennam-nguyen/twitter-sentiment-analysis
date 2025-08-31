import React, { useState, useEffect } from 'react';
import { AlertCircle, Send, Loader2, TrendingUp, TrendingDown, Minus, HelpCircle, ChevronDown, Check } from 'lucide-react';

// TypeScript interfaces
interface SentimentResult {
  text: string;
  sentiment: string;
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
    neutral: number;
    irrelevant: number;
  };
  processed_text: string;
  inference_time: number;
  model_used: string;
}

interface ApiError {
  error: string;
  detail?: string;
}

interface ModelStatus {
  total_models: number;
  loaded_models: number;
  available_models: string[];
  failed_models: Record<string, string>;
}

// API configuration
const API_URL = 'http://localhost:8888';

// Model descriptions
const MODEL_INFO = {
  'enhanced-bert': {
    name: 'Enhanced BERT',
    description: 'BERT with attention mechanism',
    color: 'blue'
  },
  'elmo-bert': {
    name: 'ELMo + BERT',
    description: 'Combined ELMo and BERT embeddings',
    color: 'purple'
  },
  'elmo-transformer': {
    name: 'ELMo Transformer',
    description: 'Transformer with ELMo embeddings',
    color: 'indigo'
  },
  '5-embedding': {
    name: '5-Embedding',
    description: 'Multi-embedding architecture',
    color: 'green'
  }
};

const SentimentAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('enhanced-bert');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  
  const [examples] = useState<string[]>([
    "I absolutely love this product! Best purchase ever!",
    "This is terrible. Complete waste of money. Never buying again.",
    "It's okay, nothing special but does the job.",
    "The weather is nice today"
  ]);

  // Fetch available models on mount
  useEffect(() => {
    fetchModelStatus();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownOpen && !(event.target as Element).closest('.dropdown-container')) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);

  // Fetch model status
  const fetchModelStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/models/status`);
      if (response.ok) {
        const data: ModelStatus = await response.json();
        setModelStatus(data);
        setAvailableModels(data.available_models);
        
        // Set default to first available model
        if (data.available_models.length > 0 && !data.available_models.includes(selectedModel)) {
          setSelectedModel(data.available_models[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch model status:', err);
      // Fallback to all models if status endpoint fails
      setAvailableModels(['enhanced-bert', 'elmo-bert', 'elmo-transformer', '5-embedding']);
    }
  };

  // Analyze sentiment
  const analyzeSentiment = async (text?: string) => {
    const textToAnalyze = text || inputText;
    
    if (!textToAnalyze.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict/${selectedModel}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textToAnalyze }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      if ('error' in data) {
        throw new Error(data.error);
      }

      setResult(data as SentimentResult);
      if (text) {
        setInputText(text);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze sentiment');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Get sentiment icon
  const getSentimentIcon = (sentiment: string) => {
    const sentimentLower = sentiment.toLowerCase();
    switch (sentimentLower) {
      case 'positive':
        return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'negative':
        return <TrendingDown className="w-6 h-6 text-red-500" />;
      case 'neutral':
        return <Minus className="w-6 h-6 text-gray-500" />;
      case 'irrelevant':
        return <HelpCircle className="w-6 h-6 text-blue-500" />;
      default:
        return null;
    }
  };

  // Get sentiment color
  const getSentimentColor = (sentiment: string) => {
    const sentimentLower = sentiment.toLowerCase();
    switch (sentimentLower) {
      case 'positive':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'negative':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'neutral':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case 'irrelevant':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Format confidence percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Capitalize first letter
  const capitalizeFirst = (str: string) => {
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Sentiment Analyzer
          </h1>
          <p className="text-gray-600">
            Multiple Model Architectures Available
          </p>
          {modelStatus && (
            <p className="text-sm text-gray-500 mt-2">
              {modelStatus.loaded_models} of {modelStatus.total_models} models loaded
            </p>
          )}
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          {/* Input Section */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter text to analyze:
            </label>
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[120px] resize-none"
                placeholder="Type or paste your text here..."
                disabled={loading}
              />
              <div className="absolute bottom-3 right-3 flex gap-2">
                {/* Model Selector Dropdown */}
                <div className="dropdown-container relative">
                  <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="bg-gray-100 text-gray-700 rounded-lg px-3 py-2 hover:bg-gray-200 transition-colors flex items-center gap-2 text-sm font-medium"
                    disabled={loading || availableModels.length === 0}
                  >
                    {MODEL_INFO[selectedModel]?.name || selectedModel}
                    <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {dropdownOpen && (
                    <div className="absolute bottom-full mb-2 right-0 w-64 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
                      <div className="py-1">
                        {availableModels.map((model) => (
                          <button
                            key={model}
                            onClick={() => {
                              setSelectedModel(model);
                              setDropdownOpen(false);
                            }}
                            className={`w-full px-4 py-2 text-left hover:bg-gray-50 flex items-start gap-3 ${
                              selectedModel === model ? 'bg-blue-50' : ''
                            }`}
                          >
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-sm">
                                  {MODEL_INFO[model]?.name || model}
                                </span>
                                {selectedModel === model && (
                                  <Check className="w-4 h-4 text-blue-500" />
                                )}
                              </div>
                              <span className="text-xs text-gray-500">
                                {MODEL_INFO[model]?.description}
                              </span>
                            </div>
                          </button>
                        ))}
                        {modelStatus && Object.keys(modelStatus.failed_models).length > 0 && (
                          <>
                            <div className="border-t border-gray-200 mt-1 pt-1">
                              <div className="px-4 py-1 text-xs text-gray-400">Unavailable:</div>
                              {Object.keys(modelStatus.failed_models).map((model) => (
                                <div key={model} className="px-4 py-1 opacity-50">
                                  <div className="text-sm text-gray-500 line-through">
                                    {MODEL_INFO[model]?.name || model}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Analyze Button */}
                <button
                  onClick={() => analyzeSentiment()}
                  disabled={loading || !inputText.trim() || availableModels.length === 0}
                  className="bg-blue-500 text-white rounded-lg px-4 py-2 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Analyze
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Example Buttons */}
          <div className="mb-6">
            <p className="text-sm text-gray-600 mb-2">Try an example:</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => analyzeSentiment(example)}
                  className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-full transition-colors"
                  disabled={loading}
                >
                  {example.substring(0, 30)}...
                </button>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
              <div>
                <p className="text-red-800 font-medium">Error</p>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className="space-y-4 animate-in fade-in duration-500">
              {/* Main Result */}
              <div className="border-t pt-4">
                <h3 className="text-lg font-semibold mb-3">Analysis Result</h3>
                
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${getSentimentColor(result.sentiment)}`}>
                  {getSentimentIcon(result.sentiment)}
                  <span className="font-bold text-lg">{capitalizeFirst(result.sentiment)}</span>
                  <span className="text-sm opacity-75">
                    ({formatPercentage(result.confidence)} confident)
                  </span>
                </div>

                {result.inference_time && (
                  <p className="text-xs text-gray-500 mt-2">
                    Analyzed by {MODEL_INFO[result.model_used]?.name || result.model_used} in {result.inference_time.toFixed(2)}s
                  </p>
                )}
              </div>

              {/* Confidence Scores */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">All Scores:</h4>
                <div className="space-y-2">
                  {result.probabilities && Object.entries(result.probabilities).map(([sentiment, score]) => (
                    <div key={sentiment} className="flex items-center gap-2">
                      <span className="w-20 text-sm text-gray-600">{capitalizeFirst(sentiment)}:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            sentiment === 'positive' ? 'bg-green-500' :
                            sentiment === 'negative' ? 'bg-red-500' :
                            sentiment === 'neutral' ? 'bg-gray-500' :
                            'bg-blue-500'
                          }`}
                          style={{ width: `${(score as number) * 100}%` }}
                        />
                        <span className="absolute right-2 top-0 h-full flex items-center text-xs text-gray-700">
                          {formatPercentage(score as number)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Processed Text (Debug) */}
              <details className="border-t pt-3">
                <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
                  View processed text (debug)
                </summary>
                <p className="mt-2 text-xs font-mono bg-gray-100 p-2 rounded">
                  {result.processed_text}
                </p>
              </details>
            </div>
          )}
        </div>

        {/* Info Card */}
        <div className="bg-white/50 rounded-lg p-4 text-sm text-gray-600">
          <p className="font-semibold mb-1">ℹ️ About the models:</p>
          <ul className="space-y-1 text-xs">
            <li>• <strong>Enhanced BERT:</strong> BERT with attention mechanism for improved accuracy</li>
            <li>• <strong>ELMo + BERT:</strong> Combines contextual ELMo with BERT embeddings</li>
            <li>• <strong>ELMo Transformer:</strong> Alternative transformer architecture with ELMo</li>
            <li>• <strong>5-Embedding:</strong> Multi-embedding approach (BERT, ELMo, GloVe, POS, Char)</li>
            <li className="pt-1 text-gray-500">All models classify into: Positive, Negative, Neutral, or Irrelevant</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalyzer;