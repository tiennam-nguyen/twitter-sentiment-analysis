import React, { useState } from 'react';
import { AlertCircle, Send, Loader2, TrendingUp, TrendingDown, Minus, HelpCircle } from 'lucide-react';

// TypeScript interfaces - Updated to match backend response
interface SentimentResult {
  text: string;
  sentiment: string; // This is what your backend actually returns
  confidence: number;
  probabilities: { // This is what your backend actually returns
    positive: number;
    negative: number;
    neutral: number;
    irrelevant: number;
  };
  processed_text: string;
}

interface ApiError {
  error: string;
}

// API configuration - Updated to match your backend port
const API_URL = 'http://localhost:8888';

const SentimentAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [examples] = useState<string[]>([
    "I absolutely love this product! Best purchase ever!",
    "This is terrible. Complete waste of money. Never buying again.",
    "It's okay, nothing special but does the job.",
    "The weather is nice today"
  ]);

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
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textToAnalyze }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SentimentResult | ApiError = await response.json();
      
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
    switch (sentiment) {
      case 'Positive':
        return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'Negative':
        return <TrendingDown className="w-6 h-6 text-red-500" />;
      case 'Neutral':
        return <Minus className="w-6 h-6 text-gray-500" />;
      case 'Irrelevant':
        return <HelpCircle className="w-6 h-6 text-blue-500" />;
      default:
        return null;
    }
  };

  // Get sentiment color
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Positive':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'Negative':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'Neutral':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case 'Irrelevant':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Format confidence percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Capitalize first letter for display
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
            Powered by Enhanced BERT Model
          </p>
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
              <button
                onClick={() => analyzeSentiment()}
                disabled={loading || !inputText.trim()}
                className="absolute bottom-3 right-3 bg-blue-500 text-white rounded-lg px-4 py-2 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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
          <p className="font-semibold mb-1">ℹ️ About this model:</p>
          <ul className="space-y-1 text-xs">
            <li>• Enhanced BERT model trained on Twitter sentiment data</li>
            <li>• Classifies text into: Positive, Negative, Neutral, or Irrelevant</li>
            <li>• Handles emojis, hashtags, and informal language</li>
            <li>• Best with English text under 280 characters</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalyzer;
