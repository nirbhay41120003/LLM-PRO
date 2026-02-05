'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Heart, AlertCircle, Stethoscope } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  disease?: string;
  confidence?: number;
  timestamp: Date;
}

interface ApiResponse {
  response: string;
  disease?: string;
  confidence?: number;
  alternatives?: { disease: string; confidence: number }[];
  severity_analysis?: {
    severity: string;
    score: number;
  };
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check API connection on mount
  useEffect(() => {
    checkConnection();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_URL}/api/status`, { 
        method: 'GET',
        mode: 'cors',
      });
      setIsConnected(response.ok);
    } catch {
      setIsConnected(false);
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/health`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        body: JSON.stringify({ message: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data: ApiResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || 'I apologize, but I could not process your request.',
        disease: data.disease,
        confidence: data.confidence,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsConnected(true);
    } catch (error) {
      console.error('Error:', error);
      setIsConnected(false);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '⚠️ Unable to connect to the health assistant server. Please make sure the backend is running.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickSymptoms = [
    'I have fever and headache',
    'Burning sensation while urinating',
    'Skin rash with itching',
    'Persistent cough with phlegm',
    'Joint pain and swelling',
  ];

  return (
    <main className="flex flex-col h-screen max-w-5xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
            <Stethoscope className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-800 dark:text-white">
              Health Assistant
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              AI-powered medical symptom analysis
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            isConnected === null ? 'bg-yellow-400' :
            isConnected ? 'bg-green-400' : 'bg-red-400'
          }`} />
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {isConnected === null ? 'Connecting...' :
             isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="p-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl mb-6">
              <Heart className="w-12 h-12 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">
              Welcome to Health Assistant
            </h2>
            <p className="text-slate-500 dark:text-slate-400 mb-8 max-w-md">
              Describe your symptoms and I'll help analyze them using advanced AI. 
              Remember, this is for informational purposes only - always consult a doctor.
            </p>
            
            {/* Quick symptom buttons */}
            <div className="flex flex-wrap gap-2 justify-center max-w-2xl">
              {quickSymptoms.map((symptom, index) => (
                <button
                  key={index}
                  onClick={() => setInput(symptom)}
                  className="px-4 py-2 text-sm bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-full border border-slate-200 dark:border-slate-600 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  {symptom}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 message-enter ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-200 shadow-sm border border-slate-100 dark:border-slate-600'
                }`}
              >
                {message.role === 'assistant' ? (
                  <div className="markdown-content">
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                    {message.disease && message.confidence && (
                      <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-600">
                        <div className="flex items-center gap-2 text-sm">
                          <AlertCircle className="w-4 h-4 text-blue-500" />
                          <span className="font-medium">Predicted:</span>
                          <span className="text-blue-600 dark:text-blue-400">{message.disease}</span>
                          <span className="text-slate-500">({message.confidence.toFixed(1)}%)</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p>{message.content}</p>
                )}
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))
        )}

        {/* Typing indicator */}
        {isLoading && (
          <div className="flex gap-3 justify-start message-enter">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="bg-white dark:bg-slate-700 rounded-2xl px-4 py-3 shadow-sm border border-slate-100 dark:border-slate-600">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
                <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
                <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="px-4 py-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-t border-slate-200 dark:border-slate-700">
        <form onSubmit={sendMessage} className="flex gap-3 max-w-3xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your symptoms..."
            className="flex-1 px-4 py-3 bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-white rounded-xl border border-slate-200 dark:border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-slate-400"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl hover:from-blue-700 hover:to-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
        <p className="text-center text-xs text-slate-400 mt-3">
          ⚠️ This AI assistant is for informational purposes only. Always consult a healthcare professional.
        </p>
      </div>
    </main>
  );
}
