'use client';

import { useState } from 'react';

interface TabDisplayProps {
  tabs: string | null;
  isLoading: boolean;
  error: string | null;
  songTitle?: string;
}

export default function TabDisplay({ tabs, isLoading, error, songTitle }: TabDisplayProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (tabs) {
      await navigator.clipboard.writeText(tabs);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownload = () => {
    if (tabs) {
      const blob = new Blob([tabs], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${songTitle || 'guitar-tabs'}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  if (isLoading) {
    return (
      <div className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl p-8">
        <div className="flex flex-col items-center justify-center gap-4">
          <div className="relative">
            <div className="w-12 h-12 border-4 border-zinc-700 border-t-emerald-500 rounded-full animate-spin" />
          </div>
          <div className="text-center">
            <p className="text-zinc-300 font-medium">Analyzing audio...</p>
            <p className="text-zinc-500 text-sm mt-1">This may take a moment</p>
          </div>
          <div className="flex gap-2 mt-2">
            {['Detecting notes', 'Building tabs', 'Formatting output'].map((step, i) => (
              <span 
                key={step}
                className="text-xs px-3 py-1 rounded-full bg-zinc-800 text-zinc-500"
                style={{ animationDelay: `${i * 0.5}s` }}
              >
                {step}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full bg-red-900/20 border border-red-800/50 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-red-500/20 rounded-lg">
            <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <p className="text-red-400 font-medium">Error generating tabs</p>
            <p className="text-red-300/70 text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!tabs) {
    return (
      <div className="w-full bg-zinc-900/30 border border-zinc-800/50 rounded-xl p-12">
        <div className="flex flex-col items-center justify-center text-center">
          <div className="p-4 bg-zinc-800/50 rounded-2xl mb-4">
            <svg className="w-10 h-10 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          </div>
          <p className="text-zinc-400">Your generated tabs will appear here</p>
          <p className="text-zinc-600 text-sm mt-2">Upload an audio file or paste a YouTube link to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-zinc-800/50 border-b border-zinc-700/50">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/80" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <div className="w-3 h-3 rounded-full bg-green-500/80" />
          </div>
          {songTitle && (
            <span className="text-zinc-400 text-sm font-medium">{songTitle}</span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg
              bg-zinc-700/50 hover:bg-zinc-700 text-zinc-300 transition-colors"
          >
            {copied ? (
              <>
                <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg
              bg-zinc-700/50 hover:bg-zinc-700 text-zinc-300 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download
          </button>
        </div>
      </div>
      
      {/* Tab content */}
      <div className="p-6 overflow-x-auto">
        <pre className="font-mono text-sm text-emerald-400 leading-relaxed whitespace-pre">
          {tabs}
        </pre>
      </div>
    </div>
  );
}
