'use client';

import { useState } from 'react';
import FileUpload from '@/components/FileUpload';
import YouTubeInput from '@/components/YouTubeInput';
import TabDisplay from '@/components/TabDisplay';

// Mock tab generation - simulates API call
const mockGenerateTabs = async (source: string): Promise<{ tabs: string; title: string }> => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2500));
  
  // Mock guitar tabs
  const mockTabs = `
╔════════════════════════════════════════════════════════════╗
║  Generated Guitar Tabs                                      ║
║  Source: ${source.slice(0, 40)}...                          
╚════════════════════════════════════════════════════════════╝

Standard Tuning: E A D G B e

[Intro]
e|---0---0---0---0---|---0---0---0---0---|
B|---1---1---1---1---|---1---1---1---1---|
G|---0---0---0---0---|---2---2---2---2---|
D|---2---2---2---2---|---2---2---2---2---|
A|---3---3---3---3---|---0---0---0---0---|
E|-------------------|-------------------|

[Verse 1]
e|---0-------0-------|---3-------3-------|
B|-----1-------1-----|-----0-------0-----|
G|-------0-------0---|-------0-------0---|
D|---------2---------|-------0-----------|
A|---3---------------|---2---------------|
E|-------------------|---3---------------|

e|---0-------0-------|---1-------1-------|
B|-----1-------1-----|-----1-------1-----|
G|-------0-------0---|-------2-------2---|
D|---------2---------|-------3-----------|
A|---3---------------|---3---------------|
E|-------------------|-------------------|

[Chorus]
e|---3---3---0---0---|---0---0---1---1---|
B|---0---0---1---1---|---1---1---1---1---|
G|---0---0---0---0---|---2---2---2---2---|
D|---0---0---2---2---|---2---2---3---3---|
A|---2---2---3---3---|---0---0---3---3---|
E|---3---3-----------|-------------------|

[Bridge]
e|---0h2p0-----------|---0h2p0-----------|
B|---------3---------|----------3--------|
G|-----------0-------|------------0------|
D|-------------------|-------------------|
A|---3---------------|---3---------------|
E|-------------------|-------------------|

Legend:
  h = hammer-on
  p = pull-off
  / = slide up
  \\ = slide down
  b = bend
  r = release
  ~ = vibrato

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated with GuitarTabs AI
`.trim();

  return {
    tabs: mockTabs,
    title: source.includes('youtube') ? 'YouTube Song' : 'Uploaded Audio',
  };
};

export default function Home() {
  const [tabs, setTabs] = useState<string | null>(null);
  const [songTitle, setSongTitle] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'youtube'>('upload');

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setTabs(null);
    
    try {
      const result = await mockGenerateTabs(file.name);
      setTabs(result.tabs);
      setSongTitle(file.name.replace(/\.[^/.]+$/, ''));
    } catch (err) {
      setError('Failed to process audio file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleYouTubeSubmit = async (url: string) => {
    setIsLoading(true);
    setError(null);
    setTabs(null);
    
    try {
      const result = await mockGenerateTabs(url);
      setTabs(result.tabs);
      setSongTitle(result.title);
    } catch (err) {
      setError('Failed to process YouTube video. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-600/5 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-4xl mx-auto px-4 pt-16 pb-12">
          {/* Logo / Title */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center gap-3 mb-6">
              <div className="p-3 bg-emerald-500/10 rounded-2xl border border-emerald-500/20">
                <svg className="w-10 h-10 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
            </div>
            
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Guitar<span className="text-emerald-500">Tabs</span> AI
            </h1>
            
            <p className="text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed">
              Transform any song into guitar tablature using AI. 
              Upload an audio file or paste a YouTube link and get accurate tabs in seconds.
            </p>
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
            {[
              {
                icon: (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                ),
                title: 'Lightning Fast',
                desc: 'Get tabs in seconds, not hours',
              },
              {
                icon: (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ),
                title: 'High Accuracy',
                desc: 'Advanced AI note detection',
              },
              {
                icon: (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                ),
                title: 'Easy Export',
                desc: 'Copy or download your tabs',
              },
            ].map((feature) => (
              <div 
                key={feature.title}
                className="flex items-start gap-3 p-4 rounded-xl bg-zinc-800/30 border border-zinc-700/50"
              >
                <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
                  {feature.icon}
                </div>
                <div>
                  <p className="text-zinc-200 font-medium text-sm">{feature.title}</p>
                  <p className="text-zinc-500 text-xs mt-0.5">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Input Section */}
          <div className="bg-zinc-900/50 backdrop-blur-sm border border-zinc-800 rounded-2xl p-6 mb-8">
            {/* Tab switcher */}
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setActiveTab('upload')}
                className={`
                  flex-1 py-3 px-4 rounded-xl font-medium text-sm transition-all
                  ${activeTab === 'upload'
                    ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                    : 'bg-zinc-800/50 text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800'
                  }
                `}
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Upload Audio
                </span>
              </button>
              <button
                onClick={() => setActiveTab('youtube')}
                className={`
                  flex-1 py-3 px-4 rounded-xl font-medium text-sm transition-all
                  ${activeTab === 'youtube'
                    ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                    : 'bg-zinc-800/50 text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800'
                  }
                `}
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                  </svg>
                  YouTube URL
                </span>
              </button>
            </div>

            {/* Input components */}
            <div className="min-h-[180px]">
              {activeTab === 'upload' ? (
                <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />
              ) : (
                <YouTubeInput onSubmit={handleYouTubeSubmit} isLoading={isLoading} />
              )}
            </div>
          </div>

          {/* Tab Display */}
          <TabDisplay 
            tabs={tabs} 
            isLoading={isLoading} 
            error={error}
            songTitle={songTitle}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 py-8 mt-16">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-zinc-600 text-sm">
            Built with ❤️ and AI • Guitar Tabs Generator
          </p>
        </div>
      </footer>
    </main>
  );
}
