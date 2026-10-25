import React from 'react';
import { Activity, Brain, Cpu, FileText } from 'lucide-react';

interface NavbarProps {
  currentTab: 'home' | 'upload' | 'dashboard';
  setCurrentTab: (tab: 'home' | 'upload' | 'dashboard') => void;
  backendHealthy: boolean;
  onOpenBenchmark: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  backendHealthy,
  onOpenBenchmark
}) => {
  return (
    <header className="sticky top-0 z-50 w-full glass-panel border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div 
          className="flex items-center space-x-3 cursor-pointer group"
          onClick={() => setCurrentTab('home')}
        >
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-sky-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-xl tracking-tight text-white group-hover:text-cyan-400 transition-colors">
                NeuroVision<span className="text-cyan-400">.Vision</span>
              </span>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                v2.4
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Brain Tumour Segmentation & Quality Assessment
            </p>
          </div>
        </div>

        {/* Navigation & Controls */}
        <nav className="flex items-center space-x-2 sm:space-x-3">
          <button
            onClick={() => setCurrentTab('home')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'home'
                ? 'bg-slate-800 text-white shadow-sm border border-slate-700'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            Overview
          </button>

          <button
            onClick={() => setCurrentTab('upload')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              currentTab === 'upload' || currentTab === 'dashboard'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            MRI Analysis
          </button>

          <button
            onClick={onOpenBenchmark}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-300 hover:text-white bg-slate-900/90 border border-slate-800 hover:border-slate-700 transition-all"
            title="Benchmark Baseline vs Lightweight Edge Models"
          >
            <Cpu className="h-4 w-4 text-cyan-400" />
            <span className="hidden md:inline">Model Benchmark</span>
          </button>

          {/* Backend Health Badge */}
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs">
            <span className={`h-2 w-2 rounded-full ${backendHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-400 text-[11px] font-mono hidden sm:inline">
              {backendHealthy ? 'Engine Online' : 'Engine Offline'}
            </span>
          </div>
        </nav>

      </div>
    </header>
  );
};
