import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { LandingPage } from './pages/LandingPage';
import { UploadPage } from './pages/UploadPage';
import { DashboardPage } from './pages/DashboardPage';
import { ModelBenchmarkModal } from './components/ModelBenchmarkModal';
import { checkBackendHealth } from './services/api';
import { AnalysisResponse } from './types/analysis';

export function App() {
  const [currentTab, setCurrentTab] = useState<'home' | 'upload' | 'dashboard'>('home');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [backendHealthy, setBackendHealthy] = useState<boolean>(true);
  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState<boolean>(false);

  useEffect(() => {
    checkBackendHealth()
      .then(() => setBackendHealthy(true))
      .catch(() => setBackendHealthy(false));

    const interval = setInterval(() => {
      checkBackendHealth()
        .then(() => setBackendHealthy(true))
        .catch(() => setBackendHealthy(false));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const handleAnalysisComplete = (result: AnalysisResponse) => {
    setAnalysisResult(result);
    setCurrentTab('dashboard');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleStartAnalysis = () => {
    setCurrentTab('upload');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleExploreCapabilities = () => {
    const el = document.getElementById('capabilities-section');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    } else {
      setCurrentTab('home');
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col justify-between selection:bg-cyan-500 selection:text-white">
      
      {/* Background Radial Glow */}
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(56,189,248,0.12),rgba(255,255,255,0))]" />

      {/* Navigation */}
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        backendHealthy={backendHealthy}
        onOpenBenchmark={() => setIsBenchmarkOpen(true)}
      />

      {/* Main Content Pages */}
      <main className="flex-1 relative z-10">
        {currentTab === 'home' && (
          <LandingPage
            onStartAnalysis={handleStartAnalysis}
            onExploreCapabilities={handleExploreCapabilities}
          />
        )}

        {currentTab === 'upload' && (
          <UploadPage onAnalysisComplete={handleAnalysisComplete} />
        )}

        {currentTab === 'dashboard' && analysisResult && (
          <DashboardPage
            analysis={analysisResult}
            onNewAnalysis={() => setCurrentTab('upload')}
          />
        )}
      </main>

      {/* Benchmark Modal */}
      <ModelBenchmarkModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />

      {/* Footer */}
      <Footer />

    </div>
  );
}

export default App;
