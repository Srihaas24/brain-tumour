import React, { useState } from 'react';
import { UncertaintyResult } from '../types/analysis';
import { Activity, AlertTriangle, CheckCircle2, HelpCircle, Eye, Sliders, SplitSquareVertical } from 'lucide-react';

interface UncertaintyViewerProps {
  uncertaintyData: UncertaintyResult;
  originalImage: string;
}

export const UncertaintyViewer: React.FC<UncertaintyViewerProps> = ({
  uncertaintyData,
  originalImage
}) => {
  const [threshold, setThreshold] = useState<number>(35.0);
  const [blendMode, setBlendMode] = useState<'overlay' | 'sideBySide' | 'heatmap'>('overlay');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.85);

  const isAboveThreshold = uncertaintyData.boundary_uncertainty > threshold;
  const overlaySrc = uncertaintyData.overlay_base64 || uncertaintyData.heatmap_base64;

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-amber-400" />
            <h3 className="font-semibold text-white text-base">Monte Carlo Dropout Uncertainty Analysis</h3>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-950 text-amber-400 border border-amber-800 font-mono">
              {uncertaintyData.num_stochastic_passes} Stochastic Passes
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Predictive variance & entropy across multiple forward stochastic passes with active dropout.
          </p>
        </div>

        {/* View Switch */}
        <div className="flex items-center space-x-1 p-1 bg-slate-900 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setBlendMode('overlay')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              blendMode === 'overlay' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
            title="Anatomical MRI overlay with glowing boundary variance"
          >
            Anatomical Overlay
          </button>
          <button
            onClick={() => setBlendMode('sideBySide')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              blendMode === 'sideBySide' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
            title="Side-by-side comparison"
          >
            Comparison
          </button>
          <button
            onClick={() => setBlendMode('heatmap')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              blendMode === 'heatmap' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
            title="Pure Jet colormap"
          >
            Jet Colormap
          </button>
        </div>
      </div>

      {/* Main Visual Display */}
      <div className="mt-4 flex flex-col items-center">
        {blendMode === 'sideBySide' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
            <div className="flex flex-col items-center">
              <span className="text-xs font-semibold text-slate-300 mb-2">Original MRI Scan</span>
              <div className="w-full max-w-[340px] aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1.5">
                <img src={originalImage} alt="Original Scan" className="w-full h-full object-contain rounded-lg" />
              </div>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-xs font-semibold text-amber-400 mb-2">Calibrated Predictive Variance</span>
              <div className="w-full max-w-[340px] aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1.5">
                <img src={uncertaintyData.heatmap_base64} alt="Uncertainty Heatmap" className="w-full h-full object-contain rounded-lg" />
              </div>
            </div>
          </div>
        ) : blendMode === 'overlay' ? (
          <div className="w-full flex flex-col items-center">
            <div className="relative w-full max-w-[420px] aspect-square rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center p-2">
              {/* Base MRI */}
              <img src={originalImage} alt="Original MRI" className="absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)] object-contain rounded-xl" />
              {/* Uncertainty Overlay with Opacity */}
              <img
                src={overlaySrc}
                alt="Uncertainty Overlay"
                style={{ opacity: overlayOpacity }}
                className="absolute inset-2 w-[calc(100%-16px)] h-[calc(100%-16px)] object-contain rounded-xl mix-blend-screen transition-opacity duration-200"
              />
            </div>

            {/* Opacity Slider */}
            <div className="w-full max-w-[420px] mt-3 px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center space-x-3 text-xs">
              <Sliders className="h-4 w-4 text-amber-400 shrink-0" />
              <span className="text-slate-300 shrink-0 font-medium">Overlay Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <span className="font-mono text-amber-300 shrink-0">{Math.round(overlayOpacity * 100)}%</span>
            </div>
          </div>
        ) : (
          <div className="relative w-full max-w-[420px] aspect-square rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center p-2">
            <img src={uncertaintyData.heatmap_base64} alt="Uncertainty Heatmap" className="w-full h-full object-contain rounded-xl" />
          </div>
        )}
      </div>

      {/* Color Scale Legend */}
      <div className="mt-4 pt-3 border-t border-slate-800 flex flex-col items-center">
        <div className="w-full max-w-[420px]">
          <div className="h-3 w-full rounded-full bg-gradient-to-r from-blue-900 via-cyan-400 via-yellow-400 to-red-600 shadow-inner" />
          <div className="flex justify-between text-[11px] font-mono text-slate-400 mt-1">
            <span>Low Uncertainty (0%)</span>
            <span>Moderate (50%)</span>
            <span className="text-red-400 font-semibold">High Margin Risk (100%)</span>
          </div>
        </div>
      </div>

      {/* Uncertainty Metrics & Review Trigger */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className={`p-4 rounded-xl border flex items-center space-x-3.5 ${
          isAboveThreshold ? 'bg-amber-950/30 border-amber-800/60' : 'bg-slate-900/60 border-slate-800'
        }`}>
          {isAboveThreshold ? (
            <AlertTriangle className="h-7 w-7 text-amber-400 shrink-0" />
          ) : (
            <CheckCircle2 className="h-7 w-7 text-emerald-400 shrink-0" />
          )}
          <div>
            <span className="text-[11px] uppercase tracking-wider text-slate-400 block font-semibold">
              Tumour Boundary Uncertainty
            </span>
            <div className="flex items-baseline space-x-2 mt-0.5">
              <span className={`text-2xl font-bold font-mono ${isAboveThreshold ? 'text-amber-400' : 'text-emerald-400'}`}>
                {uncertaintyData.boundary_uncertainty}%
              </span>
              <span className="text-xs text-slate-400 font-mono">
                (Threshold: {threshold}%)
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              {isAboveThreshold 
                ? 'Margin risk exceeds threshold. Infiltrative tumor border requires manual radiologist inspection.' 
                : 'Boundary variance is within standard acceptable parameters.'}
            </p>
          </div>
        </div>

        {/* Configurable Threshold Slider */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-center">
          <div className="flex items-center justify-between text-xs mb-2">
            <span className="text-slate-300 font-medium">Configurable Review Threshold:</span>
            <span className="font-mono text-amber-400 font-bold">{threshold}%</span>
          </div>
          <input
            type="range"
            min="10"
            max="60"
            step="1"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>Strict (10%)</span>
            <span>Standard (35%)</span>
            <span>Permissive (60%)</span>
          </div>
        </div>
      </div>

      {/* Clinical Disclaimer */}
      <div className="mt-3 p-3 rounded-lg bg-slate-900/40 border border-slate-800/80 flex items-start space-x-2 text-xs text-slate-400">
        <HelpCircle className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <span className="font-semibold text-slate-300">Interpretation Note: </span>
          Uncertainty reflects epistemic model ambiguity derived from {uncertaintyData.num_stochastic_passes} stochastic passes. High uncertainty indicates morphological ambiguity or diffuse margins; it does not constitute a calibrated statistical probability of malignancy.
        </p>
      </div>
    </div>
  );
};
