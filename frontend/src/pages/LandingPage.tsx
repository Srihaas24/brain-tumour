import React from 'react';
import { 
  Brain, 
  Layers, 
  Activity, 
  Box, 
  ShieldCheck, 
  Zap, 
  ArrowRight, 
  UploadCloud, 
  FileCheck2, 
  Eye, 
  Stethoscope, 
  AlertTriangle 
} from 'lucide-react';

interface LandingPageProps {
  onStartAnalysis: () => void;
  onExploreCapabilities: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onStartAnalysis,
  onExploreCapabilities
}) => {
  return (
    <div className="w-full flex flex-col items-center">
      
      {/* Hero Section */}
      <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20 text-center flex flex-col items-center">
        
        {/* Research Pill Badge */}
        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-cyan-950/80 border border-cyan-800/80 text-cyan-400 text-xs font-semibold mb-6 shadow-sm">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>Next-Generation Medical Neuroimaging Research Architecture</span>
        </div>

        {/* Title & Subtitle */}
        <h1 className="text-4xl sm:text-6xl font-extrabold text-white tracking-tight max-w-4xl leading-tight">
          NeuroVision<span className="text-cyan-400">.Vision</span>
        </h1>
        <p className="mt-4 text-xl sm:text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-sky-200 to-slate-200 max-w-3xl">
          Brain Tumour Segmentation & Quality Assessment
        </p>

        <p className="mt-4 text-slate-400 text-base max-w-2xl leading-relaxed">
          A modular deep learning framework combining trained 2D nnU-Net segmentation, 
          automated 8-point scan quality auditing, and interactive 3D WebGL anatomical reconstruction.
        </p>

        {/* CTA Buttons */}
        <div className="mt-8 flex flex-col sm:flex-row items-center gap-4">
          <button
            onClick={onStartAnalysis}
            className="w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-cyan-600 to-sky-500 hover:from-cyan-500 hover:to-sky-400 shadow-xl shadow-cyan-600/30 flex items-center justify-center space-x-2 transform hover:-translate-y-0.5 transition-all"
          >
            <span>Start MRI Analysis</span>
            <ArrowRight className="h-5 w-5" />
          </button>

          <button
            onClick={onExploreCapabilities}
            className="w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-slate-300 hover:text-white bg-slate-900/80 hover:bg-slate-800 border border-slate-700/80 transition-all"
          >
            Explore Capabilities
          </button>
        </div>

        {/* Clinical Disclaimer Notice */}
        <div className="mt-12 p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 max-w-2xl flex items-center space-x-3 text-xs text-slate-400 text-left">
          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
          <span>
            <strong className="text-slate-300">Research Prototype Notice: </strong>
            Designed strictly for scientific investigation, technical benchmarking, and educational demonstrations. Not certified for diagnostic use.
          </span>
        </div>
      </section>

      {/* Clinical Workflow Pipeline Illustration */}
      <section className="w-full bg-slate-950/60 border-y border-slate-800/80 py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-xs font-semibold text-cyan-400 uppercase tracking-widest">
              Standardized Diagnostic Pipeline
            </h2>
            <p className="text-2xl font-bold text-white mt-1">
              End-to-End Uncertainty-Guided Radiomics Workflow
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative">
            
            {/* Step 1 */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-xl bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400 mb-4">
                <UploadCloud className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider font-semibold">Step 01</span>
              <h3 className="text-base font-bold text-white mt-1">Upload & Validate</h3>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                2D DICOM/PNG/JPG or 3D NIfTI volumetric data decoding with slice extraction.
              </p>
            </div>

            {/* Step 2 */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-xl bg-sky-950 border border-sky-800 flex items-center justify-center text-sky-400 mb-4">
                <Brain className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-mono text-sky-400 uppercase tracking-wider font-semibold">Step 02</span>
              <h3 className="text-base font-bold text-white mt-1">Tumour Classification</h3>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                4-class tumor typing across Glioma, Meningioma, Pituitary, and Healthy Parenchyma.
              </p>
            </div>

            {/* Step 3 */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-xl bg-emerald-950 border border-emerald-800 flex items-center justify-center text-emerald-400 mb-4">
                <Layers className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-mono text-emerald-400 uppercase tracking-wider font-semibold">Step 03</span>
              <h3 className="text-base font-bold text-white mt-1">nnU-Net Segmentation</h3>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                Sub-pixel lesion boundary delineation with clinical contour extraction.
              </p>
            </div>

            {/* Step 4 */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-xl bg-amber-950 border border-amber-800 flex items-center justify-center text-amber-400 mb-4">
                <Activity className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-mono text-amber-400 uppercase tracking-wider font-semibold">Step 04</span>
              <h3 className="text-base font-bold text-white mt-1">QA & Validation</h3>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                8-point automated image integrity audit covering sharpness, contrast, domain fit and lesion plausibility.
              </p>
            </div>

            {/* Step 5 */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-xl bg-rose-950 border border-rose-800 flex items-center justify-center text-rose-400 mb-4">
                <Stethoscope className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-mono text-rose-400 uppercase tracking-wider font-semibold">Step 05</span>
              <h3 className="text-base font-bold text-white mt-1">Human Review</h3>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                Triage recommendations, calibrated spatial measurements, and 3D STL export.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* Feature Capabilities Grid */}
      <section id="capabilities-section" className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-12">
          <h2 className="text-xs font-semibold text-cyan-400 uppercase tracking-widest">
            Core System Capabilities
          </h2>
          <p className="text-3xl font-extrabold text-white mt-1">
            State-of-the-Art Medical Architecture
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          {/* Card 1 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-cyan-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-cyan-950/80 border border-cyan-800 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-105 transition-transform">
              <Brain className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Tumour Classification</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              4-class neural network categorizes brain MRI scans into Glioma, Meningioma, Pituitary Tumour, or No Tumour with probability distributions and clinical action guidelines.
            </p>
          </div>

          {/* Card 2 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-emerald-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-emerald-950/80 border border-emerald-800 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-105 transition-transform">
              <Layers className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Precision nnU-Net Segmentation</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Trained on the official BRISC 2025 dataset with InstanceNorm, strided convolutions, and paired Z-score intensity normalization. Generates sharp tumor overlays with gold border contours.
            </p>
          </div>

          {/* Card 3 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-amber-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-amber-950/80 border border-amber-800 flex items-center justify-center text-amber-400 mb-4 group-hover:scale-105 transition-transform">
              <Activity className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Automated Quality Assessment</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Rigorous 8-point check evaluates motion blur (Laplacian variance), tissue contrast dynamic range, cranial domain fit, sequence recognition, and lesion spatial plausibility. Flags scans that require specialist review.
            </p>
          </div>

          {/* Card 4 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-sky-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-sky-950/80 border border-sky-800 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-105 transition-transform">
              <Box className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Interactive 3D Reconstruction</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Marching Cubes polygon surface extraction creates watertight 3D tumor models inside an anatomical cerebral reference. Supports full rotation, anatomical views, and STL/OBJ export.
            </p>
          </div>

          {/* Card 5 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-purple-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-purple-950/80 border border-purple-800 flex items-center justify-center text-purple-400 mb-4 group-hover:scale-105 transition-transform">
              <ShieldCheck className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Automated Quality Assessment</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Rigorous 8-point check evaluates motion blur (Laplacian variance), contrast dynamic range, cranial domain fit, sequence recognition, and lesion component plausibility.
            </p>
          </div>

          {/* Card 6 */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 hover:border-cyan-500/40 transition-all group">
            <div className="h-12 w-12 rounded-xl bg-cyan-950/80 border border-cyan-800 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-105 transition-transform">
              <Zap className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Lightweight Edge Inference</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Includes FastBrain-UNet, a depthwise separable model achieving 38x parameter compression and under 35ms latency for mobile MRI units and tablet review.
            </p>
          </div>

        </div>
      </section>

    </div>
  );
};
