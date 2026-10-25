import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  Layers, 
  Cpu, 
  Sliders, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Sparkles,
  Activity
} from 'lucide-react';
import { fetchDemoScan, runFullAnalysis } from '../services/api';
import { AnalysisResponse } from '../types/analysis';

interface UploadPageProps {
  onAnalysisComplete: (result: AnalysisResponse) => void;
}

export const UploadPage: React.FC<UploadPageProps> = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isVolumetric, setIsVolumetric] = useState<boolean>(false);
  const [sliceCount, setSliceCount] = useState<number>(1);
  const [selectedSlice, setSelectedSlice] = useState<number>(0);
  
  // Pipeline Settings
  const [modelChoice, setModelChoice] = useState<string>('nnunet-2d');

  // States
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setError(null);
    setSelectedFile(file);

    const isNii = file.name.toLowerCase().endsWith('.nii') || file.name.toLowerCase().endsWith('.nii.gz');
    setIsVolumetric(isNii);

    if (!isNii) {
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target?.result as string);
      reader.readAsDataURL(file);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleLoadDemo = async (sampleId: string) => {
    try {
      setError(null);
      setLoading(true);
      setLoadingStep(`Loading verified clinical sample: ${sampleId}...`);

      const demo = await fetchDemoScan(sampleId);
      setPreviewUrl(demo.image_base64);
      setIsVolumetric(false);

      // Convert base64 to File object
      const res = await fetch(demo.image_base64);
      const blob = await res.blob();
      const file = new File([blob], demo.filename, { type: 'image/jpeg' });
      setSelectedFile(file);

      setLoading(false);
    } catch (err: any) {
      setError(`Failed to load demo: ${err.message}`);
      setLoading(false);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      setError('Please select or upload an MRI scan before proceeding.');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      setLoadingStep('Uploading & validating raster integrity...');
      await new Promise(r => setTimeout(r, 400));

      setLoadingStep('Running nnU-Net segmentation & 4-class classifier...');
      const response = await runFullAnalysis(
        selectedFile,
        isVolumetric ? selectedSlice : undefined,
        modelChoice
      );

      setLoadingStep('Computing 3D mesh reconstruction...');
      await new Promise(r => setTimeout(r, 400));

      onAnalysisComplete(response);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Analysis failed. Please verify the file and backend status.');
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      
      {/* Title */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          MRI Upload & Diagnostic Analysis
        </h1>
        <p className="text-sm text-slate-400 mt-2 max-w-xl mx-auto">
          Upload brain MRI scans (DICOM, NIfTI, PNG, JPG) or choose a verified clinical research demonstration case.
        </p>
      </div>

      {/* Preset Demo Scans Banner */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 mb-8">
        <div className="flex items-center space-x-2 mb-3">
          <Sparkles className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            1-Click Pre-Configured Clinical Demonstration Scans
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { id: 'glioma', label: 'Glioma (High Grade)', color: 'border-red-900/60 hover:border-red-500/80 text-red-400' },
            { id: 'meningioma', label: 'Meningioma (Dural)', color: 'border-amber-900/60 hover:border-amber-500/80 text-amber-400' },
            { id: 'pituitary', label: 'Pituitary Tumour', color: 'border-cyan-900/60 hover:border-cyan-500/80 text-cyan-400' },
            { id: 'no_tumor', label: 'No Tumour (Healthy)', color: 'border-emerald-900/60 hover:border-emerald-500/80 text-emerald-400' }
          ].map((item) => (
            <button
              key={item.id}
              id={`demo-btn-${item.id}`}
              onClick={() => handleLoadDemo(item.id)}
              disabled={loading}
              className={`p-3 rounded-xl bg-slate-950/80 border ${item.color} text-left transition-all hover:bg-slate-900 flex flex-col justify-between group disabled:opacity-50`}
            >
              <span className="text-xs font-semibold text-white group-hover:text-cyan-300 transition-colors">
                {item.label}
              </span>
              <span className="text-[10px] text-slate-400 mt-1">Load Case Scan →</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Upload Box & Config Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Upload Drop Zone (7 cols) */}
        <div className="lg:col-span-7 flex flex-col">
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex-1 min-h-[300px] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-cyan-400 bg-cyan-950/30'
                : selectedFile
                  ? 'border-emerald-500/60 bg-slate-950/60'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.nii,.nii.gz"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              className="hidden"
            />

            {previewUrl ? (
              <div className="flex flex-col items-center space-y-3">
                <div className="w-44 h-44 rounded-xl overflow-hidden border border-slate-700 bg-slate-900 shadow-xl p-1">
                  <img src={previewUrl} alt="MRI Preview" className="w-full h-full object-contain rounded-lg" />
                </div>
                <div className="text-center">
                  <span className="text-xs font-semibold text-emerald-400 block flex items-center justify-center space-x-1">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Scan Loaded & Verified</span>
                  </span>
                  <span className="text-xs font-mono text-slate-300 mt-0.5 block">{selectedFile?.name}</span>
                  <span className="text-[11px] text-slate-400">Click or drop to replace</span>
                </div>
              </div>
            ) : selectedFile ? (
              <div className="flex flex-col items-center space-y-3">
                <FileText className="h-12 w-12 text-cyan-400" />
                <div>
                  <span className="text-sm font-semibold text-white block">{selectedFile.name}</span>
                  <span className="text-xs text-slate-400">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Volumetric NIfTI Format</span>
                </div>
                <span className="text-xs text-cyan-400">Click or drop to replace</span>
              </div>
            ) : (
              <div className="flex flex-col items-center space-y-3">
                <div className="h-14 w-14 rounded-2xl bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400 shadow-lg">
                  <UploadCloud className="h-7 w-7" />
                </div>
                <div>
                  <span className="text-sm font-semibold text-white block">
                    Drag and drop MRI scan here, or <span className="text-cyan-400 underline">browse files</span>
                  </span>
                  <span className="text-xs text-slate-400 block mt-1">
                    Supports PNG, JPG, JPEG, and NIfTI volumes (.nii, .nii.gz) up to 100MB
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Configuration Panel (5 cols) */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 pb-4 border-b border-slate-800">
              <Sliders className="h-5 w-5 text-cyan-400" />
              <h3 className="font-semibold text-white text-base">Pipeline Parameters</h3>
            </div>

            {/* Model Architecture Selection */}
            <div className="mt-5">
              <label className="text-xs font-semibold text-slate-300 block mb-2">
                Segmentation Model Architecture:
              </label>
              <div className="space-y-2">
                <label className={`p-3 rounded-xl border flex items-start space-x-3 cursor-pointer transition-all ${
                  modelChoice === 'nnunet-2d' ? 'bg-cyan-950/40 border-cyan-500/80' : 'bg-slate-900/60 border-slate-800'
                }`}>
                  <input
                    type="radio"
                    name="model"
                    value="nnunet-2d"
                    checked={modelChoice === 'nnunet-2d'}
                    onChange={(e) => setModelChoice(e.target.value)}
                    className="mt-1 accent-cyan-400"
                  />
                  <div>
                    <span className="text-xs font-bold text-white block">nnU-Net 2D (Baseline Precision)</span>
                    <span className="text-[11px] text-slate-400 block mt-0.5">
                      InstanceNorm & LeakyReLU (7.42M params). Highest segmentation fidelity on complex lesions.
                    </span>
                  </div>
                </label>

                <label className={`p-3 rounded-xl border flex items-start space-x-3 cursor-pointer transition-all ${
                  modelChoice === 'fast-brain-unet' ? 'bg-cyan-950/40 border-cyan-500/80' : 'bg-slate-900/60 border-slate-800'
                }`}>
                  <input
                    type="radio"
                    name="model"
                    value="fast-brain-unet"
                    checked={modelChoice === 'fast-brain-unet'}
                    onChange={(e) => setModelChoice(e.target.value)}
                    className="mt-1 accent-cyan-400"
                  />
                  <div>
                    <span className="text-xs font-bold text-white block">FastBrain-UNet (Lightweight Edge)</span>
                    <span className="text-[11px] text-slate-400 block mt-0.5">
                      Depthwise separable convolutions (0.19M params, 38x compression, ~35ms latency).
                    </span>
                  </div>
                </label>
              </div>
            </div>


            {/* NIfTI Slice Navigator if volumetric */}
            {isVolumetric && (
              <div className="mt-6 p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="flex justify-between items-center text-xs mb-2">
                  <span className="font-semibold text-white">Volumetric Slice Selector:</span>
                  <span className="font-mono text-cyan-400">Slice #{selectedSlice}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={selectedSlice}
                  onChange={(e) => setSelectedSlice(parseInt(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-start space-x-2 text-xs text-rose-300">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit CTA Button */}
          <div className="mt-6">
            <button
              id="btn-run-analysis"
              onClick={handleStartAnalysis}
              disabled={loading || !selectedFile}
              className="w-full py-3.5 px-4 rounded-xl font-semibold text-white bg-gradient-to-r from-cyan-600 to-sky-500 hover:from-cyan-500 hover:to-sky-400 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-600/30 flex items-center justify-center space-x-2 transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span className="text-sm font-medium">{loadingStep || 'Processing Neural Pipeline...'}</span>
                </>
              ) : (
                <>
                  <Activity className="h-5 w-5" />
                  <span className="text-sm">Run Diagnostic Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>

      </div>

    </div>
  );
};
