import React, { useState } from 'react';
import { SegmentationStats } from '../types/analysis';
import { ZoomIn, ZoomOut, RotateCcw, Layers, FileImage, ShieldCheck, Activity } from 'lucide-react';

interface SegmentationViewerProps {
  originalImage: string;
  maskImage: string;
  overlayImage: string;
  transparentOverlayImage?: string;
  stats: SegmentationStats;
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({
  originalImage,
  maskImage,
  overlayImage,
  transparentOverlayImage,
  stats
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'split' | 'mask'>('overlay');
  const [opacity, setOpacity] = useState<number>(0.85);
  const [zoom, setZoom] = useState<number>(1);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.75));
  const handleResetZoom = () => setZoom(1);

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="h-5 w-5 text-cyan-400" />
            <h3 className="font-semibold text-white text-base">Tumour Segmentation & Lesion Mapping</h3>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
              nnU-Net 2D
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Pixel-level boundary delineation with adjustable overlay transparency and geometrical metrics.
          </p>
        </div>

        {/* View Mode Tabs */}
        <div className="flex items-center space-x-1 p-1 bg-slate-900 rounded-lg border border-slate-800 text-xs">
          <button
            id="tab-overlay-blended"
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              viewMode === 'overlay' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Overlay Blended
          </button>
          <button
            id="tab-side-by-side"
            onClick={() => setViewMode('split')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              viewMode === 'split' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Side-by-Side
          </button>
          <button
            id="tab-binary-mask"
            onClick={() => setViewMode('mask')}
            className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
              viewMode === 'mask' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Binary Mask
          </button>
        </div>
      </div>

      {/* Main Image Display */}
      <div className="mt-4 flex flex-col items-center">
        {viewMode === 'split' ? (
          /* Side-by-Side 3-Panel View */
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full">
            {/* Panel 1: Original MRI Scan */}
            <div className="flex flex-col items-center bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
              <div className="flex items-center space-x-1.5 mb-2 w-full justify-center">
                <span className="w-2 h-2 rounded-full bg-slate-400 shrink-0" />
                <span className="text-xs font-semibold text-slate-300 truncate">1. Original MRI Scan</span>
              </div>
              <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1">
                <img src={originalImage} alt="Original MRI Scan" className="w-full h-full object-contain rounded" />
              </div>
              <span className="text-[10px] text-slate-400 mt-1.5 text-center">Unprocessed grayscale acquisition</span>
            </div>

            {/* Panel 2: Predicted Binary Mask */}
            <div className="flex flex-col items-center bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
              <div className="flex items-center space-x-1.5 mb-2 w-full justify-center">
                <span className="w-2 h-2 rounded-full bg-purple-400 shrink-0" />
                <span className="text-xs font-semibold text-purple-300 truncate">2. Predicted Binary Mask</span>
              </div>
              <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1">
                <img src={maskImage} alt="Predicted Segmentation Mask" className="w-full h-full object-contain rounded" />
              </div>
              <span className="text-[10px] text-slate-400 mt-1.5 text-center">Pixel-level lesion delineation</span>
            </div>

            {/* Panel 3: MRI with Segmentation Overlay */}
            <div className="flex flex-col items-center bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
              <div className="flex items-center space-x-1.5 mb-2 w-full justify-center">
                <span className="w-2 h-2 rounded-full bg-cyan-400 shrink-0" />
                <span className="text-xs font-semibold text-cyan-300 truncate">3. Segmentation Overlay</span>
              </div>
              <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1">
                <img src={overlayImage} alt="MRI with Segmentation Overlay" className="w-full h-full object-contain rounded" />
              </div>
              <span className="text-[10px] text-slate-400 mt-1.5 text-center">Lesion core + gold contour</span>
            </div>
          </div>
        ) : (
          /* Single Viewport Mode (Overlay or Binary Mask) */
          <div className="w-full flex flex-col items-center">
            {/* Viewport Heading */}
            <div className="flex items-center space-x-2 mb-2 text-xs font-semibold">
              {viewMode === 'overlay' ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span className="text-cyan-300">MRI with Segmentation Overlay (Blended)</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                  <span className="text-purple-300">Predicted Segmentation Mask (Binary)</span>
                </>
              )}
            </div>

            <div className="relative w-full max-w-[420px] aspect-square rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center p-2">
              <div 
                className="relative w-full h-full flex items-center justify-center transition-transform duration-200"
                style={{ transform: `scale(${zoom})` }}
              >
                {viewMode === 'mask' ? (
                  <img src={maskImage} alt="Binary Mask" className="w-full h-full object-contain rounded-xl" />
                ) : (
                  <>
                    {/* Underlying pure grayscale MRI scan */}
                    <img src={originalImage} alt="Original Scan" className="absolute inset-0 w-full h-full object-contain rounded-xl" />
                    {/* Transparent lesion mask overlay (or fallback composite) with interactive opacity */}
                    <img 
                      src={transparentOverlayImage || overlayImage} 
                      alt="Segmentation Overlay" 
                      className="absolute inset-0 w-full h-full object-contain rounded-xl transition-opacity duration-150" 
                      style={{ opacity }}
                    />
                  </>
                )}
              </div>

              {/* Floating Zoom & Reset Toolbar */}
              <div className="absolute bottom-3 right-3 flex items-center space-x-1.5 p-1 rounded-lg bg-slate-900/90 border border-slate-800 backdrop-blur-md">
                <button
                  onClick={handleZoomOut}
                  className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors"
                  title="Zoom Out"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <span className="text-[11px] font-mono text-slate-400 px-1">{Math.round(zoom * 100)}%</span>
                <button
                  onClick={handleZoomIn}
                  className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors"
                  title="Zoom In"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
                <button
                  onClick={handleResetZoom}
                  className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors border-l border-slate-800 ml-1"
                  title="Reset View"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Opacity Slider for Overlay Mode */}
        {viewMode === 'overlay' && (
          <div className="w-full max-w-[420px] mt-4 flex items-center space-x-3 px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
            <span className="text-xs text-slate-400 font-medium whitespace-nowrap">Overlay Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <span className="text-xs font-mono text-cyan-400 w-10 text-right">{Math.round(opacity * 100)}%</span>
          </div>
        )}
      </div>

      {/* Color Legend */}
      <div className="mt-4 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-center gap-4 text-xs">
        <div className="flex items-center space-x-2">
          <span className="h-3 w-3 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
          <span className="text-slate-300 font-medium">Segmented Lesion Core (Crimson)</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="h-3 w-3 rounded-full bg-amber-400 shadow-sm shadow-amber-400/50" />
          <span className="text-slate-300 font-medium">Peritumoral Margin Contour (Gold)</span>
        </div>
      </div>

      {/* Geometrical Measurements Cards */}
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Segmented Pixels</span>
          <span className="text-base font-bold text-white font-mono mt-0.5 block">
            {stats.pixel_count.toLocaleString()} px
          </span>
        </div>
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Parenchyma Occupied</span>
          <span className="text-base font-bold text-cyan-400 font-mono mt-0.5 block">
            {stats.percentage_occupied}%
          </span>
        </div>
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Equiv. Diameter</span>
          <span className="text-base font-bold text-white font-mono mt-0.5 block">
            {stats.equivalent_diameter_px} px
          </span>
        </div>
        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[11px]">Physical Metric</span>
          <span className="text-base font-bold text-emerald-400 font-mono mt-0.5 block">
            {stats.physical_area_cm2 ? `${stats.physical_area_cm2} cm²` : (stats.physical_volume_ml ? `${stats.physical_volume_ml} mL` : (stats.pixel_count > 0 ? '2D Uncalibrated' : '0.0 cm²'))}
          </span>
        </div>
      </div>
    </div>
  );
};
