import React from 'react';
import { AnalysisResponse } from '../types/analysis';
import { getExportDownloadUrl, getMeshDownloadUrl } from '../services/api';
import { X, FileJson, FileSpreadsheet, Image as ImageIcon, Box, Printer, Download } from 'lucide-react';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  analysis: AnalysisResponse;
}

export const ExportModal: React.FC<ExportModalProps> = ({ isOpen, onClose, analysis }) => {
  if (!isOpen) return null;

  const handlePrintReport = () => {
    window.print();
  };

  const handleDownloadOverlayPng = () => {
    const a = document.createElement('a');
    a.href = analysis.overlay_base64;
    a.download = `neurovision_overlay_${analysis.result_id}.png`;
    a.click();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div className="glass-panel w-full max-w-xl rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl p-6 relative">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Title */}
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-10 w-10 rounded-xl bg-cyan-600/20 text-cyan-400 flex items-center justify-center">
            <Download className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Export Diagnostic & Research Data</h3>
            <p className="text-xs text-slate-400">
              Session Case ID: <span className="font-mono text-cyan-400">{analysis.result_id}</span> | {analysis.filename}
            </p>
          </div>
        </div>

        {/* Export Options Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-6">
          
          {/* JSON Export */}
          <a
            href={getExportDownloadUrl(analysis.result_id, 'json')}
            download
            className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-cyan-700/60 transition-all flex items-start space-x-3 group"
          >
            <div className="p-2 rounded-lg bg-cyan-950 text-cyan-400 group-hover:scale-105 transition-transform">
              <FileJson className="h-5 w-5" />
            </div>
            <div>
              <span className="font-semibold text-sm text-white block">JSON Data Package</span>
              <span className="text-[11px] text-slate-400 block mt-0.5">Raw pipeline metadata, class probabilities & QA array</span>
            </div>
          </a>

          {/* CSV Export */}
          <a
            href={getExportDownloadUrl(analysis.result_id, 'csv')}
            download
            className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-cyan-700/60 transition-all flex items-start space-x-3 group"
          >
            <div className="p-2 rounded-lg bg-emerald-950 text-emerald-400 group-hover:scale-105 transition-transform">
              <FileSpreadsheet className="h-5 w-5" />
            </div>
            <div>
              <span className="font-semibold text-sm text-white block">CSV Metrics Sheet</span>
              <span className="text-[11px] text-slate-400 block mt-0.5">Pixel counts, boundary variance & dimension metrics</span>
            </div>
          </a>

          {/* Overlay Image PNG */}
          <button
            onClick={handleDownloadOverlayPng}
            className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-cyan-700/60 transition-all flex items-start space-x-3 group text-left"
          >
            <div className="p-2 rounded-lg bg-red-950 text-red-400 group-hover:scale-105 transition-transform">
              <ImageIcon className="h-5 w-5" />
            </div>
            <div>
              <span className="font-semibold text-sm text-white block">Segmentation PNG</span>
              <span className="text-[11px] text-slate-400 block mt-0.5">High-contrast clinical overlay with gold margin contour</span>
            </div>
          </button>

          {/* 3D STL Mesh */}
          {analysis.mesh_3d.has_mesh && (
            <a
              href={getMeshDownloadUrl(analysis.result_id, 'stl')}
              download
              className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-cyan-700/60 transition-all flex items-start space-x-3 group"
            >
              <div className="p-2 rounded-lg bg-purple-950 text-purple-400 group-hover:scale-105 transition-transform">
                <Box className="h-5 w-5" />
              </div>
              <div>
                <span className="font-semibold text-sm text-white block">3D STL Surface Mesh</span>
                <span className="text-[11px] text-slate-400 block mt-0.5">Polygonal mesh file compatible with 3D Slicer & Blender</span>
              </div>
            </a>
          )}

          {/* Printable Report */}
          <button
            onClick={handlePrintReport}
            className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-800/60 hover:bg-cyan-900/40 transition-all flex items-start space-x-3 sm:col-span-2 text-left"
          >
            <div className="p-2 rounded-lg bg-cyan-600 text-white">
              <Printer className="h-5 w-5" />
            </div>
            <div>
              <span className="font-semibold text-sm text-white block">Print Clinical Research Summary Report</span>
              <span className="text-[11px] text-cyan-200 block mt-0.5">Generates clean printable PDF/paper document with all case metrics and disclaimers</span>
            </div>
          </button>

        </div>

      </div>
    </div>
  );
};
