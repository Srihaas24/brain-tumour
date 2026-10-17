import React, { useState } from 'react';
import { AnalysisResponse } from '../types/analysis';
import { ClassificationPanel } from '../components/ClassificationPanel';
import { SegmentationViewer } from '../components/SegmentationViewer';
import { UncertaintyViewer } from '../components/UncertaintyViewer';
import { Brain3DViewer } from '../components/Brain3DViewer';
import { QualityAssessmentPanel } from '../components/QualityAssessmentPanel';
import { HumanReviewPanel } from '../components/HumanReviewPanel';
import { ExportModal } from '../components/ExportModal';
import { 
  ArrowLeft, 
  Download, 
  Clock, 
  FileText, 
  CheckCircle2, 
  AlertTriangle,
  Layers,
  Activity,
  Box,
  ShieldCheck,
  Stethoscope
} from 'lucide-react';

interface DashboardPageProps {
  analysis: AnalysisResponse;
  onNewAnalysis: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ analysis, onNewAnalysis }) => {
  const [isExportOpen, setIsExportOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'all' | 'segmentation' | 'uncertainty' | '3d' | 'qa'>('all');

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      
      {/* Top Session Summary Bar */}
      <div className="glass-panel p-4 sm:p-5 rounded-2xl border border-slate-800 mb-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        
        {/* Left: Metadata */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onNewAnalysis}
            className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors"
            title="Upload New Scan"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400 font-mono">
                Case ID: {analysis.result_id}
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-xs text-slate-300 font-mono">{analysis.filename}</span>
              {analysis.is_volumetric && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-950 text-purple-400 border border-purple-800 font-mono">
                  3D NIfTI Volume
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Engine: <strong className="text-white">{analysis.model_used}</strong> • Total Pipeline Latency:{' '}
              <span className="text-cyan-400 font-mono">{analysis.total_pipeline_time_ms} ms</span>
            </p>
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center space-x-2 w-full md:w-auto justify-end">
          <button
            onClick={() => setIsExportOpen(true)}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all shadow-md"
          >
            <Download className="h-4 w-4 text-cyan-400" />
            <span>Export Data & Report</span>
          </button>

          <button
            onClick={onNewAnalysis}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-cyan-600 hover:bg-cyan-500 transition-all shadow-md shadow-cyan-600/30"
          >
            <span>New MRI Scan</span>
          </button>
        </div>

      </div>

      {/* Main Analysis Sections Grid */}
      <div className="space-y-8">
        
        {/* Row 1: Classification & Human Review Advisory */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7">
            <ClassificationPanel classification={analysis.classification} />
          </div>
          <div className="lg:col-span-5 flex flex-col">
            <HumanReviewPanel humanReview={analysis.human_review} />
          </div>
        </div>

        {/* Row 2: Precision Segmentation & Uncertainty Heatmap */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <SegmentationViewer
            originalImage={analysis.original_image_base64}
            maskImage={analysis.mask_base64}
            overlayImage={analysis.overlay_base64}
            transparentOverlayImage={analysis.transparent_overlay_base64}
            stats={analysis.segmentation}
          />
          <UncertaintyViewer
            uncertaintyData={analysis.uncertainty}
            originalImage={analysis.original_image_base64}
          />
        </div>

        {/* Row 3: 3D Tumour Reconstruction & Automated QA */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7">
            <Brain3DViewer meshData={analysis.mesh_3d} resultId={analysis.result_id} />
          </div>
          <div className="lg:col-span-5">
            <QualityAssessmentPanel quality={analysis.quality} />
          </div>
        </div>

      </div>

      {/* Export Modal */}
      <ExportModal
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        analysis={analysis}
      />

    </div>
  );
};
