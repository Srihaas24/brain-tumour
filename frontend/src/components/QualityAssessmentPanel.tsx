import React from 'react';
import { QualityAssessmentResult } from '../types/analysis';
import { CheckCircle2, AlertTriangle, XCircle, ShieldCheck, FileCheck } from 'lucide-react';

interface QualityAssessmentPanelProps {
  quality: QualityAssessmentResult;
}

export const QualityAssessmentPanel: React.FC<QualityAssessmentPanelProps> = ({ quality }) => {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'PASS':
        return (
          <span className="flex items-center space-x-1 text-[11px] font-semibold text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded-full border border-emerald-800">
            <CheckCircle2 className="h-3 w-3" />
            <span>PASS</span>
          </span>
        );
      case 'WARNING':
        return (
          <span className="flex items-center space-x-1 text-[11px] font-semibold text-amber-400 bg-amber-950/80 px-2 py-0.5 rounded-full border border-amber-800">
            <AlertTriangle className="h-3 w-3" />
            <span>WARNING</span>
          </span>
        );
      default:
        return (
          <span className="flex items-center space-x-1 text-[11px] font-semibold text-rose-400 bg-rose-950/80 px-2 py-0.5 rounded-full border border-rose-800">
            <XCircle className="h-3 w-3" />
            <span>FAIL</span>
          </span>
        );
    }
  };

  const isAcceptable = quality.overall_status === 'Acceptable for review';

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <ShieldCheck className="h-5 w-5 text-cyan-400" />
            <h3 className="font-semibold text-white text-base">Automated Quality Assessment & Integrity</h3>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
              8-Point Checklist
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Rule-based multi-stage evaluation assessing input suitability and segmentation reliability.
          </p>
        </div>

        {/* Overall Status Banner */}
        <div className={`px-3 py-1.5 rounded-xl border flex items-center space-x-2 text-xs font-semibold ${
          isAcceptable
            ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800'
            : 'bg-amber-950/60 text-amber-300 border-amber-800'
        }`}>
          {isAcceptable ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
          <span>{quality.overall_status}</span>
        </div>
      </div>

      {/* Suggested Action Summary */}
      <div className="mt-4 p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-start space-x-3 text-xs">
        <FileCheck className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-200">Recommended Action: </span>
          <span className="text-slate-400">{quality.recommendation}</span>
        </div>
      </div>

      {/* 8-Point Checklist Grid */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {quality.checks.map((check) => (
          <div key={check.id} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-start justify-between gap-2 text-xs">
            <div>
              <span className="font-semibold text-slate-200 block">{check.name}</span>
              <p className="text-slate-400 text-[11px] mt-0.5 leading-relaxed">{check.detail}</p>
            </div>
            <div className="shrink-0">{getStatusBadge(check.status)}</div>
          </div>
        ))}
      </div>

      {/* Warning List if any */}
      {quality.warnings.length > 0 && (
        <div className="mt-4 p-3.5 rounded-xl bg-amber-950/30 border border-amber-800/40 text-xs">
          <span className="font-semibold text-amber-300 block mb-1">Automated Quality Notices:</span>
          <ul className="list-disc list-inside space-y-1 text-amber-200/90 text-[11px]">
            {quality.warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
