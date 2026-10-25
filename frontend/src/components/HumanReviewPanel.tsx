import React from 'react';
import { HumanReviewRecommendation } from '../types/analysis';
import { UserCheck, AlertTriangle, ShieldCheck, Stethoscope } from 'lucide-react';

interface HumanReviewPanelProps {
  humanReview: HumanReviewRecommendation;
}

export const HumanReviewPanel: React.FC<HumanReviewPanelProps> = ({ humanReview }) => {
  const requiresReview = humanReview.requires_human_attention;

  return (
    <div className={`rounded-2xl p-4 sm:p-6 border transition-all ${
      requiresReview 
        ? 'bg-amber-950/20 border-amber-800/60 shadow-lg shadow-amber-950/20'
        : 'glass-panel border-slate-800'
    }`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div className="flex items-center space-x-3">
          <div className={`h-10 w-10 rounded-xl flex items-center justify-center shrink-0 ${
            requiresReview ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
          }`}>
            <UserCheck className="h-6 w-6" />
          </div>
          <div>
            <h3 className="font-semibold text-white text-base">Human Specialist Review Advisory</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Automated clinical safety gating and confidence triage for radiologists.
            </p>
          </div>
        </div>

        {/* Status Badge */}
        <div className={`px-3 py-1.5 rounded-full text-xs font-semibold border flex items-center space-x-2 ${
          requiresReview
            ? 'bg-amber-950 text-amber-300 border-amber-700 animate-pulse'
            : 'bg-emerald-950 text-emerald-300 border-emerald-800'
        }`}>
          {requiresReview ? <AlertTriangle className="h-3.5 w-3.5" /> : <ShieldCheck className="h-3.5 w-3.5" />}
          <span>{humanReview.review_status}</span>
        </div>
      </div>

      {/* Trigger Reasons */}
      <div className="mt-4">
        {humanReview.reasons.length > 0 ? (
          <div>
            <span className="text-xs font-semibold text-amber-300 uppercase tracking-wider block mb-2">
              Review Recommended Due to the Following Findings:
            </span>
            <ul className="space-y-2 text-xs text-amber-100/90">
              {humanReview.reasons.map((reason, idx) => (
                <li key={idx} className="flex items-start space-x-2 p-2.5 rounded-lg bg-amber-950/40 border border-amber-800/40">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-400 shrink-0 mt-1.5" />
                  <span className="leading-relaxed">{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 text-xs text-emerald-200 flex items-center space-x-2.5">
            <ShieldCheck className="h-5 w-5 text-emerald-400 shrink-0" />
            <span>
              All segmentation and quality metrics are within normal research boundaries. No automated warnings triggered.
            </span>
          </div>
        )}
      </div>

      {/* Medical Safety Disclaimer */}
      <div className="mt-4 p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex items-start space-x-2.5 text-xs text-slate-400">
        <Stethoscope className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed text-[11px]">
          <span className="font-semibold text-slate-300">Mandatory Clinical Notice: </span>
          {humanReview.disclaimer} Never presume a scan is medically safe merely because automated warning thresholds were not triggered.
        </p>
      </div>
    </div>
  );
};
