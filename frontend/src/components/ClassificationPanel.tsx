import React from 'react';
import { ClassificationResult } from '../types/analysis';
import { Activity, Clock, ShieldAlert, CheckCircle, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ClassificationPanelProps {
  classification: ClassificationResult;
}

export const ClassificationPanel: React.FC<ClassificationPanelProps> = ({ classification }) => {
  const chartData = classification.probabilities.map(p => ({
    name: p.display_name,
    percentage: p.percentage,
    probability: p.probability,
    classId: p.class_id
  }));

  const getBarColor = (classId: string) => {
    switch (classId) {
      case 'glioma': return '#ef4444'; // Red
      case 'meningioma': return '#f59e0b'; // Amber
      case 'pituitary': return '#38bdf8'; // Cyan
      case 'no_tumor': return '#10b981'; // Green
      default: return '#64748b';
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-cyan-400" />
            <h3 className="font-semibold text-white text-base">Tumour Classification & Typing</h3>
            <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono ${
              classification.is_trained_model 
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                : 'bg-cyan-950 text-cyan-400 border-cyan-800'
            }`}>
              {classification.status_label}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Convolutional neural network 4-class multi-category probability distribution.
          </p>
        </div>

        {/* Latency Badge */}
        <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
          <Clock className="h-3.5 w-3.5 text-cyan-400" />
          <span>{classification.inference_time_ms} ms</span>
        </div>
      </div>

      {/* Top Prediction Card */}
      <div className="mt-4 p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
            Primary Prediction
          </span>
          <div className="flex items-center space-x-3 mt-1">
            <h2 className="text-2xl font-bold text-white tracking-tight">
              {classification.display_name}
            </h2>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
              classification.predicted_class === 'no_tumor'
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                : 'bg-red-950/80 text-red-400 border-red-800/80'
            }`}>
              {classification.severity}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-2 max-w-xl leading-relaxed">
            {classification.description}
          </p>
        </div>

        {/* Confidence Percentage Badge */}
        <div className="flex flex-col items-end shrink-0">
          <span className="text-[11px] text-slate-400 font-medium">Confidence Score</span>
          <span className="text-3xl font-extrabold text-cyan-400 font-mono">
            {classification.confidence}%
          </span>
          <span className="text-[10px] text-slate-400 mt-0.5">Statistical Likelihood</span>
        </div>
      </div>

      {/* Probability Bar Chart */}
      <div className="mt-6">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
          Comparative Class Probability Distribution
        </h4>
        <div className="h-44 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
              <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" fontSize={11} />
              <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={100} />
              <Tooltip 
                formatter={(value: any) => [`${value}%`, 'Probability']}
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
              />
              <Bar dataKey="percentage" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.classId)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Suggested Clinical Actions */}
      <div className="mt-4 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-start space-x-3 text-xs">
        <ShieldAlert className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-200">Recommended Next Clinical Steps: </span>
          <span className="text-slate-400">{classification.clinical_actions}</span>
        </div>
      </div>

      {/* Distinction Note */}
      <div className="mt-2 text-[11px] text-slate-400 flex items-center space-x-1.5 px-1">
        <Info className="h-3.5 w-3.5 text-slate-400 shrink-0" />
        <span>
          Distinguish classifier confidence from clinical certainty. A high statistical percentage does not guarantee an absolute anatomical diagnosis.
        </span>
      </div>
    </div>
  );
};
