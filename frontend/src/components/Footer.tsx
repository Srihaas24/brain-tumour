import React from 'react';
import { AlertTriangle, ShieldCheck, HeartPulse } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950 text-slate-400 py-8 px-4 sm:px-6 lg:px-8 mt-16">
      <div className="max-w-7xl mx-auto">
        
        {/* Research Disclaimer Banner */}
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/30 mb-8 flex items-start space-x-3">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-200/90 leading-relaxed">
            <span className="font-semibold text-amber-300">Important Medical Research Disclaimer: </span>
            NeuroVision is an experimental research demonstration platform designed for technical evaluation of multi-task deep learning architectures and automated imaging quality assessment. It is <strong>not a certified medical device</strong> (FDA/CE-IVD) and <strong>does not provide medical diagnoses or treatment recommendations</strong>. All clinical scans must be independently reviewed by a qualified board-certified radiologist.
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8 text-xs">
          <div>
            <div className="flex items-center space-x-2 text-white font-semibold mb-2">
              <HeartPulse className="h-4 w-4 text-cyan-400" />
              <span>NeuroVision Architecture</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Automated 2D/3D brain MRI segmentation and quality assurance framework powered by PyTorch and FastAPI.
            </p>
          </div>

          <div>
            <span className="text-slate-200 font-semibold uppercase tracking-wider block mb-2">Machine Learning</span>
            <ul className="space-y-1.5 text-slate-400">
              <li>• 2D nnU-Net (BRISC 2025 Trained)</li>
              <li>• FastBrain-UNet (Edge Model)</li>
              <li>• MobileNetV3 4-Class Classifier</li>
              <li>• Lesion Coverage QA Check</li>
            </ul>
          </div>

          <div>
            <span className="text-slate-200 font-semibold uppercase tracking-wider block mb-2">Standards & Formats</span>
            <ul className="space-y-1.5 text-slate-400">
              <li>• DICOM / NIfTI (.nii, .nii.gz)</li>
              <li>• Marching Cubes 3D STL & OBJ</li>
              <li>• Z-Score Intensity Normalization</li>
              <li>• 8-Point QA Metric System</li>
            </ul>
          </div>

          <div>
            <span className="text-slate-200 font-semibold uppercase tracking-wider block mb-2">Security & Privacy</span>
            <div className="flex items-start space-x-2 text-slate-400">
              <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
              <span>
                Zero external telemetry. Uploaded patient scans are processed in volatile memory with automated cleanup.
              </span>
            </div>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400">
          <p>© 2026 NeuroVision Project. Developed for Advanced Medical Research & Hackathon Evaluation.</p>
          <p className="mt-2 sm:mt-0 font-mono text-[11px] text-slate-400">
            PyTorch 2.4 | FastAPI | Three.js | React 19
          </p>
        </div>

      </div>
    </footer>
  );
};
