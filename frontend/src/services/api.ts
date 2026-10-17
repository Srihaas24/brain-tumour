import { AnalysisResponse, BenchmarkResponse } from '../types/analysis';

const API_ORIGIN = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');
const BASE_URL = `${API_ORIGIN}/api`;

export async function checkBackendHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchDemoScan(sampleId: string) {
  const res = await fetch(`${BASE_URL}/demo/${sampleId}`);
  if (!res.ok) throw new Error(`Failed to load demo sample: ${sampleId}`);
  return res.json();
}

export async function uploadScanFile(file: File, sliceIdx?: number) {
  const formData = new FormData();
  formData.append('file', file);
  if (sliceIdx !== undefined) {
    formData.append('slice_idx', sliceIdx.toString());
  }

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function runFullAnalysis(
  file: File,
  sliceIdx?: number,
  modelChoice: string = 'nnunet-2d',
  mcDropoutPasses: number = 6
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (sliceIdx !== undefined) {
    formData.append('slice_idx', sliceIdx.toString());
  }
  formData.append('model_choice', modelChoice);
  formData.append('mc_dropout_passes', mcDropoutPasses.toString());

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Analysis request failed');
  }
  return res.json();
}

export async function fetchModelBenchmark(): Promise<BenchmarkResponse> {
  const res = await fetch(`${BASE_URL}/benchmark`, { method: 'POST' });
  if (!res.ok) throw new Error('Benchmark failed');
  return res.json();
}

export function getMeshDownloadUrl(resultId: string, format: 'stl' | 'obj') {
  return `${BASE_URL}/results/${resultId}/mesh/${format}`;
}

export function getExportDownloadUrl(resultId: string, format: 'json' | 'csv') {
  return `${BASE_URL}/results/${resultId}/export/${format}`;
}
