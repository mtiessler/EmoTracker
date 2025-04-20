export interface TemporalVAD {
  x: number[];
  v: number[];
  a: number[];
  d: number[];
}

export interface SenseInfo {
  definition: string;
  vad: [number, number, number];
  y_fitting?: number[];
}

export interface WordSenses {
  [senseId: string]: SenseInfo;
}

export interface SpectrogramData {
  years: number[];
  sense_ids: string[];
  proportions_matrix: number[][];
}

export interface WordData {
  temporal_vad: TemporalVAD;
  senses: WordSenses;
  spectrogram_data?: SpectrogramData;
}

export interface LoadedData {
  [word: string]: WordData;
}

export type VizType = '2D-V' | '2D-A' | '2D-D' | '2D-VAD' | '3D' | '4D' | 'Spectrogram';