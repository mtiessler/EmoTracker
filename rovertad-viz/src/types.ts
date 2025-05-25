export interface TemporalVAD {
  x: number[];
  v: (number | null)[];
  a: (number | null)[];
  d: (number | null)[];
}

export interface SenseInfo {
  sense_id: string;
  definition: string;
  vad: [number, number, number];
  y?: (number | null)[];
  y_fitting?: (number | null)[];
}

export interface WordSenses {
  [senseId: string]: {
    definition: string;
    vad: [number, number, number];
    y?: (number | null)[];
    y_fitting?: (number | null)[];
  };
}

export interface SpectrogramData {
  years: number[];
  sense_ids: string[];
  proportions_matrix: (number | null)[][];
}

export interface WordData {
  temporal_vad: TemporalVAD;
  senses: WordSenses;
  spectrogram_data?: SpectrogramData;
}

export interface LoadedData {
  [word: string]: WordData;
}

export type VizType = '2D-V' | '2D-A' | '2D-D' | '2D-VAD' | '3D' | '4D' | 'LSTM-Forecast';

export interface CombinedDataPoint {
  time: number;
  [dataKey: string]: number | null;
}

export interface OptionType {
  value: string;
  label: string;
}

export interface PredictedVadPoint {
  time: number;
  v: number;
  a: number;
  d: number;
}