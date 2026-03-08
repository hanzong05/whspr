export type Agent = {
  id: string;
  name: string;
  group: string;
};

export type AnalysisResult = {
  emotion: string;
  confidence: number;
  risk: string;
  transcription?: string;
  recommendation?: string;
};

export type CallRecord = {
  id: string;
  filename: string;
  agent: string;
  cluster: string;
  duration: string;
  date: string;
  emotion: string;
  risk: string;
  recommendation: string;
  result?: AnalysisResult;
};
