export type Severity = 'Critical' | 'High' | 'Medium' | 'Low';

export interface Bug {
  jira_key: string;
  summary: string;
  predicted_severity: Severity;
  severity_confidence: number;
  impact_score: number;
  rank: number;
  status: 'Open' | 'In Progress' | 'Resolved';
  component?: string;
  reporter?: string;
  description?: string;
  created_at?: string;
}

export interface ModelMetrics {
  last_trained: string;
  version: string;
  f1_macro: number;
  dataset_size: number;
  status: 'idle' | 'training' | 'completed';
}

export interface DashboardStats {
  highest_priority: Bug;
  high_risk_count: number;
  low_impact_count: number;
  avg_confidence: number;
}

export interface Insight {
  type: 'component' | 'reporter' | 'trend' | 'suggestion';
  title: string;
  description: string;
  value?: string;
}