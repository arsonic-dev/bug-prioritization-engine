export interface Bug {
  id: string;
  jira_key: string;
  summary: string;
  priority_score: number;
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  status: string;
  assignee: string;
  created_at: string;
  ai_confidence: number;
}

export interface DashboardStats {
  highest_priority: {
    jira_key: string;
    summary: string;
  };
  high_risk_count: number;
  low_impact_count: number;
  avg_confidence: number;
}

export interface Insight {
  id: string;
  type: 'trend' | 'anomaly' | 'suggestion';
  message: string;
  timestamp: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_trained: string;
}