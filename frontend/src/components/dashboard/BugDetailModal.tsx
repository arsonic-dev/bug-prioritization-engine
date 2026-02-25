import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import type { Bug } from '@/types';
import { getSeverityColor, formatConfidence } from '@/lib/utils';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';

interface BugDetailModalProps {
  bug: Bug | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const severityProbabilities = [
  { name: 'Critical', value: 0.15, color: '#ef4444' },
  { name: 'High', value: 0.65, color: '#f97316' },
  { name: 'Medium', value: 0.15, color: '#eab308' },
  { name: 'Low', value: 0.05, color: '#22c55e' },
];

const shapFeatures = [
  { feature: 'Stack trace contains "NullPointerException"', impact: 0.32, direction: 'positive' },
  { feature: 'Reported by senior engineer', impact: 0.18, direction: 'positive' },
  { feature: 'Component: Payment Gateway', impact: 0.15, direction: 'positive' },
  { feature: 'Bug age < 24 hours', impact: -0.08, direction: 'negative' },
];

export function BugDetailModal({ bug, open, onOpenChange }: BugDetailModalProps) {
  if (!bug) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl bg-slate-950 border-slate-800 text-slate-200 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="font-mono text-blue-400">{bug.jira_key}</span>
                <Badge variant="outline" className={getSeverityColor(bug.predicted_severity)}>
                  {bug.predicted_severity}
                </Badge>
              </div>
              <DialogTitle className="text-2xl text-white">{bug.summary}</DialogTitle>
            </div>
            <div className="text-right">
              <p className="text-sm text-slate-400">AI Rank</p>
              <p className="text-3xl font-bold text-white">#{bug.rank}</p>
            </div>
          </div>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-6 mt-6">
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium text-slate-400 mb-3">Severity Probability Distribution</h4>
              <div className="h-48 bg-slate-900/50 rounded-lg p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={severityProbabilities}>
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      axisLine={{ stroke: '#334155' }}
                    />
                    <YAxis
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      axisLine={{ stroke: '#334155' }}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {severityProbabilities.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-slate-400 mb-3">Why This Ranking?</h4>
              <div className="space-y-2">
                {shapFeatures.map((feature, idx) => (
                  <div key={idx} className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                    <div className={cn(
                      'w-2 h-2 rounded-full',
                      feature.direction === 'positive' ? 'bg-red-400' : 'bg-green-400'
                    )} />
                    <div className="flex-1">
                      <p className="text-sm text-slate-300">{feature.feature}</p>
                    </div>
                    <span className={cn(
                      'text-sm font-mono',
                      feature.direction === 'positive' ? 'text-red-400' : 'text-green-400'
                    )}>
                      {feature.direction === 'positive' ? '+' : ''}{feature.impact.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-800">
              <h4 className="text-sm font-medium text-slate-400 mb-3">Bug Details</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-500">Component</span>
                  <span className="text-slate-200">{bug.component}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Reporter</span>
                  <span className="text-slate-200">{bug.reporter}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Created</span>
                  <span className="text-slate-200">{new Date(bug.created_at || '').toLocaleDateString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Status</span>
                  <span className="text-slate-200">{bug.status}</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-slate-400 mb-3">Similar Historical Bugs</h4>
              <div className="space-y-2">
                {['SCRUM-12', 'SCRUM-28', 'SCRUM-31'].map((key, idx) => (
                  <div key={key} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg hover:bg-slate-800/50 transition-colors cursor-pointer">
                    <div>
                      <span className="font-mono text-sm text-blue-400">{key}</span>
                      <p className="text-xs text-slate-500 mt-0.5">Same component, similar stack trace</p>
                    </div>
                    <Badge variant="outline" className="bg-green-500/10 text-green-400 text-xs">
                      {(90 - idx * 5)}% match
                    </Badge>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 bg-blue-950/20 border border-blue-900/30 rounded-lg">
              <h4 className="text-sm font-medium text-blue-400 mb-2">AI Recommendation</h4>
              <p className="text-sm text-slate-300">
                Prioritize this bug for the next sprint. The combination of high severity confidence
                ({formatConfidence(bug.severity_confidence)}) and impact score ({bug.impact_score})
                indicates significant user impact potential.
              </p>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}