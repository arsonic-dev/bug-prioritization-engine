import { cn } from "@/lib/utils"
import { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import type { Bug } from '@/types';
import { getSeverityColor, formatConfidence } from '@/lib/utils';
import { Flame, AlertTriangle, Zap, Check, ExternalLink } from 'lucide-react';

interface RankedTableProps {
  bugs: Bug[];
  onBugClick: (bug: Bug) => void;
  loading?: boolean;
}

const severityIcons = {
  Critical: <Flame size={16} />,
  High: <AlertTriangle size={16} />,
  Medium: <Zap size={16} />,
  Low: <Check size={16} />,
};

export function RankedTable({ bugs, onBugClick, loading }: RankedTableProps) {
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-16 bg-slate-900/50 rounded-lg animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-900/30">
      <Table>
        <TableHeader>
          <TableRow className="border-slate-800 hover:bg-transparent">
            <TableHead className="w-16 text-slate-400 font-medium">Rank</TableHead>
            <TableHead className="text-slate-400 font-medium">Jira Key</TableHead>
            <TableHead className="text-slate-400 font-medium">Summary</TableHead>
            <TableHead className="text-slate-400 font-medium">AI Severity</TableHead>
            <TableHead className="text-slate-400 font-medium">Confidence</TableHead>
            <TableHead className="text-slate-400 font-medium">Impact</TableHead>
            <TableHead className="text-slate-400 font-medium">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {bugs.map((bug) => (
            <TableRow
              key={bug.jira_key}
              className="border-slate-800/50 cursor-pointer transition-all duration-200"
              style={{
                backgroundColor: hoveredRow === bug.jira_key ? 'rgba(30, 41, 59, 0.5)' : 'transparent',
              }}
              onMouseEnter={() => setHoveredRow(bug.jira_key)}
              onMouseLeave={() => setHoveredRow(null)}
              onClick={() => onBugClick(bug)}
            >
              <TableCell className="font-mono text-slate-500">#{bug.rank}</TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm text-blue-400">{bug.jira_key}</span>
                  <ExternalLink size={14} className="text-slate-600" />
                </div>
              </TableCell>
              <TableCell className="max-w-md">
                <p className="text-slate-200 truncate">{bug.summary}</p>
                <p className="text-xs text-slate-500 mt-0.5">{bug.component}</p>
              </TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={`${getSeverityColor(bug.predicted_severity)} flex items-center gap-1.5 px-2.5 py-1`}
                >
                  {severityIcons[bug.predicted_severity]}
                  {bug.predicted_severity}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="w-full max-w-[120px]">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">{formatConfidence(bug.severity_confidence)}</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        'h-full rounded-full transition-all duration-500',
                        bug.severity_confidence > 0.8 ? 'bg-green-500' :
                        bug.severity_confidence > 0.6 ? 'bg-blue-500' :
                        bug.severity_confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                      )}
                      style={{ width: `${bug.severity_confidence * 100}%` }}
                    />
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <div className="w-24">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">{bug.impact_score}</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                      style={{ width: `${bug.impact_score}%` }}
                    />
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <span className={cn(
                  'text-xs px-2.5 py-1 rounded-full',
                  bug.status === 'Open' ? 'bg-red-500/10 text-red-400' :
                  bug.status === 'In Progress' ? 'bg-yellow-500/10 text-yellow-400' :
                  'bg-green-500/10 text-green-400'
                )}>
                  {bug.status}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}