import { motion } from 'framer-motion';
import type { Bug } from '@/types';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ArrowUpRight } from 'lucide-react';

interface RankedTableProps {
  bugs: Bug[];
  onBugClick: (bug: Bug) => void;
  loading: boolean;
}

const severityColors = {
  Critical: 'bg-red-500/10 text-red-400 border-red-500/20',
  High: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  Medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  Low: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
};

export function RankedTable({ bugs, onBugClick, loading }: RankedTableProps) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-16 w-full bg-white/5 rounded-xl" />
        ))}
      </div>
    );
  }

  return (
    <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-white/5 text-slate-400 border-b border-white/10">
            <tr>
              <th className="px-6 py-4 font-medium">Rank</th>
              <th className="px-6 py-4 font-medium">Issue Key</th>
              <th className="px-6 py-4 font-medium">Summary</th>
              <th className="px-6 py-4 font-medium">Severity</th>
              <th className="px-6 py-4 font-medium">AI Score</th>
              <th className="px-6 py-4 font-medium">Status</th>
              <th className="px-6 py-4 font-medium"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {bugs.map((bug, index) => (
              <motion.tr
                key={bug.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => onBugClick(bug)}
                className="group hover:bg-white/5 transition-colors cursor-pointer"
              >
                <td className="px-6 py-4">
                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-white/5 text-slate-300 font-mono text-xs border border-white/10">
                    {index + 1}
                  </span>
                </td>
                <td className="px-6 py-4 font-medium text-blue-400">{bug.jira_key}</td>
                <td className="px-6 py-4 text-slate-300 max-w-xs truncate">{bug.summary}</td>
                <td className="px-6 py-4">
                  <Badge variant="outline" className={`${severityColors[bug.severity]} border text-xs`}>
                    {bug.severity}
                  </Badge>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                        style={{ width: `${bug.priority_score}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400">{bug.priority_score}</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="text-xs px-2 py-1 rounded-md bg-white/5 text-slate-400 border border-white/5">
                    {bug.status}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <ArrowUpRight size={16} className="text-slate-500 group-hover:text-white transition-colors" />
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}