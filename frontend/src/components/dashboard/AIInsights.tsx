import type { Insight } from '@/types';
import { Card } from '@/components/ui/card';
import { TrendingUp, Users, Component, Lightbulb } from 'lucide-react';

interface AIInsightsProps {
  insights: Insight[];
}

const typeIcons = {
  component: <Component size={20} />,
  reporter: <Users size={20} />,
  trend: <TrendingUp size={20} />,
  suggestion: <Lightbulb size={20} />,
};

const typeColors = {
  component: 'text-purple-400 bg-purple-500/10',
  reporter: 'text-blue-400 bg-blue-500/10',
  trend: 'text-green-400 bg-green-500/10',
  suggestion: 'text-yellow-400 bg-yellow-500/10',
};

export function AIInsights({ insights }: AIInsightsProps) {
  return (
    <div className="grid grid-cols-4 gap-4">
      {insights.map((insight, idx) => (
        <Card key={idx} className="p-5 bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-colors">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-3 ${typeColors[insight.type]}`}>
            {typeIcons[insight.type]}
          </div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">{insight.title}</p>
          <h4 className="text-lg font-semibold text-white mb-1">{insight.description}</h4>
          {insight.value && (
            <p className="text-sm text-slate-400">{insight.value}</p>
          )}
        </Card>
      ))}
    </div>
  );
}