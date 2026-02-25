import { motion } from 'framer-motion';
import type { Insight } from '@/types';
import { Lightbulb, AlertCircle, TrendingUp } from 'lucide-react';

interface AIInsightsProps {
  insights: Insight[];
}

const icons = {
  trend: TrendingUp,
  anomaly: AlertCircle,
  suggestion: Lightbulb,
};

const colors = {
  trend: 'text-blue-400 bg-blue-400/10',
  anomaly: 'text-red-400 bg-red-400/10',
  suggestion: 'text-amber-400 bg-amber-400/10',
};

export function AIInsights({ insights }: AIInsightsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {insights.map((insight, idx) => {
        const Icon = icons[insight.type];
        return (
          <motion.div
            key={insight.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="p-5 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all group"
          >
            <div className="flex items-start gap-4">
              <div className={`p-3 rounded-xl ${colors[insight.type]} bg-opacity-10`}>
                <Icon size={20} className={colors[insight.type].split(' ')[0]} />
              </div>
              <div>
                <h4 className="text-sm font-medium text-slate-200 mb-1 capitalize">{insight.type}</h4>
                <p className="text-sm text-slate-400 leading-relaxed">{insight.message}</p>
                <span className="text-xs text-slate-600 mt-3 block">{insight.timestamp}</span>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}