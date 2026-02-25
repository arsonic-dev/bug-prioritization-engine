import { motion } from 'framer-motion';
import { TrendingDown, TrendingUp } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ReactNode;
  variant?: 'default' | 'critical' | 'warning' | 'success';
  trend?: 'up' | 'down';
  trendValue?: string;
}

const variants = {
  default: 'bg-white/5 border-white/10 hover:bg-white/10',
  critical: 'bg-red-500/10 border-red-500/20 hover:bg-red-500/20',
  warning: 'bg-orange-500/10 border-orange-500/20 hover:bg-orange-500/20',
  success: 'bg-emerald-500/10 border-emerald-500/20 hover:bg-emerald-500/20',
};

export function StatCard({ title, value, subtitle, icon, variant = 'default', trend, trendValue }: StatCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.01 }}
      className={`relative overflow-hidden rounded-2xl border backdrop-blur-md p-6 transition-colors duration-300 ${variants[variant]}`}
    >
      <div className="flex justify-between items-start mb-4">
        <div className={`p-2.5 rounded-xl bg-white/5 ${variant === 'critical' ? 'text-red-400' : variant === 'warning' ? 'text-orange-400' : variant === 'success' ? 'text-emerald-400' : 'text-blue-400'}`}>
          {icon}
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full bg-white/5 ${trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
            {trend === 'up' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
            {trendValue}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-slate-400 text-sm font-medium mb-1">{title}</h3>
        <div className="text-2xl font-bold text-white mb-1 tracking-tight">{value}</div>
        <p className="text-xs text-slate-500 truncate">{subtitle}</p>
      </div>

      {/* Decorative Glow */}
      <div className="absolute -bottom-4 -right-4 w-24 h-24 bg-white/5 rounded-full blur-2xl pointer-events-none" />
    </motion.div>
  );
}