import { cn } from '@/lib/utils';
import { Card } from '@/components/ui/card';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  variant?: 'default' | 'critical' | 'warning' | 'success';
}

export function StatCard({
  title,
  value,
  subtitle,
  icon,
  trend,
  trendValue,
  variant = 'default'
}: StatCardProps) {
  const variants = {
    default: 'bg-slate-900/50 border-slate-800',
    critical: 'bg-red-950/20 border-red-900/30',
    warning: 'bg-orange-950/20 border-orange-900/30',
    success: 'bg-green-950/20 border-green-900/30',
  };

  return (
    <Card className={cn('p-6 border backdrop-blur-sm', variants[variant])}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-400 mb-1">{title}</p>
          <h3 className="text-3xl font-bold text-white tracking-tight">{value}</h3>
          {subtitle && <p className="text-sm text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <div className="p-3 rounded-xl bg-slate-800/50 text-slate-300">
          {icon}
        </div>
      </div>
      {trend && (
        <div className="mt-4 flex items-center gap-2">
          <span className={cn(
            'text-xs font-medium',
            trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-slate-400'
          )}>
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'} {trendValue}
          </span>
          <span className="text-xs text-slate-500">vs last week</span>
        </div>
      )}
    </Card>
  );
}