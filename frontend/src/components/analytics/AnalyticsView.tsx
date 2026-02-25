import { Card } from '@/components/ui/card';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const severityData = [
  { name: 'Critical', value: 12, color: '#ef4444' },
  { name: 'High', value: 28, color: '#f97316' },
  { name: 'Medium', value: 45, color: '#eab308' },
  { name: 'Low', value: 35, color: '#22c55e' },
];

const componentData = [
  { name: 'Backend API', failures: 42 },
  { name: 'Frontend', failures: 28 },
  { name: 'Database', failures: 15 },
  { name: 'Auth', failures: 12 },
  { name: 'Mobile', failures: 8 },
];

const agingData = [
  { days: '0-3', count: 45 },
  { days: '4-7', count: 32 },
  { days: '8-14', count: 28 },
  { days: '15-30', count: 15 },
  { days: '30+', count: 8 },
];

const resolutionData = [
  { priority: 'Critical', hours: 4.2 },
  { priority: 'High', hours: 12.5 },
  { priority: 'Medium', hours: 48.3 },
  { priority: 'Low', hours: 120.5 },
];

export function AnalyticsView() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-6">
        <Card className="p-6 bg-slate-900/50 border-slate-800">
          <h3 className="text-lg font-semibold text-white mb-4">Severity Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 bg-slate-900/50 border-slate-800">
          <h3 className="text-lg font-semibold text-white mb-4">Component Failure Rate</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={componentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="failures" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 bg-slate-900/50 border-slate-800">
          <h3 className="text-lg font-semibold text-white mb-4">Bug Aging Histogram</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={agingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="days" tick={{ fill: '#94a3b8' }} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 bg-slate-900/50 border-slate-800">
          <h3 className="text-lg font-semibold text-white mb-4">Resolution Time by Priority</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={resolutionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="priority" tick={{ fill: '#94a3b8' }} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  itemStyle={{ color: '#e2e8f0' }}
                  formatter={(value: number) => [`${value} hours`, 'Avg Resolution']}
                />
                <Bar dataKey="hours" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}